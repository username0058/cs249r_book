# -*- coding: utf-8 -*-
"""
djr_tinygpt_perf/task3_compression/run_pruning.py
=====================================================================
Task 3 Compression —— 移除冗余(幅度剪枝 / 结构化剪枝 / 重训练)
=====================================================================
【实验设计】(对应 issue 任务 1/2 与产出 1/2/3/4)
  1. 非结构化幅度剪枝:框架 magnitude_prune(全局按 |w| 排序置零),
     剪枝率 30% / 50% / 70%:
       - 记录稀疏度、held-out loss/PPL(精度损失)
       - 掩码微调(masked fine-tune,本脚本实现):剪掉的权重在微调中
         保持为 0(Han et al. 2015 的标准流程),观察精度恢复
  2. 结构化剪枝:框架 structured_prune(按输出通道 L2 范数整列置零),
     同样 30/50/70%。注意:框架实现是「置零」式,NumPy 稠密 matmul
     不会因零变快 —— 所以另做「物理裁剪」实验:真正删除神经元维度
     (收缩矩阵形状),实测 forward 加速,验证结构化剪枝的速度收益。
  3. 速度测量:
       - 非结构化:预测无加速(零仍参与乘加),实测验证;
       - 结构化物理裁剪:MLP 256 -> 128 隐层,实测单层/整块加速。
  4. 图表:pruning_accuracy.png(精度 vs 剪枝率曲线)
  5. 输出:pruning_table.csv / pruning_summary.json / stdout 结论

【与原代码关系】仅 import tinytorch.perf.compression 公开 API;
  掩码微调、物理裁剪与全部实验代码为本目录新增。
=====================================================================
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from common import (get_trained_model, eval_heldout, build_model, load_weights,
                    build_data, SEQ_LEN, BATCH_SIZE, count_params, CACHE_NPZ)
from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.layers import Linear
from tinytorch.core.optimizers import AdamW
from tinytorch.perf.compression import (magnitude_prune, structured_prune,
                                        measure_sparsity)

OUT = HERE
SPARSITIES = [0.3, 0.5, 0.7]
FT_EPOCHS = 3          # 剪枝后的微调轮数(重训练流程演示)


class LayerList:
    """结构化剪枝的适配器:structured_prune 通过 .layers 递归收集 Linear,
    TinyGPT 不是 Sequential,这里提供一个最小容器视图。
    顺序注意:lm_head 放最后 —— 框架约定最后一个 Linear 是分类头不剪。"""

    def __init__(self, linears):
        self.layers = linears


def tinygpt_linears(model):
    """按块收集 TinyGPT 的全部 Linear(顺序:q,k,v,out, mlp1, mlp2, ... lm_head)。"""
    lins = []
    for blk in model.blocks:
        a, m = blk.attention, blk.mlp
        lins += [a.q_proj, a.k_proj, a.v_proj, a.out_proj,
                 m.linear1, m.linear2]
    lins.append(model.lm_head)
    return lins


# ---------------------------------------------------------------------------
# 掩码微调:剪掉的权重在训练中保持为 0
# ---------------------------------------------------------------------------
def snapshot_masks(model):
    """记录当前非零位置掩码(剪枝后的稀疏模式)。"""
    masks = []
    for p in model.parameters():
        if p.ndim > 1:                       # 只管权重矩阵(与剪枝口径一致)
            masks.append((p, (p.data != 0).astype(np.float32)))
    return masks


def finetune_masked(model, dl, epochs, masks, lr=1e-3):
    """带掩码的微调:每步 opt.step() 后把被剪位置重新置零。

    lr 用比训练更低的学习率(1e-3):微调只需小幅恢复,避免权重分布漂移。
    """
    criterion = CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    for ep in range(epochs):
        ep_loss, n = 0.0, 0
        for xb, yb in dl:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            loss.backward()
            opt.step()
            for p, m in masks:               # 重新应用剪枝掩码
                p.data *= m
            ep_loss += float(loss.data); n += 1
        print(f"    [finetune] epoch {ep+1}/{epochs} loss={ep_loss/max(n,1):.4f}",
              flush=True)


# ---------------------------------------------------------------------------
# 物理结构化裁剪:真正删除 MLP 隐藏神经元维度(演示真实加速)
# ---------------------------------------------------------------------------
def physically_shrink_mlp(model, keep_ratio):
    """按 L2 范数保留每个 MLP 隐层的前 keep_ratio 比例神经元。

    与框架「置零」不同,这里真正收缩矩阵。
    存储约定:框架 Linear(in,out) 的 weight.data 形状是 (in, out),
    因此 MLP 的隐藏神经元是 linear1 的【列】/linear2 的【行】:
      linear1: (64,256) -> (64,K)   (按列裁)
      linear2: (256,64) -> (K,64)   (按行裁)
    K = 256*keep_ratio;计算量/内存同步减少 -> 能测到真实加速。
    """
    kept_info = []
    for bi, blk in enumerate(model.blocks):
        m = blk.mlp
        W1 = m.linear1.weight.data            # (in=64, out=256)
        norm = np.linalg.norm(W1, axis=0)     # 每个输出神经元(列)的 L2 重要性
        K = max(1, int(W1.shape[1] * keep_ratio))
        keep = np.argsort(norm, kind="stable")[::-1][:K]
        keep = np.sort(keep)                  # 保持顺序,减少缓存抖动

        new1 = Linear(W1.shape[0], K, bias=True)
        new1.weight.data = W1[:, keep].copy()             # (64, K)
        new1.bias.data = m.linear1.bias.data[keep].copy()  # bias 按 out 对齐
        W2 = m.linear2.weight.data            # (in=256, out=64)
        new2 = Linear(K, W2.shape[1], bias=True)
        new2.weight.data = W2[keep, :].copy()             # (K, 64)
        new2.bias.data = m.linear2.bias.data.copy()       # (64,) 输出侧不变
        m.linear1, m.linear2 = new1, new2
        kept_info.append((bi, K))
    return kept_info


def bench_mlp(model, x, warmup=5, iters=50):
    """测量整模型前向(主要变化在 MLP)的 wall time。"""
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


def fresh_model(vocab):
    m = build_model(vocab)
    load_weights(m, CACHE_NPZ)
    return m


def main():
    model0, tok, hx, hy, = get_trained_model()
    vocab = tok.vocab_size
    _, train_dl, _, _ = build_data()

    base_loss, base_ppl = eval_heldout(model0, hx, hy)
    n_params = count_params(model0)
    print(f"[prune] FP32 基线: loss={base_loss:.4f} ppl={base_ppl:.2f} "
          f"params={n_params:,}")

    # ---------------------------------------------------------------------------
    # 1) 非结构化幅度剪枝 + 掩码微调
    # ---------------------------------------------------------------------------
    results = []
    for s in SPARSITIES:
        m = fresh_model(vocab)
        magnitude_prune(m, sparsity=s)
        sp = measure_sparsity(m)
        l0, p0 = eval_heldout(m, hx, hy)
        # 掩码微调
        masks = snapshot_masks(m)
        _, dl, _, _ = build_data()
        finetune_masked(m, dl, FT_EPOCHS, masks)
        l1, p1 = eval_heldout(m, hx, hy)
        results.append(dict(method="magnitude", rate=s, sparsity=sp,
                            loss_after=l0, ppl_after=p0,
                            loss_ft=l1, ppl_ft=p1))
        print(f"[prune] 幅度 s={s:.0%}: sparsity={sp:.1f}% "
              f"loss {l0:.4f}(Δ{l0-base_loss:+.3f}) -> 微调后 {l1:.4f}"
              f"(Δ{l1-base_loss:+.3f})")

    # ---------------------------------------------------------------------------
    # 2) 结构化剪枝(置零版)+ 掩码微调
    # ---------------------------------------------------------------------------
    for s in SPARSITIES:
        m = fresh_model(vocab)
        structured_prune(LayerList(tinygpt_linears(m)), prune_ratio=s)
        sp = measure_sparsity(m)
        l0, p0 = eval_heldout(m, hx, hy)
        masks = snapshot_masks(m)
        _, dl, _, _ = build_data()
        finetune_masked(m, dl, FT_EPOCHS, masks)
        l1, p1 = eval_heldout(m, hx, hy)
        results.append(dict(method="structured", rate=s, sparsity=sp,
                            loss_after=l0, ppl_after=p0,
                            loss_ft=l1, ppl_ft=p1))
        print(f"[prune] 结构 s={s:.0%}: sparsity={sp:.1f}% "
              f"loss {l0:.4f}(Δ{l0-base_loss:+.3f}) -> 微调后 {l1:.4f}"
              f"(Δ{l1-base_loss:+.3f})")

    # CSV 留档
    with open(os.path.join(OUT, "pruning_table.csv"), "w", encoding="utf-8") as f:
        f.write("method,rate,sparsity,loss_after,ppl_after,loss_finetuned,"
                "ppl_finetuned,baseline_loss\n")
        for r in results:
            f.write(f"{r['method']},{r['rate']},{r['sparsity']:.4f},"
                    f"{r['loss_after']:.4f},{r['ppl_after']:.2f},"
                    f"{r['loss_ft']:.4f},{r['ppl_ft']:.2f},{base_loss:.4f}\n")

    # ---------------------------------------------------------------------------
    # 3) 速度:置零式 vs 物理裁剪
    # ---------------------------------------------------------------------------
    x_pre = Tensor(np.random.randint(0, vocab,
                  size=(BATCH_SIZE, SEQ_LEN)).astype(np.int32))
    x_dec = Tensor(np.random.randint(0, vocab, size=(1, 1)).astype(np.int32))

    m_dense = fresh_model(vocab)
    t_dense_pre = bench_mlp(m_dense, x_pre)
    t_dense_dec = bench_mlp(m_dense, x_dec)

    # 置零式(非结构化 50%):预期无加速(零仍参与稠密乘加)
    m_zero = fresh_model(vocab)
    magnitude_prune(m_zero, sparsity=0.5)
    t_zero_pre = bench_mlp(m_zero, x_pre)
    t_zero_dec = bench_mlp(m_zero, x_dec)

    # 物理裁剪(MLP 隐层 256->128):预期 MLP 部分接近 2× 加速
    m_phys = fresh_model(vocab)
    physically_shrink_mlp(m_phys, keep_ratio=0.5)
    t_phys_pre = bench_mlp(m_phys, x_pre)
    t_phys_dec = bench_mlp(m_phys, x_dec)
    lp, pp = eval_heldout(m_phys, hx, hy)
    print(f"[prune] 速度(pre/decode ms): dense={t_dense_pre:.2f}/{t_dense_dec:.2f} "
          f"| 置零50%={t_zero_pre:.2f}/{t_zero_dec:.2f} "
          f"| 物理裁剪50%={t_phys_pre:.2f}/{t_phys_dec:.2f}")
    print(f"[prune] 物理裁剪后精度: loss={lp:.4f} ppl={pp:.2f}")

    # ---------------------------------------------------------------------------
    # 4) 图表:精度 vs 剪枝率
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.axhline(base_loss, color="gray", ls=":", label=f"FP32 baseline {base_loss:.3f}")
    for method, color, marker in (("magnitude", "#2563eb", "o"),
                                  ("structured", "#dc2626", "s")):
        rs = sorted([r for r in results if r["method"] == method],
                    key=lambda r: r["rate"])
        rates = [0.0] + [r["rate"] for r in rs]
        l_after = [base_loss] + [r["loss_after"] for r in rs]
        l_ft = [base_loss] + [r["loss_ft"] for r in rs]
        ax.plot(rates, l_after, marker + "-", color=color,
                label=f"{method} pruned")
        ax.plot(rates, l_ft, marker + "--", color=color, alpha=0.6,
                label=f"{method} + masked FT {FT_EPOCHS}ep")
    ax.set_xlabel("pruning rate")
    ax.set_ylabel("held-out cross-entropy (nats/token)")
    ax.set_title("TinyGPT pruning: accuracy loss and FT recovery\n"
                 "(30%/50%/70%; magnitude=global |w| zeroing, structured=channel-L2 zeroing)")
    ax.set_xticks([0, 0.3, 0.5, 0.7],
                  ["0%", "30%", "50%", "70%"])
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pruning_accuracy.png"))
    plt.close(fig)
    print("saved:", os.path.join(OUT, "pruning_accuracy.png"))

    # ---------------------------------------------------------------------------
    # 5) 汇总
    # ---------------------------------------------------------------------------
    summary = {
        "baseline": {"loss": base_loss, "ppl": base_ppl, "params": n_params},
        "experiments": results,
        "speed": {
            "dense": {"prefill_ms": t_dense_pre, "decode_ms": t_dense_dec},
            "zeroed_50": {"prefill_ms": t_zero_pre, "decode_ms": t_zero_dec,
                          "note": "NumPy 稠密 matmul 不因零变快 —— 非结构化剪枝"
                                  "需要稀疏内核(如 CSR/spmm)才有速度收益"},
            "physical_50": {"prefill_ms": t_phys_pre, "decode_ms": t_phys_dec,
                            "note": "真实收缩 MLP 256->128,计算量减半",
                            "heldout_loss": lp, "ppl": pp},
        },
    }
    with open(os.path.join(OUT, "pruning_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved:", os.path.join(OUT, "pruning_summary.json"))


if __name__ == "__main__":
    main()
