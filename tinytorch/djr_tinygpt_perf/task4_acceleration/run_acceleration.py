# -*- coding: utf-8 -*-
"""
djr_tinygpt_perf/task4_acceleration/run_acceleration.py
=====================================================================
Task 4 Acceleration —— 算子融合与向量化(issue #33)
=====================================================================
【实验设计】(对应 issue 任务 1/2 与产出 1/2/3)
  1. 融合算子(推理路径,只动 .data,不记 autograd —— 与生产推理内核同思路):
       - fused_gelu_raw:与训练用 GELU 模块【完全同公式】(x·sigmoid(1.702x)),
         去掉 autograd/Tensor 包装 —— 公平对比,唯一差异=融合本身。
         (框架自带的 fused_gelu 用 tanh 更精确公式,在 NumPy 上反而更慢,
         见 fusion_micro.png 第三组柱 —— 课程文档自己也注明了这一点)
       - fused_add_norm:自研。融合 Pre-LN 块的「残差相加 + LayerNorm」,
         省一次全数组读写 + 省 autograd 记录与临时 Tensor 对象
       - fused block:把上面粘进 TransformerBlock 推理路径,
         输出与原块数值一致性校验(应仅浮点顺序差异,<1e-4)
  2. 逐 token 延迟对比(任务1):
       - ITL:固定上下文 (B=1,S=32) 的纯模型前向 —— 注意 generate() 端到端
         循环含采样/解码 Python 开销,会淹没融合收益,故拆开测
       - Prefill(B=32,S=32):逐 token 延迟 = wall / (B*S)
  3. 任务2:融合收益 Decode vs Prefill 的不对称 + 微基准解释机制。

【与 Task 1 的衔接】Task 1 实测插桩累计 18.56ms vs 纯净 wall 14.93ms,
  差额 ≈35% 是 Python 调度/临时对象开销 —— 融合吃掉的正是这块。
【输出】fusion_micro.png / fusion_model.png / fusion_summary.json
【与原代码关系】只 import tinytorch 公开 API;融合实现均为本目录新增。
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

from common import get_trained_model, SEQ_LEN, BATCH_SIZE, count_params
from tinytorch.core.tensor import Tensor
from tinytorch.core.transformers import GELU
from tinytorch.perf.acceleration import fused_gelu

OUT = HERE


# ===========================================================================
# 1. 融合算子(推理专用,raw numpy)
# ===========================================================================
def _sigmoid(x):
    """朴素 sigmoid(1/(1+exp(-x)),errstate 压溢出)—— 与模块路径数值等价:
    x 很负时 exp(-x)->inf -> sig->0;x 很正时 exp(-x)->0 -> sig->1,
    饱和端结果与「稳定版」完全一致,但只有一次 exp 全数组扫描(无掩码索引)。"""
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-x))


def fused_gelu_raw(x: np.ndarray) -> np.ndarray:
    """GELU 快速近似 x·sigmoid(1.702x),与模块 GELUFunction 完全同公式,
    但去掉 autograd 记录与 Tensor 包装 —— 唯一差异=融合本身。"""
    with np.errstate(over="ignore"):
        return x * _sigmoid(1.702 * x)


def fused_ln_raw(x: np.ndarray, ln) -> np.ndarray:
    """裸 LayerNorm(raw numpy):y=(x-μ)/σ·γ+β,不记 autograd。"""
    mean = x.mean(axis=-1, keepdims=True)
    d = x - mean
    var = np.mean(d * d, axis=-1, keepdims=True)
    d /= np.sqrt(var + ln.eps)
    return d * ln.gamma.data + ln.beta.data


def fused_add_norm(x: Tensor, residual: Tensor, ln):
    """融合「残差相加 + LayerNorm」(Pre-LN 块推理路径)。

    数学:y = LN(residual + x)
      unfused:Tensor 加法(临时 Tensor + autograd 记录) → LN 模块(再包装)
               ≈ 多次 Python 调度 + 2 次临时数组 + 中间结果全数组读写
      fused  :一次加法 + 原地中心化 + 一次写出,Python 层单次调用
    """
    t = residual.data + x.data
    return Tensor(fused_ln_raw(t, ln))


def make_fused_block_forward(blk):
    """返回融合版 TransformerBlock.forward(闭包捕获子模块)。

    Pre-LN:fused_ln1 → attention → 残差(与 LN2 融合) → fused_mlp → 残差加。
    注意残差状态:原块在 MLP 子层前把 x 更新为 x+attn_out,
    输出 = (x+attn_out) + mlp_out —— 融合版必须保持同一数据流。
    """
    def forward(x, mask=None):
        normed1 = Tensor(fused_ln_raw(x.data, blk.ln1))
        attn_out = blk.attention.forward(normed1, mask)
        residual = x.data + attn_out.data          # 残差状态更新
        normed2 = Tensor(fused_ln_raw(residual, blk.ln2))
        h = blk.mlp.linear1.forward(normed2)
        h = Tensor(fused_gelu_raw(h.data))         # 融合 GELU(同公式)
        mlp_out = blk.mlp.linear2.forward(h)
        return Tensor(residual + mlp_out.data)
    return forward


def fuse_model(model):
    """把每个 block.forward 换成融合版(实例属性遮蔽,可还原)。"""
    saved = []
    for blk in model.blocks:
        saved.append((blk, blk.forward))
        blk.forward = make_fused_block_forward(blk)
    return saved


def unfuse_model(saved):
    for blk, orig in saved:
        blk.forward = orig


# ===========================================================================
# 2. 计时工具
# ===========================================================================
def bench(fn, warmup=5, iters=30):
    """返回中位数 wall(ms)—— 中位数抗服务器偶发抖动。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


def bench_forward_tokens(model, batch, seq, warmup=5, iters=30):
    """纯模型前向(B,S)的 wall 中位数 —— ITL 用 (1,32) 固定上下文。"""
    x = Tensor(np.random.randint(0, 59, size=(batch, seq)).astype(np.int32))
    return bench(lambda: model(x), warmup=warmup, iters=iters)


# ===========================================================================
# 3. 主流程
# ===========================================================================
def main():
    model, tok, hx, hy = get_trained_model()
    vocab = tok.vocab_size
    print(f"[accel] params={count_params(model):,}")

    # ---------------- 数值一致性验证(融合前后输出应几乎相同) ----------------
    x = Tensor(np.random.randint(0, vocab, size=(2, SEQ_LEN)).astype(np.int32))
    y_base = model(x).data.copy()
    saved = fuse_model(model)
    y_fused = model(x).data
    max_diff = float(np.max(np.abs(y_base - y_fused)))
    rel = float(max_diff / (np.max(np.abs(y_base)) + 1e-9))
    unfuse_model(saved)
    print(f"[accel] 融合 vs 原始 输出最大绝对差={max_diff:.3e} (相对 {rel:.3e})")
    if max_diff > 1e-4:
        print("[accel] 数值不一致,终止(融合实现有误)")
        sys.exit(1)
    print("[accel] 一致(仅浮点运算顺序差异)√")

    # ---------------- 任务1:微基准(elementwise 算子,同公式公平对比) ----------------
    micro = {}
    gelu_mod = GELU()
    for label, shape, iters in (("GELU prefill (1024,256)", (1024, 256), 50),
                                ("GELU decode (1,256)", (1, 256), 300),
                                ("add+LN prefill (32,32,64)", (BATCH_SIZE, SEQ_LEN, 64), 50),
                                ("add+LN decode (1,1,64)", (1, 1, 64), 300)):
        is_gelu = label.startswith("GELU")
        xa = Tensor(np.random.randn(*shape).astype(np.float32))
        if is_gelu:
            t_unf = bench(lambda: gelu_mod.forward(xa), iters=iters)
            t_fus = bench(lambda: fused_gelu_raw(xa.data), iters=iters)
            t_frm = bench(lambda: fused_gelu(xa), iters=iters)   # 框架 tanh 式(参考)
            micro[label] = (t_unf, t_fus, t_frm)
            print(f"[accel] {label}: 模块(autograd)={t_unf:.3f}ms "
                  f"同公式raw={t_fus:.3f}ms({t_unf/t_fus:.2f}x) "
                  f"框架tanh式={t_frm:.3f}ms")
        else:
            blk = model.blocks[0]
            xb = Tensor(np.random.randn(*shape).astype(np.float32))
            def unfused_add_norm(_a=xa, _b=xb, _blk=blk):
                t = _a + _b                    # Tensor 加法(autograd 记录)
                return _blk.ln2.forward(t)
            t_unf = bench(unfused_add_norm, iters=iters)
            t_fus = bench(lambda: fused_add_norm(xa, xb, blk.ln2), iters=iters)
            micro[label] = (t_unf, t_fus, None)
            print(f"[accel] {label}: unfused={t_unf:.3f}ms "
                  f"fused={t_fus:.3f}ms({t_unf/t_fus:.2f}x)")

    # ---------------- 任务1:模型级逐 token 延迟 ----------------
    # 顺序注意:baseline(Prefill+ITL)全部测完再 fuse,避免状态串扰
    pre_base = bench_forward_tokens(model, BATCH_SIZE, SEQ_LEN)
    itl_base = bench_forward_tokens(model, 1, 32, iters=100)
    saved = fuse_model(model)
    pre_fused = bench_forward_tokens(model, BATCH_SIZE, SEQ_LEN)
    itl_fused = bench_forward_tokens(model, 1, 32, iters=100)
    unfuse_model(saved)
    tok_pre = BATCH_SIZE * SEQ_LEN
    print(f"[accel] Prefill: base={pre_base:.2f}ms fused={pre_fused:.2f}ms "
          f"({pre_base/pre_fused:.3f}x) | 逐token {pre_base/tok_pre*1000:.1f} -> "
          f"{pre_fused/tok_pre*1000:.1f} us")
    print(f"[accel] ITL(Decode, B=1,S=32): base={itl_base:.3f}ms "
          f"fused={itl_fused:.3f}ms ({itl_base/itl_fused:.3f}x)")

    # ---------------- 任务2:图表 ----------------
    # a) 微基准
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
    names = list(micro.keys())
    xs = np.arange(len(names))
    ax.bar(xs - 0.25, [micro[n][0] for n in names], 0.25,
           label="unfused (autograd modules)")
    ax.bar(xs, [micro[n][1] for n in names], 0.25,
           label="fused (raw numpy, same formula)")
    ax.bar(xs + 0.25, [micro[n][2] if micro[n][2] else 0 for n in names], 0.25,
           label="framework fused_gelu (tanh formula, ref)")
    ax.set_yscale("log")
    ax.set_xticks(xs, names, fontsize=7)
    ax.set_ylabel("op wall time (ms, log)")
    ax.set_title("Fusion microbenchmark: same-formula raw numpy vs autograd modules\n"
                 "(framework tanh fused_gelu shown for reference: different formula, "
                 "slower in NumPy)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fusion_micro.png")); plt.close(fig)
    print("saved:", os.path.join(OUT, "fusion_micro.png"))

    # b) 模型级两阶段对比
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    ax = axes[0]
    ax.bar(["unfused", "fused"], [itl_base, itl_fused],
           color=["#94a3b8", "#16a34a"])
    for i, v in enumerate([itl_base, itl_fused]):
        ax.text(i, v, f"{v:.3f} ms", ha="center", va="bottom")
    ax.set_ylabel("ITL per token (ms)")
    ax.set_title(f"Decode (B=1, ctx=32) pure forward: {itl_base/itl_fused:.2f}x\n"
                 "many small ops -> Python/alloc overhead dominates")
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    tok_lat_b = pre_base / tok_pre * 1000
    tok_lat_f = pre_fused / tok_pre * 1000
    ax.bar(["unfused", "fused"], [tok_lat_b, tok_lat_f],
           color=["#94a3b8", "#f59e0b"])
    for i, v in enumerate([tok_lat_b, tok_lat_f]):
        ax.text(i, v, f"{v:.1f} us", ha="center", va="bottom")
    ax.set_ylabel("per-token latency (us)")
    ax.set_title(f"Prefill (B=32,S=32): {pre_base/pre_fused:.2f}x\n"
                 "GEMM compute-bound -> elementwise fusion marginal")
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Operator fusion gains: Decode vs Prefill (task 2)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fusion_model.png"),
                                    bbox_inches="tight"); plt.close(fig)
    print("saved:", os.path.join(OUT, "fusion_model.png"))

    # ---------------- 汇总 ----------------
    with open(os.path.join(OUT, "fusion_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "numeric_check": {"max_abs_diff": max_diff, "rel": rel},
            "micro_ms": {k: {"unfused": float(v[0]), "fused": float(v[1]),
                             "framework_ref": (float(v[2]) if v[2] else None),
                             "speedup": float(v[0] / v[1])}
                         for k, v in micro.items()},
            "decode_itl_ms": {"unfused": itl_base, "fused": itl_fused,
                              "speedup": itl_base / itl_fused},
            "prefill_ms": {"unfused": pre_base, "fused": pre_fused,
                           "speedup": pre_base / pre_fused},
            "prefill_per_token_us": {"unfused": tok_lat_b, "fused": tok_lat_f},
            "notes": "融合=推理路径 raw numpy(无 autograd 记录),GELU 同公式公平对比;"
                     "ITL=固定上下文(1,32)纯模型前向;中位数计时",
        }, f, ensure_ascii=False, indent=2, default=float)
    print("saved:", os.path.join(OUT, "fusion_summary.json"))


if __name__ == "__main__":
    main()
