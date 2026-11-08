# -*- coding: utf-8 -*-
"""
djr_tinygpt_perf/task1_profiling/run_profiling.py
=====================================================================
Task 1 Profiling —— 看见 TinyGPT 的瓶颈
=====================================================================
【测量内容】(对应 issue 产出 3:各算子耗时占比图 + 显存占用热力图)
  A. 算子级耗时占比:包装 TinyGPT 每个子模块的 forward,用
     time.perf_counter() 累计各算子耗时。两种负载形态分开测:
       - Prefill(train-like):batch=32, seq=32,一次吃满序列 —— 计算密集
       - Decode(decode-like):batch=1, seq=1,逐 token 步进 —— 访存密集
  B. 内存(显存)热力图:逐层统计 权重/梯度/激活 三类内存占用(MB)。
     纯 NumPy 框架运行在 CPU 内存,但内存结构与 GPU 显存同构,
     结论可直接迁移到 GPU 场景。
  C. FLOPs 计数:逐算子解析式计数(与硬件无关),再结合实测耗时
     推出算术强度(FLOPs/Byte)与瓶颈类型(compute-bound /
     memory-bound),呼应 roofline 思想。

【输出】op_time_share.png / memory_heatmap.png / flops_table.csv /
        profiling_summary.json / stdout 报告
【与原代码关系】只 import tinytorch 导出的包;全部测量代码为本目录新增。
=====================================================================
"""
import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")          # 无显示环境,纯文件输出
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)       # import common

from common import get_trained_model, SEQ_LEN, BATCH_SIZE, count_params
from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import CrossEntropyLoss

OUT = HERE


# ---------------------------------------------------------------------------
# A. 算子级计时:给每个子模块的 forward 挂一个累计计时器
# ---------------------------------------------------------------------------
class OpTimer:
    """把 module.forward 换成带累计计时的版本(实例属性遮蔽,可完全还原)。

    计时口径:每个被包装 forward 单次调用的 wall-time,累加得到该算子在
    N 次前向中的总耗时。Python 调度开销同样计入 —— 这正是纯 NumPy 框架
    的真实开销,本身就是测量结论的一部分。
    """

    def __init__(self):
        self.totals = {}   # name -> 累计秒
        self._patched = []

    def patch(self, module, name):
        orig = module.forward

        def wrapped(*args, _orig=orig, _n=name, _t=self, **kwargs):
            t0 = time.perf_counter()
            out = _orig(*args, **kwargs)
            _t.totals[_n] = _t.totals.get(_n, 0.0) + (time.perf_counter() - t0)
            return out

        module.forward = wrapped
        self._patched.append((module, orig))
        return wrapped

    def unpatch_all(self):
        for mod, orig in self._patched:
            mod.forward = orig
        self._patched.clear()

    def reset(self):
        self.totals.clear()


def install_timers(model, timer):
    """按 TinyGPT 模块树逐一挂计时器;命名反映层级,如 b0.attn.q_proj。"""
    timer.patch(model.embedding_layer, "embedding")
    for i, blk in enumerate(model.blocks):
        timer.patch(blk.ln1, f"b{i}.ln1")
        timer.patch(blk.attention.q_proj, f"b{i}.attn.q_proj")
        timer.patch(blk.attention.k_proj, f"b{i}.attn.k_proj")
        timer.patch(blk.attention.v_proj, f"b{i}.attn.v_proj")
        timer.patch(blk.attention.out_proj, f"b{i}.attn.out_proj")
        timer.patch(blk.attention, f"b{i}.attn.total")   # 整体,含上面四个投影
        timer.patch(blk.ln2, f"b{i}.ln2")
        timer.patch(blk.mlp.linear1, f"b{i}.mlp.linear1")
        timer.patch(blk.mlp.gelu, f"b{i}.mlp.gelu")
        timer.patch(blk.mlp.linear2, f"b{i}.mlp.linear2")
    timer.patch(model.ln_f, "final_ln")
    timer.patch(model.lm_head, "lm_head")


def measure_op_time(model, batch, seq, warmup=3, iters=20, label=""):
    """返回 (逐算子平均单次前向耗时 ms 字典, 整次前向 wall ms)。"""
    timer = OpTimer()
    install_timers(model, timer)
    x = Tensor(np.random.randint(0, 59, size=(batch, seq)).astype(np.int32))

    for _ in range(warmup):          # 预热(缓存/JIT 无,但让 malloc/缓存热身)
        model(x)
    timer.reset()                    # 预热后清零,只统计正式测量轮
    for _ in range(iters):
        model(x)
    per_op_ms = {k: v / iters * 1000 for k, v in timer.totals.items()}
    timer.unpatch_all()

    # 整次前向墙钟(无插桩,纯净版)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    total_wall = (time.perf_counter() - t0) / iters * 1000

    # attn.total 含 q/k/v/out_proj —— 差值即 attention-core(bmm+softmax+mask)
    for i in range(len(model.blocks)):
        tot = per_op_ms.pop(f"b{i}.attn.total", 0.0)
        projs = sum(per_op_ms.get(f"b{i}.attn.{p}", 0.0)
                    for p in ("q_proj", "k_proj", "v_proj", "out_proj"))
        per_op_ms[f"b{i}.attn.core"] = max(tot - projs, 0.0)
    # b{i}.mlp 只统计 linear1/gelu/linear2,层级标签合并为 b{i}.mlp.{...}
    print(f"[{label}] 整次前向 wall = {total_wall:.2f} ms; "
          f"逐算子累计和 = {sum(per_op_ms.values()):.2f} ms "
          f"(差额 ≈ Python 调度 + 未插桩胶水代码)")
    return per_op_ms, total_wall


def plot_op_share(shares, path):
    """两种负载形态的算子耗时占比 -> 并排条形图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)
    for ax, (label, data) in zip(axes, shares.items()):
        items = sorted(data.items(), key=lambda kv: kv[1], reverse=True)
        names = [k for k, _ in items]
        vals = [v for _, v in items]
        total = sum(vals) or 1.0
        pct = [v / total * 100 for v in vals]
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(names), 1)))
        ax.barh(names[::-1], pct[::-1], color=colors[-len(names):][::-1])
        for i, p in enumerate(pct[::-1]):
            if p >= 3:
                ax.text(p + 0.4, i, f"{p:.1f}%", va="center", fontsize=7)
        ax.set_title(f"{label}\n(cumulative per-op time {total:.2f} ms/forward)")
        ax.set_xlabel("share of cumulative per-op time (%)")
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(labelsize=8)
    fig.suptitle("TinyGPT per-op time share: Prefill (compute-heavy) vs Decode (memory-heavy)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


# ---------------------------------------------------------------------------
# B. 逐层内存:权重 / 梯度 / 激活
# ---------------------------------------------------------------------------
def layer_rows(model):
    """枚举 (层名, 参数Tensor列表),供内存与 FLOPs 统计共用。"""
    rows = [("embedding", [model.embedding_layer.token_embedding.weight,
                           model.embedding_layer.pos_encoding.position_embeddings])]
    for i, blk in enumerate(model.blocks):
        rows.append((f"b{i}.ln1", [blk.ln1.gamma, blk.ln1.beta]))
        rows.append((f"b{i}.attn", [
            blk.attention.q_proj.weight, blk.attention.q_proj.bias,
            blk.attention.k_proj.weight, blk.attention.k_proj.bias,
            blk.attention.v_proj.weight, blk.attention.v_proj.bias,
            blk.attention.out_proj.weight, blk.attention.out_proj.bias]))
        rows.append((f"b{i}.ln2", [blk.ln2.gamma, blk.ln2.beta]))
        rows.append((f"b{i}.mlp", [
            blk.mlp.linear1.weight, blk.mlp.linear1.bias,
            blk.mlp.linear2.weight, blk.mlp.linear2.bias]))
    rows.append(("final_ln", [model.ln_f.gamma, model.ln_f.beta]))
    rows.append(("lm_head", [model.lm_head.weight]))
    return rows


def measure_memory(model, batch, seq, vocab):
    """逐层统计 权重/梯度/激活 内存(MB)。

    权重/梯度:参数数组 nbytes(FP32=4B);梯度先 backward 一次让它存在。
    激活:临时包装 forward 记录每个子模块输出数组的 nbytes。
    """
    x = Tensor(np.random.randint(0, vocab, size=(batch, seq)).astype(np.int32))
    y = Tensor(np.random.randint(0, vocab, size=(batch, seq)).astype(np.int32))
    crit = CrossEntropyLoss()
    logits = model(x)
    loss = crit(logits.reshape(-1, vocab), y.reshape(-1))
    loss.backward()                       # 让所有参数的 grad 就位

    # 记录各层输出激活字节数
    act = {}

    def record(name, mod):
        orig = mod.forward

        def wrapped(*args, _o=orig, _n=name, **kwargs):
            out = _o(*args, **kwargs)
            act[_n] = act.get(_n, 0) + int(np.asarray(out.data).nbytes)
            return out

        mod.forward = wrapped
        return (mod, orig)

    hooks = [record("embedding", model.embedding_layer)]
    for i, blk in enumerate(model.blocks):
        hooks.append(record(f"b{i}.ln1", blk.ln1))
        hooks.append(record(f"b{i}.attn", blk.attention))
        hooks.append(record(f"b{i}.ln2", blk.ln2))
        hooks.append(record(f"b{i}.mlp", blk.mlp))
    hooks.append(record("final_ln", model.ln_f))
    hooks.append(record("lm_head", model.lm_head))

    model(x)                              # 前向一次,激活字节数被记录
    for mod, orig in hooks:
        mod.forward = orig                # 完整还原,不留包装

    rows = []
    for name, ps in layer_rows(model):
        w_bytes = sum(int(p.data.nbytes) for p in ps)
        g_bytes = 0
        for p in ps:
            g = getattr(p, "grad", None)
            if g is not None:
                g_bytes += int(np.asarray(g).nbytes)
        rows.append((name, w_bytes / 2**20, g_bytes / 2**20,
                     act.get(name, 0) / 2**20))
    return rows


def plot_memory_heatmap(rows, path):
    """层 × 内存成分 热力图(权重/梯度/激活,MB)。"""
    names = [r[0] for r in rows]
    mat = np.array([[r[1], r[2], r[3]] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 6.8), dpi=150)
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["weights", "grads", "activations"])
    ax.set_ylabel("layer (bottom-up)")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    vmax = mat.max() or 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(j, i, f"{v:.3f}" if v < 1 else f"{v:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if v < vmax * 0.6 else "white")
    ax.set_title("TinyGPT per-layer memory heatmap (B=32, S=32, FP32)\n"
                 "MB; CPU host memory, structurally same as GPU HBM")
    fig.colorbar(im, ax=ax, label="MB")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("saved:", path)


# ---------------------------------------------------------------------------
# C. FLOPs 解析计数 + 算术强度
# ---------------------------------------------------------------------------
def flops_rows(model, batch, seq, vocab):
    """逐层解析式 FLOPs(一次乘+一次加各计 1);T=batch*seq。"""
    T, D, H, V = batch * seq, 64, 256, vocab
    S, nh = seq, 4
    rows = [("embedding", float(T * D))]
    for i in range(len(model.blocks)):
        rows.append((f"b{i}.ln1", 5 * T * D))
        rows.append((f"b{i}.attn.qkv+out", 4 * 2 * T * D * D))
        rows.append((f"b{i}.attn.core", 4 * T * S * D))    # QK^T、attn·V 各 2TSD
        rows.append((f"b{i}.ln2", 5 * T * D))
        rows.append((f"b{i}.mlp", 2 * 2 * T * D * H))
    rows.append(("final_ln", 5 * T * D))
    rows.append(("lm_head", 2 * T * D * V))
    return rows


def bytes_of_layer(name, vocab, T, S):
    """该层一次前向需搬运的字节数(权重读 + 激活读写,粗估,FP32)。"""
    D, H = 64, 256

    def lin(k, n):
        return (k * n + n) * 4

    if name == "embedding":
        return D * 4 * T
    if name.endswith(("ln1", "ln2")) or name == "final_ln":
        return (D * 2) * 4 + T * D * 4 * 3
    if name.endswith("attn.qkv+out"):
        return 4 * lin(D, D) + T * D * 4 * 4
    if name.endswith("attn.core"):
        return T * S * D * 4 * 4
    if name.endswith("mlp"):
        return lin(D, H) + lin(H, D) + T * H * 4 * 2 + T * D * 4 * 2
    if name == "lm_head":
        return lin(D, vocab) + T * D * 4 + T * vocab * 4
    return 0


def main():
    model, tok, hx, hy = get_trained_model()
    vocab = tok.vocab_size
    print(f"[profiling] model params = {count_params(model):,}, vocab={vocab}")

    # ---------------- A. 算子耗时(两种负载形态) ----------------
    prefill_ops, prefill_wall = measure_op_time(
        model, BATCH_SIZE, SEQ_LEN, label="Prefill B=32,S=32")
    decode_ops, decode_wall = measure_op_time(
        model, 1, 1, label="Decode B=1,S=1")
    plot_op_share(
        {"Prefill (B=32, S=32)": prefill_ops,
         "Decode (B=1, S=1)": {k: v for k, v in decode_ops.items()}},
        os.path.join(OUT, "op_time_share.png"))

    # ---------------- B. 内存热力图 ----------------
    mem_rows = measure_memory(model, BATCH_SIZE, SEQ_LEN, vocab)
    plot_memory_heatmap(mem_rows, os.path.join(OUT, "memory_heatmap.png"))

    # ---------------- C. FLOPs + 算术强度 + 瓶颈标签 ----------------
    label_map = {"embedding": "embedding", "final_ln": "layernorm",
                 "lm_head": "linear"}
    for i in range(len(model.blocks)):
        label_map[f"b{i}.ln1"] = "layernorm"
        label_map[f"b{i}.ln2"] = "layernorm"
        label_map[f"b{i}.attn.qkv+out"] = "linear"
        label_map[f"b{i}.attn.core"] = "attention-matmul"
        label_map[f"b{i}.mlp"] = "linear+gelu"

    fl = flops_rows(model, BATCH_SIZE, SEQ_LEN, vocab)
    summary_layers = {}
    with open(os.path.join(OUT, "flops_table.csv"), "w", encoding="utf-8") as f:
        f.write("layer,op_type,flops_per_forward_prefill,mflops,"
                "op_time_ms,arith_intensity_flops_per_byte,bound_guess\n")
        for name, fl_ops in fl:
            t_ms = prefill_ops.get(name, 0.0)
            byt = bytes_of_layer(name, vocab, BATCH_SIZE * SEQ_LEN, SEQ_LEN)
            ai = fl_ops / byt if byt else 0.0
            # roofline 粗判:AI<200 flops/byte 视为偏访存(CPU NumPy 经验阈值)
            bound = "memory-ish" if ai < 200 else "compute-ish"
            f.write(f"{name},{label_map.get(name,'other')},{fl_ops:.0f},"
                    f"{fl_ops/1e6:.3f},{t_ms:.3f},{ai:.1f},{bound}\n")
            summary_layers[name] = {"op_type": label_map.get(name, "other"),
                                    "flops": fl_ops, "time_ms": t_ms,
                                    "arith_intensity": ai, "bound": bound}
    total_fl = sum(v for _, v in fl)
    print(f"[profiling] prefill 总 FLOPs ≈ {total_fl/1e6:.2f} MFLOPs/forward, "
          f"wall {prefill_wall:.2f} ms -> {total_fl/prefill_wall/1e6:.3f} GFLOP/s")

    # ---------------- 汇总 JSON ----------------
    summary = {
        "model": {"params": count_params(model), "vocab": vocab,
                  "embed": 64, "layers": len(model.blocks), "heads": 4,
                  "max_seq": 64},
        "prefill": {"wall_ms": prefill_wall, "ops_ms": prefill_ops},
        "decode": {"wall_ms": decode_wall, "ops_ms": decode_ops},
        "memory_rows_mb": [{"layer": n, "weights": w, "grads": g, "acts": a}
                           for n, w, g, a in mem_rows],
        "flops": summary_layers,
        "notes": "NumPy CPU;op 计时含 Python 调度开销;内存结构与 GPU 显存同构",
    }
    with open(os.path.join(OUT, "profiling_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved:", os.path.join(OUT, "profiling_summary.json"))

    # ---------------- 关键结论打印(直接进报告) ----------------
    def top(d, n=6):
        items = sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]
        return ", ".join(f"{k}={v:.2f}ms" for k, v in items)

    print("\n===== 关键测量结论 =====")
    print("Prefill 最耗时算子:", top(prefill_ops))
    print("Decode  最耗时算子:", top(decode_ops))
    w_sum = sum(r[1] for r in mem_rows)
    g_sum = sum(r[2] for r in mem_rows)
    a_sum = sum(r[3] for r in mem_rows)
    lm = [r for r in mem_rows if r[0] == "lm_head"][0]
    ml = [(r[0], r[1]) for r in mem_rows if "mlp" in r[0]]
    print(f"内存: 权重合计 {w_sum:.3f} MB, 梯度 {g_sum:.3f} MB, 激活 {a_sum:.3f} MB")
    print(f"权重占用最大的层: lm_head={lm[1]:.3f}MB; "
          f"各层 MLP: {[(n, round(v, 3)) for n, v in ml]}")


if __name__ == "__main__":
    main()
