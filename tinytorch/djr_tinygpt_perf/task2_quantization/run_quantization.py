# -*- coding: utf-8 -*-
"""
djr_tinygpt_perf/task2_quantization/run_quantization.py
=====================================================================
Task 2 Quantization —— 用精度换带宽(PTQ FP32 -> INT8)
=====================================================================
【实验设计】(对应 issue 任务 1/2 与产出 1/3)
  1. 两种 PTQ 实现:
     - 对称量化(本脚本自研,带注释): scale = max|w|/127, q∈[-127,127]
     - 非对称量化(课程框架 quantize_int8): min-max + zero_point
     并逐层对比两种方案的量化误差(MSE / 最大误差 / 余弦相似度)。
  2. 整模型量化:按框架建议对自定义容器用 QuantizedLinear 逐层包装
     (TinyGPT 不是 Sequential,quantize_model() 的就地遍历不适用),
     用校准数据(calibrate)估计输入 scale,对比量化前后:
       - 模型大小(解析字节 + np.savez 实测文件大小)
       - held-out loss / PPL(精度损失)
       - 推理速度:
         (a) 实测 —— Prefill(TTFT)与 Decode(ITL)的 wall-time;
             NumPy 仿真是 float32 计算,预计无真实加速(教学要点!)
         (b) 建模 —— roofline 内存时间模型 t≈bytes/BW,
             给出真实硬件(INT8 GEMM)上可期望的加速比。
     该「实测 vs 建模」的对照直接回答产出 1:
     为什么量化对 Decode(ITL)加速远好于 Prefill(TTFT)。
  3. 图表:quant_error.png / size_compare.png / speed_compare.png
  4. 输出:quantization_summary.json + stdout 结论

【与原代码关系】仅 import tinytorch.perf.quantization 的公开 API;
  对称量化与全部实验代码为本目录新增,不修改任何课程源码。
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

from common import (get_trained_model, eval_heldout, SEQ_LEN, BATCH_SIZE,
                    count_params)
from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.transformers import generate
from tinytorch.perf.quantization import quantize_int8, dequantize_int8, QuantizedLinear

OUT = HERE


# ===========================================================================
# 1. 对称量化(自研实现,教学注释)
# ===========================================================================
def quantize_symmetric(w: np.ndarray, bits: int = 8):
    """对称 per-tensor INT8 量化。

    原理:s = max|w| / (2^(b-1) - 1),q = clip(round(w/s), -127, 127)
      - 零点固定在 0(symmetric),w=0 精确映射到 q=0,无零点漂移误差;
      - 相比非对称 min-max,当张量分布不对称(如含大离群值的一侧)时
        会浪费一半码字,但硬件实现(如 int8 GEMM 的零点处理)更简单。
    返回 (q_int8, scale);反量化 ŵ = q * s。
    """
    qmax = 2 ** (bits - 1) - 1                     # 127
    amax = float(np.max(np.abs(w)))
    scale = amax / qmax if amax > 0 else 1.0       # 常数张量保护
    q = np.clip(np.round(w / scale), -qmax, qmax).astype(np.int8)
    return q, scale


def quant_err(w: np.ndarray, what_dequant: np.ndarray):
    """量化误差三指标:MSE / 最大绝对误差 / 余弦相似度。"""
    err = what_dequant - w
    mse = float(np.mean(err * err))
    maxe = float(np.max(np.abs(err)))
    denom = (np.linalg.norm(w) * np.linalg.norm(what_dequant))
    cos = float(np.dot(w.ravel(), what_dequant.ravel()) / denom) if denom else 1.0
    return mse, maxe, cos


# ===========================================================================
# 2. 遍历 TinyGPT,把每个 Linear 包装成 QuantizedLinear(可还原)
# ===========================================================================
def linear_slots(model):
    """枚举 (所属层名, 属性宿主对象, 属性名, Linear实例)。

    TinyGPT 结构:embedding / [blocks: ln1, attention(q/k/v/out), ln2,
    mlp(linear1/linear2)] / ln_f / lm_head —— 与 Task1 的插桩路径一致。
    """
    slots = []
    for i, blk in enumerate(model.blocks):
        a = blk.attention
        slots += [(f"b{i}.attn.q_proj", a, "q_proj", a.q_proj),
                  (f"b{i}.attn.k_proj", a, "k_proj", a.k_proj),
                  (f"b{i}.attn.v_proj", a, "v_proj", a.v_proj),
                  (f"b{i}.attn.out_proj", a, "out_proj", a.out_proj)]
        m = blk.mlp
        slots += [(f"b{i}.mlp.linear1", m, "linear1", m.linear1),
                  (f"b{i}.mlp.linear2", m, "linear2", m.linear2)]
    slots.append(("lm_head", model, "lm_head", model.lm_head))
    return slots


def quantize_tinygpt(model, calib_inputs):
    """把 TinyGPT 全部 Linear 换成 QuantizedLinear,返回 (替换表, 校准信息)。

    QuantizedLinear 内部:权重/偏置走非对称 quantize_int8;
    输入 scale 用 calibrate() 从校准样本估计(激活量化参数,
    真实 int8 推理中用于激活的动态/静态量化)。
    """
    replaced = []
    for name, host, attr, lin in linear_slots(model):
        ql = QuantizedLinear(lin)
        # 校准:用真实前向中该层会看到的输入统计 input scale
        ins = [c for c in calib_inputs if c.shape[-1] == lin.weight.shape[1]]
        if ins:
            try:
                ql.calibrate(ins[:8])
            except Exception as e:
                print(f"[quantize] {name} calibrate 跳过: {e}")
        setattr(host, attr, ql)
        replaced.append((name, host, attr, lin, ql))
    return replaced


def restore_tinygpt(replaced):
    for name, host, attr, lin, ql in replaced:
        setattr(host, attr, lin)


def collect_calib_inputs(model, hx, n=8):
    """跑少量 held-out 窗口,钩出各 Linear 的真实输入(校准用)。

    简化实现:逐层用『前一层的真实输出』代价较高;此处用嵌入输出作为
    统一近似校准分布(教学演示足够;工业做法需逐层钩真实激活)。
    """
    ins = []
    for s in range(min(n, len(hx))):
        x = Tensor(hx[s:s + 1])
        ins.append(model.embedding_layer(x))
    return ins


# ===========================================================================
# 3. 实测速度:Prefill(TTFT 形态)与 Decode(ITL 形态)
# ===========================================================================
def bench_linear(lin_or_ql, x, warmup=5, iters=30):
    """单层 forward 计时中位数(ms)。"""
    for _ in range(warmup):
        lin_or_ql.forward(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        lin_or_ql.forward(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


def bench_model_forward(model, x, warmup=3, iters=20):
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model(x)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


def bench_decode(model, tok, prompt="First Citizen:", steps=20, temperature=0.8):
    """真实自回归解码:逐步 generate,统计每步(ITL)耗时中位数。"""
    ids = tok.encode(prompt)
    x = Tensor(np.array([ids], dtype=np.int32))
    # 预热一步
    generate(model, x, max_new_tokens=1, temperature=temperature)
    ts = []
    for _ in range(steps):
        t0 = time.perf_counter()
        generate(model, x, max_new_tokens=1, temperature=temperature)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1000


# ===========================================================================
# 4. 内存时间建模(roofline):真实硬件上的期望加速
# ===========================================================================
def modeled_times(param_bytes_per_layer, tokens, bw_bytes_per_s):
    """
    解码(GEMV): t ≈ weight_bytes / BW(访存受限)
    预填充(GEMM): t ≈ flops / FLOPs峰值(计算受限,与权重字节无关)
    这里只建模 decode 的带宽收益;prefill 用「计算受限 -> 不受益」作答。
    """
    return param_bytes_per_layer / bw_bytes_per_s * 1000  # ms


def main():
    model, tok, hx, hy = get_trained_model()
    vocab = tok.vocab_size
    n_params = count_params(model)
    print(f"[quant] baseline params={n_params:,} vocab={vocab}")

    # ---------- 基线精度 ----------
    base_loss, base_ppl = eval_heldout(model, hx, hy)
    print(f"[quant] FP32 基线: held-out loss={base_loss:.4f} ppl={base_ppl:.2f}")

    # ---------- 1) 逐层对称 vs 非对称误差对比 ----------
    rows = []
    for name, host, attr, lin in linear_slots(model):
        w = lin.weight.data
        # 对称(自研)
        qs, ss = quantize_symmetric(w)
        w_hat_s = qs.astype(np.float32) * ss
        mse_s, maxe_s, cos_s = quant_err(w, w_hat_s)
        # 非对称(框架 quantize_int8: Tensor -> (qTensor, scale, zero_point))
        qt, st, zt = quantize_int8(Tensor(w.copy()))
        w_hat_a = dequantize_int8(qt, st, zt).data
        mse_a, maxe_a, cos_a = quant_err(w, w_hat_a)
        rows.append((name, w.size, mse_s, maxe_s, cos_s, mse_a, maxe_a, cos_a))
    with open(os.path.join(OUT, "quant_error_table.csv"), "w", encoding="utf-8") as f:
        f.write("layer,numel,mse_sym,max_err_sym,cos_sym,mse_asym,max_err_asym,cos_asym\n")
        for r in rows:
            f.write(",".join([r[0], str(r[1])] + [f"{x:.3e}" for x in r[2:]]) + "\n")
    tot_s = np.mean([r[2] for r in rows]); tot_a = np.mean([r[5] for r in rows])
    print(f"[quant] 平均权重量化 MSE: 对称={tot_s:.3e}  非对称={tot_a:.3e}")

    # ---------- 2) 整模型 INT8(非对称 + 校准),精度与大小 ----------
    replaced = quantize_tinygpt(model, collect_calib_inputs(model, hx))
    q_loss, q_ppl = eval_heldout(model, hx, hy)
    print(f"[quant] INT8 后 : held-out loss={q_loss:.4f} ppl={q_ppl:.2f} "
          f"(Δloss={q_loss-base_loss:+.4f})")

    # 生成样例对比
    ids = tok.encode("First Citizen:")
    xt = Tensor(np.array([ids], dtype=np.int32))
    sample_q = generate(model, xt, max_new_tokens=40, temperature=0.8)
    text_q = tok.decode([int(i) for i in sample_q.data[0]])
    print(f"[quant] INT8 生成样例: {text_q!r}")

    # 模型大小:解析字节(FP32 4B/参数;INT8 权重 1B/参数 + scale/zero_point 忽略)
    fp32_bytes = n_params * 4
    # lm_head 也量化;embedding 在本实验保持 FP32(常见做法:激活敏感层保精度),
    # 因此 INT8 字节 = (Linear 参数数)×1 + (Embedding 参数数)×4
    lin_params = 0
    for name, host, attr, lin, ql in replaced:
        lin_params += lin.weight.data.size
        if lin.bias is not None:
            lin_params += lin.bias.data.size
    emb_params = n_params - lin_params
    int8_bytes = lin_params * 1 + emb_params * 4
    print(f"[quant] 大小: FP32={fp32_bytes/2**20:.2f} MB -> "
          f"INT8(Linear)+FP32(Emb)={int8_bytes/2**20:.2f} MB "
          f"({fp32_bytes/int8_bytes:.2f}×)")

    # np.savez 实测文件大小(仅 Linear 权重,展示 int8 dtype 的真实存储收益)
    lin_w = replaced[0][3].weight.data                     # 任意一层示意
    q8, s8 = quantize_symmetric(lin_w)
    np.savez(os.path.join(OUT, "_fp32_demo.npz"), w=lin_w)
    np.savez(os.path.join(OUT, "_int8_demo.npz"), q=q8)
    f32_sz = os.path.getsize(os.path.join(OUT, "_fp32_demo.npz"))
    i8_sz = os.path.getsize(os.path.join(OUT, "_int8_demo.npz"))
    print(f"[quant] 单层权重文件实测: FP32={f32_sz/1024:.1f} KB -> INT8={i8_sz/1024:.1f} KB")
    os.remove(os.path.join(OUT, "_fp32_demo.npz"))
    os.remove(os.path.join(OUT, "_int8_demo.npz"))

    # ---------- 3) 速度:实测(NumPy 仿真)+ 建模(真实硬件) ----------
    D, H, V = 64, 256, vocab
    T_pre = BATCH_SIZE * SEQ_LEN          # prefill token 数 = 1024
    speed_rows = []
    for name, host, attr, lin, ql in replaced:
        # 框架 Linear 权重存为 (in, out):x @ W,输入末维须等于 in_features
        in_dim = lin.weight.data.shape[0]
        xpre = Tensor(np.random.randn(T_pre, in_dim).astype(np.float32))
        xdec = Tensor(np.random.randn(1, in_dim).astype(np.float32))
        tp32 = bench_linear(lin, xpre)
        tq8 = bench_linear(ql, xpre)
        d32 = bench_linear(lin, xdec)
        dq8 = bench_linear(ql, xdec)
        speed_rows.append((name, tp32, tq8, d32, dq8))
    with open(os.path.join(OUT, "speed_table.csv"), "w", encoding="utf-8") as f:
        f.write("layer,prefill_fp32_ms,prefill_int8sim_ms,decode_fp32_ms,decode_int8sim_ms\n")
        for r in speed_rows:
            f.write(f"{r[0]},{r[1]:.3f},{r[2]:.3f},{r[3]:.4f},{r[4]:.4f}\n")

    pre32 = bench_model_forward(model, Tensor(
        np.random.randint(0, vocab, size=(BATCH_SIZE, SEQ_LEN)).astype(np.int32)))
    itl32 = bench_decode(model, tok, steps=15)
    pre8 = None
    # 注意:此时 model 的 Linear 已是 QuantizedLinear(量化态)
    pre8 = bench_model_forward(model, Tensor(
        np.random.randint(0, vocab, size=(BATCH_SIZE, SEQ_LEN)).astype(np.int32)))
    itl8 = bench_decode(model, tok, steps=15)
    print(f"[quant] 实测整模型: Prefill FP32={pre32:.2f}ms INT8sim={pre8:.2f}ms "
          f"({pre32/pre8:.2f}×) | ITL FP32={itl32:.2f}ms INT8sim={itl8:.2f}ms "
          f"({itl32/itl8:.2f}×)")
    print("[quant] 注:NumPy 仿真用 float32 计算 INT8 值域数值,看不到真实加速;"
          "这是课程框架刻意设计(见 QuantizedLinear 文档),真实收益见下方建模。")

    # roofline 建模:decode 是 GEMV(访存受限),t ≈ weight_bytes / BW
    BW_CPU = 200 * 2**30            # 服务器 DDR ~200 GB/s
    BW_A100 = 2039 * 2**30          # A100-80GB HBM2e 峰值
    lin_w_bytes = lin_params        # int8 后每参数 1B(fp32 为 4B)
    itl_model_fp32_cpu = lin_w_bytes * 4 / BW_CPU * 1000
    itl_model_int8_cpu = lin_w_bytes * 1 / BW_CPU * 1000
    itl_model_fp32_gpu = lin_w_bytes * 4 / BW_A100 * 1000
    itl_model_int8_gpu = lin_w_bytes * 1 / BW_A100 * 1000
    print(f"[quant] 建模 ITL(纯带宽下限): CPU FP32≈{itl_model_fp32_cpu:.3f}ms "
          f"INT8≈{itl_model_int8_cpu:.3f}ms (加速≈4×);"
          f" A100 FP32≈{itl_model_fp32_gpu:.4f}ms INT8≈{itl_model_int8_gpu:.4f}ms")
    print("[quant] 建模 TTFT: prefill 是大 GEMM,计算受限;NumPy 无 int8 GEMM,"
          "即使真实硬件也只在 int8 张量核/AVX-VNNI 上获得额外吞吐,"
          "而权重带宽占比小,故 TTFT 提升远小于 ITL —— 这就是产出 1 的核心答案。")

    # ---------- 4) 图表 ----------
    # a) 量化误差对比
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r[2] for r in rows], 0.4, label="symmetric (ours)")
    ax.bar(xs + 0.2, [r[5] for r in rows], 0.4, label="asymmetric (framework)")
    ax.set_yscale("log")
    ax.set_xticks(xs, [r[0] for r in rows], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("weight quant MSE (log)")
    ax.set_title("PTQ weight error: symmetric vs asymmetric (per layer, FP32->INT8)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "quant_error.png")); plt.close(fig)
    print("saved:", os.path.join(OUT, "quant_error.png"))

    # b) 大小对比
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.bar(["FP32 full model", "INT8 Linear\n+FP32 Emb"],
           [fp32_bytes / 2**20, int8_bytes / 2**20],
           color=["#2563eb", "#16a34a"])
    for i, v in enumerate([fp32_bytes / 2**20, int8_bytes / 2**20]):
        ax.text(i, v, f"{v:.2f} MB", ha="center", va="bottom")
    ax.set_ylabel("param storage (MB)")
    ax.set_title(f"Model size: PTQ compression {fp32_bytes/int8_bytes:.2f}x\n"
                 "(Linear weights INT8=1B/param; Embedding kept FP32)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "size_compare.png")); plt.close(fig)
    print("saved:", os.path.join(OUT, "size_compare.png"))

    # c) 速度对比(实测 + 建模)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    ax = axes[0]
    sel = [r for r in speed_rows if "mlp.linear1" in r[0] or "lm_head" == r[0] or "q_proj" in r[0]]
    xs = np.arange(len(sel))
    ax.bar(xs - 0.2, [r[3] * 1000 for r in sel], 0.4, label="FP32")
    ax.bar(xs + 0.2, [r[4] * 1000 for r in sel], 0.4, label="INT8 (NumPy simulated)")
    ax.set_yscale("log")
    ax.set_xticks(xs, [r[0] for r in sel], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("single-layer decode fwd (us, log)")
    ax.set_title("Decode shape, measured (B=1 GEMV)\nNumPy sim: no real speedup (teaching point)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.bar(["ITL CPU\n(modeled)", "ITL A100\n(modeled)"],
           [itl_model_fp32_cpu / max(itl_model_int8_cpu, 1e-9),
            itl_model_fp32_gpu / max(itl_model_int8_gpu, 1e-9)],
           color=["#f59e0b", "#8b5cf6"])
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_ylabel("expected speedup (FP32/INT8)")
    ax.set_title("Bandwidth model on real hardware\nDecode(ITL)~4x; Prefill(TTFT)~1x (compute-bound)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "speed_compare.png")); plt.close(fig)
    print("saved:", os.path.join(OUT, "speed_compare.png"))

    # ---------- 5) 汇总 ----------
    summary = {
        "baseline_fp32": {"heldout_loss": base_loss, "ppl": base_ppl,
                          "params": n_params, "bytes": fp32_bytes},
        "int8": {"heldout_loss": q_loss, "ppl": q_ppl,
                 "delta_loss": q_loss - base_loss,
                 "bytes": int8_bytes, "compression": fp32_bytes / int8_bytes,
                 "sample_text": text_q},
        "quant_error_mean_mse": {"sym": float(tot_s), "asym": float(tot_a)},
        "speed_measured": {"prefill_fp32_ms": pre32, "prefill_int8sim_ms": pre8,
                           "itl_fp32_ms": itl32, "itl_int8sim_ms": itl8,
                           "note": "NumPy 仿真=fp32 计算 int8 值域,无真实加速"},
        "speed_modeled": {"itl_cpu_fp32_ms": itl_model_fp32_cpu,
                          "itl_cpu_int8_ms": itl_model_int8_cpu,
                          "itl_a100_fp32_ms": itl_model_fp32_gpu,
                          "itl_a100_int8_ms": itl_model_int8_gpu,
                          "ttft": "compute-bound, int8 需硬件 int8 GEMM 才有吞吐收益"},
    }
    with open(os.path.join(OUT, "quantization_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("saved:", os.path.join(OUT, "quantization_summary.json"))

    # 还原模型为 FP32(缓存不受影响,只是本次进程内还原)
    restore_tinygpt(replaced)


if __name__ == "__main__":
    main()
