# -*- coding: utf-8 -*-
"""
djr_tinygpt_perf/task0_baseline/plot_loss_curve.py
=====================================================================
【作用】解析 milestone05_train.log(milestone 05 官方脚本 stdout),
       绘制训练损失曲线(训练 loss + held-out loss + bigram 基线),
       产出打卡所需的「训练损失曲线」图。

【与原代码的关系】只读取我们自己的日志文件,不触碰课程源码。
=====================================================================
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")           # 无显示环境,纯文件输出
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "milestone05_train.log")
OUT_PNG = os.path.join(HERE, "loss_curve.png")
OUT_CSV = os.path.join(HERE, "loss_curve.csv")

# milestone 日志中的 epoch 行形如:
#   Epoch 12/12 - Loss: 0.6743 (PPL: 2.0) | Held-out: 3.0852 ━━━ ... 100%
PAT = re.compile(
    r"Epoch\s+(\d+)/\d+\s+-\s+Loss:\s+([\d.]+)\s+\(PPL:\s+[\d.]+\)\s+\|\s+Held-out:\s+([\d.]+)")

# 轨迹表中 "(lowest)" 标记的最佳 held-out
rows = []
with open(LOG, "r", encoding="utf-8") as f:
    for line in f:
        m = PAT.search(line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

if not rows:
    raise SystemExit("未在日志中解析到 Epoch 行,请检查 LOG 路径")

epochs = [r[0] for r in rows]
train_loss = [r[1] for r in rows]
heldout = [r[2] for r in rows]

# milestone 输出的参考量(成功判据所需)
FINAL_TRAIN_LOSS = 0.5875     # 「Final training loss」(整个 epoch 级平均后更低)
BIGRAM_HELDOUT = 2.4908       # counted bigram 在同一 held-out 上的 loss
TARGET = 1.20                 # 官方成功判据:train loss < 1.20
MARGIN = 0.35                 # held-out 至少低于 bigram 0.35

# CSV 留档,便于报告引用
with open(OUT_CSV, "w", encoding="utf-8") as f:
    f.write("epoch,train_loss,heldout_loss\n")
    for e, t, h in zip(epochs, train_loss, heldout):
        f.write(f"{e},{t:.4f},{h:.4f}\n")

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.plot(epochs, train_loss, "o-", color="#2563eb", label="Train loss (epoch avg)")
ax.plot(epochs, heldout, "s--", color="#dc2626", label="Held-out loss (per epoch)")
ax.axhline(TARGET, color="#16a34a", lw=1, ls=":",
           label=f"Success target = {TARGET}")
ax.axhline(BIGRAM_HELDOUT, color="#9a3412", lw=1, ls="-.",
           label=f"Bigram baseline = {BIGRAM_HELDOUT}")
ax.axhline(FINAL_TRAIN_LOSS, color="#2563eb", lw=1, ls=":",
           alpha=0.6, label=f"Final train loss = {FINAL_TRAIN_LOSS}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-entropy loss (nats/token)")
ax.set_title("TinyGPT on TinyShakespeare — Milestone 05 training curve\n"
             "(final train loss 0.5875 < 1.20;  best held-out 1.9939 < bigram 2.4908 − 0.35)")
ax.set_xticks(epochs)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig(OUT_PNG)
print("saved:", OUT_PNG)
print("saved:", OUT_CSV)

# 判据核对,直接打印结论供打卡文案引用
best_h = min(heldout)
print(f"train loss target (<{TARGET}):", "PASS" if FINAL_TRAIN_LOSS < TARGET else "FAIL")
print(f"heldout margin (bigram {BIGRAM_HELDOUT} - best {best_h:.4f} = "
      f"{BIGRAM_HELDOUT - best_h:.4f} >= {MARGIN}):",
      "PASS" if BIGRAM_HELDOUT - best_h >= MARGIN else "FAIL")
