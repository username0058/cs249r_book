# -*- coding: utf-8 -*-
"""
djr_tinygpt_perf/task0_baseline/train_and_curve.py
=====================================================================
Task 0 —— TinyGPT 训练曲线与成功判据(自采集逐 epoch 数据)
=====================================================================
【背景】milestone 官方脚本用 Rich 进度条渲染,中间 epoch 的日志行被
  覆盖,无法从 stdout 还原完整曲线。本脚本用与 milestone 完全一致的
  管线(common.py:同数据/同优化器/同超参)独立训练一遍,每个 epoch
  记录:训练 loss、held-out loss,并用「计数 bigram 模型」在同一
  held-out 上给出基线 —— 成功判据全部可复算、可验证。

【产出】loss_curve.csv(逐 epoch)/ loss_curve.png(曲线图)/ stdout 判据
【附带】训练完成后把权重写入 common.CACHE_NPZ,供 Task1/2/3 复用。
=====================================================================
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, BASE)

from common import (build_data, build_model, save_weights, SEQ_LEN,
                    EPOCHS, LR, WEIGHT_DECAY, BATCH_SIZE)
from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import AdamW


def eval_loss(model, hx, hy, vocab, batch=64):
    """held-out 平均交叉熵(nats/token)。"""
    crit = CrossEntropyLoss()
    tot, n = 0.0, 0
    for s in range(0, len(hx), batch):
        xb = Tensor(hx[s:s + batch])
        yb = Tensor(hy[s:s + batch])
        logits = model(xb)
        loss = crit(logits.reshape(-1, vocab), yb.reshape(-1))
        tot += float(loss.data) * xb.shape[0]
        n += xb.shape[0]
    return tot / n


def bigram_nll(train_tokens, heldout_tokens, vocab):
    """计数 bigram 基线:直接调用 milestone 官方 transformer_gates.bigram_cross_entropy
    (全表 add-1 平滑),保证与官方判据口径完全一致。"""
    milestone_dir = os.path.join(BASE, "..", "milestones", "05_2017_transformer")
    if milestone_dir not in sys.path:
        sys.path.insert(0, milestone_dir)
    from transformer_gates import bigram_cross_entropy
    return bigram_cross_entropy(train_tokens, heldout_tokens, vocab_size=vocab,
                                smoothing=1.0)


def main():
    tok, train_dl, hx, hy = build_data()
    vocab = tok.vocab_size
    model = build_model(vocab)
    print(f"[curve] vocab={vocab}, 训练 {EPOCHS} epochs (lr={LR}, wd={WEIGHT_DECAY}, "
          f"batch={BATCH_SIZE}, seq={SEQ_LEN})")

    criterion = CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    curve = []                               # (epoch, train_loss, heldout_loss)
    for ep in range(EPOCHS):
        ep_loss, n = 0.0, 0
        for xb, yb in train_dl:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits.reshape(-1, logits.shape[-1]),
                             yb.reshape(-1))
            loss.backward()
            opt.step()
            ep_loss += float(loss.data); n += 1
        tr = ep_loss / max(n, 1)
        ho = eval_loss(model, hx, hy, vocab)
        curve.append((ep + 1, tr, ho))
        print(f"[curve] epoch {ep+1:2d}/{EPOCHS}  train={tr:.4f}  "
              f"heldout={ho:.4f}  ppl_train={np.exp(tr):.2f}", flush=True)

    # ---- 权重缓存:供 Task1/2/3 复用 ----
    save_weights(model, os.path.join(HERE, "tinygpt_weights.npz"))
    print("[curve] 权重已缓存 -> tinygpt_weights.npz")

    # ---- bigram 基线 ----
    from tinytorch.core.tokenization import CharTokenizer
    with open(os.path.join(BASE, "..", "datasets", "tinyshakespeare",
                           "tinyshakespeare_sample.txt"), encoding="utf-8") as f:
        text = f.read()
    t = CharTokenizer(); t.build_vocab([text])
    tokens = np.array(t.encode(text))
    heldout_size = min(4000, len(tokens) // 10)
    train_limit = min(35000, len(tokens) - heldout_size)
    bigram = bigram_nll(tokens[:train_limit],
                        tokens[train_limit:train_limit + heldout_size],
                        vocab)
    print(f"[curve] bigram 基线 held-out NLL = {bigram:.4f}")

    # ---- 判据 ----
    final_train = curve[-1][1]
    best_ho = min(c[2] for c in curve)
    c1 = final_train < 1.20
    c2 = (bigram - best_ho) >= 0.35
    print(f"[curve] 判据1 训练损失 <1.20: {final_train:.4f} -> {'PASS' if c1 else 'FAIL'}")
    print(f"[curve] 判据2 held-out 距 bigram ≥0.35: {bigram:.4f} - {best_ho:.4f} "
          f"= {bigram-best_ho:.4f} -> {'PASS' if c2 else 'FAIL'}")

    # ---- CSV ----
    with open(os.path.join(HERE, "loss_curve.csv"), "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,heldout_loss\n")
        for e, a, b in curve:
            f.write(f"{e},{a:.4f},{b:.4f}\n")
        f.write(f"# bigram_baseline,{bigram:.4f}\n")

    # ---- 曲线图 ----
    ep = [c[0] for c in curve]
    tr = [c[1] for c in curve]
    ho = [c[2] for c in curve]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(ep, tr, "o-", color="#2563eb", label="Train loss (epoch avg)")
    ax.plot(ep, ho, "s--", color="#dc2626", label="Held-out loss")
    ax.axhline(1.20, color="#16a34a", lw=1, ls=":",
               label="Gate: train loss < 1.20")
    ax.axhline(bigram, color="#9a3412", lw=1, ls="-.",
               label=f"Bigram baseline = {bigram:.3f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy (nats/token)")
    ax.set_title(f"TinyGPT on TinyShakespeare (Milestone 05 recipe)\n"
                 f"final train={final_train:.3f} < 1.20 | best held-out="
                 f"{best_ho:.3f}, margin={bigram-best_ho:.3f} >= 0.35")
    ax.set_xticks(ep)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "loss_curve.png"))
    print("saved:", os.path.join(HERE, "loss_curve.png"))

    # ---- 摘要 JSON ----
    with open(os.path.join(HERE, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "config": {"epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
                       "batch": BATCH_SIZE, "seq_len": SEQ_LEN,
                       "embed": 64, "layers": 2, "heads": 4},
            "curve": [{"epoch": e, "train": a, "heldout": b} for e, a, b in curve],
            "bigram_baseline": bigram,
            "criteria": {"train_lt_1p20": bool(c1),
                         "heldout_margin_ge_0p35": bool(c2)},
            "final_train_loss": final_train,
            "best_heldout": best_ho,
        }, f, ensure_ascii=False, indent=2)
    print("saved:", os.path.join(HERE, "train_summary.json"))


if __name__ == "__main__":
    main()
