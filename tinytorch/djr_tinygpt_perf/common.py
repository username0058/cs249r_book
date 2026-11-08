# -*- coding: utf-8 -*-
"""
djr_tinygpt_perf/common.py —— 四个 Task 共用的基础设施
=====================================================================
【本文件与原课程代码的关系】
  本文件是 djr_tinygpt_perf/ 工作目录的私有工具模块,全部为新增代码,
  不修改 tinytorch 仓库的任何原始文件;仅以 import 方式调用已导出的
  tinytorch 包(tinytorch.core.* / tinytorch.perf.*)。

【职责】
  1. 按 Milestone 05 的超参数构建 TinyGPT:
       vocab_size=自动(字符级), embed_dim=64, num_layers=2,
       num_heads=4, max_seq_len=64 (= seq_len 32 × 2)
  2. 按 Milestone 05 的数据管线构造训练/验证集:
       tinyshakespeare_sample, CharTokenizer, seq_len=32, stride=4,
       batch=32, 训练 token 上限 35000, held-out 4000
  3. 提供带缓存权重的 get_trained_model():所有下游实验(Task1/2/3)
     加载同一份 FP32 基线权重,保证横向可比。
  4. 提供统一的 held-out loss / PPL 评估函数。
=====================================================================
"""
import os
import sys
import time
import numpy as np

# 保证无论从哪个 task 子目录运行,都能 import 到本目录的 common
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from tinytorch.core.tensor import Tensor
from tinytorch.core.losses import CrossEntropyLoss
from tinytorch.core.optimizers import AdamW
from tinytorch.core.dataloader import TensorDataset, DataLoader
from tinytorch.core.tokenization import CharTokenizer
from tinytorch.core.transformers import TinyGPT

# ---------------------------------------------------------------------------
# 全局配置(与 milestones/05_2017_transformer/01_tinygpt_shakespeare.py 一致)
# ---------------------------------------------------------------------------
TINYTORCH_ROOT = os.path.abspath(os.path.join(HERE, ".."))
SAMPLE_TXT = os.path.join(
    TINYTORCH_ROOT, "datasets", "tinyshakespeare", "tinyshakespeare_sample.txt")

SEQ_LEN        = 32      # 训练窗口长度
STRIDE         = 4       # 滑窗步长
BATCH_SIZE     = 32      # batch 大小
TRAIN_LIMIT    = 35000   # 训练 token 上限
HELDOUT_TOKENS = 4000    # held-out token 数(从不参与训练)
EMBED_DIM      = 64
NUM_LAYERS     = 2
NUM_HEADS      = 4
MAX_SEQ_LEN    = SEQ_LEN * 2   # 位置表长度 = 64,与 milestone 相同
LR             = 2e-3
WEIGHT_DECAY   = 0.01
EPOCHS         = 12

# 权重缓存路径:task0 训练一次,task1/2/3 直接加载,保证同一基线
CACHE_NPZ = os.path.join(HERE, "task0_baseline", "tinygpt_weights.npz")


# ---------------------------------------------------------------------------
# 数据管线
# ---------------------------------------------------------------------------
def build_data():
    """复现 milestone 05 的数据管线,返回 (tokenizer, train_dl, heldout_x, heldout_y)。

    说明:
      - 字符级词表(约 59 个字符),与 milestone 完全一致;
      - held-out 段永不进入训练,用于衡量泛化(对照 bigram 基线)。
    """
    with open(SAMPLE_TXT, "r", encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer()
    tok.build_vocab([text])
    tokens = tok.encode(text)

    heldout_size = min(HELDOUT_TOKENS, len(tokens) // 10)
    train_limit = min(TRAIN_LIMIT, len(tokens) - heldout_size)
    train_tokens = tokens[:train_limit]
    heldout = tokens[train_limit:train_limit + heldout_size]

    # 滑窗切训练样本:输入 x[t:t+L],标签 y[t+1:t+L+1](next-token)
    xs, ys = [], []
    for s in range(0, len(train_tokens) - SEQ_LEN, STRIDE):
        xs.append(train_tokens[s:s + SEQ_LEN])
        ys.append(train_tokens[s + 1:s + SEQ_LEN + 1])
    x_t = Tensor(np.array(xs, dtype=np.int32))
    y_t = Tensor(np.array(ys, dtype=np.int32))
    ds = TensorDataset(x_t, y_t)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    # held-out 不重叠窗口:每字符恰好计一次 loss
    hx = np.array([heldout[s:s + SEQ_LEN]
                   for s in range(0, len(heldout) - SEQ_LEN, SEQ_LEN)], dtype=np.int32)
    hy = np.array([heldout[s + 1:s + SEQ_LEN + 1]
                   for s in range(0, len(heldout) - SEQ_LEN, SEQ_LEN)], dtype=np.int32)
    return tok, dl, hx, hy


def build_model(vocab_size):
    """按 milestone 超参数构建 TinyGPT。"""
    return TinyGPT(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        max_seq_len=MAX_SEQ_LEN,
    )


def train_model(model, train_dl, epochs=EPOCHS, log_prefix=""):
    """AdamW + CrossEntropy 的标准训练循环,返回每轮平均 loss 列表。

    与 milestone 的差异:仅去掉了 Rich 进度条(后台批处理更友好),
    优化器/损失/数据完全一致。
    """
    criterion = CrossEntropyLoss()   # 与 milestone 相同:无参构造,实例可直接调用
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    losses = []
    for ep in range(epochs):
        ep_loss, n = 0.0, 0
        for xb, yb in train_dl:
            opt.zero_grad()
            logits = model(xb)                       # [B, S, V]
            # 与 milestone 相同的展平方式:[B*S, V] 对 [B*S]
            loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
            loss.backward()
            opt.step()
            ep_loss += float(loss.data); n += 1
        avg = ep_loss / max(n, 1)
        losses.append(avg)
        print(f"{log_prefix}epoch {ep+1}/{epochs}  train_loss={avg:.4f}  ppl={np.exp(avg):.2f}",
              flush=True)
    return losses


# ---------------------------------------------------------------------------
# 权重存取(纯 NumPy npz,不依赖任何序列化库)
# ---------------------------------------------------------------------------
def _named_params(model):
    """给每个参数一个稳定名字:类型序号_属性名。"""
    out = {}
    counts = {}
    for p in model.parameters():
        # Tensor 对象身份 -> 名字:用 id 映射,遍历模块树太复杂,这里直接
        # 依赖 model.parameters() 的稳定顺序(milestone/测试同序)。
        base = type(p).__name__
        counts[base] = counts.get(base, 0) + 1
        out[f"{base}_{counts[base]}"] = p
    return out


def save_weights(model, path=CACHE_NPZ):
    arrs = {k: v.data for k, v in _named_params(model).items()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrs)


def load_weights(model, path=CACHE_NPZ):
    arrs = np.load(path)
    params = list(model.parameters())
    named = _named_params(model)
    for k, v in named.items():
        v.data = arrs[k].astype(np.float32)


# ---------------------------------------------------------------------------
# 对外主入口:训练一次,其余调用全部复用缓存
# ---------------------------------------------------------------------------
def get_trained_model(epochs=EPOCHS, force=False, verbose=True):
    """返回 (model, tokenizer, heldout_x, heldout_y)。

    首次调用:完整训练 12 epoch 并保存权重;
    之后调用:直接加载 npz 缓存(秒级),保证 Task1/2/3 用同一基线。
    """
    tok, dl, hx, hy = build_data()
    model = build_model(tok.vocab_size)
    if (not force) and os.path.exists(CACHE_NPZ):
        load_weights(model)
        if verbose:
            print(f"[common] 已加载缓存权重: {CACHE_NPZ}")
        return model, tok, hx, hy
    if verbose:
        print(f"[common] 开始训练 {epochs} epochs ...")
    t0 = time.time()
    losses = train_model(model, dl, epochs=epochs, log_prefix="[common] ")
    save_weights(model)
    if verbose:
        print(f"[common] 训练完成 {time.time()-t0:.1f}s, 最终权重已缓存。")
    return model, tok, hx, hy


# ---------------------------------------------------------------------------
# 统一评估
# ---------------------------------------------------------------------------
def eval_heldout(model, hx, hy, batch=64):
    """在 held-out 集上求平均 cross-entropy(nats/token)与困惑度。"""
    criterion = CrossEntropyLoss()
    tot, n = 0.0, 0
    for s in range(0, len(hx), batch):
        xb = Tensor(hx[s:s + batch])
        yb = Tensor(hy[s:s + batch])
        logits = model(xb)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
        b = xb.shape[0]
        tot += float(loss.data) * b
        n += b
    avg = tot / n
    return avg, float(np.exp(avg))


def count_params(model):
    return int(sum(p.data.size for p in model.parameters()))
