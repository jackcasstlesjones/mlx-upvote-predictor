# 🔧 End-to-End Manual Word2Vec (Skip-gram) with PyTorch

## Overview

This plan guides you through:

- Pretraining a **Skip-gram Word2Vec** model from scratch on a **54.4MB Wikipedia corpus**
- Fine-tuning it on **28GB of Hacker News titles**
- All using **only PyTorch, NumPy, and Python stdlib**
- Modular, GPU-compatible, reproducible

Intrinsic evaluation is skipped — embeddings will be evaluated downstream in your regression model.

---

## 🧱 Project Structure

```
word2vec_pipeline/
├── data/
│   ├── raw/
│   │   ├── wikipedia.txt
│   │   └── hn_titles.txt
│   └── processed/
│       ├── wiki_tokens.txt
│       └── hn_tokens.txt
├── src/
│   ├── tokenize.py
│   ├── vocab.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── fine_tune.py
│   └── utils.py
├── checkpoints/
│   ├── pretrain_epoch_*.pt
│   └── finetune_epoch_*.pt
├── embeddings/
│   └── word_vectors.npy
└── train_config.yaml
```

---

## 1. 📥 Load and Preprocess Corpus

**Input**: `wikipedia.txt`, `hn_titles.txt`

### Tokenization (src/tokenize.py)

```python
import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip().split()
```

### Disk streaming for large corpora

Use `yield` and `readline()` to tokenize line-by-line, keeping memory low:

```python
def stream_tokens(path):
    with open(path, 'r') as f:
        for line in f:
            yield tokenize(line)
```

---

## 2. 🧠 Build Vocabulary (src/vocab.py)

```python
from collections import Counter

def build_vocab(token_stream, min_freq=5):
    counter = Counter()
    for tokens in token_stream:
        counter.update(tokens)

    vocab = {word: i+1 for i, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
    vocab["<UNK>"] = 0
    return vocab
```

Include:

- Word → ID map
- ID → Word map
- Word frequencies
- Subsampling probabilities

---

## 3. 🪟 Windowed Skip-gram Pair Generator (src/dataset.py)

```python
import random

def generate_skipgram_pairs(tokens, vocab, window_size=5):
    indexed = [vocab.get(tok, 0) for tok in tokens]
    for i, center in enumerate(indexed):
        window = random.randint(1, window_size)
        for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
            if i != j:
                yield (center, indexed[j])
```

### Batching

Build batches using a generator that yields `(center_batch, context_batch)` as PyTorch tensors.

---

## 4. 🧮 Negative Sampling Loss (src/model.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, center, context, negatives):
        center_emb = self.in_embed(center)              # (B, D)
        context_emb = self.out_embed(context)           # (B, D)
        neg_emb = self.out_embed(negatives)             # (B, K, D)

        pos_score = torch.sum(center_emb * context_emb, dim=1)  # (B,)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_emb.neg(), center_emb.unsqueeze(2)).squeeze()  # (B, K)
        neg_loss = F.logsigmoid(neg_score).sum(dim=1)

        return -(pos_loss + neg_loss).mean()
```

Negative samples are drawn using **unigram distribution to the 3/4 power**.

---

## 5. 🚀 Pretraining Loop (src/train.py)

```python
def train(model, data_stream, optimizer, sampler, device):
    model.train()
    total_loss = 0
    for centers, contexts, negatives in sampler:
        centers, contexts, negatives = [x.to(device) for x in (centers, contexts, negatives)]
        optimizer.zero_grad()
        loss = model(centers, contexts, negatives)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss
```

### Optimization tricks:

- Subsample frequent words with probability `P(w) = 1 - sqrt(t / f(w))`
- Use torch `DataLoader` if you wrap your sampler into a Dataset
- Save checkpoints every N steps to `checkpoints/pretrain_epoch_*.pt`

---

## 6. 💾 Save & Load Embeddings

```python
# To save:
np.save("embeddings/word_vectors.npy", model.in_embed.weight.detach().cpu().numpy())

# To load:
model.in_embed.weight.data = torch.from_numpy(np.load("word_vectors.npy"))
```

---

## 7. 🔁 Fine-tuning on Hacker News Titles (src/fine_tune.py)

Same model architecture, same pipeline.

**Strategies to avoid catastrophic forgetting:**

- Lower LR (e.g., 1e-4 or 5e-5)
- Freeze `out_embed`, only update `in_embed`
- Fewer epochs
- Optional: regularize toward pretrained weights (L2 penalty)

```python
for param in model.out_embed.parameters():
    param.requires_grad = False
```

---

## 8. ⚙️ Configuration (train_config.yaml)

```yaml
embed_dim: 100
window_size: 5
min_count: 5
neg_samples: 10
learning_rate: 0.002
batch_size: 512
epochs: 5
```

---

## ✅ Reproducibility & Stability

- Set `torch.manual_seed()` and `np.random.seed()` early
- Use `torch.backends.cudnn.deterministic = True`
- Save `vocab.pkl` alongside weights for downstream alignment

---

## 📌 Notes on Trade-offs

| Design Choice               | Rationale / Trade-off                          |
| --------------------------- | ---------------------------------------------- |
| Skip-gram over CBOW         | Better with infrequent words, small corpus     |
| Negative sampling           | Scales better than full softmax (esp. on GPU)  |
| From-scratch batching       | Full control, no dependency on external tools  |
| Subsampling high-freq words | Speeds up training, improves embedding quality |
| Separate `in/out` matrices  | Improves semantic precision of embeddings      |

---

This plan is now ready for implementation.

You want to start with corpus loading/tokenization, or straight into training code?
