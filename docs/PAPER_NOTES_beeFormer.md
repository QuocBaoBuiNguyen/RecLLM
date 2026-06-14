# beeFormer — Paper Notes

**Title:** beeFormer: Bridging the Gap Between Semantic and Interaction Similarity in Recommender Systems
**Authors:** Vojtěch Vančura, Pavel Kordík, Milan Straka (Czech Technical University + Recombee)
**Venue:** RecSys 2024
**Source:** `docs/beeFormer.pdf`

---

## 1. Core idea — **CF enrichment via interaction-trained text encoder**

Unlike the 5 papers documented earlier (ILM, XIB, USER-LLM, CALRec, RA-Rec) that all **take CF as fixed input** and focus on bridge to LLM, beeFormer **enriches the CF representation itself** by training a sentence Transformer on interaction data.

**Key insight:** Sentence Transformers (e.g., MPNet) are trained for **semantic similarity** but fail to capture **interaction patterns**. beeFormer bridges this gap.

> "Users may look for a specific item (for example, batteries when buying a kid's toy, or cables when buying a new printer) with very low semantic similarity compared to other items in the catalog."

→ beeFormer text encoder learns: "These items go together based on interactions, not just text."

---

## 2. Architecture

```
Item text descriptions (T = {t_1, t_2, ..., t_I})
        ↓
  Sentence Transformer g(T, θ_g) — TRAINABLE
        ↓
  Item embedding matrix A ∈ R^(I × d)
        ↓ Use as ELSA latent factor
  Reconstruction loss on interaction matrix X
        ↓
  Gradient backprop → update θ_g (Transformer weights)
```

Uses **ELSA architecture** (Embarrassingly Shallow Linear Autoencoder, scalable variant of EASE):
```
X_pred = X_u · (AA^T - I)
loss = ||norm(X_u) - norm(X_pred)||²_F
```

→ Transformer learns to produce embeddings A such that **dot-product** between embeddings recovers user-item interactions.

---

## 3. Training challenges + solutions

### Challenge: Full-batch problem

ELSA requires ALL item embeddings for each training step → batch size = full item count (millions for some datasets) → memory infeasible.

### 3 solutions combined

1. **Gradient checkpointing** — recompute activations vs storing
2. **Gradient accumulation** — split into micro-batches
3. **Negative sampling** — subsample items per step

```python
# beeFormer Algorithm 1 (simplified)
def beeformer_step(X_batch, transformer, tokenized_texts):
    # 1. Compute A in batches WITHOUT gradients
    A = []
    for batch in tokenized_texts:
        A.append(transformer(batch).detach())
    A = concat(A)
    
    # 2. Compute predictions + loss
    X_pred = X_batch @ (A @ A.T - eye(I))
    loss = ||norm(X_batch) - norm(X_pred)||²
    
    # 3. Backprop through A only (gradient checkpoint)
    grad_A = autograd.grad(loss, A)
    
    # 4. Re-compute A with gradients for θ_g
    for batch in tokenized_texts:
        A_batch = transformer(batch)
        A_batch.backward(grad_A[batch_range])  # use checkpointed grad
        # accumulate gradients
    
    optimizer.step()  # update transformer weights
```

---

## 4. Three evaluation scenarios

### Scenario 1: Item-split (zero-shot)
- Train on subset of items, test on UNSEEN items
- Pure text-based predictions (no CF history for new items)
- → beeFormer outperforms baselines

### Scenario 2: Cold-start
- New items with text but no interactions
- Need text encoder to predict item interactions
- → beeFormer outperforms baselines

### Scenario 3: Time-split
- Train on past, test on future interactions
- → beeFormer competitive

---

## 5. Key results

### Datasets
- **MovieLens-20M** (movies)
- **GoodBooks-10K** (books)
- **GoodLens** (combined MovieLens + GoodBooks)

### Item-split zero-shot (cross-domain, Table 2)

| Method | NDCG@10 |
|---|---|
| Pure sentence-T (semantic only) | baseline |
| Pure CF (no text) | baseline |
| **beeFormer (interaction-trained text)** | **significantly outperforms all** |

### Cold-start (Table 4)
- beeFormer outperforms baselines (with Heater approach)
- Models trained on **combined datasets > single-dataset**

### Cross-domain transfer
- Train beeFormer on books → apply to movies → outperforms semantic baseline
- Training on **multiple domains together** further boosts performance
- → Suggests **universal, domain-agnostic** text encoder is possible

---

## 6. Comparison with other CF-enriching approaches

| Method | What's enriched | How |
|---|---|---|
| **beeFormer** | **Text encoder via interactions** | ELSA-style backprop on interaction matrix |
| BinLLM | Binary IP encoding | Binary representation of CF |
| SellaRec | Semantic-aware projection | Project CF to semantic space |
| Standard MF | None (pure interaction) | Matrix factorization |
| Sentence-T (vanilla) | None (pure semantic) | Pretrained on NLI |

→ **beeFormer is unique** — it combines semantic + interaction in ONE encoder.

---

## 7. Applicability to SigLLM

### Current SigLLM source pipeline
```
Item text title → bert-base-uncased (in Q-Former text branch)
Item ID → MF embedding (256-d, from CF training)
```

Both are **separate** — text encoder doesn't know interaction patterns, MF doesn't know text.

### beeFormer-style enrichment for SigLLM

**Option 1: Replace Q-Former's text encoder with beeFormer-MPNet**

```python
# Current SigLLM Q-Former
qformer_text_model_name: "bert-base-uncased"

# Proposed
qformer_text_model_name: "beeformer/movielens-mpnet"  # or similar
```

**Pros:**
- ✓ Text encoder already knows interaction patterns for movies (trained on ML-20M)
- ✓ Direct architectural compatibility (sentence Transformer drop-in)
- ✓ Stage 1 ITC/ITM/ITG losses benefit from CF-aware text representations
- ✓ Free CF enrichment without changing MF source

**Cons:**
- ✗ Need pretrained beeFormer-MPNet (or train one)
- ✗ Distribution shift may require careful Stage 1 retrain

**Effort:** 1-2 days (if pretrained available) + Stage 1+2+3 retrain.
**Probability:** **25-35%** (direct CF enrichment, addresses real gap).

**Option 2: Train beeFormer-style enrichment on SigLLM data**

Pretrain MPNet on ML-1M interactions → use as Q-Former text branch init.

**Effort:** 3-5 days (data prep + train beeFormer on ML-1M).
**Probability:** 20-30%.

### What makes beeFormer compelling for SigLLM

1. **Direct attack on CF enrichment** — None of the 5 documented papers do this
2. **Text encoder is the easiest swap** — Q-Former architecture unchanged
3. **Particularly helps cold-start** — beeFormer's main claim, matches SigLLM's user-soft-token cold-item finding
4. **Pretrained checkpoints available** — `huggingface.co/beeformer/...` if matches dataset

---

## 8. CF enrichment landscape (cross-paper)

| Approach | Source | Method | Status for SigLLM |
|---|---|---|---|
| Pure MF (current) | MF | BPR-style factorization | ✓ Used |
| LightGCN | GCN | Graph propagation | Could replace MF |
| SASRec | Transformer | Self-attention over sequences | USER-LLM style |
| beeFormer | Sentence-T + ELSA | Text encoder trained on interactions | **CF-aware text encoder** |
| BinLLM | Binary IP | Binary encoding of CF embedding | Different paradigm |
| Hybrid (Idea) | MF + beeFormer text | Multi-channel CF | Future combination |

---

## 9. Insights for SigLLM thesis

### CF enrichment as missing dimension

SigLLM has explored:
- ✓ Bridge architecture (Q-Former vs shrink variants)
- ✓ LLM integration (soft tokens, instruction-aware, user-conditioned)
- ✓ Prompt structure (v2 labeled)
- ✗ **Source enrichment** (MF kept simple)

→ beeFormer fills this gap. **Untested direction with strong literature support.**

### Practical recommendation

If using beeFormer-MPNet as Q-Former text branch:
1. Download pretrained `beeformer/movielens-mpnet` (if exists)
2. Pass to `HFQFormerAdapter(qformer_text_model_name=...)`
3. Verify token vocab compatible (MPNet uses different tokenizer than BERT)
4. Retrain Stage 1+2+3 (Q-Former text branch changes distribution)

**Could combine with current shrink + v2 prompt experiments** for stacked improvement.

---

## 10. Citation key

```bibtex
@inproceedings{vancura2024beeformer,
  title={beeFormer: Bridging the Gap Between Semantic and Interaction Similarity in Recommender Systems},
  author={Vančura, Vojtěch and Kordík, Pavel and Straka, Milan},
  booktitle={RecSys '24},
  year={2024},
  doi={10.1145/3640457.3691707}
}
```
