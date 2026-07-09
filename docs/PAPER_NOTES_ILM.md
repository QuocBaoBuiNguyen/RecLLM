# ILM — Paper Notes

**Title:** Item-Language Model for Conversational Recommendation
**Source:** `docs/ILM.pdf`
**Status:** **Direct ancestor of SigLLM** — SigLLM inherits ILM pipeline + 5-loss recipe

---

## 1. Core idea

Two-phase framework using BLIP-2's Q-Former architecture for recommendation:
- **Phase 1**: Item-language representation learning (pretrain Q-Former + text branch)
- **Phase 2**: Item-language model training (freeze LLM, train Q-Former + linear adapter)

Treats user as a **"special item"** — same Q-Former processes both user_cf and item_cf.

---

## 2. Architecture

```
Item/User CF embedding (from MF)
        ↓
    Q-Former encoder (8 transformer layers, BERT-init text branch)
        ↓ K learnable queries (cross-attend CF)
    Linear projection adapter
        ↓
    Frozen LLM (PaLM 2-S or T5 8-layer)
        ↓
    Text output (conversational rec)
```

**Key design choices:**
- Q-Former 8 layers (vs SigLLM 4)
- Linear projection adapter (single layer)
- LLM frozen, only Q-Former + adapter trained at Phase 2

---

## 3. Phase 1 — 5 contrastive losses

### 3.1 Standard BLIP-2 trio (item-text)

**ITC (Item-Text Contrastive)** — InfoNCE between Q-Former queries and text CLS:
- Select query closest to text CLS via cosine sim
- In-batch negatives

**ITM (Item-Text Matching)** — binary classifier (match/no-match):
- Hard negatives mined from ITC similarity matrix
- Joint forward (CF + text), pooled queries → binary head

**ITG (Item-Text Generation)** — causal LM:
- Q-Former queries condition text branch generation
- Predict item text caption from CF

### 3.2 ILM-specific additions

**II (Item-Item Contrastive)** — co-watched item pairs:
- Pair (i1, i2) = 2 items watched consecutively by same user
- InfoNCE between Q-Former encodings via `select_pair_by_similarity`

**UI (User-Item Contrastive)** — user-positive item pairs:
- Pair (u, i) = user u rated item i positively
- "Replace item CF with user CF" — same Q-Former encodes both
- InfoNCE between encoded user and item

### 3.3 Ablation results (Table 4, ML1M HR@5)

| Phase 1 loss config | NDCG@5 |
|---|---|
| ILM-IT (item-text only) | 0.0474 |
| ILM-IT-II (+ item-item) | 0.0479 |
| **ILM-IT-UI (+ user-item)** | **0.0485** (+2.3%) |

→ UI loss best on ML1M because **ML1M has scarce item-text but abundant user-item interactions**.

### 3.4 Regularization effect (Table 5)

| Method | Phase 1 ITG eval loss |
|---|---|
| ILM-IT (no II/UI) | 4.17 (large train-eval gap) |
| ILM-IT-UI | 4.07 (smaller gap, less overfit) |

→ UI loss **regularizes** Phase 1 training.

---

## 4. Phase 2 — generative pretraining

- LLM completely frozen
- Q-Former + linear projection trainable
- Next-token LM on item caption text
- Soft tokens prepended to caption tokens

→ This is exactly **SigLLM Stage 2** (verified identical).

---

## 5. Conversational rec setup

### Prompt examples (Figure 1)

```
"Write a long summary of the movie The Shoe (1998) {item}."
"user_123 {user} has interacted with items {history}. What is the next recommendation?"
"Based on the user's rating and tag history of {history}, what would their anticipated rating be for {item}?"
```

→ `{user}`, `{item}`, `{history}` are **placeholders** filled with Q-Former-encoded soft tokens.

→ **ILM explicitly injects user CF via `{user}` placeholder** — this is exactly what SigLLM's "re-enable user soft tokens" restores.

---

## 6. Evaluation setup

- **NOT CTR (Yes/No)** — generation-based rec
- Metrics: HR@5/@10, NDCG@5/@10 (retrieval)
- 24 ELM conversational tasks (summary, review, persuasion, etc.)
- OpenP5 sequential + straightforward rec
- Seen vs unseen prompt evaluation (zero-shot test)

→ **Multi-task + zero-shot setting** — different from SigLLM single-task CTR.

---

## 7. Key results

### ELM 24 tasks (semantic consistency)

| Method | Average SC% |
|---|---|
| MLP (CoLLM-style baseline) | 77.48 |
| ILM-rand (random init Q-Former) | 78.96 |
| **ILM** | **80.01** (+3.27% relative) |

### OpenP5 ML1M (HR@5)

| Method | Seen | Unseen (zero-shot) |
|---|---|---|
| OpenP5-R (text only) | 0.0688 | 0.0696 |
| MLP | 0.0692 | 0.0716 |
| ILM-rand | 0.0723 | 0.0710 |
| **ILM** | **0.0724** | **0.0717** |

→ ILM consistently > MLP baseline.

---

## 8. Insights inherited by SigLLM

### What SigLLM copied directly from ILM

| Component | ILM | SigLLM | Status |
|---|---|---|---|
| Q-Former architecture | 8 layers | 4 layers (shrunk) | Inherited (modified depth) |
| Phase 1 5-loss recipe | ITC + ITM + ITG + II + UI | Same 5 losses | **Inherited** |
| Phase 2 generative | Linear proj + LLM forward + LM loss | Same | **Inherited** |
| "User as special item" | Same Q-Former for user + item | Same | **Inherited (after re-enable)** |
| `{user}` slot in prompt | YES | YES (with `enable_user_soft_tokens=True`) | **Inherited** |

### What SigLLM diverges from ILM

| Aspect | ILM | SigLLM |
|---|---|---|
| Task | Conversational rec + generative retrieval | Single-task CTR (Yes/No) |
| Metric | HR/NDCG | AUC/uAUC |
| LLM | PaLM 2-S / T5 (frozen) | Qwen2-7B (frozen + LoRA) |
| Fine-tuning method | Q-Former + adapter only | + CoLLM 2-step LoRA at Stage 3 |
| Eval prompts | Seen + unseen (zero-shot) | Only seen (no zero-shot test) |

---

## 9. Why instruction-aware fails on SigLLM (per ILM context)

ILM evaluates on **multi-task + zero-shot prompts** — where instruction routing has signal.
SigLLM is **single-task CTR** — no routing signal for instruction-aware queries to learn.

→ SigLLM negative finding "instruction-aware no improvement" is **CONSISTENT with ILM's evidence** that instruction routing benefits multi-task scenarios.

---

## 10. Key takeaways for SigLLM thesis

1. **ILM Table 4 validates UI loss** on ML1M for SigLLM Stage 1 (+2.3% NDCG@5)
2. **ILM Table 5 validates regularization** — UI/II reduce train-eval gap
3. **ILM uses `{user}` slot natively** — SigLLM's re-enable user soft tokens = restoring ILM design (not novel addition)
4. **ILM 8-layer Q-Former on OpenP5 datasets** vs SigLLM 4-layer (and shrunk 2-layer test) on ML1M — depth choice differs based on data scale
5. **ILM doesn't ablate user soft tokens vs no-user** — only default design. SigLLM cold-item finding (+1.79 uAUC) is **novel evidence** that ILM paper doesn't have.

→ SigLLM's cold-item finding is **new contribution** — ILM paper didn't isolate this scenario.

---

## 11. Citation key

```bibtex
@article{ilm2024,
  title={Item-Language Model for Conversational Recommendation},
  author={...},
  year={2024}
}
```

(Verify exact bibtex from paper PDF)
