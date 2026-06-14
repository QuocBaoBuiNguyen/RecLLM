# CALRec — Paper Notes

**Title:** CALRec: Contrastive Alignment of Generative LLMs for Sequential Recommendation
**Authors:** Yaoyiran Li, Xiang Zhai, Moustafa Alzantot, Keyi Yu, Ivan Vulić, Anna Korhonen, Mohamed Hammad (Cambridge + Google)
**Venue:** RecSys 2024 (Bari, Italy)
**Source:** `docs/CALRec.pdf`

---

## 1. Core idea

Two-tower contrastive framework that finetunes a pretrained LLM with **mixture of 3 losses**:
- **NIG (Next Item Generation)** — LM loss on target item text
- **L_TT (Target-Target Conditional Alignment)** — InfoNCE
- **L_UT (User-Target Alignment)** — InfoNCE

Plus **two-stage finetuning**: multi-category joint → target-domain specific.

→ Reports **+37% Recall@1, +24% NDCG@10** improvements over SOTA baselines.

---

## 2. Two-tower architecture

### 2.1 Tower 1: Target tower
- Input: target item text only
- LLM forward
- Output: `v_T` = mean-pooled hidden state of last layer

### 2.2 Tower 2: User-target joint tower
- Input: user history + target item (joint)
- LLM forward
- Outputs:
  - `v_U` = mean-pooled hidden states corresponding to user history portion
  - `v_T|U` = mean-pooled hidden states corresponding to target portion (conditioned on user)

→ **Same LLM** processes both towers (parameter sharing).

---

## 3. Three losses

### 3.1 NIG (Next Item Generation)
```
L_NIG = E_t [Σ_{j=m+1}^l log P(t_j | t_{1:j-1}; θ)]
```

Standard causal LM loss on target item text tokens (positions m+1 to l).

→ "Generate the target item text given user history" — standard sequence generation.

### 3.2 L_TT — Target-Target Conditional Alignment
```
L_TT = -1/N · Σ log[ exp(cos(v_T|U_i, v_T_i) / τ) / Σ_j exp(cos(v_T|U_j, v_T_i) / τ) ]
```

In-batch contrastive between:
- `v_T|U_i` = target representation conditioned on user history
- `v_T_i` = target representation alone

→ **"Target conditioned on user should match target alone"** for positive pairs.

### 3.3 L_UT — User-Target Alignment
```
L_UT = -1/N · Σ log[ exp(cos(v_U_i, v_T_i) / τ) / Σ_j exp(cos(v_U_j, v_T_i) / τ) ]
```

In-batch contrastive between:
- `v_U_i` = user history representation
- `v_T_i` = target representation

→ **"User history embedding should match positive target"** in-batch negatives.

### 3.4 Final loss
```
L_CALRec = (1 - α - β) · L_NIG + α · L_TT + β · L_UT
```

α, β hyperparameters (paper finds α=0.5, β=0.5 effective).

---

## 4. Two-stage finetuning

### Stage I: Multi-Category Joint Fine-Tuning
- Use 9 Amazon categories (3.59M users)
- 7 categories used only for Stage I (3.37M users)
- Joint training across categories — model learns shared cross-domain knowledge

### Stage II: Category-Specific Fine-Tuning
- Take Stage I checkpoint
- Fine-tune on target category (0.22M users)
- Transfer learning benefit from Stage I

### Ablation results
Both stages crucial — combined > Stage II alone > Stage I alone.

---

## 5. Quasi-Round-Robin BM25 Retrieval

### Inference problem
- LLM generates text predictions for next item
- Need to map text predictions to actual item IDs (must be in catalog)

### Solution
1. **Temperature sampling**: generate N_gen candidate text predictions
2. **BM25 retrieval**: match each prediction to top-K closest catalog items
3. **Round-robin selection**: take top item from each prediction's retrieved set
4. **Score combination**: weighted by LLM log-probability

→ Bridges generation (continuous text) to retrieval (discrete catalog).

---

## 6. Architecture details

- **LLM backbone**: PaLM-2 XXS
- **Fully fine-tuned** (not frozen) — different from many LLM-Rec papers
- **Hard prompts**: text descriptions of items in user history
- **No soft tokens / no ID embeddings** — pure text-based

→ CALRec is in **ID Translation paradigm** (per RA-Rec classification), not ID Representation.

---

## 7. Key results

### Amazon datasets — main results
| Metric | Best baseline | CALRec | Δ |
|---|---|---|---|
| Recall@1 | (varies) | **+37% relative** | — |
| NDCG@10 | (varies) | **+24% relative** | — |

### Ablation: each loss component
- NIG only: weakest
- NIG + L_TT: +moderate
- NIG + L_UT: +moderate
- **NIG + L_TT + L_UT: best**

### Two-stage benefit
- Stage I only (multi-category): moderate
- Stage II only (single category): moderate
- **Stage I → Stage II**: best (transfer benefit)

---

## 8. Comparison with SigLLM design

| Aspect | CALRec | SigLLM |
|---|---|---|
| **LLM tuning** | Fully fine-tuned | Frozen + LoRA |
| **Soft tokens?** | No (text-only prompts) | Yes (CF soft tokens) |
| **Source CF model** | None | MF (matrix factorization) |
| **Forward passes** | 2-3 (target, user, joint) | 1 (joint only) |
| **Aux losses** | 2 contrastive (L_TT, L_UT) | None (only binary CE) |
| **Multi-domain pretrain** | YES (Stage I) | No (only ML-1M) |
| **Task** | Sequential rec (HR/NDCG) | CTR (Yes/No, AUC/uAUC) |

---

## 9. Ideas applicable to SigLLM

### 🥇 Idea 1 — Two-tower contrastive aux loss at Stage 3 ⭐

**The most actionable CALRec idea for SigLLM.**

**Hypothesis:** SigLLM Stage 3 currently has ONLY binary CE on Yes/No. Add CALRec-style InfoNCE aux losses to provide explicit item discrimination signal beyond match/no-match.

**Adaptation for SigLLM Stage 3:**

```python
# Forward 1: User history only (text-only LLM forward)
prompt_U = "Given user history: <ItemTitleList>. Profile:"
h_U = llm(prompt_U).hidden_states[-1]
v_U = mean_pool(h_U)

# Forward 2: Target only
prompt_T = "Target movie: <TargetItemTitle> <TargetItemID>. Profile:"
h_T = llm(prompt_T).hidden_states[-1]
v_T = mean_pool(h_T)

# Forward 3: Joint (current SigLLM)
prompt_joint = current v2 prompt
h_joint = llm(prompt_joint).hidden_states[-1]
v_TU = mean_pool(h_joint[target_positions])
ctr_logits = ...  # current binary head

# Losses
loss_ctr = CE(ctr_logits, label)
loss_TT = InfoNCE(v_TU, v_T, in_batch_neg, tau=0.07)
loss_UT = InfoNCE(v_U, v_T_positives_only, in_batch_neg, tau=0.07)

loss = 0.6 * loss_ctr + 0.2 * loss_TT + 0.2 * loss_UT
```

**Trade-offs:**
- Pros: Explicit ranking signal, in-batch negatives are "free", aux losses force discriminative representations
- Cons: 2-3x forward cost, 2-3x memory cost
- Effort: 2-3 days
- Probability: **25-35%**

**Memory mitigation:** Compute contrastive every K=4 steps (not every step) — saves compute, still gets aux signal.

### 🥈 Idea 2 — Multi-domain pretraining (Stage I)

**Hypothesis:** SigLLM only trains on ML-1M (33K samples). Pretrain Stage 1+2 on Amazon-Book or Amazon-Clothing first, then fine-tune ML-1M.

**Adaptation:**
1. Run Stage 1 (5-loss pretrain) on Amazon-Book CF data
2. Run Stage 2 (generative) on Amazon-Book item captions
3. Transfer Stage 2 ckpt as init for ML-1M Stage 3

**Effort:** 5-7 days (data prep + heavy compute).
**Probability:** 25-35% — especially helps cold-start.

### 🥉 Idea 3 — NIG-style auxiliary text generation loss

**Hypothesis:** Force Q-Former + LLM to be able to generate target item text (not just classify Yes/No). Forces representation to be richer.

**Adaptation:**
- Stage 3 add aux loss: given user history, generate target movie title text
- Auxiliary head + cross entropy on title tokens

**Effort:** 2-3 days.
**Probability:** 15-25%.

### Idea 4 — BM25-style item resolution
Not applicable to SigLLM (point-wise CTR, not retrieval — no need to map text → item ID).

---

## 10. CALRec vs RA-Rec contrastive — different alignments

| Method | Contrastive aligns | Implication |
|---|---|---|
| **CALRec L_TT** | `v_T|U` ↔ `v_T` (same sample, joint vs target-only) | "Target with user context = target alone" |
| **CALRec L_UT** | `v_U` ↔ `v_T` (in-batch, user vs positive target) | "User history near positive target" |
| **RA-Rec L_ua** | ID-derived `d^l_u` ↔ text-derived `h̃^l_u` (same sample) | "ID and text representations should match" |
| **RA-Rec L_ia** | Similar for items | |

→ **CALRec aligns cross-sample (user vs target)**, **RA-Rec aligns same-sample (text vs hybrid)**.

→ Both add InfoNCE aux losses to Stage 3 / fine-tuning. Different alignment targets.

---

## 11. Key takeaways for SigLLM thesis

### What CALRec validates for SigLLM
1. **Contrastive aux losses help LLM-Rec** — Stage 3 binary CE alone is suboptimal
2. **Two-tower / multi-pass forwards** are tractable for thesis-scale experiments
3. **In-batch negatives are free supervision**

### What CALRec challenges for SigLLM
1. **Single-task ML-1M is limited** — multi-category pretraining helps
2. **Pure text prompts (no soft tokens) can win** — challenges Q-Former value
3. **Full LLM finetuning** beats LoRA at some metric → SigLLM's frozen-LLM approach may underperform CALRec's PaLM 2-XXS full FT

### Most actionable for SigLLM
- **Idea 1 (two-tower contrastive)** = highest ROI — directly add aux losses to current Stage 3 step 2
- Idea 2 (multi-domain pretrain) = high effort but high payoff for cold-start

---

## 12. Citation key

```bibtex
@inproceedings{li2024calrec,
  title={CALRec: Contrastive Alignment of Generative LLMs for Sequential Recommendation},
  author={Li, Yaoyiran and Zhai, Xiang and Alzantot, Moustafa and Yu, Keyi and Vulić, Ivan and Korhonen, Anna and Hammad, Mohamed},
  booktitle={RecSys '24},
  year={2024},
  doi={10.1145/3640457.3688121}
}
```
