# RA-Rec — Paper Notes

**Title:** RA-Rec: An Efficient ID Representation Alignment Framework for LLM-based Recommendation
**Authors:** Xiaohan Yu, Li Zhang, Xin Zhao, Yue Wang, Zhongrui Ma (Huawei Poisson Lab + UCL)
**Venue:** arXiv:2402.04527 (Feb 2024)
**Source:** `docs/RA-Rec.pdf`

---

## 1. Core motivation — three LLM-Rec paradigms

| Paradigm | Approach | Example | Weakness |
|---|---|---|---|
| **ID Direct Usage** | IDs as text strings | P5 (`"user 15 bought items 115, 301, 24"`) | IDs carry no semantic → poor generalization |
| **ID Translation** | IDs → text titles | LLaMA4Rec (`"bought shoes, dress, watch"`) | Bounded length, cannot express CF interactions |
| **ID Representation** *(this work)* | Pretrained ID embeddings as soft prompts | RA-Rec | Need alignment between embedding space and LLM space |

→ RA-Rec is the third paradigm, complementary with the other two via **hybrid prompt**.

---

## 2. Hybrid Prompt design

**Hard prompt** (text, human-readable):
> "Your task is to recommend the next product user may be interested based on their purchase history: Geox J Arno Sneaker, Anni Coco Vintage Dress, Casio Digital Watch."

**Soft prompt** (continuous vectors):
- Pretrained user embedding `u ∈ ℝᵈ`
- Pretrained item embedding `i ∈ ℝᵈ`
- From any ID-based recommender F (e.g., SASRec, ComiRec)

**Combined:** soft prompts replace the descriptive portion of hard prompt; hard prompt provides world knowledge + cold-start stability.

---

## 3. Representation Alignment — two novel mechanisms

### 3.1 Reparameterization (layer-specific projection)

**Key insight:** Different LLM layers capture different semantic abstractions → inject ID embeddings **at each layer with layer-specific weights**, not just at input.

```
p^(l)_u = W^(l)_u · u + b^(l)_u
p^(l)_i = W^(l)_i · i + b^(l)_i
```

where `l ∈ [0, 1, ..., L]` is the layer index.

→ Each transformer layer gets its own projection of the same base user/item embedding.

### 3.2 Contextual Instruction (layer-specific prefixes)

Add **learnable continuous prefix vectors** as "virtual instructions" prepended to reparameterized ID embeddings at each layer:

```
d^(l)_u = [c^(l)_u || p^(l)_u]
d^(l)_i = [c^(l)_i || p^(l)_i]
```

where `c^(l)_u`, `c^(l)_i` are trainable prefixes, `||` is concatenation.

→ Inspired by prefix-tuning [39] — guides LLM how to use ID info at each layer.

### 3.3 Modified self-attention

Each LLM layer's self-attention is modified to attend to both hard prompt hidden states `h^(l)_u` and contextualized ID representations `d^(l)_u`:

```
Q = W_q · h^(l)_u
K = W_k · Concat(d^(l)_u, h^(l)_u)
V = W_v · Concat(d^(l)_u, h^(l)_u)
SA(Q, K, V) = softmax(Q^T K / √d) · V
```

→ ID representations and text hidden states are **jointly attended at every layer**.

### 3.4 Final prediction

After final layer L, prediction via dot product:
```
ŷ_{u,i} = h^(L)_u · h^(L)_i
```

---

## 4. Optimization

### 4.1 BPR pair-wise loss (main task)

```
L_p = Σ ln σ(ŷ_{u,i+} - ŷ_{u,i-}) - λ ||Θ||²
```

In-batch pos/neg pairs with regularization.

### 4.2 Contrastive alignment loss (auxiliary)

Two **separate forwards** through the same LLM:
1. **Hard prompt only** → get text-derived `h̃^(l)_u`, `h̃^(l)_i`
2. **Hybrid prompt** (hard + soft) → get ID-derived `d^(l)_u`, `d^(l)_i`

Then align them via InfoNCE at EACH layer:
```
L_ua = -Σ_l Σ_k log(exp(sim(d^l_{u,k}, h̃^l_{u,k}) / τ) / Σ_j exp(sim(d^l_{u,j}, h̃^l_{u,j}) / τ))
L_ia = similar for items
```

### 4.3 Final loss

```
L = L_p + λ(L_ua + L_ia)
```

→ The contrastive loss **bridges ID embedding space ↔ LLM text space** at every layer.

---

## 5. Efficient Tuning

### What's trainable

- ✓ Reparameterization weights: `W^(l)_u`, `b^(l)_u`, `W^(l)_i`, `b^(l)_i` (per layer)
- ✓ Contextual instruction prefixes: `c^(l)_u`, `c^(l)_i` (per layer)
- ✗ LLM frozen
- ✗ ID encoder F frozen

**Total trainable: ~18K params** (vs GPT-2 full finetune 774M = 0.002%).

### Data construction (key efficiency contributor)

| Strategy | Method |
|---|---|
| **Denoising** | Remove samples with **zero word overlap** between target item and history sequence |
| **Diversity (user)** | Use sequence length as proxy → bucket users by history length, uniform sample |
| **Diversity (item)** | Use item popularity as proxy → bucket items by popularity, uniform sample |

→ **6% of data → better than 100%.** Carefully selected subset > full data.

---

## 6. Results

### Datasets
- Amazon-Books: 80K users, 1.04M items, 4.2M interactions, sparsity 99.994%
- Amazon-Clothing: 100K users, 1.02M items, 3.4M interactions, sparsity 99.996%

### Main results (HR@100)

| Method | Books | Clothing | Type |
|---|---|---|---|
| ComiRec (best ID-based) | 0.1146 | 0.1142 | ID-based |
| LLaMA4Rec | 0.0257 | 0.0539 | LLM-based |
| TwinBert | 0.0831 | 0.1151 | LLM-based |
| **RA-Rec** | **0.1395** | **0.1438** | **Hybrid** |
| **Δ vs best baseline** | **+21.7%** | **+25.9%** | |

### Efficiency

| Method | Params | GPU·hrs |
|---|---|---|
| Full LLM finetune | 774M | 32 |
| LLM finetune + align | 774M | 41 |
| **RA-Rec** | **18K** | **16** |

→ 50% training time, 0.002% params, near peak at 7K steps.

### Ablations

- Without ID representation → equivalent to prefix tuning → significantly worse
- Without layer-specific reparameterization → uniform injection → worse
- Without contextual instruction → still worse than full RA-Rec
- → All 3 components contribute; reparameterization most impactful

---

## 7. Ideas applicable to SigLLM

### 🥇 Idea 1 — **Layer-specific soft token injection** (vs current input-only)

**SigLLM hiện tại:** soft tokens injected at LLM **input embedding** only — single point of contact.

**RA-Rec idea:** Inject CF representations at **EVERY LLM layer** with layer-specific projectors. Each layer gets its own projection of the same Q-Former output.

**Adaptation for SigLLM:**
```python
# Current
soft_tokens = llm_proj(qformer_out)  # [B, Q, H]
# Injected at input only

# Proposed
qformer_out = qformer(cf)  # [B, Q, 768]
self.layer_projs = nn.ModuleList([
    nn.Linear(768, H) for _ in range(num_llm_layers)
])
# At each layer i, inject layer_projs[i](qformer_out)
```

**Trade-off:**
- Pros: Deeper integration, each layer leverages CF differently (early = syntax, late = semantics)
- Cons: Need refactor LLM forward (similar to USER-LLM gated cross-attention complexity)
- Effort: 1-2 weeks
- Probability: 25-35%

### 🥈 Idea 2 — **Layer-specific learnable prefix prompts**

**RA-Rec contextual instruction** = learnable prefix at each layer concat with soft tokens. Lightweight (~18K params total).

**For SigLLM:**
```python
self.layer_prefixes = nn.ParameterList([
    nn.Parameter(torch.randn(prefix_len, H)) for _ in range(num_llm_layers)
])
# At layer i: prepend layer_prefixes[i] to soft tokens
```

Adds ~`num_layers × prefix_len × H` params. For Qwen2-7B (28 layers, prefix_len=4, H=3584) → ~400K params.

**Effort:** 1 week.
**Probability:** 20-30%.

### 🥉 Idea 3 — **Text-only baseline contrastive alignment loss** ⭐

**RA-Rec L_ua / L_ia:** Forward LLM TWICE (text-only + text+ID), align hidden states via InfoNCE.

**Adaptation for SigLLM Stage 3:**
```python
# Forward 1: text-only prompt (no soft tokens)
prompt_text_only = "Given user history: <ItemTitleList>. Target: <TargetItemTitle>. Answer:"
h_text = llm(prompt_text_only).hidden  # [B, T1, H]

# Forward 2: hybrid prompt (text + soft tokens)
prompt_hybrid = current v2 prompt
h_hybrid = llm(prompt_hybrid).hidden  # [B, T2, H]

# Align hidden states (at last layer, mean-pooled)
loss_align = InfoNCE(mean_pool(h_hybrid), mean_pool(h_text), in_batch_negatives)

loss = loss_ctr + λ * loss_align
```

**Why this is interesting:**
- Standard distillation idea but at LLM internal level
- Forces hybrid representation to stay "close to" text-only baseline → prevents CF signal corrupting text understanding
- Memory cost: 2x forward
- **Different from CALRec** — CALRec aligns target tower vs user tower; **RA-Rec aligns text-only vs hybrid for same sample**

**Effort:** 2-3 days.
**Probability:** 25-35%.

### Idea 4 — **Data filtering by overlap**

**RA-Rec denoising:** remove training samples with zero word overlap between target title and history titles.

**For SigLLM ML-1M:**
- Compute overlap between `TargetItemTitle` and `InteractedItemTitles`
- Drop "irrelevant" pairs
- Could improve training signal density

**Effort:** 0.5 day.
**Probability:** 10-15% (ML-1M small, removing data might hurt).

### Idea 5 — **Pretrain on more diverse data, filter for quality**

**RA-Rec diversity:** bucket users by history length, items by popularity, uniform sample.

For SigLLM: ML-1M is relatively small. Could pretrain on Amazon-Book + sample by RA-Rec diversity strategy → finetune on ML-1M.

**Effort:** 1 week.
**Probability:** 20-30%.

---

## 8. Comparison with related papers (cross-reference)

| Aspect | SigLLM (current) | RA-Rec | USER-LLM | CALRec |
|---|---|---|---|---|
| **Soft token injection point** | Input only | Every layer (reparam) | Cross-attn at multiple layers | N/A (text prompts) |
| **Layer-specific projectors** | ✗ Shared | ✓ Per layer | ✓ Per layer (gated) | ✗ |
| **Contrastive alignment loss** | ✗ | ✓ Text-only vs hybrid | ✗ | ✓ Target vs user-target |
| **Pretrained ID encoder** | MF (frozen) | Any ID model (frozen) | Autoregressive (pretrained) | LLM itself (no separate encoder) |
| **LLM frozen?** | ✓ Yes (LoRA on top) | ✓ Yes (only align) | ✓ Yes (only cross-attn) | ✗ Fully finetuned |
| **Trainable params** | Q-Former 50-76M + LoRA 2.5M + proj 2.8M | ~18K | Cross-attn layers (~M) | Full LLM (~B) |
| **Efficiency angle** | None | **Extreme (18K params)** | Inference speedup | Multi-stage pretraining |

---

## 9. Key takeaway for thesis

**RA-Rec validates 3 principles that SigLLM can leverage:**

1. **Layer-specific injection beats input-only** — every transformer layer has a different semantic role; injecting at all layers (with layer-specific projection) is more effective than input-only.

2. **Auxiliary contrastive alignment is cheap and helps** — forward LLM with text-only AND hybrid prompts, align via InfoNCE at hidden state level. Small compute overhead, no architecture change.

3. **Data quality > data quantity** — denoising + diversity sampling > full data. Worth filtering ML-1M training samples.

**For SigLLM specifically:**
- Idea 3 (text-only baseline contrastive alignment) = **highest ROI** — directly addresses SigLLM's "Stage 3 loss is only binary CE" gap
- Idea 1 (layer-specific injection) = **highest theoretical motivation** but requires LLM forward refactor
- Idea 4 (data filtering) = **cheapest** — apply RA-Rec denoising to ML-1M

---

## 10. Citation key

```bibtex
@article{yu2024rarec,
  title={RA-Rec: An Efficient ID Representation Alignment Framework for LLM-based Recommendation},
  author={Yu, Xiaohan and Zhang, Li and Zhao, Xin and Wang, Yue and Ma, Zhongrui},
  journal={arXiv preprint arXiv:2402.04527},
  year={2024}
}
```
