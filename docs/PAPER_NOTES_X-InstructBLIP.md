# X-InstructBLIP — Paper Notes

**Title:** X-InstructBLIP: A Framework for Aligning Image, 3D, Audio, Video to LLMs and its Emergent Cross-modal Reasoning
**Authors:** Artemis Panagopoulou, Le Xue, Ning Yu, Junnan Li et al. (Salesforce AI Research)
**Venue:** ECCV 2024
**Source:** `docs/X-Instruct-BLIP-eval-single-mutli-task.pdf`

---

## 1. Core idea

Extend InstructBLIP framework from image-only to **4 modalities** (image, 3D, audio, video) → frozen LLM with per-modality Q-Former projections. Show **emergent cross-modal reasoning** despite separate per-modality training.

→ Most relevant for SigLLM as it explicitly compares **Q-Former vs Linear Projection (LP)** for projection layer choice.

---

## 2. Two projection mechanisms compared

### 2.1 X-Instruct Projection (Q-Former based)

```
Modality input → Modality encoder (frozen)
                     ↓
              Instruction-aware Q-Former
                     ↓ K learnable queries + instruction text
              Linear projection (LP_M)
                     ↓
              [modality prefix] + soft tokens + text prompt
                     ↓
                Frozen LLM
```

- Q-Former initialized from BLIP-2 pretrained weights
- Per-modality: separate Q-Former for image, 3D, audio, video
- **Modality prefix** prepended to soft tokens (e.g., "audio:", "3D:")

### 2.2 X-LLaVA-style Projection (LP)

```
Modality input → Modality encoder
                     ↓
              Single Linear Projection: f_M : R^d_M → R^(k · d_LLM)
                     ↓
              Reshape to K tokens
                     ↓
                Frozen LLM
```

→ **No transformer, no instruction conditioning, just one linear layer.** Equivalent to CoLLM-MF approach.

---

## 3. Key finding — Q-Former vs LP trade-off

### Abstract quote
> "The Q-Former projection demonstrates **superior performance in single modality scenarios** and adaptability in joint versus discriminative reasoning involving two or more modalities. However, it exhibits **lower generalization capabilities than linear projection in contexts where task-modality data are limited**."

### Examples from Tables

| Task | LP wins? | Reason |
|---|---|---|
| 3D Classification ModelNet40 | LP loses 46.4 vs Q-Former 62.8 | Sufficient data, complex task |
| Audio ESC50 close | **LP wins 67.4 vs Q-Former 62.8** | Limited data, simpler task |
| Audio ClothoAQA | LP comparable | Mixed |
| Image VizWiz (low resource) | LP loses | LP underfits |
| DisCRn (audio-video) | **LP wins 47.1 vs Q-Former 34.0** | Multi-modal complexity |

→ Pattern: **LP wins when data is limited; Q-Former wins when sufficient data**.

→ **Implication for SigLLM:** ML-1M = 33K samples = LIMITED data → LP may compete or win vs Q-Former.

---

## 4. Modality prefix — CONSISTENT improvement

### Section 5.3 Table 7 result

Removing modality prefix (e.g., "audio:") consistently hurts:

| Task | With prefix | No prefix | Δ |
|---|---|---|---|
| Audio ClothoAQA | better | worse | + |
| 3D ModelNet40 | better | worse | + |
| MusicAVQA joint | better | worse | + |

### Quote
> "The Q-Former is **relieved from the extra burden to encode the type of modality** and instead **reserves bandwidth for semantic information**."

→ **Strong evidence:** prefix labeling soft tokens helps LLM consume them efficiently.

→ **Direct apply to SigLLM:** add labels before each soft token slot ("User feature is X", "Target item feature is Y") — matches SellaRec SOTA prompt style.

---

## 5. Other findings

### 5.1 Q-Former init transfers fast (Section 5.2)
> "the Video Q-Former component of X-InstructBLIP, initialized with the Image Q-Former's weights, reaches convergence in performance **remarkably fast, within about 1,000 iterations**."

→ BLIP-2 pretrained Q-Former weights transfer well even across modalities. Implication: ILM/BLIP-2 init for SigLLM Q-Former is valuable.

### 5.2 Stage 2 finetuning skip = mild drop (Section 5.1)
> "mild drop in performance overall, likely due to the lack of BLIP2 Stage-2 finetuning"

→ Stage 2 helps but not catastrophic to skip. Confirms SigLLM design choice to keep Stage 2.

### 5.3 Expanded prompt templates = trade-off
> "the expanded template space introduces a **trade-off of generalization and performance**"

→ More prompt templates → more robust but slightly lower peak accuracy.

### 5.4 Per-modality separate Q-Former
> "Optimize a single separate projection module f_M for each modality M"

→ XIB uses **distinct Q-Former weights per modality** (not shared).

### 5.5 Cross-modal emergent reasoning
> "discriminative cross-modal reasoning emerges naturally through individual modality alignment to LLMs"

→ Per-modality training (separate Q-Formers) → LLM combines modalities at inference without joint training.

---

## 6. Architecture comparison

| Component | InstructBLIP (image only) | X-InstructBLIP (4 modalities) |
|---|---|---|
| Q-Former count | 1 (image) | 4 (image, 3D, audio, video) |
| Q-Former init | Random | BLIP-2 pretrained (image), then transfer to other modalities |
| Modality prefix | No explicit | YES (consistent improvement) |
| Cross-modal training | N/A | Each modality trained SEPARATELY |

---

## 7. Insights applicable to SigLLM

### 🥇 Idea 1 — Modality prefix labels (Table 7 evidence)
**Most actionable.** Add explicit labels to soft tokens:
```
Before: ...<UserID> Leverage information... <TargetItemID>...
After:  User feature: <UserID>. Target item feature: <TargetItemID>.
```
→ Already implemented in SigLLM v2 prompt redesign (`qformer_prompt_movie_v2_labeled.txt`).

### 🥈 Idea 2 — LP/MLP baseline comparison
**Validates building internal baseline.** XIB shows LP competitive with Q-Former when data limited.
→ For SigLLM: LP baseline (e.g., 2-layer MLP) on ML-1M may rival Q-Former.
→ But CoLLM-MF is already external LP baseline (test uAUC 0.7179 ≈ SigLLM 0.7170) — confirms paper evidence sufficient.

### 🥉 Idea 3 — Per-domain separate Q-Former for user vs item
**Inspired by per-modality separate Q-Former.** Current SigLLM uses SAME Q-Former for user_cf and item_cf. Could split:
- Q-Former_item for items
- Q-Former_user for users (smaller, since user signal simpler)

→ Effort: 2x Q-Former params and training. May not justify.

### Idea 4 — Cross-modal init transfer
SigLLM Q-Former for item could be pretrained on **larger external rec dataset** (Amazon-Book) then transferred to ML-1M (1000-iter convergence finding).

---

## 8. Comparison with SigLLM design

| Aspect | X-InstructBLIP | SigLLM (current) |
|---|---|---|
| **Projection type** | Q-Former + LP comparison | Q-Former only (no LP baseline) |
| **Modality prefix** | YES (helps) | NO (current prompt) |
| **Q-Former per modality** | Separate per modality | Single shared Q-Former |
| **Init source** | BLIP-2 pretrained | BERT-base text branch only |
| **Multi-task training** | YES (26 datasets × 11 categories) | NO (single CTR task) |

---

## 9. Key takeaways for SigLLM thesis

### What XIB validates for SigLLM
1. **Modality prefix matters** — direct apply via prompt redesign
2. **LP competitive with Q-Former at low data** — explains why SigLLM ties CoLLM-MF on ML-1M
3. **Stage 2 generative pretrain has value** — supports keeping it
4. **BLIP-2 pretrained Q-Former weights transfer well** — supports SigLLM's text branch BERT init

### What XIB challenges for SigLLM
1. **Instruction-aware design** in InstructBLIP works for MULTI-TASK only (XIB confirms with multi-modality multi-task). SigLLM single-task setting → instruction routing has no signal to learn. Explains 4 negative findings.

### Position SigLLM in XIB framing
- SigLLM "modality" = CF embeddings
- Single modality (CF only), single task (CTR)
- → XIB predicts: **LP might compete with Q-Former** in this regime (confirmed empirically by CoLLM-MF vs SigLLM tie)
- → XIB predicts: **prefix labels should help** (untested for SigLLM as of writing this note — now being tested in v2 prompt experiment)

---

## 10. Citation key

```bibtex
@inproceedings{panagopoulou2024xinstructblip,
  title={X-InstructBLIP: A Framework for Aligning Image, 3D, Audio, Video to LLMs and its Emergent Cross-modal Reasoning},
  author={Panagopoulou, Artemis and Xue, Le and Yu, Ning and Li, Junnan and others},
  booktitle={ECCV},
  year={2024}
}
```
