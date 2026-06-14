# Paper Notes Index — Related Work for SigLLM

**Last updated:** 2026-06-05
**Purpose:** Quick-access summary of all related papers studied for SigLLM thesis. Each linked note has detailed analysis + ideas applicable to SigLLM.

---

## Quick navigation

| Paper | Note file | Year | Core idea | Focus area |
|---|---|---|---|---|
| **ILM** | [PAPER_NOTES_ILM.md](PAPER_NOTES_ILM.md) | 2024 | BLIP-2 Q-Former adapted for rec, 5-loss Phase 1 | Bridge architecture (DIRECT ANCESTOR) |
| **X-InstructBLIP** | [PAPER_NOTES_X-InstructBLIP.md](PAPER_NOTES_X-InstructBLIP.md) | ECCV 2024 | Q-Former vs LP for 4 modalities, modality prefix | Bridge type comparison |
| **USER-LLM** | [PAPER_NOTES_USER-LLM.md](PAPER_NOTES_USER-LLM.md) | AAAI 2024 | Cross-attention deep integration, Perceiver compression | LLM integration mechanism |
| **CALRec** | [PAPER_NOTES_CALRec.md](PAPER_NOTES_CALRec.md) | RecSys 2024 | Two-tower contrastive aux loss, multi-stage | Stage 3 aux loss |
| **RA-Rec** | [PAPER_NOTES_RA-Rec.md](PAPER_NOTES_RA-Rec.md) | 2024 | Layer-specific ID injection + text-only contrastive | LLM injection + distillation |
| **beeFormer** ⭐ | [PAPER_NOTES_beeFormer.md](PAPER_NOTES_beeFormer.md) | RecSys 2024 | Text encoder trained on interactions (CF-aware) | **SOURCE/CF ENRICHMENT** |

---

## Comparison table — key dimensions

### Architecture

| Paper | Source CF | Bridge | LLM | Trainable params |
|---|---|---|---|---|
| ILM | MF | 8-layer Q-Former + linear | PaLM 2-S (frozen) | Q-Former + adapter (~M) |
| X-InstructBLIP | Modality encoders | Q-Former OR Linear Projection | Vicuna/InstructBLIP base (frozen) | Q-Former + projection |
| USER-LLM | Autoregressive transformer | Perceiver + cross-attention | PaLM 2 XXS (frozen) | Cross-attn layers + α gates |
| CALRec | None (pure text) | None | PaLM 2 XXS (**fully finetuned**) | All LLM params |
| RA-Rec | Any (ComiRec, etc.) | Layer-specific projector + prefix | GPT-2 / SentenceBERT (frozen) | **~18K params only** |
| **SigLLM** | **MF** | **4-layer Q-Former + Linear+LayerNorm** | **Qwen2-7B (frozen + LoRA)** | **Q-Former 50-76M + LoRA 2.5M + proj 2.8M** |

### Integration point in LLM

| Paper | Where soft signals enter LLM |
|---|---|
| ILM | Input layer (prepended to text tokens) |
| X-InstructBLIP | Input layer (prepended) |
| USER-LLM | **Multiple layers** (cross-attention at every N layers) |
| CALRec | N/A — pure text prompt |
| RA-Rec | **Every layer** (layer-specific projection) |
| **SigLLM** | **Input layer only** (prepended) |

→ **SigLLM is "shallowest" integration** among these papers. Both USER-LLM and RA-Rec inject at multiple layers.

### Auxiliary losses at fine-tuning

| Paper | Aux loss(es) at fine-tuning stage |
|---|---|
| ILM | None at Phase 2 (just LM loss) |
| X-InstructBLIP | None |
| USER-LLM | None (just task-specific LM) |
| **CALRec** | **L_TT + L_UT** (two contrastive) |
| **RA-Rec** | **L_ua + L_ia** (text vs ID contrastive) |
| **SigLLM** | **None** (just binary CE) |

→ **SigLLM Stage 3 has weakest loss diversity.** Could borrow from CALRec or RA-Rec.

---

## Top ideas applicable to SigLLM (synthesized across all 5 papers)

### 🥇 Idea 1 — Auxiliary contrastive loss at Stage 3

**Sources:** CALRec + RA-Rec (both add aux InfoNCE)

**Two variants:**

**Variant A (CALRec-style):** Cross-sample alignment
```
v_T (target alone) ↔ v_T|U (target with user context)  # L_TT
v_U (user alone) ↔ v_T (positive target)               # L_UT
```

**Variant B (RA-Rec-style):** Same-sample alignment
```
h_text (text-only prompt forward) ↔ h_hybrid (text+soft tokens forward)
```

**Both:** add InfoNCE to current SigLLM Stage 3 step 2.

**Effort:** 2-3 days
**Probability:** 25-35%

### 🥈 Idea 2 — Layer-specific soft token injection

**Sources:** USER-LLM (cross-attention layers) + RA-Rec (per-layer projection)

**Concept:** Inject soft tokens at MULTIPLE LLM layers, not just input.

**Implementation challenge:** Major refactor of Qwen2 forward.

**Effort:** 1-2 weeks
**Probability:** 25-35%

### 🥉 Idea 3 — Modality prefix labels in prompt

**Source:** X-InstructBLIP Table 7

**Concept:** Prepend explicit text labels to soft token slots:
```
"User feature: <UserID>. Target item feature: <TargetItemID>."
```

→ Q-Former relieved from encoding modality type, reserves bandwidth for semantics.

**Effort:** 0.5 day (prompt only)
**Probability:** 30-40%

**Status for SigLLM:** **IMPLEMENTED** as `qformer_prompt_movie_v2_labeled.txt`, currently testing.

### Idea 4 — Multi-domain pretraining

**Sources:** CALRec (Stage I multi-category) + ILM (multi-dataset OpenP5)

**Concept:** Pretrain Stage 1+2 on Amazon-Book / Amazon-Clothing → fine-tune ML-1M.

**Effort:** 5-7 days (data prep + heavy compute)
**Probability:** 25-35%

### Idea 5 — Autoregressive user encoder

**Source:** USER-LLM

**Concept:** Replace MF with autoregressive transformer on user history.

**Effort:** 1-2 weeks
**Probability:** 20-30%

### Idea 6 — Data filtering by overlap

**Source:** RA-Rec denoising

**Concept:** Remove training samples with zero word overlap between target and history.

**Effort:** 0.5 day
**Probability:** 10-15% (ML-1M small, may hurt)

### Idea 7 — Zero-init gated soft token contribution

**Source:** USER-LLM (tanh(α) zero-init)

**Concept:** Gate soft token signal with trainable scalar α (zero-init) — safe init.

**Effort:** 5 lines code
**Probability:** 10-15%

### 🆕 Idea 8 — **CF-aware text encoder** (replace BERT in Q-Former) ⭐

**Source:** beeFormer

**Concept:** Current SigLLM uses `bert-base-uncased` for Q-Former text branch (pure semantic). Replace with **beeFormer-trained sentence Transformer** that is also CF-aware.

```yaml
# Current
qformer_text_model_name: "bert-base-uncased"

# Proposed
qformer_text_model_name: "beeformer/movielens-mpnet"  # if pretrained available
```

**Why this fills a missing dimension:**
- 5 other papers all treat CF source as fixed
- beeFormer is the ONLY paper that **enriches CF representation via training**
- Pretrained checkpoint for ML-20M domain may transfer well to ML-1M

**Effort:** 1-2 days (verify checkpoint availability + retrain Stage 1+2+3)
**Probability:** **25-35%**

→ **Currently the only direction touching SOURCE ENRICHMENT.**

---

## Recommended priority for SigLLM remaining time

### If 1 week remains
- **Idea 3 (modality prefix prompt)** → currently testing
- Document shrink + prompt results

### If 2-3 weeks remain
- Idea 3 + **Idea 1 (CALRec/RA-Rec aux contrastive)** — combine cheap improvements
- Both at Stage 3 step 2 level, no Stage 1+2 retrain needed

### If 1+ month remains
- All cheap ideas first
- Then **Idea 4 (multi-domain pretrain)** for cold-start improvement
- Possibly **Idea 2 (layer-wise injection)** for architectural novelty

---

## How papers position SigLLM in the landscape

### What SigLLM has that papers validate
- ✓ Q-Former bridge architecture (ILM, XIB)
- ✓ Frozen LLM + adapter approach (USER-LLM, RA-Rec)
- ✓ Soft tokens for CF signal (ILM, RA-Rec)
- ✓ 5-loss Stage 1 pretraining (ILM)
- ✓ 2-step CoLLM fine-tuning (CoLLM standard)

### What SigLLM lacks per papers
- ✗ Multi-layer injection (USER-LLM, RA-Rec)
- ✗ Auxiliary contrastive at fine-tuning (CALRec, RA-Rec)
- ✗ Modality prefix labels (XIB) — **fixing in v2 prompt**
- ✗ Multi-domain pretraining (CALRec, ILM OpenP5)
- ✗ Sequential user encoder (USER-LLM)

### Where SigLLM may innovate
- **Cold-item finding** (+1.79 uAUC test_cold via user soft tokens) — ILM/CoLLM/BinLLM/SellaRec don't have this isolated analysis
- **Shrink validation** (2L+6Q achieves 99% baseline with 34% fewer params) — novel param-efficiency study
- **Negative findings** (4 architectural extensions don't transfer to single-task CTR) — defensible scientific contribution

---

## CF / source enrichment landscape

| Approach | Where modify | Status for SigLLM |
|---|---|---|
| Pure MF (current SigLLM) | Source model | ✓ Used |
| beeFormer text encoder | Q-Former text branch | Untested, **only paper directly addressing CF enrichment** |
| LightGCN | Replace MF | Could replace source |
| SASRec / sequence encoder | Replace MF | USER-LLM-style |
| BinLLM binary IP | Replace MF encoding | Different paradigm |
| Multi-channel (MF + text) | Combine sources | Future combination |

→ **All 5 main papers documented (ILM, XIB, USER-LLM, CALRec, RA-Rec) take CF as fixed input.** Only beeFormer explicitly enriches CF representation.

---

## File map

```
docs/
├── PAPER_NOTES_INDEX.md           ← THIS FILE
├── PAPER_NOTES_ILM.md             ← Direct ancestor
├── PAPER_NOTES_X-InstructBLIP.md  ← Q-Former vs LP, modality prefix
├── PAPER_NOTES_USER-LLM.md        ← Cross-attention, Perceiver
├── PAPER_NOTES_CALRec.md          ← Two-tower contrastive
├── PAPER_NOTES_RA-Rec.md          ← Layer-specific injection
├── PAPER_NOTES_beeFormer.md       ← CF-aware text encoder ⭐ (CF enrichment)
│
├── (Original PDFs)
├── ILM.pdf
├── X-Instruct-BLIP-eval-single-mutli-task.pdf
├── USER-LLM.pdf
├── CALRec.pdf
├── RA-Rec.pdf
├── beeFormer.pdf
└── (Other related — not deeply analyzed yet)
    ├── BinLLM.pdf            ← Binary IP source encoding (also CF-enrich angle)
    ├── CoLLM.pdf             ← MLP bridge baseline (SigLLM 2-step inherits from this)
    ├── InstructBLIP_1.pdf    ← Original instruction-aware paper
    └── SellaRec.pdf          ← SOTA semantic-aware projection (likely CF-enrich angle)
```

---

## Papers NOT yet deeply analyzed (PDFs available)

These have been referenced via baseline results but not deep-read into individual notes:

| Paper | Why useful | Suggested action |
|---|---|---|
| **CoLLM.pdf** | Direct precedent — SigLLM uses CoLLM 2-step LoRA fine-tuning | Worth detailed notes (SigLLM inherits architecture) |
| **BinLLM.pdf** | Binary IP-style CF encoding — alternative source representation | Consider for thesis comparison |
| **SellaRec.pdf** | SOTA on ML-1M — semantic-aware projection | Worth detailed notes (best comparison baseline) |
| **InstructBLIP_1.pdf** | Original InstructBLIP — SigLLM's instruction-aware mode source | Worth verifying SigLLM's instruction-aware code matches paper |
| **beeFormer.pdf** | beeFormer-MPNet referenced for text encoder upgrade | Read if pursuing Idea: better text base for Q-Former |
