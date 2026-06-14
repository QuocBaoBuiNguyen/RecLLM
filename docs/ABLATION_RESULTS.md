# SigLLM Ablation Study — Results & Analysis

**Last updated:** 2026-05-30
**Dataset:** MovieLens-1M (CoLLM/BinLLM OOD2 split, 33,891 train / 10,401 valid / 7,331 test)
**Pipeline:** Stage 0 (MF) → Stage 1 (5-loss pretrain) → Stage 2 (gen pretrain) → Stage 3 step 1 (LoRA) → Stage 3 step 2 (CIE)

## Experiments index — 4 negative findings + 1 positive

| # | Extension tested | Hypothesis | Branch | val_uAUC peak | vs baseline 0.7098 | Verdict |
|---|---|---|---|---|---|---|
| ✓ | Q-Former as CF→LLM bridge | Modality bridge transfers from VLM | `feat/best-0.72-warm` | **0.7098** | — | **POSITIVE** (+4.6 uAUC vs text-only) |
| 1 | Instruction-aware (V1 paraphrase, 12 prompts) | InstructBLIP routing helps rec | `feat/ablation-instruction-aware` | 0.7098 ≈ vanilla 0.7081 | ≈ | NEGATIVE |
| 2 | Instruction-aware (task-focused, 8 prompts) | Task-vocabulary prompts unlock routing | `feat/...task-focused` | ~baseline | ≈ | NEGATIVE |
| 3a | Interaction-aware (multi-element memory) | Joint [target+history] encoding helps | `feat/interaction-aware-qformer` | 0.7036 | −0.006 | NEGATIVE |
| 3b | Interaction + instruction combined | Faithful InstructBLIP on multi-item memory | same | trend ~0.69 | < | NEGATIVE |
| 4 | **User-conditioned queries** | Per-user query shift personalizes extraction | `feat/user-conditioned-queries` | **0.7060** | **−0.004** | **NEGATIVE** |

**Consolidated finding:** 4 architectural extensions inspired by VLM Q-Former design (instruction routing, multi-element memory, user conditioning) **fail to improve** over the base Q-Former on ML-1M CTR. The base Q-Former bridge itself works (+4.6 uAUC vs text-only). The thesis story has pivoted to a rigorous architectural-transfer study (see "Consolidated negative findings" at the bottom).

---

## Section 1 — Original ablation (instruction-aware vs vanilla)

**Date:** 2026-05-29
**Branch:** `feat/ablation-instruction-aware`
**Source notebooks:** `SigLLM._best_24_05_2026ipynb.ipynb`, `SigLLM_rerun_28_5_2026.ipynb`, `SigLLM_vanilla_29_5_2026.ipynb`

## Setup — Two orthogonal ablation flags

| Flag | Mechanism | When applied |
|---|---|---|
| `model.qformer_config.instruction_aware` | When False, Q-Former `forward()` short-circuits to `encode_cf()` (vanilla BLIP-2 mode — queries cross-attend CF only, instruction text dropped from self-attention) | **Training-time** — affects Q-Former weights; requires retrain of Stage 3 step 2 |
| `model.ablate_soft_tokens` | When True, projected Q-Former soft tokens are zeroed right before LLM injection (LLM sees zero vectors at `<ItemIDList>` / `<TargetItemID>` slots) | **Inference-time** — load any trained ckpt and flip; no retrain |

Four configurations from 2×2 cross of these flags:

| Configuration | instruction_aware | ablate_soft_tokens | Architectural meaning |
|---|---|---|---|
| **Instruction-aware Q-Former** (Full SigLLM) | True | False | Replicates InstructBLIP design for recommendation |
| **Vanilla Q-Former** | False | False | Replicates ILM Phase 2 design (queries-only) |
| **Text-only baseline** (no CF) | True | True | LLM uses prompt text alone; CF signal ablated |
| Vanilla + no CF | False | True | Degenerate — equivalent to Text-only since soft tokens zeroed regardless |

## Full Results Table

All evaluations from Stage 3 step 2 `checkpoint_best.pth`, on standard test / test_warm / test_cold splits.

### test (overall, 7331 samples)

| Configuration | AUC | uAUC | val_loss | ACC@0.5 | Score gap |
|---|---|---|---|---|---|
| **Instruction-aware Q-Former** | 0.7322 | 0.7131 | 0.6102 | 0.6669 | 0.1527 |
| **Vanilla Q-Former** | **0.7325** | **0.7170** | 0.6124 | 0.6675 | 0.1538 |
| **Text-only baseline** | 0.6960 | 0.6878 | 0.7287 | 0.6489 | 0.1531 |
| Vanilla + no CF | 0.6960 | 0.6878 | 0.7287 | 0.6489 | 0.1531 |

### test_warm (3522 samples)

| Configuration | AUC | uAUC | val_loss | ACC@0.5 | Score gap |
|---|---|---|---|---|---|
| **Instruction-aware Q-Former** (best ckpt 24/05) | **0.7456** | **0.7265** | 0.6079 | 0.6671 | 0.1733 |
| **Instruction-aware Q-Former** (rerun 28/05) | 0.7448 | 0.7217 | 0.6017 | 0.6696 | 0.1667 |
| **Vanilla Q-Former** | 0.7451 | 0.7213 | 0.6041 | 0.6699 | 0.1680 |
| **Text-only baseline** | 0.7017 | 0.6750 | 0.7289 | 0.6498 | 0.1609 |
| Vanilla + no CF | 0.7017 | 0.6750 | 0.7289 | 0.6498 | 0.1609 |

### test_cold (3178 samples)

| Configuration | AUC | uAUC | val_loss | ACC@0.5 | Score gap |
|---|---|---|---|---|---|
| **Instruction-aware Q-Former** (best ckpt 24/05) | 0.7072 | 0.6478 | 0.6277 | 0.6575 | 0.1295 |
| **Instruction-aware Q-Former** (rerun 28/05) | 0.7068 | 0.6503 | 0.6222 | 0.6572 | 0.1259 |
| **Vanilla Q-Former** | 0.7072 | 0.6516 | 0.6245 | 0.6575 | 0.1266 |
| **Text-only baseline** | 0.6756 | 0.6478 | 0.7266 | 0.6413 | 0.1307 |
| Vanilla + no CF | 0.6756 | 0.6478 | 0.7266 | 0.6413 | 0.1307 |

**Note:** "Vanilla + no CF" produces identical predictions to "Text-only baseline" on all splits — confirmed at floating-point precision in cells 75 and 79 of the rerun + vanilla notebooks. Expected: with soft tokens zeroed, the LoRA-tuned LLM sees identical text-only input regardless of Q-Former's underlying weights.

## Delta Analysis

### Contribution of CF soft tokens (Text-only vs Instruction-aware Q-Former)

Quantifies how much performance the collaborative filtering signal — flowing through Q-Former → projection → LLM soft tokens — contributes beyond text prompts alone.

| Split | Instruction-aware AUC / uAUC | Text-only AUC / uAUC | Δ AUC | Δ uAUC |
|---|---|---|---|---|
| test | 0.7322 / 0.7131 | 0.6960 / 0.6878 | **+0.0362** | **+0.0253** |
| test_warm | 0.7456 / 0.7265 | 0.7017 / 0.6750 | **+0.0439** | **+0.0515** |
| test_cold | 0.7072 / 0.6478 | 0.6756 / 0.6478 | **+0.0316** | 0.0000 |

→ **CF signal contributes +5.15 pts test_warm uAUC** — the LLM substantively integrates collaborative information beyond what text prompts encode.
→ **On cold users, CF contribution collapses to 0 uAUC** (only +3.16 pts AUC) — consistent with MF embeddings being uninformative for unseen users.

### Contribution of instruction-aware Q-Former design (Vanilla vs Instruction-aware)

Quantifies whether InstructBLIP's instruction-aware routing improves over ILM-style vanilla queries-only.

| Split | Instruction-aware AUC / uAUC | Vanilla AUC / uAUC | Δ AUC | Δ uAUC |
|---|---|---|---|---|
| test | 0.7322 / 0.7131 | 0.7325 / 0.7170 | +0.0003 | **+0.0039 (vanilla slightly better)** |
| test_warm | 0.7456 / 0.7265 | 0.7451 / 0.7213 | −0.0005 | −0.0052 |
| test_cold | 0.7072 / 0.6478 | 0.7072 / 0.6516 | 0.0000 | +0.0038 |

→ **|Δ uAUC| < 0.0052 across all splits** — well within run-to-run noise (a single seed difference can shift uAUC by ±0.005-0.01). **Instruction-aware Q-Former does NOT demonstrate measurable contribution on this setting.** Vanilla even edges out on test (overall) and test_cold.

## Training Trajectories

All numbers from per-epoch validation evaluation during Stage 3.

### Stage 3 Step 1 (LoRA tuning, Q-Former frozen)

Identical for both Instruction-aware and Vanilla configurations (Stage 3 step 1 uses text-only prompts without soft-token placeholders — Q-Former forward output is computed but not injected into LLM, so `instruction_aware` flag has no effect on step 1 results).

| Epoch | AUC | uAUC | val_loss | Score gap |
|---|---|---|---|---|
| 0 | 0.7056 | 0.6749 | 0.6369 | 0.0759 |
| 1 | 0.7106 | 0.6776 | 0.6267 | 0.1442 |
| 2 | 0.7122 | 0.6841 | 0.6168 | 0.1356 |
| 3 | 0.7170 | 0.6895 | 0.6205 | 0.1727 |
| **4** | **0.7196** | **0.6917** (peak) | 0.6156 | 0.1722 |

### Stage 3 Step 2 — Instruction-aware Q-Former (best run, 24/05)

| Epoch | AUC | uAUC | val_loss | Score gap |
|---|---|---|---|---|
| 0 | 0.7194 | 0.6949 | 0.6126 | 0.1512 |
| 1 | 0.7222 | 0.7015 | 0.6149 | 0.1813 |
| 2 | 0.7212 | 0.7072 | 0.6146 | 0.1786 |
| **3** | 0.7187 | **0.7098** (peak) | 0.6177 | 0.1768 |
| 4 | 0.7224 | 0.7002 | 0.6135 | 0.1794 |

### Stage 3 Step 2 — Instruction-aware Q-Former (rerun verification, 28/05)

| Epoch | AUC | uAUC | val_loss | Score gap |
|---|---|---|---|---|
| 0 | 0.7207 | 0.6966 | 0.6124 | 0.1583 |
| 1 | 0.7192 | 0.6975 | 0.6141 | 0.1703 |
| **2** | 0.7213 | **0.7018** (peak) | 0.6177 | 0.1853 |
| 3 | 0.7144 | 0.6983 | 0.6227 | 0.1697 |
| 4 | 0.7212 | 0.6957 | 0.6139 | 0.1758 |

Note: rerun peaked at 0.7018 (epoch 2) vs original 24/05 run peaked at 0.7098 (epoch 3). The original 24/05 ckpt is treated as the primary "instruction-aware baseline" for the ablation. The 28/05 rerun confirms pipeline reproducibility within seed variance.

### Stage 3 Step 2 — Vanilla Q-Former (29/05)

| Epoch | AUC | uAUC | val_loss | Score gap |
|---|---|---|---|---|
| 0 | 0.7204 | 0.6954 | 0.6129 | 0.1561 |
| 1 | 0.7209 | 0.6893 | 0.6126 | 0.1673 |
| 2 | 0.7223 | 0.6979 | 0.6142 | 0.1814 |
| 3 | 0.7212 | 0.7008 | 0.6143 | 0.1749 |
| 4 | 0.7215 | 0.7057 | 0.6144 | 0.1774 |
| **5** | 0.7207 | **0.7081** (peak) | 0.6169 | 0.1826 |
| 6 | 0.7218 | 0.7061 | 0.6130 | 0.1773 |
| 7 | 0.7216 | 0.7052 | 0.6192 | 0.1854 |

Climbed steadily for 5 epochs, peaked at 0.7081, drifted down — same overall pattern as Instruction-aware. Peak val_uAUC 0.7081 ≈ Instruction-aware best 0.7098 (Δ = 0.0017, well within noise).

---

## Section 2 — Extension experiments (2026-05-30)

### Extension 4: User-conditioned queries (`feat/user-conditioned-queries`)

**Hypothesis:** Q-Former queries are global learnable parameters (same 8 query vectors for every user). Conditioning queries on `user_cf` so each user sees a shifted set of queries should let the Q-Former extract user-specific aspects of each item — true per-user personalization in the soft-token bridge.

**Implementation:**
- `query_tokens = self.q.expand(B, -1, -1) + user_proj(user_cf).unsqueeze(1)` where `user_proj: Linear(d_user=256, d_model=768)`
- **Zero-init** `user_proj.weight/bias` → at epoch 0 the residual is exact no-op (queries == pretrained), gradient flows through `user_cf` so weight grows only if CTR rewards it (standard LoRA / FiLM pattern)
- Stage 1/2 don't pass `user_cf` → flag is a no-op there; Stage 2 ckpts load with `strict=False` (`user_proj.*` keys missing → random init at Stage 3)
- ~197k new trainable params (256×768 + 768), <0.3% of Q-Former → negligible compute overhead

**Setup:**
- Branch from `feat/best-0.72-warm` (clean per-item baseline, val_uAUC 0.7098)
- Stage 3 step 2 only retrained (Stage 0/1/2 + step 1 LoRA reused)
- `model.qformer_config.user_conditioned=True`, `max_epoch=25`, init_lr=3e-5 (default)
- Training notebook: `notebooks/SigLLM_user_conditioned_queries.ipynb`

#### Trajectory

| Epoch | AUC | uAUC | val_loss | Score gap |
|---|---|---|---|---|
| 0 | 0.7519 | 0.6962 | 0.5877 | 0.1878 |
| 1 | 0.7582 | 0.6978 | 0.5990 | 0.2265 |
| 2 | 0.7548 | 0.7033 | 0.5862 | 0.2031 |
| **3** | 0.7568 | **0.7060** (peak) | 0.5980 | 0.2156 |
| 4 | 0.7571 | 0.6995 | 0.5837 | 0.2145 |
| 5 | 0.7523 | 0.6986 | 0.5963 | 0.2191 |
| 6 | 0.7487 | 0.6912 | 0.5938 | 0.2082 |

**Killed at epoch 7** (3 epochs of decline from peak + below baseline).

#### Verdict — NEGATIVE

| Comparison | This run (peak) | Baseline (V1 instruction-aware) | Δ |
|---|---|---|---|
| val_AUC | 0.7582 | ~0.7222 | **+0.036** ✓ |
| **val_uAUC** | **0.7060** | **0.7098** | **−0.0038** ✗ |
| Score gap | 0.219 avg | ~0.18 avg | +0.04 ✓ |

**Observations:**
- AUC and score_gap exceed baseline — model learned **better global separation** with user-conditioning
- But per-user ranking (uAUC) **did not improve** over baseline
- This calibration-vs-ranking divergence suggests: user_proj added user-popularity bias to scores (helps global AUC) without sharpening within-user item discrimination (would need true cross-user × item interaction signal, not just additive user shift)
- Peak arrived early (epoch 3) then drifted down → user_proj weight saturated quickly; further training only added noise

#### Hypothesized reasons user-conditioned fails

1. **Additive shift is too weak a personalization mechanism.** `queries = base_q + user_shift` shifts all 8 queries by the SAME user-dependent vector. This is FiLM-style **bias**, not multiplicative gating or attention modulation. To extract truly user-specific item aspects, would need user-conditioned attention WEIGHTS (e.g. hypernetwork generating per-user queries from scratch), not just user-conditioned bias.

2. **Small data scale.** 33k training samples with 839 distinct users → each user sees ~40 training pairs on average. user_proj must learn how to shift queries per-user from this sparse signal → high noise, low signal.

3. **Stage 2 Q-Former alignment is user-agnostic.** Pretrained queries learned to extract item-level features that work for "an average user". Shifting them per-user moves them off this learned manifold; Stage 3 only has 5-7 epochs to re-align before overfitting. Stage 1/2 don't train this layer → cold start at Stage 3.

4. **Per-user signal already in LLM history text.** The prompt already contains `<ItemTitleList>` (user's rated items as text) — the LLM autoregressive attention IS user-aware via this context. Adding user signal at Q-Former layer is redundant with what LLM already does.

## Key Conclusions

### What the experiments PROVE

1. **CF soft tokens carry substantial signal.** Zeroing them (Text-only baseline) costs +5.15 pts test_warm uAUC and +2.53 pts test overall uAUC. The LLM does not merely echo text prompts — it actively integrates the collaborative signal flowing through Q-Former.

2. **CF contribution localizes to warm users.** On test_cold, zeroing CF costs 0 uAUC (3.16 pts AUC). Consistent with the well-known limitation: MF embeddings are random / uninformative for users unseen during training.

3. **Soft tokens flow identically through trained model and untrained model when zeroed.** The "Vanilla + no CF" configuration produces bit-identical predictions to the "Text-only baseline" across all splits. This confirms the ablate_soft_tokens mechanism cleanly cuts the CF information path.

### What the experiments FAIL to prove — consolidated 4 negative findings

4. **Instruction-aware Q-Former does NOT contribute over vanilla.** |Δ uAUC| < 0.0052 across all splits (V1 paraphrase prompts).

5. **Task-focused instruction redesign does NOT unlock routing.** Replacing 12 paraphrase prompts with 8 CTR-vocabulary-explicit prompts ("predict / preference / personalized / binary") — uAUC tracks baseline. Rules out "paraphrase signal was the problem".

6. **Interaction-aware joint encoding underperforms per-item.** Encoding `[target, history_1, ..., history_L]` jointly in a single Q-Former forward (8 soft tokens for 11 items) peaked at 0.7036 — below baseline 0.7098. **Bandwidth bottleneck:** per-item gives 88 soft tokens (8 per item), joint gives 8 total. Information loss from compressing 11 items into 8 query slots exceeds the joint-attention benefit. Combined with instruction-aware (faithful InstructBLIP on multi-item memory) trended worse, not better.

7. **User-conditioned queries do NOT improve over global queries.** Adding `user_proj(user_cf)` shift to base queries (zero-init, FiLM-style) peaked at 0.7060 — below baseline 0.7098. AUC and score_gap improved (better global discrimination), but per-user ranking did not (additive bias added user-popularity signal, not within-user item discrimination signal).

### Cross-cutting hypothesized reasons (why 4 VLM-style extensions all fail)

- **Single-task setting**: All training is Yes/No CTR. InstructBLIP's instruction-aware design was validated on 26 datasets × 11 task categories. With one task, there is no routing signal for instruction-aware queries to learn.
- **Low information density at source**: CF embeddings are 256-d MF vectors trained with BPR loss → low entropy, single-objective compression. Q-Former (76M params) over-parameterized for what's actually present in source. Compare InstructBLIP source: 1408-d × 257 patches = 361k numbers/image, pretrained EVA at 1B+ params.
- **Small data scale**: 33k samples cannot train extensions like user_proj from cold start; they overfit or oscillate.
- **Bandwidth bottleneck**: Per-item N×Q soft tokens >> joint Q tokens; any "encode together" extension that reduces this ratio loses information.
- **Redundancy with LLM**: LLM already attends to history text autoregressively. Per-user / per-interaction structure added at Q-Former layer duplicates what LLM does natively via context.

## Comparison with Related Work

### Architectural comparison

| Paper | Q-Former at fine-tuning stage | Instruction-aware Q-Former? |
|---|---|---|
| BLIP-2 | image-only cross-attention | No |
| InstructBLIP | queries + instruction self-attention | **Yes** (their main contribution) |
| ILM | item-CF-only cross-attention | No (instruction routed to LLM directly) |
| **SigLLM (this work)** | queries + instruction self-attention | **Yes (replicates InstructBLIP)** |

→ The Vanilla configuration in this ablation matches ILM's Phase 2 design. The Instruction-aware configuration matches InstructBLIP's. Their near-equivalence on ML-1M confirms ILM's choice of vanilla queries-only Q-Former was well-justified for recommendation.

### Numerical comparison — ML-1M (test, overall split)

Baseline numbers extracted from `docs/SellaRec.pdf` Table 1 (SellaRec re-ran all baselines under unified protocol; consistent with `docs/BinLLM.pdf` Table 3 and `docs/CoLLM.pdf` Table 2 — minor differences within ±0.01).

| Category | Method | AUC | UAUC | vs SigLLM Instruction-aware |
|---|---|---|---|---|
| Collab. | MF | 0.6473 | 0.6264 | SigLLM +0.085 / +0.087 ↑ |
| Collab. | LightGCN | 0.6139 | 0.6276 | SigLLM +0.118 / +0.086 ↑ |
| Collab. | DIN | 0.7138 | 0.6595 | SigLLM +0.018 / +0.054 ↑ |
| LLMRec | Zero-shot LLM | 0.5718 | 0.5145 | SigLLM +0.160 / +0.199 ↑ |
| LLMRec | Prompt4NR | 0.7191 | 0.6839 | SigLLM +0.013 / +0.029 ↑ |
| LLMRec | TALLRec | 0.7268 | 0.6989 | SigLLM +0.005 / +0.014 ↑ |
| LM+Collab | CTRL(DIN) | 0.7155 | 0.6681 | SigLLM +0.017 / +0.045 ↑ |
| LLMRec+Collab | PersonPrompt | 0.7223 | 0.6991 | SigLLM +0.010 / +0.014 ↑ |
| LLMRec+Collab | **CoLLM-MF** | 0.7357 | 0.7179 | SigLLM −0.004 / −0.005 (tied within noise) |
| LLMRec+Collab | **BinLLM** | 0.7417 | 0.7239 | SigLLM −0.010 / −0.011 ↓ |
| LLMRec+Collab | **SellaRec** (SOTA) | **0.7606** | **0.7464** | SigLLM −0.028 / −0.033 ↓ |
| **Ours** | **SigLLM Instruction-aware Q-Former** | **0.7322** | **0.7131** | — |
| **Ours** | **SigLLM Vanilla Q-Former** | **0.7325** | **0.7170** | +0.0003 / +0.0039 vs Instruction-aware |

### Position of SigLLM in the field

- **Outperforms:** MF, LightGCN, DIN, Zero-shot LLM, Prompt4NR, TALLRec, CTRL(DIN), PersonPrompt
- **Tied within noise:** CoLLM-MF (Δ uAUC = −0.005)
- **Underperforms:** BinLLM (Δ uAUC ~−0.011), SellaRec (Δ uAUC ~−0.033)

### Comparison with ILM (different evaluation protocol)

ILM (`docs/ILM.pdf`) evaluates on **ELM 24 conversational tasks** (Semantic Consistency via Sentence-T5 11B, log perplexity) and **OpenP5 generative retrieval** (HR@K, NDCG@K), NOT on CTR-style AUC/UAUC. Direct numerical comparison is not possible. Qualitative comparison:

| Aspect | ILM | SigLLM |
|---|---|---|
| Task | Conversational rec + generative retrieval | Binary CTR (Yes/No) |
| Dataset | MovieLens 25M + OpenP5 (ML-1M, Beauty, Clothing) | ML-1M (CoLLM/BinLLM OOD2 split) |
| LLM | PaLM 2-S (frozen) | Qwen2-7B-Base (LoRA fine-tuned) |
| Q-Former Phase 1 | ITC + ITM + ITG + item-item contrastive | + user-item contrastive (SigLLM extension) |
| Q-Former Phase 2 / Stage 3 | Vanilla queries-only (instruction → LLM directly) | Instruction-aware (queries + instruction concat) ← but ablation shows this matches vanilla on rec |
| LLM tuning | Frozen | CoLLM-style 2-step LoRA |

→ The ablation finding that Vanilla ≈ Instruction-aware on recommendation **independently validates ILM's choice** to use vanilla queries-only Q-Former at Phase 2.

### Honest position statement (for thesis defense)

> "SigLLM achieves AUC 0.7322 / UAUC 0.7131 on ML-1M overall test, exceeding baselines that operate purely on text prompts (TALLRec: 0.7268 / 0.6989) or simple collaborative encoders (DIN: 0.7138 / 0.6595), and tying CoLLM-MF (0.7357 / 0.7179, Δ ≤ 0.005) within run-to-run noise. The gap to BinLLM (0.7417 / 0.7239) and SellaRec (0.7606 / 0.7464) suggests their specific encoding strategies (binary IP-style encoding; semantic-aware projection) capture signal that SigLLM's Q-Former + LoRA design does not. Our two ablations isolate the contributions: (1) zeroing CF soft tokens at inference costs +5.15 pts test_warm uAUC, confirming the modality bridge transmits meaningful collaborative signal; (2) replacing instruction-aware Q-Former with vanilla queries-only produces near-identical performance (|Δ uAUC| < 0.005), indicating InstructBLIP's instruction routing innovation does not transfer to this single-task recommendation setting."

## Implications for Thesis

The original thesis claim — "instruction-aware Q-Former adapted for recommendation" — is **not supported by 4 architectural experiments**. The 4 negative findings are independent (different mechanisms tested) and consistent (all fail). This rules out "we just picked the wrong extension" — the limitation is at the paradigm level.

### Final recommended pivot — "On the Limits of Vision-Language Q-Former Design in Recommendation"

This pivot **subsumes the 3 earlier pivot options** (1: honest negative, 2: prompt redesign attempt, 3: drop instruction-aware claim). With 4 confirmatory experiments now in hand, the thesis can defend a **systematic architectural transferability finding** rather than report a single null result.

#### Proposed thesis narrative

> **Positive contribution.** Q-Former functions as an effective CF→LLM modality bridge on ML-1M CTR. Soft tokens projected from a 4-layer Q-Former (trained via BLIP-2-style 5-loss pretraining + CoLLM 2-step LoRA fine-tuning) contribute **+5.15 pts test_warm uAUC** over text-only prompts (0.7265 vs 0.6750), confirming the paradigm transfer is non-trivial.
>
> **Negative contributions (systematic architectural transferability study).** Four design choices that proved effective in vision-language Q-Formers (InstructBLIP, BLIP-2, VideoBLIP) do NOT transfer to single-task CTR recommendation:
>
> 1. **Instruction-aware self-attention routing** (V1 paraphrase prompts) — Δ uAUC < 0.005 vs vanilla queries-only across test / test_warm / test_cold splits.
> 2. **Task-focused instruction redesign** (CTR-vocabulary-explicit prompts) — uAUC tracks baseline, ruling out "prompt content was the issue".
> 3. **Multi-element interaction-aware encoding** (joint Q-Former forward over `[target, history]`) — peak uAUC 0.7036 < baseline 0.7098 due to soft-token bandwidth compression (88→8 tokens for 11 items).
> 4. **User-conditioned queries** (additive FiLM-style shift `base_q + user_proj(user_cf)`) — peak uAUC 0.7060 < baseline 0.7098; AUC and score_gap improve (global discrimination) but per-user ranking does not (additive bias added user-popularity signal, not within-user discrimination).
>
> **Root cause analysis.** Cross-cutting reasons identified: (a) single-task setting eliminates instruction routing signal; (b) MF-encoded CF source has lower information density than vision encoder output; (c) ML-1M scale (33k pairs) insufficient to train added parameters from cold start; (d) per-item soft-token bandwidth dominates joint-encoding benefit; (e) LLM autoregressive context already provides per-user / per-interaction conditioning that Q-Former extensions duplicate.
>
> **Implication for the field.** ILM's choice to use vanilla queries-only Q-Former at their Phase 2 is independently validated. Future work on Q-Former for recommendation should focus on (i) richer source encoders (LightGCN, SASRec, content+CF dual-channel) to address information density, or (ii) multi-task tuning (CTR + rating + explanation) to provide instruction routing signal, rather than further architectural extensions to the standard Q-Former.

#### What this story defends well

- **Rigor:** 4 independent experiments, all rigorous (zero-init safety, baseline reuse, proper validation tracking, killed at clear plateau not random)
- **Honesty:** Doesn't claim positive results that aren't there
- **Field contribution:** Negative findings that close off architectural directions ARE publishable contributions (cf. ILM Section 7 "Limitations and Future Work"; CoLLM ablation tables)
- **Defensible at thesis defense:** Examiner cannot dismiss as "you didn't try X" — 4 X's tried

#### What this story does NOT claim (and should not)

- SigLLM is NOT SOTA on ML-1M (BinLLM, SellaRec exceed it)
- Q-Former is NOT proven useless for recommendation (the base bridge contributes substantively)
- VLM techniques NEVER transfer to rec (only these 4 specific design choices were tested; richer source encoders or multi-task tuning not ruled out)

## Reproducibility

### Notebook → result mapping

| Result | Notebook | Cell |
|---|---|---|
| Instruction-aware Stage 3 step 1 trajectory (best run) | `SigLLM._best_24_05_2026ipynb.ipynb` | 59 |
| Instruction-aware Stage 3 step 2 trajectory (best run, peak val_uAUC 0.7098) | `SigLLM._best_24_05_2026ipynb.ipynb` | 61 |
| Instruction-aware test_warm + test_cold eval (best ckpt) | `SigLLM._best_24_05_2026ipynb.ipynb` | 70 |
| Text-only baseline (best ckpt + ablate flag) eval | `SigLLM._best_24_05_2026ipynb.ipynb` | 69 |
| Instruction-aware Stage 3 step 2 rerun (peak val_uAUC 0.7018) | `SigLLM_rerun_28_5_2026.ipynb` | 67 |
| Text-only baseline eval (test + warm + cold) | `SigLLM_rerun_28_5_2026.ipynb` | 75 |
| Instruction-aware eval (test + warm + cold, rerun ckpt) | `SigLLM_rerun_28_5_2026.ipynb` | 76 |
| Vanilla Stage 3 step 2 trajectory (8 epochs, peak val_uAUC 0.7081) | `SigLLM_vanilla_29_5_2026.ipynb` | 69 |
| Vanilla + no CF eval (degenerate) | `SigLLM_vanilla_29_5_2026.ipynb` | 79 |
| Vanilla eval (test + warm + cold) | `SigLLM_vanilla_29_5_2026.ipynb` | 80 |

### Commands

**Instruction-aware Q-Former — train Stage 3 step 2** (default config has `instruction_aware: True`):
```bash
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie \
    --cfg-path configs/config.yaml
```

**Instruction-aware Q-Former — eval** (default config):
```bash
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie \
    --cfg-path configs/config.yaml \
    --options run.evaluate=True
```

**Text-only baseline — eval** (Instruction-aware ckpt + flip ablate flag at inference):
```bash
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie \
    --cfg-path configs/config.yaml \
    --options model.ablate_soft_tokens=True run.evaluate=True
```

**Vanilla Q-Former — train Stage 3 step 2** (retrain required; Stage 0/1/2 reused, Stage 3 step 1 reused since prompt is text-only and Q-Former output is not injected):
```bash
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie \
    --cfg-path configs/config.yaml \
    --options model.qformer_config.instruction_aware=False
```

**Vanilla Q-Former — eval:**
```bash
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie \
    --cfg-path configs/config.yaml \
    --options model.qformer_config.instruction_aware=False run.evaluate=True
```

**User-conditioned queries — train Stage 3 step 2** (branch `feat/user-conditioned-queries`; Stage 0/1/2/step1 ckpts reused; only Stage 3 step 2 retrains because `user_proj` is the only new param and it only receives gradient when soft tokens are injected — Step 1 prompts are text-only):
```bash
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie \
    --cfg-path configs/config.yaml \
    --options \
        model.qformer_config.user_conditioned=True \
        run.qformer_stage3_step2.max_epoch=25 \
        run.qformer_stage3_step2.output_dir=/content/SigLLM/ckpt/qformer_stage3_step2_cie_userq_qwen2/
```

Verify in startup log:
- `USER-CONDITIONED MODE | user_conditioned=True → Q-Former queries are shifted per-user...`
- `QFormer ckpt missing keys (kept at init) | user_proj.weight, user_proj.bias`
- `Trainable parameter counts | qformer=76211712 (= 76M base + 197k user_proj)`

### Checkpoint locations

- **Instruction-aware best ckpt** (val_uAUC 0.7098, 24/05 run): backed up to user Drive (local copy overwritten by Vanilla training on 29/05)
- **Vanilla best ckpt** (val_uAUC 0.7081, 29/05 run): `/content/SigLLM/ckpt/qformer_stage3_step2_cie_qwen2/qwen2-7b-base/checkpoint_best.pth`

To restore Instruction-aware for eval, copy ckpt from Drive back to that path and run with `instruction_aware=True` (default).

### Important caveat — checkpoint / flag matching

The `instruction_aware` flag is a model hyperparameter, NOT stored in checkpoint weights. Loading mismatch produces invalid evaluation:

- Vanilla ckpt + `instruction_aware=True` flag → mismatch (vanilla weights forced through instruction-aware forward) → wrong numbers
- Instruction-aware ckpt + `instruction_aware=False` flag → mismatch → wrong numbers

Always set the flag to match the ckpt origin. Verify in log start:
- `Q-Former mode | instruction-aware (InstructBLIP-style)` → matches Instruction-aware ckpt
- `Q-Former mode | vanilla queries-only (BLIP-2-style) [ABLATION]` → matches Vanilla ckpt

## Next Steps

1. **Decision point:** Choose pivot strategy (1, 2, or 3) based on:
   - Time remaining to thesis defense
   - Advisor's tolerance for negative findings
   - Willingness to redesign prompts and rerun (Pivot 2)

2. **If Pivot 2:** Modify `QFORMER_ITEM_INSTRUCTIONS` in `src/sigllm/models/multimodal/qformer_rec_llm.py` lines 69-82 with diverse semantic angles. Retrain both Instruction-aware (`instruction_aware=True`) and Vanilla (`instruction_aware=False`). Re-evaluate.

3. **If Pivot 1 or 3:** Write thesis with current results. Use Text-only ablation as primary evidence of CF contribution (+5.15 pts test_warm uAUC). Frame Vanilla ≈ Instruction-aware as honest investigation finding with hypothesized reasons.
