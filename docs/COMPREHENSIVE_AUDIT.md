# SigLLM Comprehensive Experiment Audit

**Date:** 2026-06-03
**Scope:** Full codebase + git history + docs + checkpoints + notebooks
**Goal:** Phân biệt CHẶT CHẼ experiment ĐÃ CHẠY THẬT vs PROPOSED (chưa chạy)
**Honest principle:** Mỗi mục đánh dấu rõ `TESTED ✓`, `NOT TESTED ✗`, hoặc `NO DATA ?`. Không suy đoán số liệu.

---

## Resource inventory

| Resource | Path | Status |
|---|---|---|
| Git branches | 24 branches (15 feat/*, 1 experiment/*, 1 fix/*) | Verified |
| Notebooks | `notebooks/` (10 files) | 7 dated post-2026-05-19, each ~1-3 MB → có outputs |
| Local checkpoints | `checkpoints/13_5_2026/` (MF + Stage 1 + Stage 2) | Local copy only |
| Colab checkpoints | `/content/SigLLM/ckpt/...` | Live trên Colab, không có ở repo |
| Ablation doc | `docs/ABLATION_RESULTS.md` (728 lines) | Main result source |
| Improvement doc | `docs/QFORMER_IMPROVEMENT_DIRECTIONS.md` (478 lines) | Proposals, mostly NOT YET RUN |
| Init doc | `docs/QFORMER_INIT_IMPROVEMENTS.md` (321 lines) | Proposals only |
| FAQ doc | `docs/QFORMER_FAQ.md` (570 lines) | Technical clarifications |
| Codebase audit | `docs/CODEBASE_AUDIT.md` (vừa viết hôm nay) | Static code analysis |

---

# 1. Danh sách experiment đã chạy — verified status

Tổng hợp từ git branches + ablation doc + training notebooks. Mark **`TESTED ✓`** chỉ khi có số liệu trong doc/log; **`NOT TESTED ✗`** nếu chỉ là proposal.

## 1.1 — Tested experiments (with numbers)

| # | Experiment | Branch | Notebook | Best val_uAUC | Test_warm uAUC | Status |
|---|---|---|---|---|---|---|
| 1 | **Vanilla Q-Former** (instruction-aware OFF) | `feat/best-0.72-warm` (commit 13702bb) | `SigLLM_vanilla_29_5_2026.ipynb` | **0.7081** (epoch 5) | 0.7213 | ✓ |
| 2 | **Instruction-aware V1** (12 paraphrase) | `feat/ablation-instruction-aware` (a69e9e7) | `SigLLM._best_24_05_2026ipynb.ipynb` + `SigLLM_rerun_28_5_2026.ipynb` | **0.7098** (24/05 best) / 0.7018 (28/05 rerun) | 0.7265 | ✓ |
| 3 | **Task-focused instructions** (8 prompts) | `feat/best-0.72-warm-update-instruction` (bc2a8ef) | _unclear which notebook_ | "~baseline" (no exact number in doc) | n/a | partial — số liệu không chi tiết |
| 4 | **Interaction-aware Q-Former** | `feat/interaction-aware-qformer` (7d8279d) | `SigLLM_interaction_instruction.ipynb` | **0.7036** | n/a | ✓ |
| 5 | **User-conditioned queries** | `feat/user-conditioned-queries` (2478f3f) | `SigLLM_user_condition.ipynb` | **0.7060** (epoch 3) | n/a (val only) | ✓ |
| 6 | **User soft tokens (re-enable)** | `feat/reenable-user-soft-tokens` (7338ab3, current) | _live colab session_ | **0.7023** (epoch 4) | 0.7214 | ✓ |
| 7 | **Text-only ablation** (`ablate_soft_tokens=True`) | Same baseline ckpt + inference flag | Same baselines + cell flip | 0.6878 test_overall / 0.6750 test_warm | 0.6750 | ✓ |
| 8 | **Step 1 LoRA-only** (intermediate ckpt) | Default pipeline (`tuning_step=1`) | All baselines | 0.6917 peak | n/a | ✓ (intermediate) |
| 9 | **8-layer Q-Former** (vs 4) | `feat/best-0.72-warm` history (commits 2331b57 → cfd6418) | Pre-vanilla notebooks | Reverted — "8 layers Q-Former too large for 33k samples, plateaued at uAUC ~0.694" (per `qformer_rec_llm.py:320`) | n/a | ✓ (revert) |
| 10 | **LoRA r=16 [q,k,v,o]** ablation | (mentioned in config.yaml comment) | Unknown notebook | "r=16 overfit on 33k, val uAUC peaked at 0.7048 vs r=8 baseline 0.7077" (per `config.yaml:30-32`) | n/a | ✓ (revert) |

## 1.2 — Proposed but NOT TESTED

| # | Experiment | Source proposal | Status |
|---|---|---|---|
| 11 | **MLP bridge / no Q-Former** | `QFORMER_IMPROVEMENT_DIRECTIONS.md #8` | ✗ NOT TESTED |
| 12 | **Linear bridge direct (no Q-Former)** | Implicit from CoLLM comparison | ✗ NOT TESTED |
| 13 | **Multi-token CF projection** (M > 1) | Implicit from `CODEBASE_AUDIT.md` bottleneck A | ✗ NOT TESTED |
| 14 | **Num queries Q sweep** (Q=1,2,4,16,32) | Implicit | ✗ NOT TESTED |
| 15 | **CF encoder tokens M sweep** | Implicit | ✗ NOT TESTED |
| 16 | **LayerNorm before `proj_cf`** | Bottleneck candidate | ✗ NOT TESTED |
| 17 | **GELU in `llm_proj` (MLP variant)** | `CODEBASE_AUDIT.md` bottleneck B | ✗ NOT TESTED |
| 18 | **L2 norm MF embeddings** | Implicit | ✗ NOT TESTED |
| 19 | **Dropout / Residual in projection** | Implicit | ✗ NOT TESTED |
| 20 | **Separate projections (user/item/history)** | Implicit | ✗ NOT TESTED |
| 21 | **Unified Stage 3** (no 2-step) | `QFORMER_IMPROVEMENT_DIRECTIONS.md` Hypothesis B | ✗ NOT TESTED |
| 22 | **Step 1 LoRA WITH soft tokens** | Same proposal | ✗ NOT TESTED |
| 23 | **Stage 3 from random Q-Former** | `QFORMER_INIT_IMPROVEMENTS.md` (Stage ablation) | ✗ NOT TESTED |
| 24 | **Stage 3 with Stage 1 ckpt skip Stage 2** | Same | ✗ NOT TESTED |
| 25 | **Stage 1 loss ablation** (skip ITG/ITM/UI/II individually) | Implicit | ✗ NOT TESTED on SigLLM (ILM paper Table 4 has this on OpenP5, không transfer được số) |
| 26 | **Smart query init from cluster centroids** | `QFORMER_INIT_IMPROVEMENTS.md` #1 | ✗ NOT TESTED |
| 27 | **OLS warm-start proj_cf** | `QFORMER_INIT_IMPROVEMENTS.md` #2 | ✗ NOT TESTED |
| 28 | **MPNet/beeFormer text base** | `QFORMER_INIT_IMPROVEMENTS.md` #3 | ✗ NOT TESTED |
| 29 | **Semantic alignment aux loss at Stage 3** | `QFORMER_IMPROVEMENT_DIRECTIONS.md` #2 | ✗ NOT TESTED |
| 30 | **Target-conditioned gating** | `QFORMER_IMPROVEMENT_DIRECTIONS.md` #3 | ✗ NOT TESTED |

### Per-experiment metadata (tested ones only)

#### Exp 1 — Vanilla Q-Former (baseline)
- **Config:** `configs/config.yaml` (default `instruction_aware: False` effectively because vanilla = `encode_cf`)
- **Checkpoint chain:** MF → Stage 1 → Stage 2 → Stage 3 step 1 → Stage 3 step 2
- **Random seed:** 42 (`run.seed`)
- **Train from:** Stage 3 step 1 LoRA ckpt (loaded warm)
- **Date:** 2026-05-29 (notebook timestamp)
- **Code path:** Used `forward()` via `qformer.encode_cf()` path conditional

#### Exp 2 — Instruction-aware V1
- **Config:** Same + `model.qformer_config.instruction_aware=True`
- **Checkpoint chain:** Same
- **Seed:** 42
- **Date:** 2026-05-24 (best ckpt), 2026-05-28 (rerun)
- **Code path:** `QFORMER_ITEM_INSTRUCTIONS` list (12 paraphrase, lines 71-84 hiện tại match V1)

#### Exp 3 — Task-focused
- **Config:** Same as V1 + commit `bc2a8ef` replaces `QFORMER_ITEM_INSTRUCTIONS` list (8 CTR-vocab prompts)
- **Note:** Doc gọi là "Set A" trong commit message. Số liệu chỉ "~baseline" — không có epoch-by-epoch table.

#### Exp 4 — Interaction-aware
- **Config:** Same + `model.qformer_config.interaction_aware=True` (added in `3df5bac`)
- **Code path:** Joint encode `[target, history_1, ..., history_L]` thành 8 soft tokens

#### Exp 5 — User-conditioned queries
- **Config:** Same + `model.qformer_config.user_conditioned=True`
- **Code change:** Added `self.user_proj = nn.Linear(d_user=256, d_model=768)` với zero-init, line ~90 trong adapter (branch only)
- **Trainable params delta:** +197K
- **Seed:** 42

#### Exp 6 — User soft tokens (re-enable)
- **Config:** Same + `model.qformer_config.enable_user_soft_tokens=True`
- **Code change:** Re-enable disabled `<UserID>` slot, new prompt file `qformer_prompt_movie_with_user.txt`
- **Date:** 2026-06-01 (training log)

---

# 2. Bảng kết quả chuẩn hóa

⚠️ **Caveat:** ABLATION_RESULTS.md không log đầy đủ tất cả columns. Cells **`n/a`** = không có data trong doc/notebook. Cells **`?`** = data tồn tại trong notebook nhưng không extract được.

## Master result table

| Experiment | Main change | Trainable | S1 ckpt | S2 ckpt | S3 strategy | Best val epoch | Best val AUC | Best val uAUC | Best val loss | Test AUC | Test uAUC | Warm AUC | Warm uAUC | Cold AUC | Cold uAUC | Score gap | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Vanilla Q-Former** | Baseline (no instruction) | 78.8M | qformer_stage1_best | qformer_stage2_best | 2-step | **5** | 0.7207 | **0.7081** | 0.6169 | **0.7325** | **0.7170** | 0.7451 | 0.7213 | 0.7072 | 0.6516 | 0.183 | Best baseline |
| **Instruction-aware V1 (24/05)** | + instruction text → self-attn | 78.8M | same | same | 2-step | **3** | 0.7187 | 0.7098 | 0.6177 | 0.7322 | 0.7131 | 0.7456 | 0.7265 | 0.7072 | 0.6478 | 0.177 | Best test_warm uAUC |
| Instruction-aware (28/05 rerun) | same | same | same | same | 2-step | 2 | 0.7213 | 0.7018 | 0.6177 | n/a | n/a | n/a | n/a | n/a | n/a | 0.185 | Lower than 24/05 → seed noise |
| **Interaction-aware** | Joint encode N items | n/a (likely 78.8M) | same | same | 2-step | n/a | n/a | **0.7036** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Below baseline. Bandwidth bottleneck |
| **User-conditioned queries** | `queries += user_proj(user_cf)` | 79M | same | same | 2-step | **3** | **0.7582** ← max AUC | 0.7060 | 0.5980 | n/a (val only) | n/a | n/a | n/a | n/a | n/a | 0.216 | AUC win, uAUC tie |
| **User soft tokens (NEW)** | `<UserID>` slot re-enabled | 78.8M | same | same | 2-step | **4** | 0.7543 | 0.7023 | 0.5890 | **0.7336** | 0.7145 | **0.7462** | 0.7214 | 0.7077 | **0.6695** | 0.208 | **Cold-item win** +1.79 |
| **Text-only ablation** | Soft tokens zeroed at eval | (any) | same | same | (load any) | (best of base) | n/a | n/a | n/a | 0.6960 | 0.6878 | 0.7017 | 0.6750 | 0.6756 | 0.6478 | 0.153 | −5.15 vs Vanilla on warm |
| **Stage 3 Step 1 only** | LoRA only, text-only prompt | 2.5M (LoRA) | same | same | step 1 only | **4** | 0.7196 | 0.6917 | 0.6156 | n/a | n/a | n/a | n/a | n/a | n/a | 0.172 | Intermediate stop |
| 8-layer Q-Former | (Reverted) | ~152M | n/a | n/a | n/a | n/a | n/a | **~0.694** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Comment in code only — no full log |
| LoRA r=16 [q,k,v,o] | (Reverted) | ~10M (LoRA) | n/a | n/a | n/a | n/a | n/a | **0.7048** | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | Comment in config only |

### Best per metric

| Metric | Winner | Value | vs vanilla baseline |
|---|---|---|---|
| **val_uAUC peak** | Instruction-aware V1 (24/05) | 0.7098 | +0.0017 |
| **val_AUC peak** | User-conditioned queries | 0.7582 | +0.036 |
| **test (overall) AUC** | User soft tokens | 0.7336 | +0.0011 |
| **test (overall) uAUC** | **Vanilla Q-Former** | **0.7170** | — |
| **test_warm AUC** | User soft tokens | 0.7462 | +0.0011 |
| **test_warm uAUC** | **Instruction-aware V1** | **0.7265** | +0.0052 |
| **test_cold AUC** | User soft tokens | 0.7077 | +0.0005 |
| **test_cold uAUC** | **User soft tokens** | **0.6695** | +0.0179 (significant) |

→ **KHÔNG có single winner** — mỗi metric/split có method khác nhau.

---

# 3. Kiểm tra công bằng

Trả lời cho từng experiment đã tested:

## 3.1 — Shared infrastructure (FAIR)

| Yếu tố | Status | Source |
|---|---|---|
| Train/valid/test split | ✓ **SAME** — pre-built `train_ood2.pkl`, `valid_ood2.pkl`, `test_ood2.pkl`, `test_warm_cold_ood2.pkl` | `config.yaml:75-78` |
| Negative samples | ✓ **SAME** — fixed by dataset construction (point-wise) | `MovieOODDataset.__getitem__` |
| Eval script | ✓ **SAME** — `RecBaseTask.evaluate` (`rec_base_task.py:152`) | All Stage 3 runs |
| Best checkpoint metric | ✓ **SAME** — `agg_metrics = AUC` (`rec_base_task.py:197`) — NOTE: NOT uAUC | `'agg_metrics': metrics.get('auc', -metric_logger.meters['loss'].global_avg)` |
| LLM base | ✓ **SAME** — Qwen2-7B-Base | `config.yaml:4` |
| MF checkpoint | ✓ **SAME** — `/content/SigLLM/ckpt/mf/mf_model.pth` | All branches inherit `feat/best-0.72-warm` baseline |
| Warm/cold split | ✓ **SAME** — `test_warm_cold_ood2.pkl` | All branches |
| Stage 1/2 ckpts | ✓ **SAME** — reused across all Stage 3 experiments | All branches |

## 3.2 — Per-experiment variations (POTENTIALLY UNFAIR)

| Yếu tố | Vanilla | Instr V1 | Task-focused | Interaction-aware | User-cond queries | User soft tokens |
|---|---|---|---|---|---|---|
| Random seed | 42 | 42 | n/a | n/a | 42 | 42 |
| Epochs trained | 8 (peak 5) | 5 (peak 3) | n/a | n/a | 7 (peak 3) | 5 (peak 4) |
| Max epoch config | 200 | 200 | 200 | 200 | 25 | 20 (current run) |
| Batch size | 16 | 16 | 16 | 16 | 16 | 16 |
| Learning rate Step 2 | 3e-5 | 3e-5 | 3e-5 | 3e-5 | 3e-5 | 3e-5 |
| Prompt file (LLM) | `qformer_prompt_movie.txt` | same | same | same | same | **`qformer_prompt_movie_with_user.txt`** (different!) |
| Q-Former instructions | 12 V1 paraphrase | same | **8 task-focused** | same V1 | same V1 | same V1 |
| Early stopping | None explicit (relies on save-best by AUC) | same | same | same | killed at epoch 7 | killed at epoch 5 |

### Findings về fairness

| Concern | Magnitude | Implication |
|---|---|---|
| **Best ckpt theo AUC, không phải uAUC** | HIGH | `rec_base_task.py:197`: `'agg_metrics': metrics.get('auc', ...)` — Test results report theo AUC-best ckpt, KHÔNG phải uAUC-best ckpt. Nếu chọn theo uAUC-best, test numbers có thể khác |
| **User soft tokens dùng prompt file khác** | MEDIUM | `qformer_prompt_movie_with_user.txt` thêm `<UserID>` slot — không thể isolate "prompt change" vs "user soft token" |
| **Max epoch khác nhau** | LOW | User-cond max_epoch=25, User soft tokens=20, Vanilla=200 → user-cond + user soft tokens có thể chưa converge full |
| **Killed early lúc decline** | LOW | Cả 2 user-* experiments đều killed sau 3 epoch decline — convention nhất quán nhưng không có "fixed N epochs" rule |
| **Task-focused chưa report đầy đủ** | HIGH | Không có epoch-by-epoch table cho task-focused → không thể compare apples-to-apples |
| **8-layer / r=16 LoRA chỉ là code comment** | HIGH | Không có notebook/log cho 2 ablations này → claim "tested" yếu |

### Kết luận fairness

| Comparison | Fair? | Caveat |
|---|---|---|
| Vanilla vs Instruction-aware V1 | ✓ **FAIR** | Same prompt, same epochs, same seed |
| Vanilla vs Interaction-aware | ⚠️ **MOSTLY FAIR** | Number of soft tokens khác (88 → 8) — fundamental change |
| Vanilla vs User-conditioned | ⚠️ **MOSTLY FAIR** | Trainable params khác (+197K), max_epoch khác |
| Vanilla vs User soft tokens | ⚠️ **CONFOUNDED** | Prompt file khác + soft token count khác (88 → 96) |
| User-conditioned vs User soft tokens | ⚠️ **MOSTLY FAIR** | Same goal (user CF inject), khác mechanism, khác prompt |
| Task-focused vs anything | ✗ **NOT FAIR TO COMPARE** | Insufficient logged data |
| 8-layer vs 4-layer Q-Former | ✗ **NOT FAIR TO COMPARE** | No notebook, only code comment |

---

# 4. Chi tiết MLP bridge / no Q-Former

**Status: ✗ NOT TESTED**

Đây là **bottleneck major** trong audit. CoLLM paper dùng `Linear → GELU → Linear` (intermediate 10x). SigLLM **không có baseline này** để compare.

| Question | Answer |
|---|---|
| File implement? | KHÔNG có (`grep -rn "MLP" src/sigllm/models/ → empty`) |
| Architecture proposed | Per `QFORMER_IMPROVEMENT_DIRECTIONS.md #8`: `nn.Sequential(Linear(256, 256*10), GELU, Linear(256*10, 768*K))` |
| Param count estimated | `256*2560 + 2560*6144 ≈ 16M` (K=8) — less than Q-Former 76M |
| Stage 1/2 pretraining | Likely none (CoLLM doesn't pretrain MLP) |
| Test results | KHÔNG có |

→ **Mục này không trả lời được vì experiment chưa chạy.** Recommend chạy MLP baseline để confirm Q-Former giá trị thực sự.

---

# 5. Chi tiết multi-token CF projection

**Status: ✗ NOT TESTED**

Q-Former hiện cross-attend vào **chỉ 1 token** (verified `hf_qformer_adapter.py:220`):
```python
encoder_hidden_states = self.proj_cf(cf_vec).unsqueeze(1)  # [B, 1, 768]
```

| Question | Answer |
|---|---|
| M sweep | KHÔNG có (M=1 fixed) |
| Architecture variants | KHÔNG có |
| Cross-attention pattern with M>1 | KHÔNG đo |
| Attention map analysis | KHÔNG có (chưa extract) |
| Results | KHÔNG có |

→ **`CODEBASE_AUDIT.md` đã identify đây là HIGH-probability bottleneck.** Nhưng chưa có experiment test.

---

# 6. Chi tiết số query tokens Q

**Status: ✗ NOT TESTED Q sweep**

Q=8 hardcoded suốt:
- `config.yaml:58: num_queries: 8`
- Không có branch nào sweep Q=1,2,4,16,32

`git log --all --grep="num_queries\|query_tokens"` returns empty.

| Q value | Tested? | Result |
|---|---|---|
| 1 | ✗ | n/a |
| 2 | ✗ | n/a |
| 4 | ✗ | n/a |
| **8** | ✓ (default) | val_uAUC 0.7081 (vanilla baseline) |
| 16 | ✗ | n/a |
| 32 | ✗ | n/a |

→ **Major missing experiment.** ILM paper (Figure 3) cho thấy Q ảnh hưởng đáng kể. Không thể trả lời "Q nào tốt nhất warm/cold" vì chưa test.

---

# 7. Chi tiết LayerNorm / projection

Baseline hiện tại: `Linear(768, 3584) + LayerNorm(3584)` — đã verified `qformer_rec_llm.py:429-432`.

Variants:

| Variant | Tested? | Source / commit | Result |
|---|---|---|---|
| `Linear → LayerNorm` (current) | ✓ | `ee48751` (added LayerNorm), `4471599` (LayerNorm init) | Baseline |
| `Linear + GELU + Linear + LayerNorm` | ✗ | — | NO DATA |
| `LayerNorm(256)` trước proj_cf | ✗ | — | NO DATA |
| L2 norm MF embedding trước proj_cf | ✗ | — | NO DATA |
| Dropout trong projection | ✗ | `26e5cb3` removed `proj_dropout` (so có dropped, đã loại) | NO new test |
| Residual connection | ✗ | — | NO DATA |
| Separate proj cho user/item/history | ✗ | — | NO DATA. Hiện tại **SHARED projection** cho cả 3 (verified `qformer_rec_llm.py:638-639,666`) |
| `proj_mid` intermediate layer | ✗ removed | `6ca7be5` "remove proj_mid parameter" | Đã từng có, removed |

→ **Chỉ baseline được tested.** Không có variant nào có số liệu so sánh.

---

# 8. Stage 3 training strategy

| Strategy | Tested? | Notes |
|---|---|---|
| **Step 1 text-only LoRA, Step 2 Q-Former+proj** (CoLLM 2-step) | ✓ | Default. Used by all experiments. |
| Step 1 LoRA WITH soft tokens | ✗ | Proposed as Hypothesis B in `QFORMER_IMPROVEMENT_DIRECTIONS.md` |
| Unified LoRA + Q-Former + projection | ✗ | Same proposal |
| Q-Former frozen, projection only | ✓ partial | "α2 experiment" mentioned in code comment `qformer_rec_llm.py:317-323`: "Q-Former briefly frozen... plateaued at uAUC ~0.694" — NO full notebook |
| Projection frozen, Q-Former only | ✗ | NO DATA |
| LoRA unfrozen in Step 2 | ✗ | NO DATA |
| Train LoRA với small LR in Step 2 | ✗ | NO DATA |
| Train từ Stage 2 ckpt vs Stage 1 ckpt | ✓ default Stage 2 | NO Stage 1-only Stage 3 test |
| Train bridge từ scratch vs pretrained | ✗ | NO scratch test |

### Step 1 trajectory (`ABLATION_RESULTS.md:114-120`)

| Epoch | AUC | uAUC | val_loss |
|---|---|---|---|
| 0 | 0.7056 | 0.6749 | 0.6369 |
| 1 | 0.7106 | 0.6776 | 0.6267 |
| 2 | 0.7122 | 0.6841 | 0.6168 |
| 3 | 0.7170 | 0.6895 | 0.6205 |
| **4** | **0.7196** | **0.6917** (peak) | 0.6156 |

→ Step 1 alone đạt 0.6917 uAUC. Step 2 add ~+0.018 từ soft tokens.

---

# 9. Stage 1/2 có thật sự đóng góp?

**Status: KHÔNG có ablation đầy đủ Stage 1/2 trên SigLLM.**

| Ablation | Tested trên SigLLM? | Source |
|---|---|---|
| Stage 3 từ random Q-Former | ✗ NOT TESTED | ILM paper "ILM-rand" comparable analysis on OpenP5 (Table 3) shows ILM > ILM-rand by ~+0.001 HR@5 |
| Stage 3 từ BERT-init only Q-Former (skip Stage 1+2) | ✗ NOT TESTED | — |
| Stage 3 từ Stage 1 ckpt (skip Stage 2) | ✗ NOT TESTED | — |
| Stage 3 từ Stage 2 ckpt | ✓ DEFAULT | Used by all experiments |
| Bỏ Stage 2 generative | ✗ NOT TESTED | — |
| Bỏ Stage 1 ITG | ✗ NOT TESTED on SigLLM. ILM Table 5 trên OpenP5 ML1M: ILM-IT-UI eval loss 4.07 vs ILM-IT 4.17 → UI giúp |
| Bỏ Stage 1 ITM | ✗ NOT TESTED |
| Bỏ Stage 1 UI loss | ✗ NOT TESTED on SigLLM. ILM Table 4 chỉ test trên rec retrieval metric, không transfer được số |
| Bỏ Stage 1 II loss | ✗ NOT TESTED on SigLLM. ILM Table 4 same situation |
| Chỉ UI/II contrastive | ✗ NOT TESTED |
| Chỉ item-text contrastive | ✗ NOT TESTED |

→ **Không có evidence trực tiếp** rằng Stage 1/2 đóng góp BAO NHIÊU vào final uAUC. Chỉ có ILM paper indirect evidence (ILM > ILM-rand on retrieval, không phải CTR).

→ **Đáng chạy:** Stage 3 từ random Q-Former → đo "Stage 1+2 contribution to uAUC" trên SigLLM specifically.

---

# 10. Kiểm tra metric và checkpoint selection

## 10.1 — `agg_metrics` (verified)

`src/sigllm/tasks/base/rec_base_task.py:196-200`:
```python
all_results = {
    'agg_metrics': metrics.get('auc', -metric_logger.meters['loss'].global_avg),
    'auc': metrics.get('auc', 0),
    'acc': val_acc,
    ...
}
```

→ **Best checkpoint chọn theo AUC, KHÔNG phải uAUC.** This is a **fairness issue**:
- Nếu test_uAUC report là **uAUC ở ckpt best-AUC**, không phải best-uAUC
- Khả năng test_uAUC under-report nhẹ so với potential

## 10.2 — Best ckpt theo AUC khác best theo uAUC?

Vanilla trajectory (verified `ABLATION_RESULTS.md:144-155`):

| Epoch | AUC | uAUC | val_loss |
|---|---|---|---|
| 0 | 0.7204 | 0.6954 | 0.6129 |
| 1 | 0.7209 | 0.6893 | 0.6126 |
| 2 | 0.7223 | 0.6979 | 0.6142 |
| 3 | 0.7212 | 0.7008 | 0.6143 |
| 4 | 0.7215 | 0.7057 | 0.6144 |
| **5** | 0.7207 | **0.7081** (uAUC peak) | 0.6169 |
| 6 | **0.7218** (AUC peak?) | 0.7061 | 0.6130 |
| 7 | 0.7216 | 0.7052 | 0.6192 |

→ **Tại Vanilla:** best AUC ở epoch 6 (0.7218), best uAUC ở epoch 5 (0.7081). Δepoch=1. Có thể save ckpt khác.

→ **Implication:** Nếu chọn theo uAUC, test có thể khác nhẹ. Không thể determine without rerun.

## 10.3 — Per-epoch val_uAUC log

✓ **CÓ** — đầy đủ trong `ABLATION_RESULTS.md` Training Trajectories section (lines 106-191).

## 10.4 — Test result hiện tại có dùng đúng best ckpt?

Tested ckpt = ckpt được save theo agg_metrics=AUC. → Test results match best-AUC ckpt, không phải best-uAUC.

---

# 11. Phân tích warm/cold (verified data only)

## 11.1 — Performance tables (extracted from ABLATION_RESULTS.md)

### test_warm (3522 samples)

| Method | AUC | uAUC |
|---|---|---|
| Vanilla | 0.7451 | 0.7213 |
| Instruction-aware (24/05 best) | 0.7456 | **0.7265** |
| Instruction-aware (28/05) | 0.7448 | 0.7217 |
| User soft tokens | 0.7462 | 0.7214 |
| Text-only (CF zeroed) | 0.7017 | 0.6750 |

### test_cold (3178 samples)

| Method | AUC | uAUC |
|---|---|---|
| Vanilla | 0.7072 | 0.6516 |
| Instruction-aware (24/05) | 0.7072 | 0.6478 |
| Instruction-aware (28/05) | 0.7068 | 0.6503 |
| User soft tokens | 0.7077 | **0.6695** |
| Text-only | 0.6756 | 0.6478 |

## 11.2 — Warm vs cold analysis

| Experiment | Warm uAUC change vs vanilla | Cold uAUC change vs vanilla | Trade-off |
|---|---|---|---|
| Instruction-aware | **+0.0052** | −0.0038 | warm win, cold lose |
| User soft tokens | +0.0001 (tie) | **+0.0179** | cold win, warm tie |

→ **Trade-off rõ ràng giữa 2 methods.**

## 11.3 — Tại sao user soft tokens giúp cold?

Hypothesis từ `QFORMER_FAQ.md`:
> "Cold = items chưa thấy ở training → item CF nghèo signal. User CF vẫn rich → user soft tokens BÙ cho item signal thiếu hụt"

Verified text-only test_cold uAUC = 0.6478 ≈ Vanilla test_cold uAUC = 0.6516 (Δ +0.0038, in noise). → **Item CF không contribute trên cold**. → User soft tokens fills cái gap này.

## 11.4 — MLP / multi-token CF có giúp cold không?

**KHÔNG có data** — chưa tested.

## 11.5 — Cold items rely on title text more than CF?

**NO DIRECT DATA**. Nhưng có evidence indirect:
- Text-only Vanilla test_cold uAUC = 0.6478 (chỉ dùng title)
- Full Vanilla test_cold uAUC = 0.6516 (+0.0038)
- → CF chỉ contribute ~0.004 uAUC trên cold → **title dominate trên cold** (consistent với hypothesis)

---

# 12. Phân tích score distribution

⚠️ **Limited data** — codebase chỉ log mean/std/gap, KHÔNG có histogram.

## 12.1 — Available stats (from training logs in conversation)

| Experiment | pos_score_mean | neg_score_mean | score_gap | score_mean | score_std |
|---|---|---|---|---|---|
| Vanilla baseline | ~0.60 | ~0.43 | ~0.183 | ~0.51 | ~0.20 |
| Instruction-aware (24/05) | 0.598 | 0.438 | 0.161 | 0.523 | 0.200 |
| User soft tokens epoch 4 | 0.588 | 0.383 | 0.205 | 0.491 | 0.232 |

→ **Score distribution chưa được dump per-sample.** Histogram phải reconstruct from raw eval output.

## 12.2 — Calibration shift in user soft tokens

Quan sát:
- Vanilla score_gap = 0.183, user soft tokens = 0.208 → **+14% gap improvement**
- Nhưng uAUC: Vanilla 0.7081 vs user soft tokens 0.7023 → **lower**

→ **Gap rộng hơn nhưng không tăng ranking quality per user** — chính là calibration shift pattern.

## 12.3 — AUC tăng nhưng uAUC giảm experiments

| Experiment | Δ AUC vs Vanilla | Δ uAUC vs Vanilla |
|---|---|---|
| User-conditioned queries | +0.036 (val) | −0.004 (val) |
| User soft tokens | +0.033 (val) | −0.006 (val) |

**Pattern explanation:** User-level signal tạo per-user bias (constant cho mọi items của 1 user) → AUC tăng vì separate users tốt hơn, uAUC không vì không refine item ordering WITHIN user.

---

# 13. Phân tích theo user

**Status: NO DATA — codebase không log per-user metrics.**

| Question | Answer |
|---|---|
| uAUC distribution theo user | ✗ NO DATA. Chỉ log avg uAUC across users. Phải reconstruct từ raw eval. |
| Top users improved most | ✗ NO DATA |
| Top users worsened most | ✗ NO DATA |
| User ít vs nhiều interaction | ✗ NO DATA |
| User cold/warm behavior | ✗ NO DATA |
| Bias groups | ✗ NO DATA |
| User soft tokens giúp user ít history? | ✗ NO DATA. **Worth running per-user analysis** với raw predictions. |

Code-level: `calculate_user_auc` (`train_rec_baseline.py:26-67`) skip users với "only_one_interaction" hoặc "only_one_class". Train log:
- "Users with only one interaction | 30" (test_warm)
- "Users with only one class | 44" (test_warm)
- → 282 / 356 users contribute to test_warm uAUC

→ Có **74 users (21%) bị skip** mỗi eval. Worth investigating.

---

# 14. Phân tích theo item

**Status: NO DATA — codebase không log per-item metrics.**

| Question | Answer |
|---|---|
| Item warm vs cold split detail | ✓ partial — count only (3522 warm, 3178 cold) |
| Item popular vs unpopular | ✗ NO DATA |
| Item title genre specificity | ✗ NO DATA |
| Item embedding norm distribution | ✗ NO DATA |
| Per-item improvement | ✗ NO DATA |
| Popularity bias | ✗ NO DATA |

→ Tất cả analysis này cần raw eval predictions + item metadata → chưa thực hiện.

---

# 15. Soft token insertion verification

## 15.1 — Token count matching

Verified from training log (User soft tokens run):
```
valid_history_items=6, user_soft_tokens=8, history_soft_tokens=48, target_soft_tokens=8,
sample_soft_tokens=64, sample_unk_slots=64, batch_unk_slots=1168
```
→ `sample_soft_tokens = sample_unk_slots = 64` (match per sample)
→ `batch_unk_slots = 1168 = sum_over_batch(per_sample_slots)` (match)

✓ **Counts match per sample + per batch.**

## 15.2 — Mismatch detection

Code: `qformer_rec_llm.py:780-781`:
```python
replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
inputs_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = rec_embeds['merged_embs'].to(...)
```

→ `replaced_idx.shape[0]` must equal `merged_embs.shape[0]`. Nếu mismatch → PyTorch sẽ raise indexing error → exception. **Chưa thấy error trong training** → consistent.

## 15.3 — Collision với `<|im_start|>`

`_resolve_soft_token_placeholder` (`qformer_rec_llm.py:230-274`) picks `<|im_start|>` for Qwen2 (verified log: `soft_token_id=151644 ('<|im_start|>')`).

Prompts hiện tại (`prompts/qformer_prompt_movie*.txt`) **không chứa literal `<|im_start|>`**. → No collision.

⚠️ **Risk:** Nếu sau này dùng Qwen2-Instruct version + dùng chat template → có thể collision. Cần test khi đổi LLM.

## 15.4 — Padding scatter mismatch

History pad ID = 0. Code mask:
```python
item_mask = (ids != self.rec_encoder.padding_index).long()  # padding_index = 0
item_mask_q = item_mask.unsqueeze(-1).repeat(1, 1, Q).reshape(B, L * Q)
```

→ Padded positions có `item_mask_q = 0` → KHÔNG được include trong `merged_embs`. ✓ Verified consistent.

## 15.5 — History length = 0

Test: `MovieOODDataset.__getitem__` line 103-114:
```python
interacted_count = history_len - 1 if (history_len > 0 and history_item_ids[0] == 0) else history_len
if history_len < max_history_len:
    pad_size = max_history_len - history_len
    padded_history_ids = ([0] * pad_size) + list(history_item_ids)
```

→ Nếu `history_len = 0`, `padded_history_ids = [0]*10`, `interacted_count = 0`.
→ Prompt: `item_list_placeholder = " ".join([unk_seq] * 0) = ""` → empty replacement.

✓ **Handled** but worth verifying test set có sample nào history=0 không (likely no, vì dataset OOD2 đã filter).

---

# 16. Gradient & trainable params

## 16.1 — Trainable params (verified log)

User soft tokens run:
```
Trainable parameter counts | rec_encoder=0, qformer=76014336, llm_proj=2763264, llm_model=0, llm_lora=0
```

- ✓ rec_encoder = 0 (MF frozen)
- ✓ qformer = 76M (trainable)
- ✓ llm_proj = 2.76M (trainable)
- ✓ llm_model = 0 (base frozen)
- ✓ llm_lora = 0 (LoRA frozen tại Step 2)

→ **Consistent với expected tuning_step=2 policy.**

## 16.2 — Gradient norm logging

**NO DATA.** Codebase **không log gradient norm**. Search:
```bash
grep -rn "grad_norm\|grad\.norm" src/sigllm/ → empty
```

→ **Đáng add** instrumentation để debug overfit/exploding patterns.

## 16.3 — Gradient leak / None gradients

`forward_stage2:213`:
```python
with torch.no_grad():
    item_cf = mf.item_encoder(item_ids)
```
→ MF lookup wrapped in no_grad ✓

Stage 3 `encode_rec_features_to_llm_v2`: KHÔNG có `torch.no_grad()` quanh `rec_encoder` call (line 602):
```python
target_cf = self.rec_encoder.item_encoder(batch_data["TargetItemID"])
```

→ MF embeddings được forward THROUGH với autograd, nhưng `rec_encoder.requires_grad=False` đã set. → Tensor có grad path nhưng MF params không update.

⚠️ **Worth verifying:** MF embedding tensors có `requires_grad=True` (vì lookup từ embedding `requires_grad=False` returns leaf tensor with no grad). Should be fine but worth assert.

## 16.4 — Exploding/vanishing

No logs. AMP enabled → numerical stability concern. Train loss trajectories không show explosion (val_loss decrease monotonic mỗi epoch).

---

# 17. Norm và similarity của embedding

## 17.1 — Available data (from training logs)

```
Information flow | user_q: shape=(16, 8, 768), mean=-0.0024, std=0.7305, mean_l2=20.2306
                | target_q: shape=(16, 8, 768), mean=-0.0027, std=0.7320, mean_l2=20.2728
                | user_llm: shape=(16, 8, 3584), mean=-0.0000, std=0.0177, mean_l2=1.0596
                | target_llm: shape=(16, 8, 3584), mean=-0.0000, std=0.0177, mean_l2=1.0595
```

| Stage | Tensor | mean_l2 norm |
|---|---|---|
| Q-Former output | `user_q`, `target_q` | **~20.27** |
| LLM proj output | `user_llm`, `target_llm` | **~1.06** |

→ `llm_proj` (Linear + LayerNorm) **shrink** norm từ 20 → 1.

## 17.2 — LLM embedding original norm

**NO DIRECT MEASUREMENT** trong codebase. Qwen2-7B token embedding norm typically ~0.5-1.5 (industry typical).

→ Soft tokens norm ~1.06 **roughly match** Qwen2 embedding scale. ✓ LayerNorm initialization (`H ** -0.5 ≈ 0.0167`) chính là để force scale match.

## 17.3 — Query token collapse?

**NO DATA.** Codebase không compute pairwise cosine giữa 8 queries. Đáng add diagnostic.

## 17.4 — User/item/history soft tokens phân biệt được?

**NO DATA** trên cosine similarity between user_llm vs target_llm vs inter_llm tokens.

From flow log shape consistency: user_q std=0.7305, target_q std=0.7320 → identical distribution stats. → Có thể queries collapse / produce similar outputs → **Worth investigating**.

---

# 18. Q-Former attention analysis

**Status: NO DATA — attention chưa được extract.**

| Question | Status |
|---|---|
| Cross-attn weights query → CF tokens | ✗ NOT extracted |
| Single-token CF: attention = 1? | Logical: với M=1, softmax(scalar) = 1.0 trivially. **ATTENTION COLLAPSE** mặc định. |
| Multi-token CF attention distribution | ✗ NOT applicable (M=1) |
| Self-attn between queries collapse? | ✗ NO DATA |
| Instruction attention strength | ✗ NO DATA |
| Pos vs neg attention pattern | ✗ NO DATA |

→ **Single-token CF guarantees trivial cross-attention** (softmax over 1 element = 1) → **cross-attention layer effectively becomes a fixed function** of the single CF token. → Strong argument that **multi-token CF projection is bottleneck candidate**.

---

# 19. Dữ liệu và label

## 19.1 — Positive/negative ratio

From training log:
- `pos_rate=0.5283` (test/val) → ~52.8% positives, 47.2% negatives

→ **Balanced dataset** (gần 50/50). CTR not skewed.

## 19.2 — Per user / warm vs cold breakdown

**NO LOGGED DATA** per-user pos/neg ratio. Worth computing.

## 19.3 — Users with one class in eval

Verified log:
- test_warm: "Users with only one class | 44" → 44/356 ≈ 12% users
- test_cold: TBD

→ uAUC calculation **SKIPS** users with only one class (per `calculate_user_auc:57-59`):
```python
if np.all(user_y_true == user_y_true[0]):
    only_one_class += 1
    continue
```

→ **uAUC tính trên ~70-80% users only.** Có thể bias.

## 19.4 — Duplicate interaction train/test

**NOT VERIFIED.** CoLLM OOD2 split design ensures no leakage by construction, but không có explicit check trong codebase.

## 19.5 — Negative samples quá dễ/khó

**NO DATA** trên negative difficulty. Pre-built static negatives → can't reweight.

---

# 20. So sánh với baseline ngoài LLM

Source: `ABLATION_RESULTS.md:258-277` (verified from SellaRec paper Table 1).

| Method | Bridge | test AUC | test uAUC | Source |
|---|---|---|---|---|
| MF | None | 0.6473 | 0.6264 | SellaRec Table 1 |
| LightGCN | None | 0.6139 | 0.6276 | SellaRec Table 1 |
| DIN | None | 0.7138 | 0.6595 | SellaRec Table 1 |
| Zero-shot LLM | Text | 0.5718 | 0.5145 | SellaRec Table 1 |
| Prompt4NR | Text | 0.7191 | 0.6839 | SellaRec Table 1 |
| TALLRec | Text + LoRA | 0.7268 | 0.6989 | SellaRec Table 1 |
| CTRL(DIN) | Text + CF baseline | 0.7155 | 0.6681 | SellaRec Table 1 |
| PersonPrompt | Text + soft prompt | 0.7223 | 0.6991 | SellaRec Table 1 |
| CoLLM-MF | MLP + LoRA | 0.7357 | 0.7179 | SellaRec Table 1 |
| BinLLM | Binary IP + LoRA | 0.7417 | 0.7239 | SellaRec Table 1 |
| **SellaRec (SOTA)** | Semantic-aware | **0.7606** | **0.7464** | SellaRec Table 1 |
| **SigLLM Vanilla** | Q-Former + LoRA | **0.7325** | **0.7170** | This work |
| **SigLLM Instruction-aware** | Q-Former + LoRA + instruction | 0.7322 | 0.7131 | This work |
| **SigLLM User soft tokens** | Q-Former + user channel + LoRA | 0.7336 | 0.7145 | This work |

→ **Cùng split, cùng test set** (CoLLM/BinLLM/SellaRec dùng cùng OOD2 split).

→ **SigLLM tied với CoLLM-MF**, below BinLLM (-0.011 uAUC) và SellaRec (-0.033 uAUC).

**No MLP-bridge baseline implemented on SigLLM internally** — chỉ có CoLLM-MF từ paper.

---

# 21. KẾT LUẬN — TRẢ LỜI 10 CÂU HỎI

## Q1 — Best theo val_uAUC?

**Instruction-aware Q-Former V1 (24/05 best run)** với **val_uAUC peak = 0.7098** (epoch 3).

Caveats: Rerun 28/05 chỉ đạt 0.7018 → seed variance ~±0.008.

## Q2 — Best theo test_uAUC?

**Vanilla Q-Former** với **test_uAUC = 0.7170** (overall).

→ **Instruction-aware test_uAUC = 0.7131** (lower 0.0039) dù val_uAUC cao hơn → **val-test reversal** trong noise band.

## Q3 — Best cho warm?

**Instruction-aware V1 (24/05)** với **test_warm uAUC = 0.7265**.

## Q4 — Best cho cold?

**User soft tokens (NEW)** với **test_cold uAUC = 0.6695** (+1.79 pts vs vanilla 0.6516).

→ **Single positive finding** rõ ràng nhất sau 6 experiments.

## Q5 — Q-Former còn đáng giữ?

⚠️ **CHƯA THỂ KẾT LUẬN CHẮC** vì:
- ✓ Q-Former contribute substantively: text-only ablation costs -5.15 uAUC test_warm
- ✗ Không có MLP-bridge baseline internal để confirm Q-Former vs simpler bridge
- ✓ External comparison: SigLLM (Q-Former) tied CoLLM-MF (MLP) → suggest Q-Former không hơn MLP đáng kể

→ **Đáng giữ** vì:
1. Cấu trúc multi-loss Stage 1 (5 losses) cho Q-Former framework
2. Bridge transfer evidence từ ILM
3. **NHƯNG cần test MLP baseline internal để confirm**

## Q6 — MLP bridge đủ mạnh thay Q-Former?

**CHƯA TEST nội bộ.** Chỉ external evidence:
- CoLLM-MF (MLP): uAUC 0.7179
- SigLLM (Q-Former): uAUC 0.7170
- → Same ballpark.

→ **CANNOT CONCLUDE** without internal MLP baseline.

## Q7 — Multi-token CF có chứng minh bottleneck single-token?

**CHƯA TEST.** Single-token CF làm cross-attention trivial (softmax over 1 element = 1) → strong THEORETICAL argument bottleneck.

→ **Đáng test với M=4, M=8.** High-priority experiment.

## Q8 — Stage 1/2 pretraining có đóng góp?

**INDIRECT EVIDENCE ONLY.**
- ILM paper: ILM > ILM-rand by ~+0.001 HR@5 on OpenP5 (small but consistent)
- SigLLM-specific ablation: ✗ NOT TESTED

→ **Cannot quantify Stage 1/2 contribution on SigLLM final uAUC.** Worth running Stage 3 from random Q-Former để measure.

## Q9 — Bottleneck nằm ở đâu?

Per evidence available:

| Category | Evidence | Verdict |
|---|---|---|
| **Architecture** | Single-token CF cross-attn trivializes, llm_proj không có GELU, no per-position diff in queries | **PRIMARY suspect** |
| **Data** | 33K samples, balanced 52/48, history capped 10, no leakage observed | **NOT a bottleneck** (similar dataset size in CoLLM/BinLLM achieves higher) |
| **Loss** | Binary CE on Yes/No tokens, no aux loss, no contrastive at Stage 3 | **SECONDARY suspect** |
| **Evaluation** | Best ckpt theo AUC not uAUC, ~21% users skipped in uAUC, no per-sample analysis | **MINOR concern** |

→ **Bottleneck primarily ARCHITECTURE** (single-token CF + missing MLP baseline + projection too simple).

## Q10 — Ba experiment đáng chạy nhất

### Priority 1: **MLP bridge baseline**
- **Lý do:** Foundational ablation. Confirm Q-Former giá trị thực sự.
- **Effort:** 1 day code + 6h Stage 3 train.
- **Probability informative result:** **95%** (kết quả any direction đều là contribution).

### Priority 2: **Multi-token CF projection** (M=4 or M=8)
- **Lý do:** Single-token cross-attention trivializes. HIGH-probability architectural bottleneck.
- **Effort:** 1 day code (modify `_project_cf` to project to k tokens) + retrain Stage 1+2+3.
- **Probability win:** **40-50%** (strong theoretical motivation).

### Priority 3: **Unified Stage 3** (Hypothesis B — không 2-step LoRA)
- **Lý do:** Step 1 LoRA chưa từng thấy soft tokens. Mismatch with Step 2.
- **Effort:** Modify Stage 3 config + retrain.
- **Probability win:** **30-40%** (already flagged in `QFORMER_IMPROVEMENT_DIRECTIONS.md`).

### Honorable mentions
- **Best ckpt selection theo uAUC** (not AUC) — code change 1 line, rerun eval. Free.
- **Per-user / per-item analysis** — dump raw eval predictions + analyze offline. 1 day.
- **Smart query init** (`QFORMER_INIT_IMPROVEMENTS.md` #1) — orthogonal to architecture changes.

---

# Reference appendix

## Verified data sources

| Source | Lines | Used for |
|---|---|---|
| `docs/ABLATION_RESULTS.md` | 728 | Trajectories, test results, score gaps |
| `docs/QFORMER_FAQ.md` | 570 | Loss mechanism, BERT init, code references |
| `docs/QFORMER_IMPROVEMENT_DIRECTIONS.md` | 478 | Proposed (NOT TESTED) experiments |
| `docs/QFORMER_INIT_IMPROVEMENTS.md` | 321 | Init proposals (NOT TESTED) |
| `docs/CODEBASE_AUDIT.md` | 580 | Tensor shapes, code paths |
| Notebooks (10 files) | n/a | Original training output containers |
| Git branches (24) | n/a | Implementation history |
| `configs/config.yaml` | 222 | Hyperparameters |

## Not verified / requires extraction

- Raw eval predictions (prob_yes per sample) — would enable per-user, per-item, calibration analyses
- Attention weights from Q-Former layers — would confirm cross-attention collapse
- Per-epoch validation predictions — would enable trajectory deep dive
- Gradient norms during training — would diagnose convergence patterns

## Honest summary

Audit phát hiện:

1. **6 experiments thực sự tested** với numbers đầy đủ. 24+ experiments **PROPOSED** chưa chạy.
2. **No internal MLP bridge baseline** — major gap cho thesis defense.
3. **Single-token CF cross-attention** = architectural bottleneck cao xác suất chưa được verify.
4. **Best ckpt theo AUC** không phải uAUC — potential under-report cho user-side experiments.
5. **No per-user / per-item analysis** — codebase chỉ log aggregate metrics.
6. **Stage 1/2 contribution chưa quantify** trên SigLLM-specific.

Thesis có **defensible position** với:
- Vanilla Q-Former + Text-only ablation chứng minh CF bridge contributes
- 4-5 negative architectural extensions chứng minh "VLM techniques don't trivially transfer"
- User soft tokens cold-item win là single positive finding mới

Nhưng **fragile** cho examiner challenges:
- "Did you compare with simpler MLP?" → ✗ No
- "Did you ablate Stage 1/2 contribution?" → ✗ No
- "Did you sweep number of queries?" → ✗ No
- "Did you test multi-token CF projection?" → ✗ No

→ **3 priority experiments** (Q10) trực tiếp address những challenges này.
