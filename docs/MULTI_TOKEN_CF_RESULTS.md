# Multi-Token CF — Results

**Date:** 2026-06-07
**Branch:** `feat/multi-token-cf`
**Baseline compared:** `feat/best-0.72-warm` (Vanilla Q-Former, see [ABLATION_RESULTS.md](ABLATION_RESULTS.md))

## Architecture summary

Cross-attention K/V chuyển từ single CF token → 12-token sequence per sample:
- 1 user token = `proj_cf(MF_user) + type_emb[USER]`
- 1 target token = `proj_cf(MF_target) + type_emb[TARGET]`
- 10 history tokens = `proj_cf(MF_history[i]) + type_emb[HISTORY] + pos_emb[i]` (chronological, left-padded; padding masked by `history_mask`)

- Shared `proj_cf: nn.Linear(256, 768)` cho cả 3 type
- `type_emb = nn.Parameter(randn(3, 768))`, `pos_emb = nn.Parameter(randn(10, 768))`
- `num_queries: 8 → 16` (more bandwidth)
- Single Q-Former forward, queries cross-attend full 12-token sequence
- Prompt 1 placeholder duy nhất `<CFTokens>` (thay 3 placeholder cũ `<UserID>/<ItemIDList>/<TargetItemID>`)
- Stage 1/2 trainers chuyển sang dùng helper `pack_item_context` / `pack_user_context` (mask=0 cho slot không có)

## Test results

Best ckpt (Stage 3 Step 2, multi-token CF):

### test (overall, 7331 samples)

| Configuration | AUC | uAUC | val_loss | ACC@0.5 |
|---|---|---|---|---|
| **Baseline (Vanilla Q-Former)** | 0.7325 | **0.7170** | 0.6124 | 0.6675 |
| **Multi-token CF** | **0.7334** | 0.7122 | 0.6040 | 0.6705 |
| **Δ** | **+0.0009** | **−0.0048** | −0.0084 | +0.0030 |

### test_warm (3522 samples)

| Configuration | AUC | uAUC | val_loss | ACC@0.5 |
|---|---|---|---|---|
| **Baseline (Vanilla Q-Former)** | 0.7451 | **0.7213** | 0.6041 | 0.6699 |
| **Multi-token CF** | **0.7455** | 0.7212 | 0.5941 | 0.6842 |
| **Δ** | **+0.0004** | **−0.0001** | −0.0100 | +0.0143 |

### test_cold (3178 samples)

| Configuration | AUC | uAUC | val_loss | ACC@0.5 |
|---|---|---|---|---|
| **Baseline (Vanilla Q-Former)** | 0.7072 | 0.6516 | 0.6245 | 0.6575 |
| **Multi-token CF** | **0.7081** | **0.6705** | 0.6167 | 0.6575 |
| **Δ** | **+0.0009** | **+0.0189** ✅ | −0.0078 | 0.0000 |

## Delta summary

| Split | Δ AUC | Δ uAUC |
|---|---|---|
| test | +0.0009 | −0.0048 |
| test_warm | +0.0004 | −0.0001 |
| test_cold | **+0.0009** | **+0.0189** |

**AUC**: Essentially tied across all 3 splits (+0.0004 to +0.0009 — within noise band ±0.005).
**uAUC**: test_warm tied, test_cold gains +0.019, test loses 0.005.

## Stage 3 Step 2 training trajectory (val split)

| Epoch | val_loss | AUC | uAUC | pred_pos@0.5 | score_gap |
|---|---|---|---|---|---|
| 0 | 0.6167 | 0.7190 | **0.6951** | 0.5308 | 0.1678 |
| 1 | 0.6139 | 0.7192 | 0.6863 | 0.5980 | 0.1642 |
| 2 | 0.6140 | 0.7183 | 0.6883 | 0.6202 | 0.1513 |
| 3 | 0.6073 | 0.7302 | 0.6881 | 0.6337 | 0.1792 |
| 4 | 0.6114 | 0.7310 | 0.6853 | 0.6173 | 0.1990 |
| 5 | **0.6053** | 0.7311 | 0.6916 | 0.6147 | 0.1774 |
| 6 | 0.6190 | 0.7329 | 0.6889 | 0.4662 | 0.1990 |
| 7 | 0.6020 | 0.7364 | 0.6886 | 0.5493 | 0.1902 |
| 8 | 0.6067 | **0.7373** | 0.6887 | 0.5642 | 0.2099 |

**Observation**: val_uAUC plateau quanh 0.685-0.695 từ epoch 1, không vượt 0.70. Pattern AUC↑ uAUC→ trong 8 epoch trông giống popularity bias, **nhưng test set không xác nhận** — val_uAUC vs test_uAUC trên ML-1M có gap ~+0.02-0.03 (baseline cũng vậy: val peak 0.7098 → test_warm 0.7213). val không phải predictor tốt cho test ở setup này.

## Interpretation

### What gained
- **test_cold uAUC +0.019**: explicit user_cf token + type embedding cung cấp user identity signal khi history sparse → cold user benefit nhiều nhất
- **test_cold AUC +0.0009**: gain nhỏ ở AUC, nhưng đáng kể ở uAUC → personalization signal mới (không phải popularity)
- Architectural cleanliness: 1 Q-Former forward thay 3 (target + history flat + optional user)
- val_loss thấp hơn baseline ở cả 3 split (−0.008 đến −0.010)

### What stayed flat
- **test_warm uAUC −0.0001**: essentially identical
- **test_warm AUC +0.0004**: identical
- **test AUC tổng +0.0009**: identical
- → Warm-user performance **không bị regress** bởi multi-token CF

### What lost
- **test uAUC −0.0048**: slight regression on overall test mix. Lý do likely: test mix có cả warm + cold + new combinations; tăng cold dampened by test-distribution effects.

### Mechanism hypothesis
Baseline `feat/best-0.72-warm` đã disable `<UserID>` placeholder (TEMP_DISABLED_USER_CF) — không có user soft token. Multi-token CF tái-thêm user signal nhưng theo cách kiến trúc hơn (typed token trong cross-attn K/V thay vì soft token prepended). Cold-start users hưởng lợi rõ vì lúc đó history nghèo nàn nên user_cf là tín hiệu chính khả dụng.

## Verdict

- **Multi-token CF không phải universal win**: warm tied, test tổng nhỏ regression, cold rõ ràng win.
- **Defensible thesis claim**: *"Multi-token CF with typed user/target/history encoder tokens specifically improves cold-start uAUC (+0.019) while maintaining warm performance (Δ ≈ 0). The explicit user-identity token in cross-attention K/V provides signal when history is sparse."*
- **AUC story**: tied across all splits (+0.0004 to +0.0009 — within noise). Multi-token KHÔNG cải thiện population-level AUC measurably.

Vẫn defensible cho thesis vì cold-start gain là real signal, nhưng story bị giới hạn ở cold-start chứ không phải "universal architectural improvement". 

## Configuration snapshot

```yaml
# configs/config.yaml (multi-token CF)
qformer_config:
  num_queries: 16          # was 8 in baseline
  num_heads: 8
  num_layers: 4
  max_history_length: 10   # new

model.prompt_path: prompts/qformer_prompt_movie_mt.txt   # single <CFTokens> placeholder
qformer_stage3_step2.prompt_path: prompts/qformer_prompt_movie_mt.txt
qformer_stage2.epoch: 50                                  # was 20
qformer_stage2.early_stopping_patience: 8                 # was 5
```

## Files changed (this branch)

- `src/sigllm/models/q_former/hf_qformer_adapter.py`: multi-token encoder, type/pos embeddings, `pack_item_context`/`pack_user_context` helpers
- `src/sigllm/models/multimodal/qformer_rec_llm.py`: single Q-Former call, `<CFTokens>` placeholder
- `src/sigllm/models/projection/qformer_alignment_model.py`: Stage 1 losses adapted to multi-token (mask=0 for absent slots)
- `src/sigllm/pipelines/multimodal/train_qformer_stage2_generative.py`: same adaptation for Stage 2
- `configs/config.yaml`: num_queries 16, Stage 2 epochs 50
- `prompts/qformer_prompt_movie_mt.txt`: new prompt with single `<CFTokens>` block
