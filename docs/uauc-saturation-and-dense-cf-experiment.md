# uAUC saturation & dense-CF experiment — full log of results

> Self-contained note for handing to another model. Captures the whole
> investigation into why the RecLLM pipeline's **uAUC** is stuck ~0.70–0.72,
> the dense-CF-pretraining experiment (a negative result), and open hypotheses.
> Branch: `feat/dense-cf-pretrain-uauc`. Dataset: ML-1M, CoLLM-style OOD split.

---

## 0. TL;DR

- The **uAUC metric is correct** — `calculate_user_auc` matches CoLLM/BinLLM exactly (unweighted mean of per-user `roc_auc_score`, dropping users with <2 interactions or a single class). Not a bug.
- Pipeline **uAUC is stuck ~0.70–0.72** on the OOD test set across every architecture variant tried. AUC and uAUC are **decoupled**: things that raise AUC leave uAUC flat.
- **Root cause is NOT the CF-teacher data.** We rebuilt the MF teacher on 28× more (leakage-free) history: MF-alone uAUC jumped **0.671 → 0.715** (warm test). But retraining the full pipeline (Stage 1→2→3) on this dense teacher **did not help** — test uAUC went **down** (0.717 → 0.700), AUC flat (0.7325 → 0.7331).
- **A plain MF on full history (uAUC 0.715, AUC 0.774 warm) BEATS the entire 7B-LLM pipeline (0.710 / 0.749 warm)** on both metrics. The LLM adds nothing over a well-fed MF; it slightly subtracts.
- Conclusion: the pipeline's uAUC ceiling is **architectural** (Q-Former compression / injection path), and/or a **data ceiling**. Teacher quality is exhausted as a lever.

---

## 1. Setup & metric

- **Backbone:** Qwen2-7B-Instruct. LoRA r=8 on `q_proj,v_proj`. Q-Former (4 layers, 16 queries, d=768) → linear proj → LLM hidden 3584. CF injection mode `both` (soft-token + CoRA-style weight delta), alpha=4.0.
- **Pipeline stages:** Stage 1 (Q-Former representation: ITC/ITM/ITG + item-item `ii` + user-item `ui` contrastive on MF embeddings) → Stage 2 (generative pretrain) → Stage 3 Step 1 (LoRA on **text-only** prompt, no `<CFTokens>`) → Step 2 (CIE: train Q-Former+proj+CF-injector on full prompt with `<CFTokens>`, LoRA frozen).
- **Split (CoLLM OOD, "ood2"):** temporal. `train_ood2` = months 14–23, `valid_ood2` = 24–28, `test_ood2` = 29–33. Labels: `rating >= 4 → 1` else 0 (no sampled negatives). Sizes: train 33,891 / valid 10,401 / test 7,331.
- **uAUC** (`train_rec_baseline.calculate_user_auc`): per-user `roc_auc_score`, drop users with <2 interactions or single-class, unweighted mean. **Identical to CoLLM** `minigpt4/tasks/rec_base_task.py`.
- **warm/cold** (`preprocess_test_cold_warm.py`): warm = uid AND iid have >3 interactions in `train_ood2`; cold (strict) = uid AND iid both absent from `train_ood2`.

---

## 2. Reference numbers (prior runs, from COMPREHENSIVE_AUDIT.md)

Weak-MF pipeline (MF trained on months 14–23 only):

| Method | test AUC | test uAUC | warm AUC | warm uAUC | cold AUC | cold uAUC |
|---|---|---|---|---|---|---|
| **Vanilla Q-Former** (best baseline) | 0.7325 | **0.7170** | 0.7451 | 0.7213 | 0.7072 | 0.6516 |
| Improvement bundle (CoRA+Qwen2-Instruct+…) | 0.7508 | 0.7041 | — | — | — | — |

Published CoLLM on ML-1M (Vicuna-7B backbone, for reference only — backbone differs):

| Model | AUC | uAUC |
|---|---|---|
| TALLRec (LLM, no CF) | 0.7097 | 0.6818 |
| CoLLM-MF | 0.7295 | 0.6875 |

Observation across ~10 audited variants: val_uAUC always lands 0.69–0.71; raising AUC (e.g. user-conditioned queries val AUC 0.758) leaves uAUC ~0.706. Seed noise on uAUC ≈ 0.01 (documented: same config reran 0.7098 → 0.7018).

**Why AUC↑ but uAUC flat:** uAUC only measures *within-user* ordering and is invariant to any monotone per-user transform. Improvements that strengthen global / cross-user discrimination (popularity, text priors, instruction-following) raise AUC but are orthogonal to (or dilute) within-user ranking.

---

## 3. MF-alone (CF teacher) — the decisive baseline

MF = 2-tower matrix factorization, embedding size 256, BCE on `rating>=4`, selected by **valid uAUC**. Evaluated on **warm** test subset (`warm_or_cold='warm'`).

| MF teacher | train window | valid AUC | valid uAUC | test AUC | **test uAUC** | best epoch |
|---|---|---|---|---|---|---|
| Baseline | months 14–23 (`train_ood2`) | 0.6667 | 0.6368 | 0.7256 | **0.6706** | 341 |
| **Extended (dense)** | months 0–23 (`train_ext_ood2`) | 0.7516 | 0.7096 | 0.7740 | **0.7149** | 29 |

- The dense MF teacher lifts warm-test uAUC by **+0.044** (0.671 → 0.715) and AUC by **+0.048** (0.726 → 0.774), changing **one variable** (CF training data).
- ⚠️ These are **warm-test** numbers. The overall-test MF uAUC (`warm_or_cold=null`) was not re-measured — should be run for a headline-comparable figure.
- Selection was changed to valid-**uAUC** (was valid_auc) in `train_rec_baseline.py` so the teacher is the best per-user ranker.

---

## 4. The dense-CF data lever — diagnostic

The MF was starved: the CoLLM preprocessing (`data_preprocessing.build_ml1m`, `train_slot=range(14,24)`) keeps only the flat tail (months 14–33) and **discards the huge collection "burst" (months 0–13)** — visible in the ratings-over-time histogram as a spike (~290k at month 7). Months 0–13 are temporally **before** the train window ⇒ usable as extra CF pretraining with **no leakage**.

`scripts/diagnose_burst_overlap.py` on the real data:

| Window | interactions | share |
|---|---|---|
| HIST (0–13, discarded burst) | 947,090 | 94.7% |
| TRAIN (14–23) | 33,891 | 3.4% |
| VALID (24–28) | 10,401 | 1.0% |
| TEST (29–33) | 7,331 | 0.7% |

- Burst is **27.9×** the current train pool. **The MF was trained on 3.4% of available interactions.**
- Test users = 320: 82.8% already in train; 17.2% cold, of which **98.2% appear in the burst** (rescuable). Only 1 user + 3 items irreducibly unseen.
- Median interactions per test-user: **20.5 → 316** (×15) when adding burst. +75 test users cross the warm threshold. Strict-cold test interactions 26 → 0.
- Verdict: burst densifies CF history massively and leakage-free.

`scripts/build_ext_mf_data.py` builds `train_ext_ood2.pkl` = months 0–23, remapped through the **existing** `users_map`/`items_map` (drops burst-only entities), so `user_num=839 / item_num=3256` are unchanged and the MF checkpoint stays dimension-compatible with the pipeline.

---

## 5. Full pipeline retrained on the dense teacher — TEST results

Retrained Stage 1 → Stage 2 → Stage 3 Step 2 on `mf_ext_model.pth` (Step 1 LoRA reused: it uses a **text-only** prompt with no `<CFTokens>`, so it is MF-independent — verified). Best checkpoint selected by **val uAUC** (`agg_metrics=uauc`), which landed on epoch 34.

Training sanity: Stage 1 contrastive learned well (train ITC@1 0.07→0.70, II@1 0.03→0.41, UI@1 0.02→0.53; phase transition ~epoch 14) — the dense MF is "eating." Chain loaded correctly (mf_ext → stage1 → stage2 → step1 LoRA). The `_IncompatibleKeys(missing_keys=…)` at load is **expected** (step1 ckpt holds only LoRA; base LLM/MF/Q-Former/proj load from their own sources).

**eval_test.py results (best ckpt):**

| Split | AUC | **uAUC** | vs vanilla uAUC |
|---|---|---|---|
| overall | 0.7331 | **0.7007** | −0.0163 |
| warm | 0.7488 | 0.7099 | −0.0114 |
| cold | 0.7035 | 0.6524 | +0.0008 |

- **uAUC dropped on every split; AUC flat** (overall 0.7331 vs baseline 0.7325).
- During training, **val AUC peaked ~0.759 (epoch 28) but did NOT hold on test** (0.733) — a real val→test gap. So the apparent AUC gain was validation-only.
- Side-by-side, **warm**: dense MF-alone = AUC 0.774 / uAUC 0.715; dense pipeline = AUC 0.749 / uAUC 0.710. **The raw MF beats the full pipeline on both metrics.**

**Verdict:** the dense-CF pipeline is a **negative result** — it improves neither AUC nor uAUC on test. The stronger teacher's per-user signal is *lost* (or diluted) when routed through the LLM pipeline.

### Caveats on the comparison
1. **Selection confound:** the baseline (0.7170) was AUC-selected (old `agg_metrics=auc`); this run was uAUC-selected and landed on epoch 34, a noisy epoch (val AUC dipped to 0.72). BUT no epoch's val uAUC exceeded ~0.711, so no selection would have beaten 0.717.
2. **Seed noise** on uAUC ≈ 0.01; the −0.016 drop is at the edge of "flat vs slightly worse."
3. **Protocol deviation:** dense-CF trains the teacher on months 0–23, not CoLLM's 14–33. Not directly comparable to published CoLLM; report both.
4. **"cold" definition** is still relative to `train_ood2` (14–23); the dense MF now has embeddings for many "cold" users via the burst, so the cold bucket is no longer strictly unseen-by-CF. Document this if reporting cold.

---

## 6. Code / config changes (branch `feat/dense-cf-pretrain-uauc`)

- `qformer_rec_llm.py::_per_user_pairwise_loss` → **user-weighted** aggregation (mean within user, then across users) to align the BPR surrogate with uAUC (was pair-weighted ≈ AUC).
- `rec_base_task.py` → `agg_metrics = uauc` (was `auc`) so pipeline best-ckpt/early-stop track uAUC.
- `config.yaml` → `run.user_grouped_batch.items_per_user: 4→8`; `run.ranking_loss.weight = 0.2, tau = 1.0`.
- `train_rec_baseline.py` → `EarlyStopping ref_metric valid_auc→valid_uauc`; new `run.rec_baseline.train_file` flag.
- `config.yaml::model.rec_config.pretrained_path` → `mf_ext_model.pth` (revert to `mf_model.pth` for the sparse baseline).
- New: `scripts/diagnose_burst_overlap.py`, `scripts/build_ext_mf_data.py`.

---

## 7. Open hypotheses — why does the pipeline SATURATE / DESTROY the MF's per-user signal?

The sharp puzzle: MF-alone = 0.715 uAUC, but pipeline+same-MF = 0.700. The LLM path *degrades* within-user ranking. Candidate mechanisms (with cheap tests):

- **H1 — Text prior homogenizes within-user scores.** Prompt carries `<ItemTitleList>`+`<TargetItemTitle>`; the LLM may rank by "is this movie generally likeable / genre-matches history" (a global text prior, ~constant in effect within a user) instead of user-specific CF → compresses within-user score variance. MF has no text → purely personalized → higher uAUC.
- **H2 — Joint 16-token compression drowns the target signal.** `<CFTokens>` compresses **user+target+history** into 16 tokens. Within a user only the *target* varies; if the 16 tokens are user/history-dominated, the target's marginal contribution gets a tiny budget slice. MF's `dot(user, target)` gives the target its full 256-dim interaction → preserves within-user resolution. Directly explains MF > pipeline.
- **H3 — CF soft tokens under-scaled vs text embeddings.** Info-flow log: `cf_llm std≈0.019, mean_l2≈1.13`. If Qwen2 text-token embeddings have larger norm, attention favors text → CF is a whisper → text (global) drives ranking.
- **H4 — Yes/No head loses resolution → within-user ties.** score = softmax(logit)[Yes]; the LLM clusters probabilities → items within a user get near-equal scores → ties hurt per-user AUC. Observed: within a user, positives 0.5–0.9 and negatives 0.1–0.7 overlap heavily.
- **H5 — Data/statistical ceiling ~0.72.** Only 224–282 users scored (many dropped), few items/user, noisy `rating>=4` label. Both MF (0.715) and pipeline (0.70) sit near a possibly-irreducible ceiling.

---

## 8. Recommended next experiments (cheap → informative)

1. **Soft-token-only ablation (do first).** Eval the pipeline with **blank titles** (only `<CFTokens>`). If uAUC rises to ~0.715 → **text (H1)** is the diluter. If it stays ~0.70 → **Q-Former compression (H2/H3)** destroys within-user signal even for pure CF. One eval-only run splits the two dominant hypotheses.
2. **Late fusion at eval.** `score = α·σ(pipeline_logit) + (1−α)·MF_dot(user,item)`, grid-search α on valid, report on test. Now that dense MF (0.715) ≥ pipeline (0.710) on uAUC, fusion may exceed the 0.717 baseline. No retraining. Strongest lever to *raise* the reported number.
3. **Bootstrap CI on uAUC.** Resample users, get a 95% CI. Tests H5: if CI ≈ ±0.01–0.02, all differences discussed here are within noise / a data ceiling, and the honest story is "saturated at the data limit."
4. (Diagnostic only) Re-measure MF-alone **overall** uAUC (`warm_or_cold=null`) for a headline-comparable teacher number.

## 9. Thesis framing (given a negative result)

- Keep the **weak-MF vanilla pipeline (test uAUC 0.717)** as the headline.
- Present dense-CF as an **ablation + finding**: *the LLM's contribution to uAUC is teacher-dependent — it helps a weak CF (0.671→0.717) but vanishes/reverses once the CF is well-fed (MF 0.715 > pipeline 0.710)*, evidencing that the pipeline's within-user ranking is bounded by the Q-Former injection path (bandwidth bottleneck), not by CF data. This is a defensible, publishable nuance about **when LLM4Rec helps**.
- Report warm/cold uAUC separately; the honest gains are on global AUC and (weak-MF) warm uAUC, not overall uAUC.
