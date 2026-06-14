# Q-Former Instructions Redesign — Task-Focused Variant

**Date:** 2026-05-30
**Branch:** `feat/swap-llm-qwen2` (original SigLLM main path, per-item Q-Former, instruction concat enabled at Stage 3)
**Trigger:** Instruction-aware ablation in `docs/ABLATION_RESULTS.md` showed |Δ uAUC| < 0.005 between instruction-aware and vanilla queries-only configurations — the original 12 paraphrase-style prompts did not give the Q-Former a discriminative task signal.
**Goal:** Test whether replacing paraphrase prompts with **CTR-task-focused prompts** unlocks a measurable instruction-aware contribution.

## Naming convention used in this document

- **Original instructions (V1):** the 12 paraphrase templates that were in place during the failed instruction-aware ablation. All paraphrase the same idea — "represent this movie for recommendation".
- **Task-focused instructions:** the 8 new templates introduced in this experiment, explicitly mentioning the downstream Yes/No CTR task vocabulary.

## Hypothesis

The original V1 instructions failed for two compounding reasons:

1. **Paraphrase-style, no semantic diversity.** All 12 V1 templates were syntactic rewordings of "represent this movie for recommendation". Q-Former queries saw nearly-identical text every batch, so the instruction-aware routing mechanism had no signal to learn — there was nothing to route differently between batches.

2. **Vocabulary collision with pretraining.** Words like *"title and genres"*, *"metadata"*, *"item embedding space"*, *"language-aligned"* matched the Stage 1 alignment builder's `TEMPL_ITEM_TEXT` distribution. The V1 instructions effectively told Q-Former to repeat what it had already learned to do during Stage 1 pretraining, providing no Stage 3-specific task signal.

The redesigned task-focused instructions hypothesise that the failure was about **task framing**, not architecture. If instructions explicitly mention the downstream Yes/No CTR vocabulary — *"user"*, *"predict"*, *"preference"*, *"matching"*, *"binary"*, *"personalized"* — Q-Former queries gain a routing signal that is genuinely new at Stage 3 (because Stage 1/2 pretraining used different vocabulary).

## Task-focused instructions (current code)

Located at `src/sigllm/models/multimodal/qformer_rec_llm.py` lines 69-87. Reduced from 12 paraphrases to 8 templates grouped into three semantic axes:

```python
QFORMER_ITEM_INSTRUCTIONS = [
    # User-item compatibility framing
    "Extract features of this movie for matching against a specific user's viewing history.",
    "Identify aspects of this movie that signal compatibility with user preferences.",
    "Encode this movie's characteristics relevant for predicting individual user enjoyment.",
    # Binary decision framing
    "Represent this movie's features that distinguish positive from negative user responses.",
    "Extract signals predictive of binary user preference for this movie.",
    # Personalization framing
    "Encode aspects of this movie that drive personalized recommendation decisions.",
    "Extract user-specific relevance signals from this movie's content.",
    "Represent this movie's features for individual taste-based preference scoring.",
]
```

## V1 vs task-focused — vocabulary contrast

| Signal | V1 (paraphrase) | Task-focused (new) |
|---|---|---|
| Number of templates | 12 | 8 |
| Mentions "user" | 1 / 12 | 6 / 8 |
| Mentions "predict" / "prediction" | 0 / 12 | 2 / 8 |
| Mentions "preference" | 0 / 12 | 3 / 8 |
| Mentions "binary" / "matching" | 0 / 12 | 2 / 8 |
| Mentions "personalized" | 0 / 12 | 1 / 8 |
| Uses Stage 1 vocabulary (*"title and genres"*, *"metadata"*) | 8 / 12 | 0 / 8 |
| Verb framing | "represent / encode" (pretraining objective) | "extract / identify FOR matching / predicting" (downstream objective) |

## What needs to be rebuilt or retrained

### Dataset — NO rebuild needed

Q-Former instructions live inside the Python class `QRecLLM` and are sampled at forward time during Stage 3 step 2. They are **not** baked into any cached dataset artifact:

- Stage 1 dataset (`build_qformer_dataset.py`) builds item-text pairs and item-item / user-item co-occurrence pairs from MF embeddings + item captions. It uses `TEMPL_ITEM_TEXT` (a separate list inside `qformer_alignment_builder.py`), not `QFORMER_ITEM_INSTRUCTIONS`.
- Stage 3 dataset (`MovieOODDataset`) provides `(uid, iid, label, his, his_title, title)` tuples. No instruction text is stored.
- Instructions are randomly sampled per batch at runtime by `_build_qformer_instructions()` in `qformer_rec_llm.py`.

→ **Reuse the existing Stage 1 alignment dataset and Stage 3 OOD pickles. No rebuild.**

### Stage retraining — only Stage 3 step 2

| Stage | Uses `QFORMER_ITEM_INSTRUCTIONS`? | Action |
|---|---|---|
| 0 — MF baseline | No (does not touch Q-Former) | **Reuse ckpt** |
| 1 — 5-loss pretraining | No (uses `TEMPL_ITEM_TEXT`, a different list in the alignment builder) | **Reuse ckpt** |
| 2 — generative pretraining | No (uses `encode_cf`, queries only, no instruction input) | **Reuse ckpt** |
| 3 step 1 — LoRA tuning | Forward calls `self.qformer(target_cf, ins_list)` but output is **not injected** (text-only prompt has no soft-token slots), so LoRA gradients are unaffected by instruction text | **Reuse ckpt** |
| **3 step 2 — CIE** | **Yes** — `ins_list` flows into Q-Former self-attention alongside queries; output is injected at `<ItemIDList>` / `<TargetItemID>` slots | **Retrain** |

Effort: **~6-7 hours A100** for Stage 3 step 2 (200 epochs max with patience-20 early stop).

## `num_layers` — already correct

Verified `configs/config.yaml` line 60: `num_layers: 4` — matches Stage 1/2 pretraining. No action needed.

## Reproducibility — command to run

```bash
PYTHONPATH=src python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie \
    --cfg-path configs/config.yaml \
    --options run.qformer_stage3_step2.output_dir=/content/SigLLM/ckpt/qformer_stage3_step2_cie_task_focused_qwen2/
```

The custom `output_dir` keeps the task-focused result separate from the V1 Main ckpt. If the result is worse, the V1 Main ckpt at `qformer_stage3_step2_cie_qwen2/` is still intact for direct comparison.

## Baselines for comparison

From `docs/ABLATION_RESULTS.md`:

| Setting | val_uAUC peak | test_warm AUC | test_warm uAUC |
|---|---|---|---|
| Instruction-aware (V1, 12 paraphrases) | 0.7098 | 0.7456 | 0.7265 |
| Vanilla queries-only (no instruction) | 0.7081 | 0.7451 | 0.7213 |
| Text-only (CF soft tokens zeroed) | — | 0.7017 | 0.6750 |

## Outcome interpretation matrix

| Task-focused test_warm uAUC | Verdict | Defensible claim |
|---|---|---|
| > 0.7320 | **Strong win** (Δ ≥ +0.01) | "Task-focused instructions unlock the instruction-aware contribution. Original paraphrase prompts were the bottleneck, not the InstructBLIP design itself." |
| 0.7250 – 0.7320 | **Marginal win** (Δ ≈ +0.005) | "Task framing provides small but consistent improvement. Instruction-aware design works when prompts connect to the downstream task." |
| 0.7150 – 0.7250 | **Inconclusive** (|Δ| < 0.005) | "Prompt redesign did not unlock the instruction-aware contribution. The negative finding from the original ablation is robust across two prompt families. Hypothesised remaining causes: single-task setting, insufficient prompt-output coupling." |
| < 0.7150 | **Regression** (Δ < -0.01) | "Task-focused prompts hurt; pretraining-aligned vocabulary in V1 was actually mildly helpful. Revert to V1." |

## What this experiment tests vs does NOT test

**Tests:**
- Whether *task framing* in instruction text matters for instruction-aware Q-Former on CTR.
- Whether the original ablation's negative result was due to prompt content (testable here) vs architecture (not testable here).

**Does NOT test:**
- Whether *semantic aspect diversity* matters — that would require instructions each focused on a different item aspect (genre / era / mood / etc).
- Whether *multi-task instruction tuning* matters — would need adding auxiliary tasks like rating prediction or explanation generation, a much bigger change.
- Whether *prompt count* matters — 8 vs 12 is held roughly constant; testing 4 vs 16 would be a separate experiment.

## Follow-up plan (after task-focused result)

| Outcome | Next step |
|---|---|
| Strong / marginal win | Update thesis claim: "Instruction-aware Q-Former contribution depends on prompt-task alignment, not just prompt diversity." Update `ABLATION_RESULTS.md` to include the task-focused row. Stop here. |
| Inconclusive | Try a **semantic-aspect-diversity variant** — 8 instructions each focused on a different item aspect (genre / era / audience / mood / popularity / franchise / narrative / critical). If still inconclusive, accept the robust negative finding and pivot the thesis narrative per `ABLATION_RESULTS.md` Pivot 3 (drop instruction-aware as the main claim). |
| Regression | Investigate why the prompts violate Stage 1-pretrained text distribution. Try a **hybrid variant** that keeps some V1 vocabulary while adding task framing. |

## File locations

| Purpose | Path |
|---|---|
| Instruction list (task-focused, current code) | `src/sigllm/models/multimodal/qformer_rec_llm.py` lines 69-87 |
| Stage 3 step 2 training script | `src/sigllm/pipelines/multimodal/train_qformer_stage3_step2_cie.py` |
| Main config | `configs/config.yaml` (line 60: `num_layers: 4`) |
| Existing ablation results | `docs/ABLATION_RESULTS.md` |
| Architecture summary | `docs/PAPER_SUMMARY.md` |
| This document | `docs/QFORMER_INSTRUCTIONS_TASK_FOCUSED.md` |
