# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SigLLM integrates collaborative-filtering signals into LLMs for recommendation. Architecture is inspired by CoLLM, MiniGPT-4, and BLIP-2: a frozen rec encoder (MF) feeds embeddings through a Q-Former into a LLaMA-family LLM. Some scaffolding (runner base, dataset utils) is adapted from Salesforce LAVIS (BSD-3-Clause).

Training is split into three Q-Former stages (BLIP-2 inspired) on top of the MF baseline. Stage 3 follows the CoLLM 2-step recipe:
- **Stage 1 — representation** (`pipelines/multimodal/train_qformer_stage1_representation.py`) — train Q-Former with ITC + ITM + ITG (item-text) and an item-item contrastive on co-watch pairs. Outputs Q-Former weights.
- **Stage 2 — generative** (`pipelines/multimodal/train_qformer_stage2_generative.py`) — load Stage 1 Q-Former, attach a fresh `Linear + LayerNorm` projection into LLaMA's hidden size, and train Q-Former + projection with next-token LM on item-text while keeping the LLM fully frozen. Q-Former runs uni-modal here (queries cross-attend to the CF vector only). Outputs Q-Former + projection weights for Stage 3.
- **Stage 3 Step 1 — LoRA tuning** (`pipelines/multimodal/train_qformer_stage3_step1_lora.py`) — attach a fresh LoRA module to LLaMA and train it alone on Yes/No instruction prompts using a **text-only** prompt template (no `<ItemIDList>` / `<TargetItemID>` placeholders, soft tokens disabled). Q-Former, projection, MF and the base LLM are all frozen. The script applies overrides from `run.qformer_stage3_step1`. Output: `checkpoint_best.pth` under `qformer_stage3_step1_lora/`, which contains LoRA weights only.
- **Stage 3 Step 2 — CIE tuning** (`pipelines/multimodal/train_qformer_stage3_step2_cie.py`) — load Step 1 LoRA into `QRecLLM`, freeze it (along with base LLM and MF), and train Q-Former + projection only on the **full prompt** with soft tokens active. This is the `Ω = ϕ` variant from the paper (mapping-layer only; MF stays frozen). The script applies overrides from `run.qformer_stage3_step2`. Output: `qformer_stage3_step2_cie/checkpoint_best.pth` with Q-Former + projection weights.

A separate baseline pipeline (`pipelines/rec/train_rec_baseline.py`) pretrains the MF rec encoder; its checkpoint feeds all three Q-Former stages.

Stage 1 and Stage 2 both consume the same Q-Former alignment pickles (`{train,valid,test}_qformer_ood2.pkl`). Pkl building is a separate, one-shot step (`build_qformer_dataset.py`, see Commands) — neither training script builds them anymore.

## Commands

There are no CLI scripts in `scripts/` yet — pipelines run as modules. `build_qformer_dataset`, Stage 1, Stage 2, and both Stage 3 steps all take `--cfg-path`; the MF baseline is still driven through notebooks (`notebooks/SigLLM-*.ipynb`).

Run order (full pipeline):

```powershell
# 0. Build Q-Former alignment pickles once per dataset/seed change. Reads
#    `run.qformer_stage1` (seed, item_pair_window, max_item_item_pairs,
#    max_user_item_pairs, include_user_item) + first dataset path, writes
#    {train,valid,test}_qformer_ood2.pkl into that path.
python -m sigllm.pipelines.multimodal.build_qformer_dataset --cfg-path configs/config.yaml

# 1. Stage 1 — representation (ITC + ITM + ITG + item-item)
python -m sigllm.pipelines.multimodal.train_qformer_stage1_representation --cfg-path configs/config.yaml

# 2. Stage 2 — generative pretraining (frozen LLM, item-text LM)
python -m sigllm.pipelines.multimodal.train_qformer_stage2_generative --cfg-path configs/config.yaml

# 3a. Stage 3 Step 1 — LoRA tuning on text-only prompt
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step1_lora --cfg-path configs/config.yaml

# 3b. Stage 3 Step 2 — CIE tuning on full prompt, LoRA loaded frozen from Step 1
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie --cfg-path configs/config.yaml

# Pull base LLM weights once. `pull_model` defaults to Qwen2-7B-Base on this
# branch (`./ckpt/llm/qwen2-7b-base`); pass model_path + save_dir to pull
# Vicuna ("lmsys/vicuna-7b-v1.5", "./ckpt/llm/base") instead.
python -c "from sigllm.pipelines.llm.pull_llm_model import pull_model; pull_model()"
```

Skip step 0 only if the pkls already exist with the right parameters. Stage 1's `main()` used to also build the pkls; it no longer does (see Active work).

The MF baseline is still driven through the notebooks (it reads the same `configs/config.yaml` via `omegaconf`). When porting it to a script, pass the `run.rec_baseline` subtree.

There is no test suite (`tests/` is empty), no lint config, and no Makefile/Justfile. The README claims Ruff + Black at line length 100, but neither is configured in-repo.

## Configuration

Everything is driven by **one** YAML: `configs/config.yaml`. It has three top-level sections that are split apart by `sigllm.common.config.Config`:
- `model` → `cfg.model_cfg` (arch, LoRA, Q-Former, rec_config, ckpt paths)
- `datasets` → `cfg.datasets_cfg` (keyed by builder name, e.g. `movie_ood`)
- `run` → `cfg.run_cfg` (task, optimizer, schedules, plus nested `rec_baseline`, `qformer_stage1` subtrees)

OmegaConf interpolations (`${...}`) are heavily used — e.g. Stage 1 reads `embedding_size` from `model.rec_config.embedding_size`. When changing a value, check whether downstream sections interpolate from it before duplicating.

The config is currently authored with absolute Colab paths (`/content/SigLLM/...`). On Windows/local, override via `--options key=value` or edit before running.

## Registry-based plugin system

`sigllm.common.registry.Registry` maps string names → classes for builders, models, tasks, runners, and lr schedulers. New components are wired in by:
1. Decorating with `@registry.register_<kind>("name")`.
2. Importing the module so the decorator runs (typically via the parent package's `__init__.py`).
3. Referencing the name in YAML (`run.task`, `model.arch`, dataset key, etc.).

`RecPretrainTask.build_*` methods look up classes through the registry — never construct task/model/runner directly in pipeline code.

## Module layout (non-obvious bits)

- `models/multimodal/qformer_rec_llm.py` — `QRecLLM` (registered as `mini_gpt4rec_v2`), the unified Stage 3 model. Inherits `Rec2Base` (in `models/multimodal/base/`).
- `models/q_former/hf_qformer_adapter.py` — current Q-Former; the older `q_former.py` is being phased out (imports are commented out, not deleted).
- `models/projection/qformer_alignment_model.py` — Stage 1 alignment wrapper combining MF, Q-Former, and a BERT text encoder.
- `datasets/qformer/qformer_alignment_*` — Stage 1 / Stage 2 dataset + builder (instruction-conditioned (u, i_pos, i_negs, instruction, item_text) samples).
- `datasets/qformer/qformer_loader.py` — shared DataLoader factory: `qformer_collate`, `build_qformer_loader`, `build_qformer_loaders`. Used by both Stage 1 (full samples) and Stage 2 (passes `filter_fn=_filter_item_text` to keep only `item_text` samples). New stages reading the same pkls should go through this too rather than re-implementing collate.
- `datasets/movie/movie_ood_*` — Stage 3 dataset/builder for the ML-1M OOD splits (`train_ood2.pkl`, `valid_ood2.pkl`, `test_ood2.pkl` under `data/processed/ml-1m/`).
- `pipelines/multimodal/build_qformer_dataset.py` — one-shot script that reads `run.qformer_stage1` + the first dataset path and writes `{train,valid,test}_qformer_ood2.pkl`. Stage 1 and Stage 2 no longer build pickles inside their `main()` — they only load.
- `pipelines/multimodal/train_qformer_stage3_step1_lora.py` and `train_qformer_stage3_step2_cie.py` — the two Stage 3 entry points, mirrored in shape. Each calls `apply_step{N}_overrides(cfg)` which pulls 6 keys (`tuning_step`, `prompt_path`, `ckpt`, `output_dir`, `init_lr`, `max_epoch`) from its own `run.qformer_stage3_step{N}` block before building model + runner. Everything else (warmup, weight decay, batch sizes, etc.) is inherited from the top-level `run.*` defaults. Top-level `model.tuning_step` and `model.ckpt` default to `null`, so an ablation run that bypasses both scripts gets vanilla frozen-LLM behaviour.
- `prompts/qformer_prompt_movie.txt` (full, with `<ItemIDList>` + `<TargetItemID>` soft-token placeholders) vs `prompts/qformer_prompt_movie_text_only.txt` (Step 1, soft-token placeholders removed). The placeholder presence is what triggers the Q-Former forward in `build_llm_inputs_from_prompt_v2`; text-only prompts skip Q-Former entirely.
- `runners/runner_base.py` (LAVIS-derived) → `runner_base_rec.py` (`RecRunnerBase`, registered as `rec_runner_base`) drives Stage 3 training/eval.
- `tasks/base/rec_base_task.py` — the only task base; `train_step` calls `model(samples)["loss"]`, `valid_step` calls `model.generate_for_samples(samples)`.

Anything under `evaluation/` is currently empty — metrics live inline in tasks/runners.

## Active work / gotchas

- InstructBLIP-style Q-Former with `bert-base-uncased` tokenizer is the live setup. User-CF contrastive has been **re-enabled** (`include_user_item: True`, `w_ui: 1.0`, `tau_ui: 0.07` in `qformer_stage1`); comments tagged `TEMP_DISABLED_USER_CF` in source are stale and refer to a brief ablation window — Stage 1 now optimizes ITC + ITM + ITG + item-item + user-item (5 losses).
- Q-Former is **8 layers** (`model.qformer_config.num_layers: 8`), bumped from 4 to match ILM paper config. Stage 1 retrain required when changing this.
- **Pkl build is decoupled from training.** `build_qformer_dataset.py` reads its parameters from `run.qformer_stage1` (seed, `item_pair_window`, `max_item_item_pairs`, `max_user_item_pairs`, `include_user_item`). Re-run it whenever you change any of those — Stage 1 / Stage 2 will silently train on stale pkls otherwise.
- **Stage 3 follows the CoLLM 2-step recipe with LoRA.** Step 1 trains LoRA (r=8, alpha=16, `[q_proj, v_proj]`, dropout=0.05) on the text-only prompt with everything else frozen — paper Equation (4). Step 2 freezes LoRA + LLM + MF and trains Q-Former + projection on the full prompt — paper Equation (5), `Ω = ϕ` variant. The base LLM and MF stay frozen across both steps. Switching `model.lora_config.use_lora` to False reverts to the old frozen-LLM behaviour (kept for ablation comparison only).
- **Diagnosed Stage 3 failure mode (pre-LoRA)**: AUC plateaus around ~0.70 while `pred_pos_rate@0.5` drifts from val's true `pos_rate=0.5283` (e.g., 0.53 → 0.65 over 3 epochs), shrinking the score gap and dragging ACC down. The ablation `ablate_soft_tokens=True` traced this to LLaMA's Yes-prior dominating without CF signal (`pred_pos_rate=0.9997` when soft tokens are zeroed). LoRA in Step 1 is the targeted fix — it lets the LLM relax that prior on the recommendation distribution. If the symptom persists post-LoRA, fall back to `pos_weight` in CE or a post-hoc threshold sweep on val.
- **Branch `feat/swap-llm-qwen2` swaps LLM backbone to Qwen2-7B-Base** (vs Vicuna-7B on main). Motivation: Vicuna's Yes-prior comes from its RLHF/ShareGPT tuning, and Vicuna's training corpus heavily contains ML-1M discussion — both inflate the text-channel ceiling. Qwen2-7B-Base is a *base* (not instruction-tuned) model with a different pretraining mix. **Stage 1 weights are reusable** (Q-Former is LLM-independent); Stage 2 must retrain (hidden_size 4096 → 3584 invalidates the projection layer), Stage 3 Step 1 + Step 2 must retrain. `_init_llm_model` now uses `AutoTokenizer` / `AutoModelForCausalLM` so any HF causal LM works, but variable names stay `llama_*` as legacy. The Vicuna chat wrapper in `prompt_template` is replaced with a bare `'{} Answer:'` completion cue.
- `requirements.txt` and `env.yaml` are now aligned to the Unsloth stack (Python 3.11, torch>=2.4, transformers>=4.46, unsloth>=2025.7). The legacy pins (Python 3.9 / torch 1.12 / transformers 4.38) are gone — torchao>=0.13 (transitive dep of modern transformers) requires Python 3.10+ syntax and won't import on 3.9.
- `eval_configs/` exists but is empty; evaluation runs out of `run.test_splits` in the main config.
- `src/sigllm/pipelines/llm/instruction_tuning_llm.py` is currently 100% commented-out — a stale standalone QLoRA sketch that pre-dated the Q-Former pipeline. The live LoRA integration is inside `QRecLLM._attach_lora` (`models/multimodal/qformer_rec_llm.py`); ignore the standalone file.

## Conventions (from `.github/COPILOT_INSTRUCTIONS.md`)

- English-only across code, configs, docs, prompts, tests, commits.
- No inline code comments unless asked. Public APIs get NumPy-style docstrings.
- Python 3.10+, full type hints. snake_case funcs/vars, PascalCase classes, UPPER_CASE constants.
- New components go behind the registry + a YAML-driven factory; don't hardcode class refs.

## Recent updates (2026-05)

- **Branch `feat/unsloth-qwen2`** — migrated LLM + LoRA to Unsloth's `FastLanguageModel`:
  - `_init_llm_model` loads `unsloth/Qwen2-7B-bnb-4bit` (pre-quantized 4-bit, ~5GB on disk) via `FastLanguageModel.from_pretrained(load_in_4bit=True)` instead of `AutoModelForCausalLM` + `BitsAndBytesConfig`.
  - `_attach_lora` uses `FastLanguageModel.get_peft_model(use_gradient_checkpointing="unsloth", ...)` — drop-in for `peft.get_peft_model` with Triton-accelerated LoRA kernels + ~30% activation memory savings.
  - `UNSLOTH_DISABLE_FAST_GENERATION=1` is set at import time to dodge Unsloth issue #3309 (`inputs_embeds=` crash in fast-generate). SigLLM never calls `model.generate()` in training/eval (forward-only), so the bug doesn't affect the pipeline.
  - Expected Stage 3 wall-time: -30% vs HF + bitsandbytes baseline. Fits 48GB A6000 with ~25 GB headroom.
- **Stage 3 Step 2 policy** in `_apply_tuning_step_policy`: Q-Former + projection are **trainable**, LoRA + base LLM + MF frozen. A brief α2 experiment (freeze Q-Former, only train `llm_proj`) plateaued at val uAUC ~0.694 and was reverted — see the comment block at `qformer_rec_llm.py:_apply_tuning_step_policy(step=2)`.
- **Best results so far (Qwen2-7B + Q-Former 4L + Unsloth's predecessor 4-bit QLoRA)**: val uAUC=0.7098 (epoch 3 peak), test_warm AUC=0.7456 / uAUC=0.7265, test_cold AUC=0.7072 / uAUC=0.6478. Ablation `ablate_soft_tokens=True` confirms Q-Former contributes +0.0515 warm uAUC and 0 cold uAUC. Cold-start is the next bottleneck.
- **Comparison targets** (paper numbers on ML-1M): CoLLM-MF AUC=0.7295 / uAUC=0.6875; BinLLM AUC=0.7425 / uAUC=0.6956; **SellaRec AUC=0.7606 / uAUC=0.7464** (SellaRec uniquely *unfreezes MF* with a Stage 2 alignment loss — ILM/CoLLM/BinLLM all freeze it).
- **Path portability**: `configs/config.yaml` paths use OmegaConf env-var interpolation `${oc.env:SIGLLM_ROOT,/content/SigLLM}` — Colab works with the fallback, Thunder/local override via `export SIGLLM_ROOT=$REPO_DIR`. `src/sigllm/datasets/data_preprocessing.py` and `preprocess_test_cold_warm.py` derive their default dirs from `Path(__file__).resolve().parents[3] / "data" / ...`.
- **Test/warm/cold eval**: `rec_base_dataset_builder.RecBaseDatasetBuilder.build_datasets(evaluate_only=True)` builds `test` (full), `test_warm` and `test_cold` from `test_warm_cold_ood2.pkl` (produced by `preprocess_test_cold_warm.py`). `MovieOODDataset` filters warm/cold via the `subset=` constructor arg. `run.test_splits` lists which to evaluate.
- **Thunder Compute migration**: `notebooks/SigLLM_thundercompute_24_05_2026.ipynb` is the Thunder-targeted notebook (parameterized via `WORKDIR`/`REPO_DIR`/`SIGLLM_ROOT`, no Google Drive); `scripts/thunder_snapshot_and_delete.sh` is a local-machine helper that snapshots a Thunder instance and deletes it after verifying snapshot success. The Colab notebook `notebooks/SigLLM._best_24_05_2026ipynb.ipynb` is preserved unchanged.
- **Q-Former scaled to 8 layers** (was 4). Cascade: rebuilding pkls is NOT required (Stage 1 hyperparams unchanged), but Stage 1 → Stage 2 → Stage 3 all need retrain.
