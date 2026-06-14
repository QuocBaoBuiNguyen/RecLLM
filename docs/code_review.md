# SigLLM Code Review

**Scope:** every Python file under `src/sigllm/`, `configs/config.yaml`, `prompts/`, top-level metadata files (`README.md`, `env.yaml`, `requirements.txt`, `.github/COPILOT_INSTRUCTIONS.md`, `docs/ROADMAP.md`).

**Branch reviewed:** `feat/opt-loss-upgrade-to-instructblip-model-using-bert-base-tokenizer`.

The codebase is a CoLLM/MiniGPT-4-style two-stage pipeline (Stage 1 trains a Q-Former with rec↔text alignment; Stage 2 fine-tunes an LLM with rec embeddings projected into prompt slots). The architecture is sound, but the implementation has several latent runtime bugs (the distributed path, the eval-only path, and early stopping all break in observable ways), large amounts of dead/commented code from the LAVIS port, and an English-only convention that is silently violated. Below is a prioritized punch list.

---

## P0 — Bugs that will (or do) blow up at runtime

### 1. `from torch import dist` actually imports `torch.dist`, not `torch.distributed`
**File:** `src/sigllm/datasets/base/rec_base_dataset_builder.py:10`

`torch.dist` is the built-in p-norm distance function (a callable on the `torch` module), **not** `torch.distributed`. The subsequent `dist.barrier()` call at line 35 will raise `AttributeError` the moment `is_dist_avail_and_initialized()` returns `True`.

`configs/config.yaml` has `run.distributed: True`, so any multi-process launch hits this immediately.

**Action:** replace with `import torch.distributed as dist` (or use the project's own `sigllm.common.dist_utils`).

### 2. `RecBaseDataset.__init__` is broken and never called
**File:** `src/sigllm/datasets/base/rec_base_dataset.py:8-23`

- `dataset_config.build_info.storage / filename` — `storage` is a YAML string (`"/content/SigLLM/data/processed/ml-1m/"`), strings have no `/` operator.
- `ann_path.exists()` and `ann_path.with_suffix(".pkl")` — strings have no such methods.
- Never calls `super().__init__()`.

In practice `MovieOODDataset.__init__` shadows it entirely (it correctly wraps `Path(...)` first), so the bug is hidden. Either fix the base class to be the canonical Path-based implementation that subclasses can call via `super()`, or delete the body and keep only the abstract contract.

**Action:** rewrite `__init__` against `Path(...)` and remove the duplicated logic from `MovieOODDataset`.

### 3. `RunnerBase.train()` early-stop counter increments on every epoch
**File:** `src/sigllm/runners/runner_base.py:257-270, 287-289`

```python
if agg_metrics > best_agg_metric and split_name == "valid":
    ...
    not_change = 0

val_log.update({"best_epoch": best_epoch})
self.log_stats(val_log, split_name)
not_change += 1   # <-- runs in BOTH branches
```
`not_change` is reset to 0 on improvement and then immediately incremented to 1 on the same epoch. So `not_change` grows monotonically regardless of best-epoch progress, and the `if not_change > 20` guard at line 287 will trip after at most 21 evaluations even if validation keeps improving.

**Action:** indent `not_change += 1` into the `else` branch, or use the dedicated `EarlyStopping` helper already present in `sigllm.common.early_stopping`.

### 4. `RunnerBase.train()` references `cur_epoch` after the training loop never ran
**File:** `src/sigllm/runners/runner_base.py:280-296`

When `evaluate_only=True`, the outer `if not self.evaluate_only:` block is skipped entirely, so `cur_epoch` is never bound. The post-loop block then does:

```python
test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch
```

`NameError` whenever `valid_splits` is empty in eval-only mode.

**Action:** initialize `cur_epoch = self.start_epoch` before the conditional, or guard with `cur_epoch = "best"`.

### 5. `RecBaseTask.evaluate()` — wrong indentation, wrong return container
**File:** `src/sigllm/tasks/base/rec_base_task.py:152-210`

- `all_results = []` initialized as a list, then inside the loop reassigned to a dict — the list initialization is dead.
- The dict assignment `all_results = { ... }` is **inside** `for data_loader in data_loader.loaders:`, so only the **last** loader's metrics survive; multi-loader semantics are silently lost.
- `return all_results` is outside the loop and returns the dict for whichever loader iterated last.

Combined with #6 below, the eval path is fragile.

**Action:** decide what evaluation over multiple loaders should mean, then either iterate one loader (drop `.loaders`) or accumulate.

### 6. `RecBaseTask.evaluate()` assumes `data_loader.loaders`
**File:** `src/sigllm/tasks/base/rec_base_task.py:158`

```python
for data_loader in data_loader.loaders:
```
`RunnerBase.eval_epoch` passes the dataloader from `self.dataloaders[split_name]`, which `dataloader_builder._create_loader` returns as a `PrefetchLoader` (or `IterLoader` for train). Neither has a `.loaders` attribute — only `MultiIterLoader` does, and that is only built when the dataset is a list/tuple of datasets, which is not the current configuration.

This is the "is the eval path ever exercised?" smell — if it ran today, it would `AttributeError`.

**Action:** drop the outer `for data_loader in data_loader.loaders:` loop and iterate the loader directly. Add a unit test for a single-dataset eval pass.

### 7. `BaseModel.load_checkpoint_from_config` — assert is a no-op
**File:** `src/sigllm/models/multimodal/base/base_model.py:91-93`

```python
pretrain_path = cfg.get("pretrained", None)
assert "Found load_finetuned is False, but pretrain_path is None."
```
The assert tests truthiness of a string literal — always `True`. The intended check `assert pretrain_path is not None, "..."` is missing. Loading "pretrained" mode with a `None` path silently passes through and then `load_from_pretrained(None)` crashes with a less-clear message.

**Action:** replace with `assert pretrain_path is not None, "..."`.

### 8. `from html import parser` in Stage 2 trainer
**File:** `src/sigllm/pipelines/multimodal/train_qformer_stage2_llm_alignment.py:2`

Imported by accident (likely IDE auto-fill collision with `parser = argparse.ArgumentParser(...)`), unused. Harmless but indicative.

**Action:** delete.

### 9. `MetricLogger_v2.update` calls `extend(value * n)` on scalars
**File:** `src/sigllm/common/logger.py:210`

`SmoothedValue_v2.total` is initialized as a list, but `total.extend(value * n)` only works if `value` is iterable. Scalar update would raise `TypeError`. This class isn't used yet in production paths, but it's a footgun if someone reaches for it.

**Action:** delete `SmoothedValue_v2`/`MetricLogger_auc` (the v1 versions are sufficient and correct), or finish them.

---

## P1 — Design / correctness concerns to address before next training run

### 10. Stage 2 prompt is fed as Q-Former instruction
**File:** `src/sigllm/models/multimodal/qformer_rec_llm.py:444-454, 717`

In `build_llm_inputs_from_prompt_v2`, `instruction_list = [prompt_template] * batch_size`. `prompt_template` is the **full Stage 2 question with `<ItemIDList>`/`<TargetItemID>` placeholders embedded**. That string is then tokenized by the Q-Former's BERT tokenizer and used as Q-Former conditioning. So the Q-Former sees literal `<ItemIDList>`/`<TargetItemID>` as text (which BERT will split into pieces), instead of the short instructions the Stage 1 alignment was trained on (`"Represent this movie for recommendation..."`).

This likely degrades the alignment Stage 1 worked to learn. Compare with Stage 1, which feeds `instruction` strings from the alignment dataset.

**Action:** decide what the Stage 2 instruction should be (a short fixed template, the row's own instruction, or sample from `TEMPL_ITEM_TEXT`), and pass that through `instruction_list` instead of the verbose prompt body.

### 11. `RecPretrainTask.train_epoch` divides loss but keeps backward outside the accum guard
**File:** `src/sigllm/tasks/base/rec_base_task.py:122-139`

`loss = loss / accum_grad_iters` is done before backward, but `loss.backward()` runs every step (correct). However `optimizer.zero_grad()` is also inside the `if (step + 1) % accum_grad_iters == 0:` block — so gradients from the previous accumulation are not zeroed before the next sub-step accumulates again. Looks correct on paper (zeroing happens after the optimizer step, before the new accumulation window starts), but only by accident: if the loop exits mid-window the optimizer never sees those last gradients. Acceptable; flag for later.

Also: `metric_logger.update(loss=loss.item() * accum_grad_iters)` — multiplying back is fine for display, but the metric still smooths post-divided values for `.global_avg`. Verify display intent.

**Action:** add a comment about accumulation semantics or use `torch.optim.Optimizer.step()` with explicit gradient checkpointing helpers.

### 12. `QRecLLM` triple-loads MF weights
**File:** `src/sigllm/models/multimodal/qformer_rec_llm.py:128-143, 824-831`

1. `_init_rec_model` loads `pretrained_rec` via `load_state_dict` (strict by default).
2. `from_config` then loads the QRecLLM checkpoint, which contains `rec_encoder.*` keys, overwriting (1).
3. If `os.path.exists(rec_config['pretrained_path'])` and `freeze_rec`, it loads MF weights *again*, overwriting (2).

This is wasteful and confusing — and step 3 could overwrite a fine-tuned rec encoder embedded in a saved QRecLLM checkpoint. Decide a single source of truth.

**Action:** load MF only once based on a clear rule: "if the QRecLLM ckpt has rec_encoder weights, trust those; otherwise fall back to the standalone MF file."

### 13. `_init_lora` is fully commented out but `execute_llm_forward` still references `self.llama_model_lora`
**File:** `src/sigllm/models/multimodal/qformer_rec_llm.py:162-181, 659`

```python
def execute_llm_forward(self, embeds, atts, targets):
    with self.maybe_autocast():
        model = self.llama_model_lora if self.use_lora else self.llama_model
```
`use_lora` defaults `False`, so today this never trips, but if a config sets `use_lora: True` (the YAML does have a `lora_config.use_lora` switch), `AttributeError: 'QRecLLM' object has no attribute 'llama_model_lora'`.

**Action:** either restore `_init_lora` (the README/ROADMAP both promise LoRA), or remove the `use_lora` branch and the `lora_config` from the YAML so the surface area matches what's actually wired.

### 14. Optimizer param-group split is too loose
**File:** `src/sigllm/runners/utils/optimizer_builder.py:13`

```python
if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
    p_non_wd.append(p)
```
`"ln"` and `"bn"` are substring-matched anywhere in the parameter name. Anything containing the bigram (e.g., `qformer.embeddings.position_embeddings`, `proj_align_n.weight` — not real here, but plausible) gets misclassified into the no-weight-decay group.

**Action:** match on path components: `n.endswith(".bias") or any(p in n.split(".") for p in ("ln", "bn", "norm"))`.

### 15. `RecRunnerBase.eval_epoch_pre` is dead code
**File:** `src/sigllm/runners/runner_base_rec.py:36-70`

Identical body to `eval_epoch` minus the leading `self.model.eval()`. Never called.

**Action:** delete.

### 16. `tasks/__init__.py` exports `BaseTask` that doesn't exist
**File:** `src/sigllm/tasks/__init__.py:16`

```python
__all__ = ["BaseTask", "RecPretrainTask", "setup_task"]
```
`BaseTask` is not imported (the actual class is `RecBaseTask` from `tasks.base.rec_base_task`). `from sigllm.tasks import *` will fail.

**Action:** remove `"BaseTask"` from `__all__` or import the real `RecBaseTask` and alias if a public name is desired.

### 17. `dist_utils.main_process` decorator silently swallows non-rank-0 results
**File:** `src/sigllm/common/dist_utils.py:29-36`

`@main_process` returns `None` for non-rank-0 callers. Used on `_save_checkpoint`/`log_stats` (fine, side-effect-only). But anyone reaching for it on a function that returns a value will get `None` on workers — easy to miss in code review.

**Action:** rename to `@main_process_only` or add a docstring.

### 18. `qformer_rec_llm._init_qformer` calls `.to(self.device)` before all sub-modules exist
**File:** `src/sigllm/models/multimodal/qformer_rec_llm.py:215`

`self.device` is `list(self.parameters())[0].device`, which depends on whatever sub-module was created first. `_init_rec_model` runs first, so `self.device` is the rec_encoder's device — but `_init_llm_model` uses `device_map="auto"` which can be CUDA (or split if multi-GPU). In a single-GPU Colab they line up, but the contract is fragile.

**Action:** in `from_config` or `__init__` end, do an explicit `model.to(target_device)` at the end and stop relying on `self.device` mid-construction.

### 19. `QFormerAlignmentDataset` uses `weights_only=False`
**File:** `src/sigllm/datasets/qformer/qformer_alignment_dataset.py:7`

PyTorch ≥2.4 emits a security warning by default; explicit `weights_only=False` is correct only because the file is built locally by `QFormerAlignmentBuilder` and trusted. Worth a comment so future maintainers don't toggle it.

**Action:** add a short comment, or save a plain `pickle` and load with explicit unpickler.

---

## P2 — Dead or duplicated code

### 20. `models/q_former/q_former.py` (entire file)
The hand-rolled `QFormer` is unused — every importer is commented out, replaced by `HFQFormerAdapter`.

**Action:** delete the file and the commented imports in `qformer_rec_llm.py`, `train_qformer_stage1_representation.py`.

### 21. `pipelines/llm/instruction_tuning_llm.py` (entire file)
100% commented-out boilerplate.

**Action:** delete; bring it back from git history if/when the standalone instruction-tuning pipeline is implemented.

### 22. `common/logger.py` duplicates `SmoothedValue` / `MetricLogger`
The `_v2` / `_auc` variants are an in-progress AUC logger that appears unused (RecBaseTask uses sklearn AUC directly) and contains the bug from #9.

**Action:** delete the `_v2`/`_auc` classes or finish them. The clean v1 versions cover current needs.

### 23. `set_seed`, `disabled_train`, `log_step` redefined per file
- `set_seed`: `pipelines/rec/train_rec_baseline.py:78`, `pipelines/multimodal/train_qformer_stage1_representation.py:31`, `pipelines/multimodal/train_qformer_stage2_llm_alignment.py:33`.
- `disabled_train`: same three plus `models/multimodal/qformer_rec_llm.py:26` and `models/multimodal/base/rec_base_model.py` (implied).
- `log_step`: ~10 different files, each with the same body.

**Action:** move all three to `sigllm.common.utils` (or `common/logging_utils.py` for `log_step`) and import.

### 24. Star imports everywhere
`from sigllm.common.dist_utils import *` in `runner_base.py`, `runner_base_rec.py`, `rec_base_task.py`. Pollutes namespace, hides where `download_cached_file`/`is_dist_avail_and_initialized` come from.

**Action:** use explicit imports.

---

## P3 — Conventions, hygiene, repo state

### 25. English-only convention is silently violated
The Copilot instructions and CLAUDE.md both mandate English everywhere.

Found Vietnamese in:
- `qformer_rec_llm.py:198, 217, 220, 231` (`# 1) init qformer kiến trúc giống stage1`, `# load checkpoint stage1`, `# nếu bạn save thẳng state_dict ...`).
- `rec_base_task.py:124` (`# Chia loss để hỗ trợ Gradient Accumulation`).
- `notes.txt` is mostly Vietnamese — fine as a personal notebook but it's checked in at repo root.

**Action:** translate the Vietnamese comments. Move `notes.txt` to `docs/notes-personal.md`, or add to `.gitignore`.

### 26. `prompts/qformer_prompt_movie.txt` has literal `\n`
Line 1: `... Answer with "Yes" or "No". \n#Answer:`

The file is read with `open().read().splitlines()` and the lines are formatted with `prompt_template.format(p)` where the template is `"{}"`. The `\n` is **not** a newline — it survives as the two characters `\` and `n` and ends up in the LLaMA prompt as literal text.

**Action:** decide intent. If you want a newline, write the file with an actual newline (or post-process with `.replace("\\n", "\n")`); if you want the literal escape sequence (unusual), document it.

### 27. `eval_configs/` is empty
README and architecture documents reference it. Either populate it or remove the references.

### 28. `tests/` is empty
README claims pytest; nothing exists. Several of the bugs above (#1, #3, #4, #5, #6, #7) would have been caught by a basic smoke test that imports the module and runs `RunnerBase` for one epoch on a tiny synthetic dataset.

**Action:** add at minimum:
- `tests/test_imports.py` — import every public module.
- `tests/test_runner_smoke.py` — instantiate `RecRunnerBase` against a 4-sample fake dataset on CPU; run one epoch with `evaluate=False`, then one with `evaluate=True`.
- `tests/test_distributed_path.py` — at least exercise `RecBaseDatasetBuilder.build_datasets()` so #1 surfaces.

### 29. `requirements.txt` is incomplete; `env.yaml` pins incompatible versions
`requirements.txt`:
```
omegaconf==2.3.0
torch
scikit-learn
webdataset
timm
transformers
sentencepiece
peft
```
Missing: `pandas`, `numpy`, `rich`, `bitsandbytes` (used in commented LoRA path), `accelerate` (needed by HF for `device_map="auto"`).

`env.yaml` pins `transformers==4.28.0`, but `transformers.InstructBlipQFormerConfig` / `InstructBlipQFormerModel` were added in 4.31. So `env.yaml` is stale relative to the current Q-Former adapter.

**Action:** add the missing dependencies to `requirements.txt` and either delete `env.yaml` or bump the pins to a tested set (`transformers>=4.36`, `accelerate>=0.25`).

### 30. Hardcoded Colab paths in config
`/content/SigLLM/...` everywhere in `configs/config.yaml` (model paths, prompt path, data paths). Combined with no override pattern documented in the README, anyone running outside Colab will hit "file not found" before their first epoch.

**Action:** make paths relative to a single `${oc.env:SIGLLM_ROOT,/content/SigLLM}` interpolation, or document the `--options` override pattern explicitly in the README.

### 31. `notes.txt` and notebooks at repo root
`notebooks/SigLLM-2-5-2026.ipynb` and `notebooks/SigLLM-6-5-2026.ipynb` are dated work artifacts. `notes.txt` is a thinking-out-loud doc.

**Action:** decide whether to keep them in version control. If yes, move under `docs/journal/` and add a short header to each so a stranger can tell which is current.

---

## P4 — Smaller items / tidy-ups

- `runner_base.py:294` — bare `print("training finish or just evaluation...")` mid-method; replace with `logging`.
- `data_utils.py:60-90` — `ChainDataset` uses `import logging` then `logging.info` inside a tight loop; once per dataset it's fine.
- `data_utils.py:18` — `[_apply(x) for x in x]` shadows the outer `x` in the comprehension. Rename to `_apply(item) for item in x`.
- `early_stopping.py` — `is_improved` is computed but `improvement` flag isn't returned for callers that want both "improved + best snapshot"; current pipelines hand-track `is_improved`. Consider returning a richer record.
- `rec_base_task.py:228` — `samples['UserID'].detach()` assumes `UserID` is always a tensor in eval samples. The dataset returns `int(user_id)` as a Python int (from `row["UserID"]` in `MovieOODDataset.__getitem__`), so collation produces a tensor only because PyTorch's default collate stacks ints into a tensor. Works today; brittle if dataset changes.
- `MovieOODDataset` indents with **tabs** (line 23 onward); every other file uses 4-space indents. Pick one.
- `models/q_former/text_encoder.py` — encodes inside `forward` by tokenizing on every call. With `freeze_text_encoder=True` (current config) this still re-tokenizes per batch. For Stage 1's small `TEMPL_*` set you could cache token IDs.
- `pipelines/multimodal/train_qformer_stage1_representation.py` rebuilds `train_qformer_ood2.pkl`/`valid_*`/`test_*` on every `main()` invocation. Add an "if exists, skip" guard or a `--rebuild` flag.
- `qformer_rec_llm._sample_prompt` weights — `[5] * (len(prompt_list) - 1) + [1]` puts low weight on the last prompt. With the current 3 active prompts (others marked `# DISABLED_USER_CF`), the third prompt is almost never sampled. Probably accidental.
- `models/multimodal/base/base_model.py:198` — `register_path` already enforces uniqueness with `assert isinstance(path, str), "All path must be str."` then the `KeyError` on duplicate. Anyone re-running training in the same Python session (e.g. a notebook cell) will hit `KeyError "library_root"` from `__init__.py:7`. Add a "register-or-overwrite" path, or skip if already set.
- `runner_base.py:412` — the bare-except path on `RuntimeError` falls through to `strict=False` reload, which is a reasonable recovery, but the warning message is multi-line indented and reads oddly in logs.

---

## Suggested order of execution

1. **Smoke test the runtime path** (tests in #28) — this will surface #1, #3, #4, #5, #6 immediately on a tiny CPU fake.
2. **Fix the P0 list (#1–#9)** — none of them are large changes; #2 and #3 are 1-line fixes.
3. **Decide on the LoRA situation (#13)** — the codebase claims to support LoRA in three places; either wire it up or drop it from configs/docs.
4. **Resolve the Stage 2 instruction question (#10)** — this might be a real model-quality issue, not just a code smell.
5. **Cleanup pass (P2 + P3 #25, #26)** — drop dead files (`q_former.py`, `instruction_tuning_llm.py`, `MetricLogger_v2`), translate the Vietnamese comments, fix the prompt file's literal `\n`.
6. **Repo hygiene (#27, #28, #29, #30, #31)** — flesh out tests, requirements, eval_configs, paths.

Items in P4 are quality-of-life — do them as you touch the surrounding code.
