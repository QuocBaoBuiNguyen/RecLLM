# Stage 2 uAUC Bottleneck — Likely Causes

**Symptom:** Stage 2 (Q-Former + frozen LLaMA + soft tokens) plateaus at uAUC ≈ 0.68.

uAUC = 0.68 on a frozen-LLaMA + soft-tokens-only setup is exactly the ceiling that regime tends to hit. Suspects in priority order:

## 1. LoRA is OFF — there are no LLM-side trainable params

**File:** `src/sigllm/models/multimodal/qformer_rec_llm.py:162-181, 659`

`_init_lora` is fully commented out, `lora_config.use_lora: False` in YAML, all `llama_model` params get `requires_grad=False`. The only things that train are `qformer` and `llama_proj`. So Stage 2 is **pure soft-prompt tuning** of a frozen 3B LLM — and the literature is consistent: this regime caps around 0.65–0.70 uAUC on rec tasks while LoRA adds 5–10 points. The 0.68 number sits exactly on that ceiling.

**Fix:** uncomment `_init_lora`, set `use_lora: True` in YAML, target `["q_proj", "v_proj", "k_proj", "o_proj"]` (not just q/v). This is the single biggest knob.

## 2. The Q-Former gets the full Stage 2 prompt as its "instruction"

**Files:** `qformer_rec_llm.py:717` and `hf_qformer_adapter.py:122-129`

```python
instruction_list = [prompt_template] * batch_size  # the FULL prompt, ~150+ tokens
```

This goes to `_tokenize_instruction` with `max_length=48`. So the Q-Former always sees the same 48-token truncated prefix (`"#Question: A user has given high ratings to the following movies: <ItemTitleList>. The interaction-history signals..."`) — **regardless of which user or item is in the batch**.

Result: the Q-Former's text-conditioning channel is essentially constant across the batch. The model can only differentiate users/items via `target_cf` and `inter_cf` rec embeddings, throwing away most of what Stage 1's instruction-conditioning was supposed to provide. This is also a hard distribution mismatch with Stage 1, which trained on short instructions like *"Represent this movie for recommendation using its title and genres."*.

**Fix:** pass a short, item-specific instruction to the Q-Former. Either:
- Sample one of `QFormerAlignmentBuilder.TEMPL_ITEM_TEXT` per batch, or
- Use a single fixed short string (`"Represent this movie for recommendation using its title and genres."`) for all items.

The Stage 2 *prompt* and the Q-Former *instruction* do not have to be the same string.

## 3. Soft-token embeddings probably don't match LLaMA's input embedding distribution

**File:** `qformer_rec_llm.py:287-292`

```python
self.llama_proj = nn.Sequential(
    nn.LayerNorm(d_q),    # at the INPUT
    nn.Linear(d_q, hidden),
    nn.GELU(),
    nn.Linear(hidden, H),  # output unnormalized
)
```

LayerNorm is at the input; the output of the second Linear has no normalization. LLaMA's frozen `embed_tokens.weight.std()` is ~0.02; freshly initialized `Linear(hidden, H)` produces vectors with much larger magnitude. Frozen LLaMA's attention has never seen vectors like this, so it routes them poorly. The classic prefix-tuning fix is to add an output LayerNorm (or scale to match `embed_tokens` statistics).

**Fix:** add `nn.LayerNorm(H)` at the end of `llama_proj`, and optionally initialize the final Linear's weights with std matched to the LLM embeddings (~0.02).

## 4. The text titles in the prompt overwhelm the soft tokens

The current prompt:

```
A user has given high ratings to the following movies: <ItemTitleList>. The interaction-history signals are: <ItemIDList>. ... titled <TargetItemTitle>. <TargetItemID> Answer with "Yes" or "No".
```

`<ItemTitleList>` is replaced with the actual movie titles ("Toy Story", "Aladdin", ...) — the LLM already has all the recommendation signal it needs from text alone. The soft tokens (`<ItemIDList>` and `<TargetItemID>`) are competing with explicit textual titles for the LLM's attention. With LLM frozen, the soft tokens essentially have to "outshine" the text — they don't.

**Fix (one of):**
- Drop `<ItemTitleList>` and `<TargetItemTitle>` from the prompt entirely so soft tokens carry the load.
- Or keep titles but enable LoRA (the LLM can learn to fuse both).

## 5. The early-stop counter bug

**File:** `runner_base.py:270, 287-289`

`not_change += 1` runs on every epoch, including epochs where validation improves (it sits outside the `if improved` block). So at epoch 21 the run stops regardless of whether validation is still climbing. If best-val epoch was 6 and the curve was still moving slowly upward, training stopped before the model saturated.

**Fix:** indent `not_change += 1` into an `else:` after the improvement block — or just use the `EarlyStopping` class already in `src/sigllm/common/early_stopping.py`.

## 6. Stage 1 transferred objective ≠ Stage 2 objective

Stage 1 trained Q-Former for **item-text contrastive alignment** (BERT-space). Stage 2 needs Q-Former outputs that bias a **frozen LLaMA** toward Yes/No. These are different geometries. The Q-Former is being asked to do double duty:

1. Produce BERT-aligned embeddings (Stage 1).
2. Produce LLaMA-prompt-friendly embeddings (Stage 2).

The `llama_proj` MLP has to bridge that gap, with only Yes/No CE supervision. That's a thin signal.

**Fix:** unfreeze the Q-Former in Stage 2 (already done — `freeze_qformer=False` in `_init_qformer`) but also lower its learning rate (currently shared `init_lr: 1e-4` for everything). Try `lr_qformer = 1e-5`, `lr_proj = 1e-4`, `lr_lora = 1e-4` — needs a param-group split in the optimizer builder.

## 7. The MF baseline ceiling

If the MF-only baseline (pretrained `mf_model.pth`) gives uAUC ≈ 0.65–0.67, then 0.68 means the entire LLM stack adds ~0.01–0.03 of lift. **Run the MF baseline alone first** (`pipelines/rec/train_rec_baseline.py`) and look at its test uAUC.

- If MF alone is at 0.66, headroom is small without architectural changes (LoRA #1, prompt #2/#4 above).
- If MF is at 0.55 and you're at 0.68, the LLM is doing real work and the rest of this list applies.

## 8. Sanity check: soft-token slot alignment

Add a debug assertion in `wrap_prompt_with_soft_tokens_v2` after tokenization:

```python
assert replaced_idx.shape[0] == sum_expected_unk_slots, \
    f"Soft-token slot mismatch: tokenizer={replaced_idx.shape[0]}, expected={sum_expected_unk_slots}"
```

LLaMA's BPE can split ` <unk>` into 2 tokens depending on context, which would silently misalign the injection. The "Prompt injection stats" log already prints `sample_unk_slots` vs `sample_soft_tokens` — verify they match.

---

## Recommended order of fixes

1. **Turn on LoRA (#1)** and **fix the Q-Former instruction (#2)** together. Expected lift: ~0.05–0.08 uAUC.
2. **Add output LayerNorm on `llama_proj` (#3)** and **fix the early-stop counter (#5)** — free wins.
3. **Decide the prompt-text question (#4)** — try both ways once LoRA is in.
4. **Verify the MF baseline (#7)** to know how much headroom actually exists.
