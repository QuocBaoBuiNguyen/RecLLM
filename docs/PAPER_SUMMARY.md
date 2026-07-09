# SigLLM — Paper-Style System Summary

**Purpose:** Self-contained technical summary of SigLLM architecture, training procedure, prompts, and configuration. Intended as context document for thesis writing, future agents, and reviewers. Reflects state on `feat/ablation-instruction-aware` branch, 2026-05-29.

## Research Scope

**Thesis title (Vietnamese):** Căn chỉnh thông tin cộng tác vào hệ tư vấn sử dụng QFormer
**Thesis title (English working translation):** Aligning collaborative information into recommendation systems using Q-Former

**Research question:** Can the BLIP-2 / InstructBLIP modality-bridge paradigm (Q-Former) be transferred from vision-language tasks to recommendation, by replacing the visual encoder with a collaborative filtering encoder?

**Investigation, not benchmark chase.** The thesis investigates whether the paradigm transfers, what design choices matter, and characterizes the limits of the approach. Beating state-of-the-art is not the primary objective.

## High-Level Architecture

SigLLM is a four-stage pipeline that bridges a Matrix Factorization (MF) collaborative encoder to a frozen-then-LoRA-tuned LLM via a Q-Former modality adapter.

```
┌──────────────────┐
│ Stage 0          │  MF: learn (user, item) → embedding vectors
│ rec_baseline     │  Train: BCE on (uid, iid, label)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Stage 1          │  5-loss Q-Former pretraining (ILM-style + extensions)
│ representation   │    - ITC: item-text contrastive
│ learning         │    - ITM: item-text matching
│                  │    - ITG: item-grounded text generation
│                  │    - Item-item contrastive (ILM extension)
│                  │    - User-item contrastive (SigLLM extension)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Stage 2          │  BLIP-2-style generative pretraining
│ generative       │  Q-Former (queries-only) → linear proj → frozen LLM
│ pretraining      │  Loss: next-token LM on item caption ("Title: X. Genres: Y.")
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Stage 3 Step 1   │  CoLLM Step 1 — LoRA tuning only
│ LoRA tuning      │  Prompt: text-only (no soft tokens)
│                  │  Frozen: Q-Former, projection, base LLM, MF
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Stage 3 Step 2   │  CoLLM Step 2 — CIE (Collaborative Info Embedding)
│ CIE tuning       │  Prompt: full (with <ItemIDList>, <TargetItemID> soft-token slots)
│                  │  Trainable: Q-Former, projection
│                  │  Frozen: LoRA (from Step 1), base LLM, MF
└──────────────────┘
```

## Component Details

### Matrix Factorization (Rec Encoder)

- **File:** `src/sigllm/models/rec/matrix_factorization.py`
- **Architecture:** Plain dot-product MF (no bias terms): `score = dot(user_emb, item_emb)`
- **Embedding dim:** 256 (ML-1M); 64 (Amazon-Book, reduced due to scale)
- **Padding index:** 0 (sentinel for history padding)
- **Training:** BCE with logits, Adam optimizer
- **Output of training:** `mf_model.pth` containing user_embedding and item_embedding tables

### Q-Former Adapter

- **File:** `src/sigllm/models/q_former/hf_qformer_adapter.py`
- **Backbone:** HuggingFace `InstructBlipQFormerModel` (literal class import)
- **Cross-attention input:** Single projected CF vector `[B, 1, d_model]` (vs InstructBLIP's 257 image patches)
- **Number of queries:** 8 learnable embeddings (vs InstructBLIP's 32)
- **Number of layers:** 4 (vs InstructBLIP's 12)
- **Cross-attention frequency:** Every other layer (same as InstructBLIP)
- **Text branch init:** From pretrained `bert-base-uncased` (BLIP-2 convention); cross-attention layers keep random init
- **Hidden size:** `d_model = 768`
- **Max instruction length:** 48 tokens

#### Four forward modes exposed:

1. **`encode_cf(cf_vec)`** — queries-only forward. No text branch input. Used in Stage 1 ITC item branch, Stage 1 item-item / user-item contrastive, Stage 2.
2. **`encode_text(text)`** — text-only forward, no queries, no cross-attention. Used in Stage 1 ITC text branch.
3. **`forward_multimodal(cf_vec, text, causal_text)`** — joint forward returning both query and text hidden states. `causal_text=True` for ITG (causal mask on text); `False` for ITM and instruction-aware forward.
4. **`forward(cf_vec, instruction)`** — LLM-feeding mode: queries cross-attend CF, instruction concatenated with queries in self-attention. **This is the instruction-aware pathway and is the focus of the ablation study.**

When `instruction_aware=False`, `forward()` short-circuits to `encode_cf()` + `out_proj` (vanilla BLIP-2 mode, no instruction).

### Linear Projection (Q-Former → LLM)

- **Implemented as 2-layer projection** (proj_token_num=8, mirroring `num_queries`)
- **Input:** Q-Former hidden states `[B, Q, d_model=768]`
- **Output:** LLM-compatible soft tokens `[B, Q, H=3584]` (3584 = Qwen2-7B hidden size)
- **Trainable:** Stage 2 (initial), Stage 3 step 2 (re-tuned)

### LLM Backbone

- **Model:** `Qwen/Qwen2-7B-Base`
- **Loading:** HuggingFace `AutoModelForCausalLM` at fp16, no quantization (current branch)
- **Tokenizer:** Qwen2 tokenizer with appended `<|im_start|>` (`id=151644`) as the soft-token placeholder
- **Frozen base:** All Qwen2 layers frozen across all stages
- **LoRA adapter:** Attached after Stage 2, trained in Stage 3 step 1, frozen in step 2

### LoRA Configuration

- **Library:** PEFT
- **Rank (r):** 8
- **Alpha:** 16
- **Target modules:** `q_proj`, `v_proj` (attention query and value projections only)
- **Dropout:** 0.05
- **Trainable LoRA parameters:** 2,523,136 (verified in training logs)

**Decision rationale** (from `THESIS_ROADMAP.md`):
- r=16 with `[q_proj, k_proj, v_proj, o_proj]` was tested (10M trainable params) → overfit on 33k ML-1M samples, val uAUC peaked at 0.7048 vs r=8 baseline 0.7077
- Conclusion: LoRA capacity is not the bottleneck; reverted to r=8 with `[q_proj, v_proj]` only

## Prompts

Two distinct prompt families used in different stages.

### Q-Former item instructions (Stage 3 step 2, instruction-aware Q-Former input)

Defined as `QFORMER_ITEM_INSTRUCTIONS` in `src/sigllm/models/multimodal/qformer_rec_llm.py` lines 69-82. **12 paraphrase templates**, randomly sampled per batch:

1. "Represent this movie for recommendation using its title and genres."
2. "Align this movie metadata with its collaborative filtering representation."
3. "Given the movie metadata, extract recommendation-relevant item features."
4. "Use the title and genres to describe this movie in the item embedding space."
5. "Map this movie's textual attributes to its collaborative recommendation signal."
6. "Identify the movie preferences implied by its title and genre metadata."
7. "Create a language-aligned representation of this movie for recommendation."
8. "Summarize this movie as an item a recommender system can compare."
9. "Based on the title and genres, represent what kind of users may like this movie."
10. "Encode the semantic information of this movie for item-language alignment."
11. "Use a few metadata cues to align this movie with behavioral item signals."
12. "Produce a recommendation-aware representation from this movie description."

**Note:** All 12 are paraphrases of "represent this movie for recommendation". The semantic content is essentially identical across templates. This is hypothesized in the ablation results as a possible reason the instruction-aware design does not demonstrate measurable contribution over vanilla queries-only.

### LLM input prompts (Stage 3 step 1 — text-only)

File: `prompts/qformer_prompt_movie_text_only.txt`. Three templates randomly sampled per batch. No soft-token placeholders.

```
A user has given high ratings to the following movies: <ItemTitleList>.
Leverage the information above to predict whether the user would enjoy
the movie titled <TargetItemTitle>. Answer with "Yes" or "No".

A user has highly rated these movies: <ItemTitleList>. Based on this
preference history, predict whether the user would like the movie
<TargetItemTitle>. Answer with "Yes" or "No".

A user previously gave strong ratings to: <ItemTitleList>. Use this
information to decide whether the user would enjoy <TargetItemTitle>.
Answer with "Yes" or "No".
```

Placeholders:
- `<ItemTitleList>` — comma-joined string of recent positively-rated item titles
- `<TargetItemTitle>` — the candidate item being scored

### LLM input prompts (Stage 3 step 2 — full with soft-token slots)

File: `prompts/qformer_prompt_movie.txt`. Same three templates as step 1, augmented with soft-token placeholders.

```
A user has given high ratings to the following movies: <ItemTitleList>.
The interaction-history signals are: <ItemIDList>. Leverage the
information above to predict whether the user would enjoy the movie
titled <TargetItemTitle>. <TargetItemID> Answer with "Yes" or "No".
```

Soft-token placeholders:
- `<ItemIDList>` — replaced by `L × Q = L × 8` projected Q-Former soft tokens (one Q-Former forward per history item)
- `<TargetItemID>` — replaced by `Q = 8` projected Q-Former soft tokens (one Q-Former forward on the target item)

History length `L` is dataset-dependent (max 10 for ML-1M, padded with sentinel `id=0`).

### Trailing answer cue

Configured in `configs/config.yaml`:
```yaml
prompt_template: '{} Answer:'
```

This appends `Answer:` after the chosen prompt template, prompting the base LLM (which is NOT instruction-tuned — Qwen2-7B-Base, not -Instruct) to continue with `Yes` or `No` as plain language-model completion.

## Training Procedure

### Stage 0: MF baseline

- **Script:** `src/sigllm/pipelines/rec/train_rec_baseline.py`
- **Objective:** BCE with logits on `(uid, iid, label)` triples
- **Optimizer:** Adam, lr=1e-3, weight_decay=1e-4
- **Batch size:** 1024
- **Epochs:** Up to 5000 with early stopping on validation uAUC, patience=100
- **Output:** `mf_model.pth` (user + item embedding tables)

### Stage 1: Q-Former representation learning (5 losses)

- **Script:** `src/sigllm/pipelines/multimodal/train_qformer_stage1_representation.py`
- **Trainable:** Q-Former only
- **Frozen:** MF (loaded from Stage 0)

Five losses, summed with configurable weights:

| Loss | Weight (default) | Description |
|---|---|---|
| `w_itc` (ITC) | 1.0 | Item-text contrastive: align item CF representation with item caption text CLS |
| `w_itm` (ITM) | 1.0 | Item-text matching: binary classification on (item, text) pair (positive vs in-batch hard negative) |
| `w_itg` (ITG) | 1.0 | Item-grounded text generation: causal LM loss on text branch given item CF |
| `w_ii` (item-item) | 1.0 | Item-item contrastive (ILM extension): co-watch positive pairs |
| `w_ui` (user-item) | 1.0 | User-item contrastive (SigLLM extension): user-history positive pairs |

Temperature for contrastive losses: `tau_itc=0.07`, `tau_ii=0.05`, `tau_ui=0.07`.

- **Optimizer:** Adam, lr=5e-5, weight_decay=1e-3
- **Batch size:** 256
- **Epochs:** Up to 100 with early stopping (patience=10, min_delta=1e-4)
- **Output:** `qformer_stage1_best_qformer.pth`

### Stage 2: Generative pretraining

- **Script:** `src/sigllm/pipelines/multimodal/train_qformer_stage2_generative.py`
- **Architecture:** Q-Former (queries-only via `encode_cf`) → linear projection → frozen LLM
- **Objective:** Next-token language modeling on item caption (`"Title: <title>. Genres: <genres>."`)
- **Trainable:** Q-Former, linear projection
- **Frozen:** MF, base LLM (no LoRA yet)
- **Optimizer:** Adam, lr=1e-4, weight_decay=1e-3
- **Batch size:** 32
- **Max caption length:** 64 tokens
- **Epochs:** Up to 20 with early stopping (patience=5, min_delta=1e-4)
- **Output:** `qformer_stage2_best_qformer.pth`, `qformer_stage2_best_proj.pth`

### Stage 3 Step 1: LoRA tuning

- **Script:** `src/sigllm/pipelines/multimodal/train_qformer_stage3_step1_lora.py`
- **Prompt:** Text-only (`qformer_prompt_movie_text_only.txt`)
- **Trainable:** LoRA adapter only
- **Frozen:** Q-Former, linear projection (loaded from Stage 2), base LLM, MF
- **Note:** Q-Former forward still runs but output is not injected into LLM prompt (text-only prompt has no soft-token slots) → `instruction_aware` flag has no effect on Step 1 results
- **Optimizer:** Adam with `linear_warmup_cosine_lr` scheduler
- **init_lr:** 2e-4
- **min_lr / warmup_lr:** 8e-5 / 1e-5
- **warmup_steps:** 200
- **Batch size (train / eval):** 16 / 64
- **Iters per epoch:** 400
- **Max epochs:** 5
- **AMP:** enabled (bf16)
- **Output:** `checkpoint_best.pth` (LoRA adapter weights)

### Stage 3 Step 2: CIE (Collaborative Info Embedding)

- **Script:** `src/sigllm/pipelines/multimodal/train_qformer_stage3_step2_cie.py`
- **Prompt:** Full (`qformer_prompt_movie.txt`) with `<ItemIDList>` and `<TargetItemID>` soft-token slots
- **Trainable:** Q-Former, linear projection (re-tuned on full prompt)
- **Frozen:** LoRA (loaded from Step 1), base LLM, MF
- **Loss:** Cross-entropy on next-token Yes/No prediction
- **Optimizer:** Adam with `linear_warmup_cosine_lr`
- **init_lr:** 3e-5
- **Max epochs:** 200 with early stopping (patience=20, ref_metric=valid_auc)
- **Iters per epoch:** 400
- **Output:** `checkpoint_best.pth` (Q-Former + projection weights)

### Per-stage parameter count

| Stage | Trainable params | Frozen params |
|---|---|---|
| Stage 0 (MF) | ~2.5M (user + item embeddings, ML-1M) | 0 |
| Stage 1 (Q-Former pretraining) | ~76M (Q-Former) | MF: 2.5M |
| Stage 2 (gen pretraining) | ~76M (Q-Former) + 2.76M (projection) | MF + LLM: ~7.5B |
| Stage 3 Step 1 (LoRA) | 2.52M (LoRA adapter) | Q-Former + proj + LLM + MF: ~7.6B |
| Stage 3 Step 2 (CIE) | ~76M (Q-Former) + 2.76M (proj) | LoRA + LLM + MF: ~7.6B |

## Dataset and Splits

**MovieLens-1M (OOD2 split):**
- 33,891 train / 10,401 validation / 7,331 test samples
- 839 users / 3,256 items
- Test split is further partitioned by warm/cold:
  - test_warm: 3,522 samples (both user and item seen in training)
  - test_cold: 3,178 samples (user OR item not seen in training)
- Split protocol matches CoLLM and BinLLM for direct baseline comparison

**Amazon-Book (attempted, parked):**
- 727,463 train / 25,747 valid / 25,747 test
- 22,967 users / 34,154 items
- MF baseline did not converge in current configuration; pipeline parked for future work

## Ablation Study Summary

Two orthogonal flags tested in a 2×2 cross. Full details in `ABLATION_RESULTS.md`.

**Key finding 1 (CF contribution is real):** Zeroing CF soft tokens at inference (`ablate_soft_tokens=True`) costs +5.15 pts test_warm uAUC. The LLM substantively integrates collaborative signal beyond text prompts alone.

**Key finding 2 (instruction-aware contribution is not demonstrated):** Replacing instruction-aware Q-Former with vanilla queries-only produces near-identical performance (|Δ uAUC| < 0.005 across all splits). Hypothesized reasons: single-task setting, paraphrase-style prompts, recommendation task does not require feature-extraction diversity.

## Configuration File

Single source of truth: `configs/config.yaml`. Key settings as of 2026-05-29:

```yaml
model:
  arch: mini_gpt4rec_v2
  llm_model: "/content/SigLLM/ckpt/llm/qwen2-7b-base"
  prompt_path: "/content/SigLLM/prompts/qformer_prompt_movie.txt"
  prompt_template: '{} Answer:'
  freeze_rec: True
  freeze_proj: False
  ablate_soft_tokens: False     # Ablation flag (inference-time)
  tuning_step: null              # Overridden per stage by pipeline script

  lora_config:
    use_lora: True
    r: 8
    alpha: 16
    target_modules: [q_proj, v_proj]
    dropout: 0.05

  qformer_config:
    use_qformer: True
    qformer_d_model: 768
    qformer_output_dim: 768
    qformer_text_model_name: "bert-base-uncased"
    max_instruction_length: 48
    instruction_aware: True       # Ablation flag (training-time)
    num_queries: 8
    num_heads: 8
    num_layers: 4

  rec_config:
    user_num: 839
    item_num: 3256
    embedding_size: 256

datasets:
  movie_ood:
    path: "/content/SigLLM/data/processed/ml-1m/"
```

Pipeline scripts override stage-specific settings from `run.qformer_stage{1,2}` and `run.qformer_stage3_step{1,2}` blocks (`apply_step{1,2}_overrides` functions in the Stage 3 scripts).

## Inheritance from Prior Work

| Component | Source | Adaptation |
|---|---|---|
| Q-Former architecture | InstructBLIP (Dai et al. 2023) | Same `InstructBlipQFormerModel`, reduced from 12L/32q to 4L/8q |
| Instruction-aware self-attention | InstructBLIP | Identical pathway |
| 5-loss Stage 1 pretraining | ILM (Yang et al. 2024) + BLIP-2 (Li et al. 2023) | ITC/ITM/ITG from BLIP-2; item-item from ILM; user-item is SigLLM extension |
| Stage 2 generative pretraining | BLIP-2 | Captions from item title + genres instead of image captions |
| Stage 3 2-step LoRA tuning | CoLLM (Zhang et al. 2023) | Same Step 1 (text-only LoRA) + Step 2 (full prompt with CIE) protocol |
| LoRA implementation | PEFT (Hugging Face) | Standard, r=8, [q_proj, v_proj] |
| LLM backbone | Qwen2 (Yang et al. 2024) | Qwen2-7B-Base (deliberately base, not Instruct, to avoid RLHF bias) |
| MF baseline | Standard | Plain dot-product, BCE training |

**Synthesis claim:** SigLLM is the first work to combine ILM's 5-loss pretraining, InstructBLIP's instruction-aware Q-Former, and CoLLM's 2-step LoRA tuning into a unified pipeline for recommendation. Each individual component is borrowed; the combination and the empirical investigation of which design choices matter is the contribution.

## Key Files (cheat sheet)

| Purpose | Path |
|---|---|
| Q-Former adapter | `src/sigllm/models/q_former/hf_qformer_adapter.py` |
| Main model wrapper (Stage 3) | `src/sigllm/models/multimodal/qformer_rec_llm.py` |
| Stage 1 alignment model | `src/sigllm/models/projection/qformer_alignment_model.py` |
| Stage 0 training | `src/sigllm/pipelines/rec/train_rec_baseline.py` |
| Stage 1 training | `src/sigllm/pipelines/multimodal/train_qformer_stage1_representation.py` |
| Stage 2 training | `src/sigllm/pipelines/multimodal/train_qformer_stage2_generative.py` |
| Stage 3 Step 1 training | `src/sigllm/pipelines/multimodal/train_qformer_stage3_step1_lora.py` |
| Stage 3 Step 2 training | `src/sigllm/pipelines/multimodal/train_qformer_stage3_step2_cie.py` |
| Single config file | `configs/config.yaml` |
| Prompts | `prompts/qformer_prompt_movie.txt`, `prompts/qformer_prompt_movie_text_only.txt` |
| Ablation results | `docs/ABLATION_RESULTS.md` |
| Thesis strategy notes | `docs/THESIS_ROADMAP.md` |
| This summary | `docs/PAPER_SUMMARY.md` |
