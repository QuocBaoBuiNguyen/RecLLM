# Thesis Roadmap & Architecture Notes

Working notes for future agent/reviewer to reason about current state, defensible novelty claims, and open improvement directions. Update as work progresses.

## Current state (2026-05-27)

- **Branch**: `feat/swap-llm-qwen2`, commit `13702bb`
- **Q-Former**: 4 layers, 8 queries (reverted from 8 layers — 8L overfit on 33k ML-1M samples)
- **LLM**: Qwen2-7B-Base loaded fp16 (no quantization on current commit)
- **Best result so far** (ML-1M, Q-Former 4L + fp16): val uAUC 0.7098, test_warm AUC 0.7456 / uAUC 0.7265, test_cold AUC 0.7072 / uAUC 0.6478

## Architecture analysis: SigLLM vs InstructBLIP

SigLLM uses HuggingFace's literal `InstructBlipQFormerModel` (see `hf_qformer_adapter.py:12-13, 96`). The instruction-aware self-attention pathway is **identical** to InstructBLIP:

1. **Self-attention layer**: `[queries; instruction_tokens]` concat → attend each other → queries "read" instruction first
2. **Cross-attention layer**: only queries attend to encoder output → instruction does NOT cross-attend

The architectural pattern matches Figure 3 of InstructBLIP paper exactly.

### Where SigLLM diverges from InstructBLIP

| Aspect | InstructBLIP | SigLLM |
|---|---|---|
| Cross-attention encoder input | Image patches `[B, 257, 768]` from ViT-g/14 | Single CF vector `[B, 1, 768]` projected from MF |
| Queries | 32 | 8 |
| num_layers | 12 | 4 |
| cross_attention_frequency | 2 | 2 (same) |
| Pretrained init | Full BLIP-2 weights (vision-language pretraining) | BERT-base text branch only; cross-attention random init |
| Task | Vision-language QA / captioning | Yes/No CTR recommendation |
| LLM | FlanT5 / Vicuna | Qwen2-7B-Base |

### Key insight (defensible thesis claim)

The functional role of Q-Former components shifts when moving from vision-language to recommendation:

- **InstructBLIP**: cross-attention is the primary mechanism (compress 257 patches → 32 queries via selective attention)
- **SigLLM**: with only 1 CF vector as encoder input, cross-attention has no "selection" to do; the **self-attention pathway between queries and instruction tokens becomes the primary task-conditional feature extractor**

This explains observed behavior:
- Scaling num_layers (4 → 8) did NOT improve performance — cross-attention has nothing more to extract from 1 vector
- Instruction diversity / templates DO matter — that's where the queries actually learn task-specific features

### Defensible thesis narrative (proposed)

> We adapt InstructBLIP's instruction-aware Q-Former architecture from vision-language to collaborative filtering for recommendation. The self-attention pathway, where learnable queries read the task instruction before cross-attending to the encoder, is preserved exactly. The key adaptation is replacing ViT-encoded image patch sequences (length 257) with a single projected CF embedding (length 1) as the cross-attention target. As a consequence, the self-attention pathway between queries and instruction becomes the primary mechanism for task-conditional feature extraction in our setting, rather than selective compression of dense visual features as in vision-language Q-Formers.

## Open improvement directions (priority-ordered, no-new-loss constraint)

User constraint: no inventing new loss functions; novelty must come from architecture composition, training procedure, or data.

### 1. Switch to Amazon-Book dataset (HIGH priority — next step)

ML-1M is saturated. Amazon-Book provides:
- Larger scale (727K train vs 33K) → can justify higher Q-Former capacity
- Less LLM contamination (book titles less in pretraining than movie titles)
- Direct comparison vs BinLLM (AUC 0.8429, UAUC 0.7073) and SellaRec (AUC 0.8837, UAUC 0.7459)
- ILM also tested on Amazon (Beauty, Clothing) — closer reference

See "Amazon-Book dataset plan" section below.

### 2. Diversify instruction templates (LOW effort, MEDIUM impact)

Current: 12 generic templates in `QFORMER_ITEM_INSTRUCTIONS` (`qformer_rec_llm.py:69-82`). Paper recommends 10-15 per task. Could extend to 25-30 covering more aspects (genre-aware, era-aware, audience-aware) — but **without injecting metadata dynamically** (that creates training/inference distribution shift).

Effort: 30 min edit + retrain Stage 1+2+3 (~3h A100).
Expected: +0.005-0.010 uAUC.

### 3. InstructBLIP Q-Former pretrained init (MEDIUM effort, MEDIUM impact)

Current: Q-Former text branch initialized from BERT; cross-attention random init.

Proposal: load full Q-Former weights from `Salesforce/instructblip-vicuna-7b` checkpoint. Self-attention + text branch get vision-language pretraining head start. Cross-attention will still be random-init because encoder dim differs (ViT 1408 vs SigLLM 256) — would need a fresh projection layer.

Effort: ~1 day code (handle dim mismatch in cross-attention init).
Expected: +0.005-0.015 uAUC.

### 4. iALS replace plain MF (MEDIUM effort, MEDIUM impact)

ILM uses iALS (Section 4.1 of paper). SigLLM uses BPR/MSE MF. iALS proven more robust on ML-1M (Rendle et al., 2022).

Effort: ~1-2 days (implement iALS trainer or use `implicit` library).
Expected: +0.005-0.015 uAUC.

### 5. Increase max_history_length 10 → 20 (LOW effort, LOW-MEDIUM impact)

Current dataset truncates user history to 10 items. ML-1M has users with 50+ ratings. More history → richer item-item co-watch signal in Stage 1 contrastive.

Effort: 5 min config + rebuild pkl + retrain Stage 1 (~2h).
Expected: +0.005-0.015 uAUC.

### 6. Increase Stage 1 training duration (LOW effort, LOW-MEDIUM impact)

Current Stage 1: 100 epochs × 132 batches ≈ 13K steps.
ILM Stage 1: 259K steps (20× more).
Bump `epoch: 100 → 500`, let early stopping trigger naturally.

Effort: 1 config edit + ~6h retrain.
Expected: +0.005-0.015 uAUC.

## Amazon-Book dataset plan

### Why Amazon-Book

1. Direct baselines: BinLLM Table 3 reports AUC=0.8429, UAUC=0.7073 on Amazon-Book
2. Larger scale supports investigation of Q-Former capacity (4L vs 8L tradeoff different from 33K-sample ML-1M)
3. Different domain → tests generalization of architecture
4. Less LLM training contamination than ML-1M (book titles less indexed)

### Implementation steps (rough)

1. **Download Amazon Books 2014 metadata + reviews** from https://nijianmo.github.io/amazon/index.html
2. **Preprocess**:
   - Same OOD2 temporal split logic as ML-1M
   - Extract metadata: title, description, brand, features → item-text for Q-Former alignment
   - Item count target: ~34K (BinLLM setup)
   - User count target: ~23K with >=20 interactions
3. **Add `book_ood` builder** in `src/sigllm/datasets/book/` mirroring `movie_ood` structure
4. **Update config**: add `datasets.book_ood` block; new output paths under `ckpt/...book/`
5. **Adjust Q-Former instructions** to book domain (replace "movie" → "book", add book-specific aspects like author/genre)
6. **Run pipeline**: Stage 0 MF → Stage 1 Q-Former → Stage 2 generative → Stage 3 Step 1 LoRA → Stage 3 Step 2 CIE
7. **Eval**: report on full test + warm/cold subsets, compare with BinLLM Table 3 + Figure 2

### Open questions for Amazon-Book

- Does Q-Former 8 layers help here (727K samples vs 33K)? Run ablation 4L vs 8L.
- Is `num_queries: 8` still optimal at scale? Try 16, 32.
- Does cold-start gap (current 0 contribution from Q-Former on ML-1M cold) reproduce on Amazon-Book?

### Estimated time

- Preprocessing + builder: 1-2 days
- Full pipeline first run: ~4-6h GPU on A100
- Hyperparameter tuning: 3-5 days

## Open questions for future agent review

These are unresolved decisions that benefit from a fresh look:

1. **Q-Former depth on bigger dataset**: 8L overfit on 33K ML-1M. Does it work on 727K Amazon-Book? Run controlled ablation.
2. **Cold-start mechanism**: ablation showed Q-Former contributes 0 to cold uAUC on ML-1M. Is this fundamental (CF embeddings are noise for cold items) or fixable via instruction design / text fallback?
3. **InstructBLIP weights init**: worth the effort? Cross-attention dim mismatch (ViT 1408 vs MF 256) means cross-attn weights still need to be re-trained — only self-attn + text branch transfer cleanly.
4. **Generalization story**: if SigLLM works well on ML-1M but poorly on Amazon-Book (or vice versa), what does that say about the architecture? Need to discuss in thesis.
5. **Comparison fairness**: BinLLM/SellaRec use Vicuna, SigLLM uses Qwen2-Base. Apples-to-apples requires re-running baselines with same LLM — feasible?
6. **Soft-token slot count**: currently `num_queries=8` per item, history has 10 items → 80 soft tokens for history alone in prompt. At 512 max_txt_len this fits, but limits prompt budget. Worth reducing query count?
7. **Instruction-aware at INFERENCE**: training picks random instruction per sample, eval uses `instructions[0]` (deterministic). Should eval ensemble over multiple instructions? Could give +0.005 uAUC with minimal cost.

## Constraints / non-goals (per user direction)

- **Q-Former is the central method** — must stay, cannot be replaced by MLP/binary encoding even if alternatives match SOTA
- **No new loss functions** — novelty must come from architecture composition, training procedure, instruction design, or dataset
- **No Unsloth** — integration attempted, ran into multiple dtype issues (see git history `feat/unsloth-qwen2` branch). Standard HF + bitsandbytes is the proven path.
- **A100 80GB is the training GPU** — current setup uses 38-49GB peak, fits comfortably without aggressive optimization

## Reference papers in `docs/`

- `BLIP-2` (original Q-Former) — `docs/` doesn't include but referenced via InstructBLIP
- `InstructBLIP_1.pdf` — instruction-aware Q-Former, current architecture reference
- `ILM.pdf` — Q-Former for recommendation, item-item + user-item contrastive in Stage 1
- `BinLLM.pdf` — text-like binary encoding (alternative approach, not Q-Former), reports ML-1M numbers
- `SellaRec.pdf` — pre-aligned MF with LLM-distilled semantic (alternative approach, not Q-Former), reports best ML-1M numbers
