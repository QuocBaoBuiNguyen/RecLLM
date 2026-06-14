# Interaction-Aware Q-Former — Design Proposal

**Date:** 2026-05-30
**Status:** Phase 1 implemented on branch `feat/interaction-aware-qformer`; training pending
**Target branch:** `feat/interaction-aware-qformer` (created from `feat/ablation-instruction-aware`)
**Timeline estimate:** 3-4 weeks (Phase 1 implementation + Phase 2 training + Phase 3 comparison)

## Background and Rationale for Switching Direction

### Where SigLLM stood before this proposal

The thesis original main claim was **instruction-aware Q-Former adapted for recommendation** — replicating InstructBLIP's instruction-aware design (queries concat with instruction text in self-attention) in the recommendation domain. Stage 1 (5-loss pretraining), Stage 2 (generative pretraining), Stage 3 step 1 (LoRA), and Stage 3 step 2 (CIE) all ran successfully. Test_warm uAUC reached 0.7217 on ML-1M, competitive with CoLLM-MF baseline (0.7179) within noise.

### What the ablation revealed

To validate the "instruction-aware" claim, two ablation flags were introduced (see `docs/ABLATION_RESULTS.md`):

| Comparison | Δ test_warm uAUC | Conclusion |
|---|---|---|
| Instruction-aware vs Text-only (zero soft tokens) | **+0.0467 pts** | CF signal flowing through Q-Former contributes meaningfully (PROVEN) |
| Instruction-aware vs Vanilla queries-only | **−0.0004 pts** | Instruction routing provides no measurable contribution (NEGATIVE) |

The first finding validates that the modality bridge works — CF signal is meaningfully integrated by the LLM. The second finding undermines the original thesis claim: instruction-aware Q-Former does not demonstrate value over vanilla queries-only on this single-task CTR setting.

### Why instruction-aware likely failed (hypothesized)

- **Single-task setup**: Only Yes/No CTR. InstructBLIP's instruction-aware was validated across 26 datasets × 11 task categories where instructions had to discriminatively route between tasks.
- **Paraphrase-style prompts**: All 12 templates in `QFORMER_ITEM_INSTRUCTIONS` are semantic paraphrases of "represent this movie for recommendation" — no diversity for routing to exploit.
- **Recommendation task is fundamentally simpler than VQA/captioning**: a single feature extraction angle suffices, unlike vision where different tasks need different visual features.

### The core insight that motivated this pivot

After the instruction-aware ablation came back inconclusive, we examined the architectural structure more carefully and noticed that **the Q-Former under the per-item layout is essentially under-utilized as a sequence reasoner**.

In the existing per-item layout:

```
target item  → Q-Former → 8 tokens
history[1]   → Q-Former → 8 tokens
history[2]   → Q-Former → 8 tokens
...
history[10]  → Q-Former → 8 tokens
                                   ─────────────────────────────
                                   → 88 soft tokens fed to LLM
                                   → LLM self-attention figures out
                                     whether history matches target
```

Each Q-Former forward independently encodes a single CF vector. The cross-attention sequence at the Q-Former encoder side has length 1 — there is **nothing to reason over**. The queries are effectively just projecting a single vector.

The crucial question — *"is this user's history compatible with the target item?"* — is answered entirely at the LLM level via self-attention over 88 soft tokens plus per-item text titles. This means:

1. **Q-Former's reasoning capacity is wasted.** It was designed (in BLIP-2 / InstructBLIP / VideoBLIP) to compress and reason over long sequences (256 image patches, sequences of frames). When given a length-1 input it just projects.
2. **Target-history compatibility is offloaded to the LLM.** A 7B-parameter LLM does heavy attention work that a lightweight Q-Former (76M params) could pre-compute.
3. **The modality bridge produces per-item representations, not joint representations.** The "alignment" — meaning of thesis title "căn chỉnh thông tin cộng tác" — happens at LLM level, not Q-Former level.

### The proposed pivot in plain terms

Instead of running Q-Former 11 times (once per item) producing 88 independent soft tokens, run Q-Former **once** with the entire interaction sequence:

```
[target CF, history[1] CF, history[2] CF, ..., history[10] CF]
              ↓
         Q-Former (single forward, queries cross-attend the whole sequence)
              ↓
         8 soft tokens that jointly encode user-target-history compatibility
              ↓
         LLM Yes/No
```

This way:

- Q-Former queries can directly attend over all 11 items at once and learn relational patterns:
  - *"target is an action movie, history is all comedies → low compatibility"*
  - *"target is a sequel, history contains the prequel → high compatibility"*
  - *"target genre matches dominant history genre → high compatibility"*
- The 8 output soft tokens are **interaction features**, not per-item features.
- LLM receives a more compositional signal in fewer tokens (8 instead of 88), freeing prompt budget for text titles.

### Why this is a better thesis direction than refining instruction-aware

After the instruction-aware negative finding, three paths were considered:

| Path | Effort | Outcome quality |
|---|---|---|
| Defend negative finding as-is (Pivot 1 in ABLATION_RESULTS.md) | 0 | Defensible but weak — "we tried, it didn't help" |
| Redesign prompts with diverse semantic angles (Pivot 2) | 1 week | Risky — paraphrase issue may persist; small architectural change |
| **Interaction-aware Q-Former (this proposal)** | **3-4 weeks** | **Real architectural shift; defendable regardless of outcome** |

The interaction-aware pivot addresses a **structural concern** (Q-Former under-utilization, LLM-level reasoning offload) rather than tweaking prompt text. This gives the thesis a genuine architectural contribution to defend.

### Decision: instruction is intentionally dropped from interaction-aware mode

In the Phase 1 implementation, `forward_interaction()` does **not** accept an instruction parameter. The queries see only the CF sequence in cross-attention; no instruction concat in self-attention. Rationale:

1. **Test pure interaction effect first.** If instruction-aware contributed nothing in the per-item ablation, adding instruction back into the interaction-aware mode would conflate two design axes and obscure attribution.
2. **Keep the implementation minimal.** Phase 1 aims to verify the interaction-aware mechanism works; combining flags is a Phase 4 refinement if Phase 1 results warrant it.
3. **Aligns with the thesis pivot narrative.** The story shifts from "instruction-aware adaptation" (failed) to "interaction-aware adaptation" — we want to isolate and measure the latter cleanly.

If Phase 3 results show interaction-aware works, Phase 4 may explore the orthogonal 2×2 (instruction × interaction) for a complete ablation table. For now, code defaults to `instruction_aware=False` when `interaction_aware=True` is used.

## Motivation

The current SigLLM architecture treats Q-Former as a **per-item encoder**: for each (target item) and each (history item), Q-Former runs an independent forward pass producing 8 soft tokens. The user's history of length L produces `L × 8` soft tokens, and the target item adds another 8, for a total of `(L+1) × 8 = 88` soft tokens injected into the LLM prompt (with L=10 for ML-1M).

This setup has two architectural concerns:

### Concern 1: Q-Former is under-utilized as an interaction reasoner

The Q-Former architecture — learnable queries cross-attending to encoder hidden states — is designed for **compression and reasoning over sequences**. In InstructBLIP, 32 queries compress 257 image patches into 32 informative tokens, performing selective attention over the visual sequence. In SigLLM's per-item mode, the encoder input is a **single CF vector** of length 1; cross-attention has no sequence to reason over. The queries effectively just project + transform a single vector — much of Q-Former's reasoning capacity is wasted.

### Concern 2: Target-history interaction reasoning is offloaded to the LLM

In the per-item setup, the LLM receives `(L+1) × 8` independent soft tokens plus per-item text titles. The LLM must use its own self-attention to figure out which history items are relevant for predicting the target item's CTR. This means:

1. **Target-history compatibility reasoning** happens in the 7B-parameter LLM via expensive attention, rather than in the lightweight (~76M parameter) Q-Former.
2. **Token budget** is dominated by per-item soft tokens, limiting how much textual context fits in the prompt.
3. The **modality bridge purpose** — aligning collaborative signal into the LLM's representation space — is performed item-by-item, not as a joint user-item-history representation.

## Proposed Architecture

Replace per-item Q-Former forwards with a **single joint forward** that takes the entire interaction context as the cross-attention encoder input.

### Encoder input construction

```
target_cf:  [B, d_cf]          -- target item's MF embedding
history_cf: [B, L, d_cf]       -- L history item MF embeddings, padded
            (L=10 for ML-1M, padded with sentinel id=0)

Stack:      [B, L+1, d_cf]     -- target first, then history items in temporal order
            interaction_cf = cat([target_cf.unsqueeze(1), history_cf], dim=1)
```

### Q-Former forward

```
1. Project to d_model:  proj_cf(interaction_cf) -> [B, L+1, d_model=768]
2. Add positional encoding (learnable):
       pos_emb[0]      -> target slot (always position 0)
       pos_emb[1..L]   -> history slots (temporal order)
3. Build encoder attention mask:
       mask[target slot]            = 1 always
       mask[history slot i]         = 1 if history[i] is not padding (id != 0)
4. Q-Former cross-attention:
       queries (Q=8 learnable) cross-attend to [B, L+1, d_model]
       respecting encoder_attention_mask
5. Output: [B, Q=8, d_model] joint interaction representation
6. llm_proj projects to LLM hidden size: [B, 8, H=3584]
```

The 8 (or optionally 16) output soft tokens encode the **target-history compatibility** directly, instead of leaving this reasoning to the LLM.

### Prompt change

Current Stage 3 step 2 prompt:
```
A user has highly rated these movies: <ItemTitleList>. The
interaction-history signals are: <ItemIDList>. Based on this preference
history, predict whether the user would like the movie
<TargetItemTitle>. <TargetItemID> Answer with "Yes" or "No".
```

→ 80 soft tokens at `<ItemIDList>` + 8 at `<TargetItemID>` = 88 soft tokens

Proposed Stage 3 step 2 prompt:
```
A user has highly rated these movies: <ItemTitleList>. The collaborative
interaction context is: <InteractionContext>. Based on this preference
history, predict whether the user would like the movie
<TargetItemTitle>. Answer with "Yes" or "No".
```

→ 8 soft tokens at `<InteractionContext>` (single placeholder, encoded jointly)
→ Removed `<TargetItemID>` slot (target is already in InteractionContext)

## Comparison Table

| Aspect | Current (Per-item) | Proposed (Interaction-aware) |
|---|---|---|
| Q-Former forward passes per sample | L+1 (=11 with L=10) | 1 |
| Encoder input shape | [B, 1, d_cf] each pass | [B, L+1, d_cf] once |
| Soft tokens in LLM prompt | (L+1) × 8 = 88 | 8 (or 16 if 2Q output) |
| Target-history reasoning | LLM self-attention | Q-Former cross-attention |
| Q-Former cross-attention sequence length | 1 (no reasoning needed) | L+1 (real sequence reasoning) |
| Positional encoding on encoder side | Not used | Required (temporal order matters) |
| Variable L handling | Per-item, no masking | Encoder attention mask |
| LLM compute per sample | Higher (more tokens to attend) | Lower (fewer tokens) |
| Q-Former compute per sample | Higher (L+1 forwards) | Lower (1 forward) |
| Token budget left for text | Less (88 soft + text) | More (8 soft + text) |

## Theoretical Grounding

### Direct precedent: VideoBLIP / Video-LLaMA

VideoBLIP applies Q-Former to video by treating frames as the cross-attention sequence:

```
T frames × P spatial patches each = T*P encoder hidden states
Q-Former queries (32) cross-attend to this sequence
Output: 32 tokens encoding the whole video
```

The proposed SigLLM interaction-aware Q-Former is the recommendation analogue:

```
L+1 items (target + history) = L+1 CF vectors as encoder hidden states
Q-Former queries (8) cross-attend to this sequence
Output: 8 tokens encoding the user-target-history compatibility
```

The architectural pattern is identical — Q-Former as **sequence reasoner**, not just per-item encoder.

### Indirect precedent: Sequential recommendation (SASRec, BERT4Rec)

These models process user history as a sequence with self-attention:
```
SASRec: history sequence -> self-attention -> next item prediction
```

SigLLM interaction-aware is similar but:
- Includes target item explicitly in the sequence (for compatibility scoring, not next-item prediction)
- Uses cross-attention (queries as separate stream) rather than self-attention
- Hands output to an LLM rather than directly predicting

### Departure from current SigLLM (instruction-aware)

Note this is a **different design axis** than instruction-aware:

- **Instruction-aware** (current Main config): how queries READ task description
- **Interaction-aware** (proposed): how queries ENCODE the input signal

These are orthogonal. The ablation showed instruction-aware contribution is negligible; interaction-aware tackles a different concern entirely (Q-Former under-utilization).

## Implementation Plan

### Phase 1 — Minimal interaction-aware (Week 1)

**Goal:** Train and evaluate interaction-aware Q-Former at Stage 3 step 2 only, reusing Stage 0/1/2/Step1 checkpoints.

**Files to modify:**

1. **`src/sigllm/models/q_former/hf_qformer_adapter.py`**
   - Add learnable positional embedding: `nn.Parameter(torch.randn(MAX_L + 1, d_model))` where MAX_L=10
   - Add new method `forward_interaction(target_cf, history_cf, history_mask)`:
     - Stack target + history → `[B, L+1, d_cf]`
     - Project via `proj_cf` → `[B, L+1, d_model]`
     - Add positional embedding
     - Build attention mask from `history_mask` (1 for valid, 0 for padding) plus always-1 for target
     - Q-Former forward with this encoder input
     - Return `out_proj(query_hidden)` → `[B, Q, output_dim]`

2. **`src/sigllm/models/multimodal/qformer_rec_llm.py`**
   - Add init flag `interaction_aware: bool = False` (default False to preserve current behavior)
   - In `forward()` (around line 600), if `interaction_aware`:
     - Compute `target_cf = self.rec_encoder.item_encoder(target_id)` → `[B, d_cf]`
     - Compute `history_cf = self.rec_encoder.item_encoder(history_ids)` → `[B, L, d_cf]`
     - Compute `history_mask` from `history_ids != padding_index`
     - Call `interaction_q = self.qformer.forward_interaction(target_cf, history_cf, history_mask)` → `[B, Q, d_model]`
     - Project: `interaction_llm = self.llm_proj(interaction_q)` → `[B, Q, H]`
     - Inject as single `<InteractionContext>` soft-token slot in prompt

3. **`src/sigllm/models/multimodal/qformer_rec_llm.py`** (PLACEHOLDERS_FOR_EMBED)
   - Add conditional branch: when `interaction_aware=True`, use `PLACEHOLDERS_FOR_EMBED = ["<InteractionContext>"]`
   - When False, keep `["<ItemIDList>", "<TargetItemID>"]`

4. **`prompts/qformer_prompt_movie_interaction.txt`** (new file)
   - 3 templates with `<InteractionContext>` placeholder

5. **`configs/config.yaml`**
   - Add `model.qformer_config.interaction_aware: False` (default)
   - Document: True switches to interaction-aware mode

6. **`src/sigllm/pipelines/multimodal/train_qformer_stage3_step2_cie.py`**
   - No code changes; route via config flag

### Phase 2 — Verification and tuning (Week 2)

- Train interaction-aware Stage 3 step 2 with `interaction_aware=True`
- Initial config: 8 output tokens (Q=8), reuse Stage 1/2 ckpts
- Monitor: convergence trajectory, val_uAUC peak, training loss
- If val_uAUC < 0.65 at epoch 10 → tuning needed (lr, init scale, positional encoding init)
- If val_uAUC > 0.70 at epoch 5-10 → on track

### Phase 3 — Comparison and ablation (Week 3)

- Eval interaction-aware on test / test_warm / test_cold
- Compare with per-item Instruction-aware (0.7322 / 0.7131) and Vanilla (0.7325 / 0.7170)
- Run sanity ablation: interaction-aware + `ablate_soft_tokens=True` (should drop to ~0.69 like Text-only baseline, confirming CF flows through)
- Compute efficiency comparison:
  - Q-Former FLOPs / sample: L+1 forwards × cost vs 1 forward × cost
  - LLM input tokens: 88 + text vs 8 + text
  - Training wall-time per epoch

### Phase 4 — Optional refinements (Week 4, if time)

If Phase 3 shows Δ uAUC > 0 over per-item:
- Try Q=16 output tokens (less aggressive compression)
- Try sinusoidal vs learnable positional encoding
- Multi-seed runs for statistical confidence

If Phase 3 shows Δ uAUC ≈ 0 or worse:
- Sanity check: does interaction-aware match per-item when L+1=1 (only target, no history)?
- Probe what queries learned (attention weight analysis)
- Document as second-direction-explored finding

## Open Design Decisions

### Decision 1: Output dimension (Q=8 vs Q=16)

- **Q=8 (default):** Same as current per-item — maintains parameter compatibility with existing `llm_proj`
- **Q=16:** Less aggressive compression — but requires retraining `llm_proj`

Recommend start with Q=8 for direct comparability; bump to Q=16 if results suggest under-capacity.

### Decision 2: Positional encoding type

- **Learnable PE:** `nn.Parameter(torch.randn(MAX_L+1, d_model))` — flexible, can learn target-vs-history distinction
- **Sinusoidal PE:** standard transformer PE — generalizes to longer L at inference

Recommend learnable PE — recommendation L is bounded (10 for ML-1M), no generalization needed; learnable lets target slot (position 0) develop a distinct embedding.

### Decision 3: Target placement

- **Target at position 0 (recommended):** Distinct positional embedding signal indicating "this is the candidate"
- **Target at position L (end):** Treats target as "next item" in sequence — SASRec-style
- **Target as separate stream:** Use 2 sets of queries (target queries + history queries) — more parameters, more complex

Recommend target-at-position-0. Clean separation, simple to implement.

### Decision 4: Stage 1/2 retraining

- **Reuse current Stage 1/2 ckpts (Phase 1 default):** Q-Former was pretrained for single-item encoding. Using it for joint encoding is technically a distribution shift, but Stage 3 step 2 will adapt it.
- **Retrain Stage 1 with multi-item objectives:** Build interaction-aware pretraining (e.g., predict whether target matches history). Major work, 1-2 weeks alone.

Recommend Phase 1 reuse approach. If results disappointing, consider retraining Stage 1.

### Decision 5: Handle missing history (L=0)

Some test samples may have no history (cold-start users). Encoder input becomes `[B, 1, d_cf]` (only target). This degenerates to per-item case, which is acceptable. Just ensure padding mask handles correctly.

## Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Q-Former pretraining mismatch (trained on single item, now sees sequence) | Medium | Stage 3 step 2 has 76M trainable Q-Former params + 200 epochs to adapt |
| Information bottleneck (L+1→Q=8 too aggressive) | Medium | Try Q=16 if Q=8 underperforms |
| LLM loses per-item granularity in soft tokens | Low | Per-item text titles still in prompt; LLM has both |
| Positional encoding init bad | Low | Standard init, can tune if needed |
| Training instability (multi-item gradient flow) | Low | Adam + gradient clipping; same loss as Stage 3 step 2 |
| No improvement over per-item | Medium | Still defendable as efficiency win (1 forward vs 11, 8 tokens vs 88) |
| Worse than per-item | Low-Medium | Document as finding, frame as "joint encoding is harder than LLM-level reasoning in this setting" |

## Expected Outcomes (hypotheses)

### Hypothesis 1 (preferred): Interaction-aware beats per-item

If Δ uAUC > 0.01: **strong thesis claim**. Architecturally meaningful contribution.
> "Joint encoding of target and history in Q-Former, rather than per-item encoding with LLM-level interaction reasoning, improves CTR uAUC by X.Y points while reducing LLM input token budget 11x. The interaction-aware design better utilizes Q-Former's sequence-reasoning capacity."

### Hypothesis 2: Interaction-aware matches per-item

If |Δ uAUC| < 0.005: **efficiency win** narrative.
> "Interaction-aware Q-Former matches per-item performance at 11x compute reduction and 11x token budget compression. LLM-level interaction reasoning is sufficient, but Q-Former-level reasoning is equally viable and more efficient."

### Hypothesis 3: Interaction-aware underperforms

If Δ uAUC < -0.01: **honest negative finding** with hypothesis.
> "Joint encoding compresses interaction context too aggressively (L+1 items into Q=8 tokens). LLM-level attention over per-item soft tokens preserves more discriminative signal. Future work should explore intermediate compression (Q=16, 32) or hybrid approaches."

All three outcomes are defendable. Unlike instruction-aware (where Δ ≈ 0 was inconclusive due to setup), here Δ ≈ 0 is itself a meaningful efficiency claim.

## Defense Narrative Integration

This proposal addresses the gap left by the instruction-aware negative finding:

**Before this experiment:**
> "Investigation of instruction-aware Q-Former showed no measurable contribution over vanilla queries-only in single-task CTR setting. We hypothesized this is due to task and instruction homogeneity."

**After this experiment (Hypothesis 1 outcome):**
> "Building on our finding that instruction-aware design does not transfer to single-task recommendation, we identified a second under-explored axis of Q-Former design for this domain: per-item vs interaction-aware encoding. We propose interaction-aware Q-Former that jointly encodes target+history, demonstrating X.Y pts uAUC improvement over per-item encoding."

→ Thesis claim shifts from "instruction-aware adaptation" (failed) to "interaction-aware adaptation" (succeeded). Cleaner narrative for defense.

## Connection to Related Work

| Paper | Q-Former encoder input | This proposal |
|---|---|---|
| BLIP-2 | Image patches (257) | Item sequence (L+1) |
| InstructBLIP | Image patches + instruction | Item sequence (instruction-aware orthogonal, can be combined) |
| VideoBLIP | Frame sequence | **Direct analogue** |
| ILM | Single item CF | **Per-item baseline this work compares against** |
| SigLLM (current) | Single item CF (per-item) | This proposal generalizes to L+1 items |

## Timeline

| Week | Milestone |
|---|---|
| 1 | Implement `forward_interaction` in HFQFormerAdapter; modify qformer_rec_llm.py; create new prompt file; smoke test (forward pass works, shapes correct) |
| 2 | Train interaction-aware Stage 3 step 2 (uses Stage 0/1/2/Step1 ckpts from current Main); monitor convergence |
| 3 | Eval on test/test_warm/test_cold; compare with per-item baselines; run sanity ablation (ablate_soft_tokens=True) |
| 4 | Document results; optional Q=16 ablation; multi-seed if time |

Total: ~3-4 weeks for Phase 1 verification. Phase 2 (Stage 1 retraining with interaction objectives) is optional follow-up if Phase 1 shows promise.

## Backwards Compatibility

The `interaction_aware` flag defaults to `False`, preserving all current per-item behavior. All existing checkpoints (Instruction-aware, Vanilla, Text-only baseline) remain valid and re-runnable. Both modes can coexist in the same codebase.

## Open Questions

1. Should we also test interaction-aware **with** instruction-aware (orthogonal flags), or only as a replacement?
2. Should target be encoded at position 0 with a special "target indicator" embedding, or position-only encoding?
3. Should we retrain Stage 2 (generative pretraining) with multi-item input, or keep single-item Stage 2?
4. How to handle `<UserID>` slot (currently disabled in code) — should interaction-aware also encode user CF in the sequence?

These can be deferred to Phase 4 ablations if Phase 1 shows promise.
