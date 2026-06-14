# Improvement Proposals — Synthesizing X-InstructBLIP + SigLLM Audit

**Date:** 2026-06-03
**Sources:**
- `docs/X-Instruct-BLIP-eval-single-mutli-task.pdf` (Salesforce, 2024, ECCV)
- `docs/COMPREHENSIVE_AUDIT.md` (just compiled)
- `docs/CODEBASE_AUDIT.md`
- `docs/QFORMER_IMPROVEMENT_DIRECTIONS.md`

**Goal:** Đề xuất các cải tiến cụ thể, có evidence từ X-InstructBLIP paper + audit findings, ranked theo ROI.

---

## Key insights từ X-InstructBLIP

X-InstructBLIP (XIB) extend InstructBLIP cho 4 modalities (image, 3D, audio, video) → cùng pattern với SigLLM extend cho rec (1 modality: CF). Findings rất transferable.

### Insight 1 — **Modality prefix CONSISTENT improvement** (Section 5.3, Table 7)

Trích paper:
> "In both audio and 3D single modality tasks removing the prefix consistently hurts the performance. This improvement is likely due to the fact that the Q-Former is **relieved from the extra burden to encode the type of modality** and instead **reserves bandwidth for semantic information**."

→ Prepend "audio:", "video:", "3d:" trước soft tokens **luôn cải thiện**. Lý do: Q-Former không phải tốn capacity encode "đây là modality gì" → reserve cho semantic.

**Áp dụng cho SigLLM:** Hiện tại `<UserID>`, `<TargetItemID>`, `<ItemIDList>` đều dùng **SAME Q-Former + SAME projection** với KHÔNG có prefix gì để LLM biết slot nào là user vs item vs history. → **Strong evidence prefix sẽ giúp.**

### Insight 2 — **Q-Former vs Linear Projection trade-off** (Abstract + Table 1-4)

Trích paper:
> "The Q-Former projection demonstrates **superior performance in single modality scenarios** and adaptability in joint versus discriminative reasoning involving two or more modalities. However, it exhibits **lower generalization capabilities than linear projection in contexts where task-modality data are limited**."

→ Q-Former win khi đủ data. **Linear Projection (LP) win khi data limited.**

**Áp dụng:** SigLLM có 33K samples (very limited compared to BLIP-2 129M pairs) → LP có thể competitive. **Confirms `COMPREHENSIVE_AUDIT.md` Priority 1 (MLP baseline).**

### Insight 3 — **Q-Former init transfer fast** (Section 5.2)

Trích paper:
> "the Video Q-Former component of X-InstructBLIP, initialized with the Image Q-Former's weights, reaches convergence in performance **remarkably fast, within about 1,000 iterations**."

→ Q-Former weights từ pretrained model transfer well across modalities. Implication: BLIP-2 stage-1 pretrained Q-Former có thể được dùng làm init cho SigLLM Stage 1 (thay vì random + BERT only).

### Insight 4 — **Each modality has SEPARATE Q-Former** (Section 3, Figure 3a)

Trích paper:
> "Optimize a single separate projection module $f^M$ for each modality M"

→ XIB dùng **một Q-Former riêng cho mỗi modality** (image, 3D, audio, video Q-Formers tách biệt).

**Áp dụng cho SigLLM:** Hiện tại 1 Q-Former xử lý cả user_cf + target_cf + history_cf. **Có thể tách:**
- 1 Q-Former cho user
- 1 Q-Former cho item (target + history)

→ Mỗi Q-Former specialize cho role riêng. Trade-off: 2x params.

### Insight 5 — **Stage 2 (generative) matters but có thể skip** (Section 5.1)

Trích paper:
> "While X-InstructBLIP outperforms InstructBLIP on VizWiz there is a **mild drop in performance overall, likely due to the lack of BLIP2 Stage-2 finetuning**"

→ Skip Stage 2 = "mild drop" — không phải catastrophic. **Confirms `COMPREHENSIVE_AUDIT.md` Stage 1/2 contribution chưa quantify trên SigLLM** (đáng test).

### Insight 6 — **Prompt template diversity trade-off** (Section 5.1)

Trích paper:
> "the expanded template space which introduces a **trade-off of generalization and performance** as shown by the increased prompt robustness of X-InstructBLIP in the supplement."

→ Nhiều prompt templates = robust hơn nhưng accuracy có thể giảm nhẹ.

**SigLLM hiện tại có 3 LLM prompts + 12 Q-Former instructions.** Có thể balance lại.

### Insight 7 — **Cross-modal emergent từ separate training**

Trích paper:
> "discriminative cross-modal reasoning emerges naturally through individual modality alignment to LLMs"

→ Train từng modality riêng → khả năng cross-modal reasoning xuất hiện tự nhiên ở LLM.

**Áp dụng cho SigLLM:** User và item là 2 "modalities" trong rec sense. Train tách biệt + tin tưởng LLM combine ở prompt level.

---

## Đề xuất cải tiến — ranked theo ROI

### 🥇 Proposal 1 — **Modality prefix labels in prompt** (XIB Insight 1)

**Hypothesis:** Hiện tại Q-Former phải tốn capacity encode "user vs item vs history" via context. Add explicit prefix labels → free up bandwidth cho semantic content.

**Concrete change:** Prompt template từ:
```
A user has highly rated: <ItemTitleList>. The signals are: <ItemIDList>.
<UserID> Predict <TargetItemTitle>. <TargetItemID> Answer:
```

Thành (SellaRec/XIB style):
```
A user has given high ratings to the following movies. Given the user's
collaborative-filtering feature, the user's history (titles + signals), and
a target movie's title + collaborative-filtering feature, predict whether
the user would like the target movie.

User feature: <UserID>
History titles: <ItemTitleList>
History signals: <ItemIDList>
Target movie title: <TargetItemTitle>
Target movie feature: <TargetItemID>

Answer (Yes/No):
```

**Mechanism:** Each soft token slot có explicit text label trước nó → LLM biết rõ "user feature" vs "movie feature" → Q-Former không cần encode modality type.

**Evidence from XIB Table 7:**
- Audio task: with prefix > without prefix
- 3D task: with prefix > without prefix
- Cross-modal (MusicAVQA, VATEX): prefix needed for joint reasoning

**Effort:** **30 phút** (chỉ sửa file prompt) + 6h retrain Stage 3 Step 2.

**Probability win:** **30-40%** (XIB confirms direction, but rec is different task).

**Crosses with audit Section 5** (current prompt analysis).

**Files to modify:**
- `prompts/qformer_prompt_movie_with_user.txt` — đổi structure
- (Optional) `prompts/qformer_prompt_movie.txt` — variant không user

---

### 🥈 Proposal 2 — **MLP/Linear Projection baseline** (XIB Insight 2)

**Hypothesis:** Q-Former 76M params over-parameterized cho 33K samples → LP với fewer params có thể generalize tốt hơn.

**Concrete:** Implement 3 baselines song song với Q-Former:
1. **Linear Projection (LLaVA-style):** `Linear(256 → 8 * 3584)` reshape thành 8 soft tokens.
   - Params: ~7.3M
2. **MLP (CoLLM-style):** `Linear(256 → 2560) → GELU → Linear(2560 → 8 * 3584)`
   - Params: ~28M
3. **Q-Former (current baseline):** 76M params.

**Evidence from XIB:**
- Table 3 (Audio): LP win ESC50 close (67.4 > 62.8 Q-Former khi data limited)
- Table 6 (DisCRn A-V): LP win (47.1 > 34.0 Q-Former)
- Pattern: LP win khi modality data limited

**Áp dụng SigLLM:** 33K samples = limited data → LP có thể tied or win.

**Effort:** 2-3 ngày code (LP/MLP class + integrate Stage 3 trainer) + 12h retrain (2 baselines × 6h).

**Probability win (LP/MLP better than Q-Former):** **30-40%**.
**Probability win (LP/MLP tied = confirmation Q-Former overkill):** **30-40%**.
**Probability win (Q-Former still best):** **20-30%** — even outcome confirms Q-Former value (thesis defense).

**Crosses with audit Q10 Priority 1.**

**Files to create:**
- `src/sigllm/models/projection/linear_bridge.py` — LP baseline class
- `src/sigllm/models/projection/mlp_bridge.py` — MLP baseline class
- Trainer override `model.qformer_config.bridge_type=linear|mlp|qformer`

---

### 🥉 Proposal 3 — **Multi-token CF projection** (Audit bottleneck A + XIB image patches)

**Hypothesis:** Hiện tại cross-attention attend vào **1 token duy nhất** → softmax(scalar) = 1.0 → cross-attention trivializes. XIB dùng nhiều input tokens cho image (257 patches), video (multi-frame).

**Concrete:** Thay `proj_cf: Linear(256, 768)` thành multi-token projection:

```python
class MultiTokenCFProjection(nn.Module):
    def __init__(self, d_cf=256, d_model=768, num_cf_tokens=8):
        super().__init__()
        self.num_cf_tokens = num_cf_tokens
        # Project 256-d MF vec into M tokens, mỗi token là feature group
        self.proj = nn.Linear(d_cf, d_model * num_cf_tokens)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cf_vec):  # cf_vec: [B, 256]
        B = cf_vec.size(0)
        out = self.proj(cf_vec).reshape(B, self.num_cf_tokens, -1)  # [B, M, 768]
        out = self.norm(out)
        return out
```

Sau đó Q-Former cross-attend vào M tokens thay vì 1 → cross-attention thực sự non-trivial.

**Sweep M:** {2, 4, 8, 16}.

**Evidence:**
- Audit: `softmax over 1 element = 1` → trivial
- XIB: image Q-Former cross-attend vào 257 patches → meaningful attention
- ILM uses Q-Former trên 1 token CF cũng → result tied with CoLLM-MLP → suggests single-token bottleneck

**Effort:** **2 ngày code** + retrain Stage 1+2+3 (~3 ngày GPU).

**Probability win:** **40-50%** (strong theoretical motivation + cross-validated by XIB).

**Crosses with audit Q10 Priority 2.**

---

### Proposal 4 — **Separate Q-Formers for user vs item** (XIB Insight 4)

**Hypothesis:** User và item là 2 "modalities" trong rec context (semantic role khác biệt). XIB dùng Q-Former riêng cho mỗi modality. Thử same cho SigLLM.

**Concrete:**

```python
class QRecLLM(Rec2Base):
    def __init__(self, ...):
        self.qformer_item = HFQFormerAdapter(...)   # item path
        self.qformer_user = HFQFormerAdapter(...)   # user path
        # Share queries? Or separate?
```

**Trade-off:**
- ✓ Each Q-Former specializes
- ✗ 2x params (76M → 152M trainable)
- ✗ Need Stage 1 train both Q-Formers

**Variant nhẹ hơn:** **Shared Q-Former + role embedding** — add learnable role vector cho user vs item:
```python
self.role_user = nn.Parameter(torch.randn(1, 1, d_model))
self.role_item = nn.Parameter(torch.randn(1, 1, d_model))
# Add role to cf before Q-Former
cf_with_role = proj_cf(cf) + self.role_user  # or role_item
```
Fewer params (~768 each role), shared Q-Former. **Hybrid approach.**

**Effort:**
- Full separation: 3 ngày code + ~5 ngày Stage 1+2+3 retrain
- Role embedding: 1 ngày code + 6h retrain Step 2

**Probability win:** 15-25% (XIB validates per-modality Q-Former, nhưng SigLLM single-task khác)

---

### Proposal 5 — **Stage 1/2 ablation** (XIB Insight 5)

**Hypothesis:** XIB note "skip Stage 2 = mild drop". SigLLM chưa quantify Stage 1+2 contribution → đáng làm.

**Concrete experiments:**

| Variant | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| A | ✓ 5-loss | ✓ generative | ✓ 2-step (baseline) |
| B | ✗ random Q-Former | ✗ | ✓ 2-step |
| C | ✓ 5-loss | ✗ skip | ✓ 2-step |
| D | ✓ ITC only | ✓ | ✓ 2-step |
| E | ✓ 5-loss but w_ui=0 | ✓ | ✓ 2-step |
| F | ✓ 5-loss but w_ii=0 | ✓ | ✓ 2-step |

**Effort:** Variant B chỉ cần skip 2 stages → fastest. Mỗi variant ~3h Stage 3 step 2.

**Probability informative:** **95%** (always gives data point cho thesis).

**Crosses with audit Section 9.**

---

### Proposal 6 — **Unified Stage 3 + soft tokens in Step 1** (Existing proposal, XIB indirect)

XIB train LoRA + Q-Former + projection **đồng thời** (Section 3 "Optimize a single separate projection module"):
> "Train projection $f^M$ on instruction tuning data D^M"

Không có Step 1/Step 2 tách. CoLLM 2-step là **SigLLM-specific** convention.

**Hypothesis:** XIB unified approach work — SigLLM 2-step có thể suboptimal vì LoRA chưa từng thấy soft tokens.

**Already in `QFORMER_IMPROVEMENT_DIRECTIONS.md` Hypothesis B.** Combine với XIB evidence → confirm worth testing.

**Effort:** 1 day code + 6h retrain.

**Probability win:** **30-40%**.

---

### Proposal 8 — **Reduce Q-Former capacity** (audit overlooked direction)

**Hypothesis:** Q-Former 76M params / 33K samples = ratio 2300:1, cực over-parameterized. Direction "8 layers" đã tested → regress. Direction "smaller" CHƯA tested. XIB Insight 2 hỗ trợ: LP (essentially 0 layers) win khi data limited.

**Concrete sweep:**

| Variant | num_layers | num_queries | qformer_d_model | Approx params | Hypothesis |
|---|---|---|---|---|---|
| Current | 4 | 8 | 768 | ~76M | Baseline |
| **A: shallow** | **2** | 8 | 768 | ~38M | Half-depth, keep BERT layer 0-1 |
| **B: narrow queries** | 4 | **4** | 768 | ~75M | Tighter bottleneck (small param savings) |
| **C: narrow hidden** | 4 | 8 | **384** | ~19M | Big savings, lose BERT init |
| **D: combined** | 2 | 4 | 384 | ~10M | Approach LP territory |
| **E: extreme** | 1 | 4 | 384 | ~5M | Near-LP |

**Connection với 8-layer regression** (code comment `qformer_rec_llm.py:320`):
- 8 layers: ~0.694 uAUC (overfit)
- **4 layers (current): 0.7081**
- 2 layers: ??? untested
- 1 layer: ??? untested
- LP (0 layers via Proposal #2): ??? untested

→ Sweep 1/2/4/8 = **clean depth ablation curve** cho thesis.

**Evidence from XIB:**
- LP win audio task ESC50 (67.4 vs 62.8 Q-Former) — LP = no transformer = 0 layers
- Confirm "less is more" khi data limited

**Effort:** Config-only change + Stage 1+2+3 retrain (Q-Former weights are layer/dim-specific).

| Variant | GPU time per variant |
|---|---|
| A (2 layers) | ~10h (Stage 1+2 = 4h, Stage 3 = 6h) |
| B (4 queries) | ~10h |
| C (d=384) | ~10h |
| D, E | ~10h each |

**Total sweep:** 4 variants ≈ 40h GPU.

**Probability win:**
- Single variant: 20-30%
- **Probability ≥ 1 variant win**: 60-70%
- **Probability of clean depth curve** (informative regardless): 95%

**Crosses with audit Section 13 bottleneck E** (param/data ratio extreme).

---

### Proposal 7 — **Expand prompt template diversity** (XIB Insight 6)

**Hypothesis:** XIB note diverse prompts → robust hơn. SigLLM hiện 3 LLM prompts (low diversity).

**Concrete:**
- Tạo 8-12 prompt templates với structure variations (different orderings, different framings, CoT-style, structured-list-style, etc.)
- Update `prompts/qformer_prompt_movie_with_user.txt`

**Trade-off (theo XIB):** Generalization ↑, peak accuracy có thể ↓ nhẹ. Đáng test cho **robustness story** trong thesis.

**Effort:** 1h code + retrain.
**Probability marginal win:** 10-15%.

---

## Combined recommendations ranked

| Rank | Proposal | Time | Probability win | XIB evidence | Audit evidence |
|---|---|---|---|---|---|
| **1** | **#1 Modality prefix prompt** | **0.5 ngày** | **30-40%** | **Strong (Table 7)** | Confirms current prompt analysis |
| **2** | **#3 Multi-token CF projection** | 2-3 ngày | 40-50% | **Strong (image patches)** | **Top priority bottleneck** |
| **3** | **#2 LP/MLP baseline** | 2-3 ngày | 30-40% (any way win) | **Strong (Table 3,6)** | **Priority 1 audit** |
| **4** | **#8 Q-Former shrink sweep** | 1-3 ngày (per variant) | 20-30% per variant / 60-70% ≥1 | **Strong (Table 3)** | **Param/data ratio extreme** |
| 5 | #5 Stage 1/2 ablation (variant B random Q-Former) | 1 ngày | n/a (informative) | Validates direction | Quantifies Section 9 |
| 6 | #6 Unified Stage 3 | 1 ngày | 30-40% | Indirect | Existing proposal |
| 7 | #4 Role embedding (hybrid variant) | 1 ngày | 15-25% | Moderate | Novel direction |
| 8 | #7 Expanded prompts | 0.5 ngày | 10-15% | Indirect | Engineering polish |

---

## Combo proposal — **"X-InstructBLIP-style upgrade"**

**Combine 3 highest-ROI:**

### Combo A — "Quick wins" (~3 ngày total)
- **Day 1:** #1 Modality prefix prompt + #7 expanded templates
- **Day 2-3:** Retrain Stage 3 Step 2 với prompt mới
- **Outcome:** Single retrain, 2 changes stack
- **Probability ≥ 1 improvement:** ~50-55%

### Combo B — "Architecture overhaul" (~7 ngày)
- **Day 1-3:** #3 Multi-token CF projection + #2 LP baseline (parallel implement)
- **Day 4-5:** Retrain Stage 1+2+3 cho multi-token CF
- **Day 6:** Retrain Stage 3 only cho LP baseline
- **Day 7:** Compare 3 architectures (vanilla, multi-token CF, LP)
- **Outcome:** Major architectural ablation cho thesis
- **Probability ≥ 1 architecture win:** ~60-70%

### Combo C — "Stage-level study" (~3 ngày)
- **Day 1-2:** #5 Stage 1/2 ablations (3 variants: random, no-Stage2, ITC-only)
- **Day 3:** Compare + write up
- **Outcome:** Quantify Stage 1/2 contribution definitively
- **Probability informative:** 95%

---

## RECOMMENDATION cụ thể

### Nếu còn ≥ 2 tuần trước deadline:

**Pick Combo A + Combo B sequentially:**
1. Tuần 1: Combo A (quick wins) — confirm hoặc loại prompt redesign
2. Tuần 2: Combo B (architecture ablation) — main thesis contribution

### Nếu còn < 2 tuần:

**Pick Combo A only:**
- Cheapest, highest ROI ratio
- Single retrain
- Direct evidence from XIB

### Nếu còn < 1 tuần:

**Pick Proposal #1 ONLY:**
- 0.5 ngày code
- 6h retrain
- Direct apply XIB Table 7 finding
- Single binary test: prefix vs no-prefix

---

## Thesis defense narrative cải thiện

Với các experiments này, có thể frame thesis:

**Hiện tại (limited story):**
> "SigLLM dùng Q-Former bridge từ ILM/BLIP-2, đạt 0.7170 uAUC trên ML-1M, tied với CoLLM-MF. 4 architectural extensions không transfer (negative findings). User soft tokens cải thiện cold-start (+1.79 uAUC)."

**Sau Combo B (stronger story):**
> "We adopt the X-InstructBLIP framework's projection comparison methodology to systematically evaluate bridge architectures for CF→LLM. On ML-1M (33K samples, low-resource regime):
> - **Q-Former (76M params)**: 0.7170 uAUC
> - **MLP bridge (28M)**: [X] uAUC
> - **Linear projection (7M)**: [Y] uAUC
> - **Multi-token CF Q-Former (M=4)**: [Z] uAUC
>
> Findings align with X-InstructBLIP's observation that Q-Former excels with sufficient data, while linear projections offer better generalization at low-data regimes. We further demonstrate that single-token CF cross-attention is a architectural bottleneck — multi-token projection lifts performance by Δ..."

→ **Much stronger thesis defense:** systematic study, references SOTA paper (X-InstructBLIP ECCV 2024), addresses examiner challenges ("did you compare with MLP?", "did you test multi-token?").

---

## Action items cụ thể (TODO list)

### Phase 1 — Implement (in priority order)

1. **#1 Modality prefix prompt** — 30 phút
   - File: `prompts/qformer_prompt_movie_with_user.v2.txt` (new, SellaRec/XIB style)
   - Test: Stage 3 Step 2 với `prompt_path=...v2.txt`

2. **#3 Multi-token CF projection** — 2 ngày
   - File: `src/sigllm/models/q_former/hf_qformer_adapter.py` line 81 — modify `proj_cf`
   - Add config flag `qformer_config.num_cf_tokens: int (default=1)`
   - Need Stage 1+2 retrain (Q-Former forward changes)

3. **#2 LP/MLP baseline** — 3 ngày
   - File: `src/sigllm/models/projection/linear_bridge.py` (new)
   - File: `src/sigllm/models/projection/mlp_bridge.py` (new)
   - File: `src/sigllm/models/multimodal/qformer_rec_llm.py` — add `bridge_type` selector
   - Need Stage 1 (or skip — LP doesn't need pretraining) + Stage 3 retrain

### Phase 2 — Run experiments

| Day | Experiment | GPU time |
|---|---|---|
| 1 | #1 modality prefix retrain Stage 3 Step 2 | ~6h |
| 2-3 | #3 multi-token CF M=4: Stage 1 (4h) + Stage 2 (3h) + Stage 3 (6h) | ~13h |
| 4 | #3 multi-token CF M=8: full retrain | ~13h |
| 5-6 | #2 LP baseline: Stage 3 only (no pretraining) | ~6h × 2 (LP + MLP) = 12h |
| 7 | Eval all on test/warm/cold + write up | — |

**Total:** ~50h GPU time, 7 days wall-clock.

### Phase 3 — Document

- Update `ABLATION_RESULTS.md` với numbers từ 3-4 mới experiments
- Update thesis narrative với X-InstructBLIP framing
- Update slides cho final defense

---

## Honest disclaimer

XIB findings từ **multi-modality multi-task setting** (4 modalities, multiple QA tasks). SigLLM = **single-modality single-task** (CF, CTR Yes/No). Direct transfer của findings **có giới hạn**:
- ✓ Modality prefix should still help (basic LLM understanding pattern)
- ⚠️ LP vs Q-Former trade-off applies but ML-1M might be "enough" data trong category
- ⚠️ Per-modality Q-Former không trực tiếp apply (single modality)

→ **Pick experiments có nhiều base evidence overlap:** Proposal #1 (prefix), #3 (multi-token CF), #2 (LP baseline) đều có **dual evidence** (XIB + audit). Confidence cao hơn.
