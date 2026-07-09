# Q-Former Improvement Directions — Consolidated Roadmap

**Date:** 2026-05-31
**Source:** ChatGPT proposal (review by user) + SigLLM ablation evidence + earlier code audit findings
**Thesis title constraint:** "Căn chỉnh thông tin cộng tác vào hệ tư vấn sử dụng mô hình ngôn ngữ lớn"
**Direction:** Q-Former + alignment side. **KHÔNG touch LLM prompt structure / LLM model.**

---

## Context (verified từ ablation evidence)

- ✓ Q-Former vanilla có đóng góp rõ: soft tokens +4.6 đến +5.15 uAUC vs text-only / CF-zeroed (test_warm split)
- ❌ InstructBLIP-style instruction-aware routing **không có gain** trong single-task binary CTR (|Δ uAUC| < 0.005)
- ❌ 4 architectural extensions đã thử (instruction V1, task-focused, interaction-aware, user-conditioned) đều không break ceiling 0.71
- ⚠️ Audit phát hiện `<UserID>` soft-token slot đang **TEMP_DISABLED** — user side CF không inject vào LLM
- 📊 Stage 3 Step 1 peak uAUC = 0.6917, Stage 3 Step 2 peak = 0.7098 → soft tokens chỉ contribute +0.018 in Step 2 sau full pipeline → **soft tokens đang underutilized**

**Goal:** giữ Q-Former như collaborative-to-language bridge. Focus vào:
- Personalization
- Semantic alignment preservation
- Bandwidth preservation
- Bridge quality (không phải instruction routing)

---

## ChatGPT's 8 proposals — evaluated

### ✓ #1 — Partial tuning / projection-only baseline

**Proposal:** Test (a) freeze Q-Former, train projection only; (b) unfreeze last 1-2 Q-Former layers + projection. Goal: check full Q-Former tune có overfit / representation drift không.

**Evaluation:** **AGREE — diagnostic giá trị cao, effort thấp.**

**Lý do:**
- Hypothesis valid: 76M params / 33k samples = ratio 2300:1 → overfit risk
- Training trajectory cho thấy peak epoch 3-5 rồi drift → **đáng test xem có phải drift là do Q-Former over-tune không**
- Nếu projection-only / partial tune cho cùng uAUC → confirm Q-Former không cần full tune → có thể là "negative finding" mạnh ("Q-Former 76M overkill, projection 2.7M đủ")
- Nếu partial tune **tốt hơn** full tune → fix real issue

**Effort:** 0.5 ngày (chỉ đổi `requires_grad` flag) + 2 retrain (~6h mỗi cái).
**Probability success:** 15-25%.
**Defensibility:** **Very high** — diagnostic ablation luôn defensible.

---

### ✓ #2 — Semantic alignment loss (aux loss at Stage 3)

**Proposal:** Thêm aux loss align Q-Former output với text encoder của item:
```
L_total = L_yesno + λ_align * L_align
L_align = InfoNCE(QFormer(item_cf), TextEncoder("Title: X. Genres: Y"))
λ_align = 0.05, tau = 0.07
```

**Evaluation:** **STRONGLY AGREE — addresses Stage 1→Stage 3 drift, không touch LLM.**

**Lý do:**
- Stage 1 ITC loss đã align Q-Former với text. Stage 3 CTR-only loss có thể drift Q-Former khỏi alignment
- Aux InfoNCE preserve alignment → soft tokens nằm gần "language space" mà LLM hiểu
- Standard distillation/preservation pattern
- 100% Q-Former side, không touch LLM

**Implementation:**
```python
# Stage 3 step 2 forward
target_q = self.qformer(target_cf, ins_list)        # [B, Q, d]
target_text_emb = self.text_encoder("Title: ... Genres: ...")  # frozen text encoder

loss_ctr = main_ctr_loss(logits, labels)
loss_align = info_nce(target_q.mean(1), target_text_emb, tau=0.07)
loss = loss_ctr + 0.05 * loss_align
```

**Effort:** 1 ngày code (cần load frozen text encoder song song) + 3-6h retrain.
**Probability success:** 20-30%.
**Defensibility:** **High** — distillation pattern proven trong VLM literature.

---

### ✓ #3 — Target-conditioned gating over per-item tokens

**Proposal:** Giữ per-item design (88 soft tokens) nhưng thêm gate để target reweight history item soft tokens. Tránh bandwidth bottleneck của interaction-aware.

**Evaluation:** **STRONGLY AGREE — addresses bandwidth issue mình đã raise trong audit, novel.**

**Lý do:**
- Interaction-aware fail vì collapse 88 tokens xuống 8 → bandwidth bottleneck
- Target-conditioned gating giữ **88 tokens** nhưng cho target attend đến history
- Mỗi history item được reweight theo relevance với target
- Genuinely novel — chưa thấy trong ILM / CoLLM / SellaRec

**Implementation sketch:**
```python
# After Q-Former forward
target_q  = self.qformer(target_cf)            # [B, Q, d]
history_q = self.qformer(history_cf_flat)      # [B*L, Q, d] -> [B, L, Q, d]

# Compute gate: target attends to each history item
gate = softmax(target_q @ history_q.mean(2).transpose(-1, -2), dim=-1)  # [B, Q, L]

# Reweight history tokens
history_q_weighted = (gate.unsqueeze(-1) * history_q.unsqueeze(1)).sum(dim=2)  # [B, Q, L?, d]
# Or simpler: scale each history item's tokens by its gate
history_q_gated = history_q * gate.transpose(1, 2).unsqueeze(-1)  # [B, L, Q, d]
```

**Effort:** 1-2 ngày code + 6h retrain.
**Probability success:** 20-25%.
**Defensibility:** **High** — combines benefits of interaction-aware (target-history interaction) with vanilla (bandwidth preserved).

---

### ⚠️ #4 — User-conditioned queries (bản nhẹ với α=0.03-0.1)

**Proposal:** `query = global_query + α * user_query_proj(user_cf)`, α init 0.03-0.1.

**Evaluation:** **ALREADY TESTED, FAILED.**

**Đã làm trên branch `feat/user-conditioned-queries`:**
- Implementation: `query_tokens = self.q.expand(B,-1,-1) + user_proj(user_cf).unsqueeze(1)`
- Zero-init `user_proj.weight/bias` (≈ α start at 0, grow with gradient)
- Result: peak val_uAUC = **0.7060 < baseline 0.7098** sau 7 epoch, killed

**Sự khác biệt với ChatGPT proposal:** ChatGPT đề xuất fixed α=0.03-0.1 (explicit scalar). Mình dùng zero-init (α effective start at 0, free to grow).

**Re-test với fixed α?** Có thể có khác nhau nhỏ:
- Zero-init: user_proj có thể stuck ở magnitude nhỏ nếu gradient yếu
- Fixed α=0.1: bias mạnh hơn từ ngày đầu

**Worth re-testing?** ⚠️ Low priority. Đã có 1 negative finding, sub-variant có thể marginal gain.

**Effort nếu re-test:** 0.5 ngày code + 3h train. **Probability:** 10-15%.

---

### ⚠️ #5 — Query diversity / decorrelation loss

**Proposal:** Loss penalize off-diagonal cosine similarity giữa 8 query outputs:
```
L_div = sum of |cos(q_i, q_j)| for i ≠ j
L_total = L_yesno + λ_div * L_div, λ_div = 1e-4 to 1e-3
```

**Evaluation:** **DIAGNOSTIC FIRST.**

**Lý do cần diagnostic trước:**
- Hypothesis: 8 queries có thể collapse vào cùng direction → redundant
- Không biết có collapse thật sự không

**Diagnostic 30 phút:** Load baseline ckpt, eval 1 batch, compute pairwise cosine giữa 8 query outputs:
```python
qs = qformer(item_cf)  # [B, 8, d]
qs_norm = F.normalize(qs, dim=-1)
sim = qs_norm @ qs_norm.transpose(-1, -2)  # [B, 8, 8]
off_diag_sim = (sim - torch.eye(8)).abs().mean()
print(f"Avg off-diag cos sim: {off_diag_sim:.4f}")
# Near 0 → diverse, no collapse → diversity loss won't help
# Near 0.5+ → some collapse → diversity loss could help
```

**Nếu collapse:** Add diversity loss, probability win ~15-20%.
**Nếu diverse:** Skip — không cần loss này.

**Effort total:** 1 ngày (diagnostic + implement + retrain).

---

### ⚠️ #6 — Query grouping / semantic-role queries

**Proposal:** Chia 8 queries thành semantic groups (item/CF/preference/decision), train với role-specific loss.

**Evaluation:** **MEDIUM PRIORITY — interesting research direction nhưng effort cao.**

**Lý do:**
- Cần thiết kế per-group loss
- Mỗi group cần ground truth (e.g. genre label cho semantic queries)
- Research extension, không phải minimal change

**Effort:** 3-4 ngày (cần aux labels từ metadata) + retrain.
**Probability success:** 15-25%.
**Defensibility:** **High novelty** — publishable nếu work.

**Verdict:** Skip cho thesis thời gian gấp. Đáng làm nếu có thêm 1-2 tuần.

---

### 🥇 #7 — User-level soft tokens (re-enable `<UserID>` slot)

**Proposal:** Thêm 2-4 soft tokens từ user_cf trực tiếp vào prompt.

**Evaluation:** **STRONGLY AGREE — top priority. Aligned with audit finding + thesis title.**

**Lý do (đây là khám phá lớn từ code audit):**
- Code hiện tại có `# TEMP_DISABLED_USER_CF` (line 60-62, 598-606, 646-651 của `qformer_rec_llm.py`)
- Pipeline original có `<UserID>` slot trong prompt — bị tắt
- **Thesis title nói "căn chỉnh thông tin cộng tác"** nhưng user-side collaborative info HIỆN KHÔNG đi qua LLM
- Re-enable này = **làm code match đúng idea của thesis**

**Implementation:** Uncomment ~5 dòng:
```python
# qformer_rec_llm.py:60-62
PLACEHOLDERS_FOR_EMBED = ["<UserID>", "<ItemIDList>", "<TargetItemID>"]   # re-add UserID

# qformer_rec_llm.py:598-606 (in forward)
user_cf = self.rec_encoder.user_encoder(batch_data["UserID"])    # [B, d_cf]
user_q = self.qformer(user_cf, ins_list)                          # [B, Q, d_model]
user_llm = self.llm_proj(user_q)                                  # [B, Q, H]

# qformer_rec_llm.py:646-651 (in merge)
ph2emb = {
    "<UserID>": user_llm,                  # re-add
    "<ItemIDList>": interacted_llm_flat,
    "<TargetItemID>": target_llm
}
```

Cần thêm `<UserID>` slot trong prompt template:
```text
"This user (described by <UserID>) has highly rated: <ItemTitleList><ItemIDList>. 
Will they like <TargetItemTitle><TargetItemID>? Answer:"
```

**Effort:** 1-2h code (uncomment) + 1h prompt update + 6h retrain Stage 3 step 2.
**Probability success:** **25-35%** (highest among realistic options).
**Defensibility:** **Highest** — fixes architectural mismatch, aligns code với thesis claim.

---

### ✓ #8 — MLP-CIE baseline (Bonus, không phải improvement)

**Proposal:** Implement CoLLM-style baseline: `CF emb → MLP → soft tokens → LLM`. So sánh xem Q-Former có thực sự hơn MLP đơn giản không.

**Evaluation:** **AGREE — essential cho thesis story.**

**Lý do:**
- Ablation hiện tại so Q-Former vs text-only → +4.6 uAUC
- Nhưng KHÔNG so Q-Former vs MLP bridge → không thể claim "Q-Former is the right architecture"
- CoLLM benchmark trong literature đạt 0.7179 uAUC (test overall) — close với Q-Former 0.7131
- Nếu MLP-CIE baseline của bạn cũng đạt ~0.71 → cần điều chỉnh thesis claim
- Nếu Q-Former > MLP → **strong positive finding cho thesis**

**Implementation:** Thay `self.qformer = HFQFormerAdapter(...)` bằng:
```python
self.cf_mlp = nn.Sequential(
    nn.Linear(d_cf, d_cf * 10),       # intermediate 10x (CoLLM style)
    nn.GELU(),
    nn.Linear(d_cf * 10, d_model * Q), # output Q soft tokens
)
# In forward: soft_tokens = self.cf_mlp(cf).reshape(B, Q, d_model)
```

**Effort:** 1 ngày code + 3-6h train.
**Defensibility:** **Essential** — strengthens thesis story regardless of outcome.

---

## Re-prioritized roadmap

ChatGPT's priority (1-7) là hợp lý nhưng dưới góc nhìn của mình + evidence audit:

| Mình rank | Item | ChatGPT rank | Effort | Prob | Defensibility |
|---|---|---|---|---|---|
| **🥇 1** | **#7 — User-level soft tokens (re-enable `<UserID>`)** | 6 | **2-3h code + 6h train** | **25-35%** | **Highest** (fixes audit gap) |
| **🥈 2** | **#3 — Target-conditioned gating** | 3 | 1-2 ngày | 20-25% | High novelty |
| **🥉 3** | **#2 — Semantic alignment aux loss** | 2 | 1 ngày | 20-30% | High |
| 4 | #1 — Partial tuning diagnostic | 1 | 0.5 ngày | 15-25% | Diagnostic |
| 5 | #8 — MLP-CIE baseline | 7 | 1 ngày | n/a (baseline) | Essential |
| 6 | #5 — Query diversity (diag first) | 5 | 1 ngày | 10-20% | Conditional |
| ❌ Skip | #4 — User-conditioned queries | 4 | — | Already tested, failed | — |
| ❌ Defer | #6 — Query grouping | (not numbered) | 3-4 ngày | 15-25% | High but high effort |

### Lý do re-prioritize

**#7 lên #1:**
- Audit phát hiện `<UserID>` disabled — đây là **mismatch giữa code và thesis title**
- Re-enable là **minimal change** (~5 lines)
- Aligned 100% với thesis: "căn chỉnh thông tin cộng tác (user-side included) vào LLM"
- Untested → fresh signal

**#3 và #2 cũng cao:**
- #3 fix bandwidth bottleneck của interaction-aware (mình raise issue, ChatGPT propose fix)
- #2 prevent Stage 3 drift, semantic preservation

**#1 (partial tuning) là diagnostic, không phải improvement:**
- Worth running nhưng không phải main bet
- ChatGPT prioritize cao có lẽ vì cheap, nhưng outcome chủ yếu confirmatory

---

## Recommended execution plan

### Phase 1 — Quick wins (3 ngày)

**Day 1:** Re-enable user soft tokens (#7) — minimal code change
- Uncomment TEMP_DISABLED_USER_CF block
- Update prompt template với `<UserID>` slot
- Retrain Stage 3 step 2 only (~6h)
- Compare val_uAUC vs baseline 0.7098

**Day 2-3:** Phụ thuộc kết quả #7:
- Nếu #7 win → consolidate, viết thesis section
- Nếu #7 không win → chạy #2 (semantic alignment loss) hoặc #3 (target-conditioned gating)

### Phase 2 — Diagnostic + bonus (2 ngày, if time)

**Day 4:** Partial tuning sweep (#1) + query diversity diagnostic (#5)
- Cả 2 đều cheap diagnostic
- Cho data point cho thesis discussion

**Day 5:** MLP-CIE baseline (#8)
- Essential cho thesis claim "Q-Former is the right bridge"

### Phase 3 — Optional research extension (3+ ngày)

**Nếu Phase 1+2 win + còn thời gian:**
- #3 target-conditioned gating (novel contribution)
- #6 query grouping (high-novelty research)

### Skip entirely

- #4 user-conditioned queries — đã test, fail
- LLM prompt redesign (CoT, scaffolded) — out of scope per user constraint

---

## Updated core thesis claim

**Per ChatGPT (slightly modified):**

> Q-Former is effective as a CF-to-LLM bridge for recommendation, contributing +5.15 uAUC over text-only baselines on ML-1M warm split. However, InstructBLIP-style instruction-aware routing does NOT transfer to single-task binary CTR recommendation (|Δ uAUC| < 0.005 across two prompt families tested). Future improvements should focus on **personalized conditioning** (re-enabling user-side soft tokens, target-conditioned gating over per-item soft tokens), **semantic alignment preservation** (auxiliary alignment loss at fine-tuning stage), and **collaborative bandwidth preservation** (avoiding multi-element memory compression that collapses N×Q tokens to Q tokens).

---

## File references

- Existing ablation: `docs/ABLATION_RESULTS.md`
- Earlier user-conditioned proposal: branch `feat/user-conditioned-queries`
- Earlier init proposal: `docs/QFORMER_INIT_IMPROVEMENTS.md` (orthogonal — could combine với items here)
- Code audit findings: TEMP_DISABLED_USER_CF in `src/sigllm/models/multimodal/qformer_rec_llm.py` lines 60-62, 598-606, 646-651
- Q-Former adapter: `src/sigllm/models/q_former/hf_qformer_adapter.py`
- Stage 3 step 2 trainer: `src/sigllm/pipelines/multimodal/train_qformer_stage3_step2_cie.py`

---

## Decision matrix — what to do NOW

| Time available | Action |
|---|---|
| < 2 ngày | Chỉ #7 (re-enable user soft tokens) |
| 3-5 ngày | #7 → #2 hoặc #3 → #1 diagnostic |
| 1 tuần | Phase 1 + Phase 2 đầy đủ |
| > 1 tuần | Phase 1 + 2 + #3 (gating) + #6 (grouping) |

**Recommended first action:** **Implement #7 (re-enable user soft tokens)** — 2-3 hours code, addresses audit finding, aligned với thesis title, untested, lowest effort + highest defensibility.

---

## Execution log — branch strategy

### 2026-05-31 — Setup branch cho #7

**Decision:** Checkout về stable baseline `feat/best-0.72-warm` rồi tạo nhánh mới `feat/reenable-user-soft-tokens` để implement #7.

**Lý do branch từ `feat/best-0.72-warm` (KHÔNG phải từ `feat/user-conditioned-queries`):**
- `feat/best-0.72-warm` là baseline ổn định (val_uAUC peak 0.7098)
- `feat/user-conditioned-queries` đã thêm `user_proj` layer cho query shift — orthogonal với #7 (re-enable user soft tokens). Nếu branch từ đây, 2 experiments bị conflate.
- #7 cần baseline cleanest để measure impact riêng của user soft tokens

**Branch chain:**
```
feat/best-0.72-warm (stable baseline, val_uAUC 0.7098)
        ↓
feat/reenable-user-soft-tokens (NEW — implement #7)
```

**Files dự kiến touch (~5 dòng uncomment + prompt update):**
- `src/sigllm/models/multimodal/qformer_rec_llm.py`:
  - Line 60-62: re-add `<UserID>` to `PLACEHOLDERS_FOR_EMBED`
  - Line 598-606: uncomment `user_cf`, `user_q`, `user_llm`
  - Line 646-651: re-add `<UserID>: user_llm` vào `ph2emb`
  - Line 652+: re-add mask cho `<UserID>` slot
- `prompts/qformer_prompt_movie.txt`: thêm `<UserID>` slot vào prompt template
- `configs/config.yaml`: không cần đổi (PLACEHOLDERS quản lý ở model side)

**Training command dự kiến:**
```bash
PYTHONPATH=src python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie \
    --cfg-path configs/config.yaml \
    --options \
        run.qformer_stage3_step2.max_epoch=20 \
        run.qformer_stage3_step2.output_dir=/content/SigLLM/ckpt/qformer_stage3_step2_cie_userslot_qwen2/
```

**Verdict matrix:**

| val_uAUC peak | Verdict | Action |
|---|---|---|
| > 0.715 | **Win** — user-side CF matters | Document, consolidate thesis |
| 0.708 – 0.715 | Marginal | Try combine với #3 (gating) |
| 0.700 – 0.708 | Tie với baseline | Confirm user soft tokens không add value |
| < 0.700 | Regression | Investigate prompt structure / soft token interference |

---

## Section 2 — Top 4 proposals from X-InstructBLIP + Audit synthesis

**Source:** `docs/IMPROVEMENT_PROPOSALS_FROM_XIB.md` + `docs/COMPREHENSIVE_AUDIT.md`
**Date added:** 2026-06-03
**Context:** Sau khi đọc X-InstructBLIP (ECCV 2024) + comprehensive audit, identify 4 cải tiến cao priority có **dual evidence** (XIB paper + audit findings).

### 🥇 Proposal #1 — Modality prefix labels in prompt

**Hypothesis:** Soft token slots (`<UserID>`, `<TargetItemID>`) hiện đứng **trơ trọi** trong prompt → LLM phải tự đoán đâu là user vs item vs history. Q-Former tốn capacity encode "modality type" thay vì semantic content.

**Evidence dual:**
- **XIB Table 7:** Adding modality prefix ("audio:", "3d:") **consistently improve** audio + 3D tasks. Paper note: *"Q-Former is relieved from the extra burden to encode the type of modality and instead reserves bandwidth for semantic information"*.
- **Audit Section 5:** Current SigLLM prompt lacks structural labels. SellaRec (SOTA) explicitly labels: "The user's feature is X. The target movie's feature is Y."

**Concrete change:** Đổi prompt template từ inline soft tokens sang labeled blocks:

```text
# CURRENT
A user has highly rated: <ItemTitleList>. Signals: <ItemIDList>. <UserID>
Predict <TargetItemTitle>. <TargetItemID> Answer:

# PROPOSED (SellaRec/XIB style)
A user has given high ratings to the following movies. Given the user's
collaborative feature, the user's history, and a target movie's title +
collaborative feature, predict whether the user would like the target.

User feature: <UserID>
History titles: <ItemTitleList>
History signals: <ItemIDList>
Target title: <TargetItemTitle>
Target feature: <TargetItemID>

Answer (Yes/No):
```

**Effort:** 0.5 ngày — chỉ sửa file prompt + retrain Stage 3 Step 2 (~6h GPU).
**Probability win:** 30-40%. Best-case Δ uAUC: +0.01-0.02.
**Risk:** Low — fallback về prompt cũ dễ.
**Files to modify:** `prompts/qformer_prompt_movie_with_user.txt`

---

### 🥈 Proposal #2 — Multi-token CF projection (M > 1)

**Hypothesis:** Hiện tại cross-attention attend vào **CHỈ 1 token** (projected CF). `softmax(1 element) = 1.0` luôn → cross-attention TRIVIAL. Q-Former lose key VLM mechanism.

**Evidence dual:**
- **XIB:** Image Q-Former cross-attend vào **257 patches**, video vào multi-frame tokens. Meaningful attention distribution là nguyên nhân Q-Former hoạt động trong VLM.
- **Audit Section 13 bottleneck A:** Marked as HIGH probability bottleneck.

**Concrete change:** Modify `proj_cf` để project MF vec thành M tokens thay vì 1:

```python
class MultiTokenCFProjection(nn.Module):
    def __init__(self, d_cf=256, d_model=768, num_cf_tokens=4):
        super().__init__()
        self.num_cf_tokens = num_cf_tokens
        self.proj = nn.Linear(d_cf, d_model * num_cf_tokens)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, cf_vec):  # [B, 256]
        B = cf_vec.size(0)
        out = self.proj(cf_vec).reshape(B, self.num_cf_tokens, -1)  # [B, M, 768]
        return self.norm(out)
```

**Sweep:** M = {2, 4, 8, 16}.

**Effort:** 2-3 ngày code + Stage 1+2+3 retrain (~3 ngày GPU per M variant).
**Probability win:** 40-50% (strong theoretical motivation).
**Risk:** Medium — Stage 1 alignment có thể behave differently với multi-token CF.
**Files to modify:**
- `src/sigllm/models/q_former/hf_qformer_adapter.py` line 81 (replace `proj_cf`)
- `configs/config.yaml` add `qformer_config.num_cf_tokens: int (default=1)`

---

### 🥉 Proposal #3 — LP / MLP bridge baseline (no Q-Former)

**Hypothesis:** Q-Former 76M params over-parameterized cho 33K samples. Simpler bridge (Linear hoặc MLP) có thể competitive. Quan trọng cho thesis: **confirm Q-Former giá trị thực sự** vs simpler alternative.

**Evidence dual:**
- **XIB Table 3, 6:** LP win audio (ESC50 close 67.4 vs Q-Former 62.8) khi data limited. Pattern: LP better generalization low-resource.
- **Audit Q5, Q6:** Cannot conclude Q-Former value without internal MLP baseline. Major gap cho thesis defense.

**Concrete change:** Implement 2 bridge alternatives song song với Q-Former:

**A. Linear Projection (LLaVA-style):**
```python
class LinearBridge(nn.Module):
    def __init__(self, d_cf=256, d_llm=3584, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Linear(d_cf, d_llm * num_tokens)
    def forward(self, cf):
        return self.proj(cf).reshape(cf.size(0), self.num_tokens, -1)
# Params: 256 × (3584×8) ≈ 7.3M
```

**B. MLP (CoLLM-style, 10x intermediate):**
```python
class MLPBridge(nn.Module):
    def __init__(self, d_cf=256, d_llm=3584, num_tokens=8, hidden_mult=10):
        super().__init__()
        h = d_cf * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_cf, h), nn.GELU(),
            nn.Linear(h, d_llm * num_tokens)
        )
        self.num_tokens = num_tokens
    def forward(self, cf):
        return self.net(cf).reshape(cf.size(0), self.num_tokens, -1)
# Params: 256×2560 + 2560×(3584×8) ≈ 73M (large) or với intermediate 2x = ~14M
```

**Effort:** 2-3 ngày code + Stage 3 only (no pretraining cần) (~6h GPU per variant).
**Probability "win" (LP/MLP > Q-Former):** 30-40%.
**Probability "tie" (LP/MLP ≈ Q-Former):** 30-40% — **STILL CONTRIBUTION** ("Q-Former overkill" finding).
**Probability "Q-Former wins":** 20-30% — **CONFIRMS Q-Former value** cho thesis.

→ **All 3 outcomes are publishable contributions.**

**Files to create:**
- `src/sigllm/models/projection/linear_bridge.py`
- `src/sigllm/models/projection/mlp_bridge.py`
- Modify `qformer_rec_llm.py` để add `bridge_type` selector.

---

### 🏅 Proposal #4 — Q-Former shrink sweep (this branch!)

**Hypothesis:** 76M params / 33K samples = 2300:1 over-parameterized. 8-layer Q-Former đã REGRESS (uAUC ~0.694 per code comment). Direction "smaller" CHƯA test.

**Evidence dual:**
- **XIB Insight 2:** LP (0 layers, ~7M params) competitive với Q-Former khi data limited.
- **Audit Section 13:** Param/data ratio extreme. CoLLM-MF (~17M params MLP) tied SigLLM Q-Former — confirms "less may be enough".

**Sweep table:**

| Variant | num_layers | num_queries | num_heads | d_model | Params | Branch / Status |
|---|---|---|---|---|---|---|
| Baseline | 4 | 8 | 8 | 768 | 76M | `feat/best-0.72-warm` ✓ |
| **A: This sweep** | **2** | **4** | **12 (BERT match)** | 768 | **50M** | `feat/qformer-shrink-2L-4Q` ✓ ready |
| B: narrower hidden | 4 | 8 | 12 | 384 | ~19M | Future sweep |
| C: minimal | 1 | 4 | 12 | 384 | ~5M | Future sweep, near-LP territory |

**Status:** ✓ Variant A config committed (2 commits: shrink + heads→12 for BERT compat).

**Connection:** Sweep `num_layers: 1, 2, 4, 8` cùng với LP (0 layers) → **clean depth ablation curve** cho thesis.

**Effort per variant:** ~13h GPU (Stage 1: 4h + Stage 2: 3h + reuse Step 1 LoRA + Stage 3 Step 2: 6h).
**Probability single variant win:** 25-30%.
**Probability ≥1 variant win:** 60-70%.
**Probability informative (clean curve regardless):** 95%.

**Critical insight về `num_heads`:** Q/K/V weights là `[768, 768]` regardless of heads — **heads chỉ là reshape view**, không add params. Để BERT init faithful, **MUST match BERT's 12 heads** (head_dim=64). 8 heads (current baseline) thực ra đã suboptimal cho BERT transfer.

**Files modified:**
- `configs/config.yaml` — qformer_config block + 4 output_dirs (avoid baseline overlap)

---

## Combined recommendation — ranking by ROI

| Rank | Proposal | Effort | Probability win | XIB evidence | Audit evidence |
|---|---|---|---|---|---|
| **1** | **#1 Modality prefix prompt** | **0.5 ngày** | **30-40%** | **Strong (Table 7)** | Section 5 prompt analysis |
| **2** | **#3 LP/MLP baseline** | 2-3 ngày | 30-40% (any outcome informative) | **Strong (Table 3, 6)** | **Priority 1** (foundational gap) |
| **3** | **#2 Multi-token CF projection** | 2-3 ngày | 40-50% | **Strong (image patches)** | **Bottleneck A** (single-token trivial) |
| **4** | **#4 Q-Former shrink** | 1-3 ngày/variant | 25-30% per / 60-70% ≥1 | **Strong (Table 3)** | **Param/data ratio** extreme |

## Implementation status (2026-06-03)

| Proposal | Status |
|---|---|
| #1 Modality prefix | ✗ Not started |
| #2 Multi-token CF | ✗ Not started |
| #3 LP/MLP baseline | ✗ Not started |
| **#4 Shrink (variant A)** | **✓ Config committed on `feat/qformer-shrink-2L-4Q`** |

## Execution plan tùy thời gian

| Time available | Strategy |
|---|---|
| < 1 week | #1 only (0.5d) — cheapest, fastest direct evidence |
| 1-2 weeks | #4 (already ready) + #1 sequential |
| 2-3 weeks | **Combo:** #4 + #1 + #3 LP baseline — covers shrink + prompt + foundational gap |
| > 3 weeks | All 4 proposals — **comprehensive architecture study** matches XIB methodology |

→ Thesis defense vững nhất với **all 4 proposals tested**. Pre-empt mọi examiner challenge.
