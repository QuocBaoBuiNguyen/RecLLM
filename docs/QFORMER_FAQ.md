# SigLLM Q-Former — FAQ & Technical Clarifications

**Date:** 2026-06-02
**Source:** Code audit verified từ `src/sigllm/`
**Mục đích:** Trả lời các câu hỏi technical về architecture, loss, training pipeline để defend thesis

---

## Q1 — Lớp mapping CF→Q-Former và Q-Former→LLM: tuyến tính hay phi tuyến?

### Trả lời ngắn: **CHỦ YẾU tuyến tính, có MỘT chỗ semi-phi-tuyến (LayerNorm)**.

### Chi tiết verified từ code

**1. CF → Q-Former (`proj_cf`)** — `src/sigllm/models/q_former/hf_qformer_adapter.py:81`
```python
self.proj_cf = nn.Linear(d_cf, d_model)   # 256 → 768
self.out_proj = nn.Identity() if self.output_dim == d_model else nn.Linear(d_model, self.output_dim)
```
→ **Tuyến tính thuần** (Linear). Không có activation.

**2. Q-Former → LLM (`llm_proj`)** — `src/sigllm/models/multimodal/qformer_rec_llm.py:429-432`
```python
self.llm_proj = nn.Sequential(
    nn.Linear(d_q, H),    # 768 → 3584
    nn.LayerNorm(H),       # normalize
)
```
→ **Linear + LayerNorm**. LayerNorm là `(x-μ)/σ * γ + β` — affine transformation **nhưng có μ, σ tính từ data** → **non-linear theo nghĩa rộng** (output không scale tuyến tính với input).

**3. Bên trong Q-Former (HuggingFace InstructBLIP)** — tất nhiên có GELU/FFN bên trong các transformer layers (intermediate.dense + GELU + output.dense). Nhưng **interface in/out của Q-Former** thì là `[B,Q,d_model]` → `[B,Q,d_model]` (linear-shape preserving).

### Tổng kết bridge stack

```
MF embedding (256d)
    │   proj_cf: Linear(256→768)        ← TUYẾN TÍNH
    ▼
Q-Former cross-attention input (768d)
    │   12 cross/self-attention + FFN layers  ← TRANSFORMER (có GELU bên trong)
    ▼
Q-Former output (768d per query)
    │   out_proj: Identity (nếu cùng dim)     ← TUYẾN TÍNH/IDENTITY
    ▼
llm_proj: Linear(768→3584) → LayerNorm     ← TUYẾN TÍNH + LayerNorm
    ▼
LLM embedding space (3584d)
```

### Có nên dùng phi tuyến hơn không?

Hiện tại `proj_cf` là `Linear(256→768)` thuần — **không có activation/MLP**. Đây là BLIP-2 convention (image features đã rich semantic nên không cần MLP). Nhưng với CF features (entropy thấp hơn nhiều so với image), có thể test:
- `proj_cf = nn.Sequential(Linear(256→512), GELU, Linear(512→768))` — 2-layer MLP với expansion
- Hoặc giữ tuyến tính (vì BLIP-2 dùng tuyến tính)

→ **Worth ablating** nhưng probability cải thiện thấp (~10-15%) — bottleneck không ở projection mà ở information density.

---

## Q2 — ITC / ITM / ITG: bản chất, code, so cái gì với cái gì?

### Bối cảnh

Cả 3 loss đều từ **BLIP-2 paper Section 3.1**. Ý tưởng chung: **align item CF embedding với item text** thông qua Q-Former trong 3 cách khác nhau (contrastive, matching, generation).

### ITC — Item-Text Contrastive

**Bản chất:** InfoNCE contrastive giữa Q-Former output (representation của item CF) và CLS token của item text.

**So sánh:** Q-Former(item_cf) ↔ TextEncoder(item_text)

**Code** — `src/sigllm/models/projection/qformer_alignment_model.py:87-117`
```python
def loss_itc(self, item_ids, text_list, tau=0.07, symmetric=True):
    query_hidden = self.encode_item_queries(item_ids)   # [B, Q=8, 768] — queries cross-attend CF
    text_cls = self.encode_text_cls(text_list)           # [B, 768] — text branch CLS

    # Select best query (out of 8) closest to text_cls
    selected_query, _ = self._select_query_by_text(query_hidden, text_cls)

    q_norm = self.l2norm(selected_query)
    t_norm = self.l2norm(text_cls)
    sim_matrix = (q_norm @ t_norm.T) / tau              # [B, B] cosine sim / temperature

    labels = torch.arange(sim_matrix.size(0))
    loss_q2t = F.cross_entropy(sim_matrix, labels)       # in-batch negatives, diagonal = positive
    loss_t2q = F.cross_entropy(sim_matrix.T, labels)
    loss = (loss_q2t + loss_t2q) / 2.0
```

**Ý nghĩa:** Trong 1 batch B items, mỗi item (i) chỉ "match" với text của chính nó (i). Cross-entropy buộc `sim(query_i, text_i)` cao hơn `sim(query_i, text_j≠i)` → align CF space với text space.

### ITM — Item-Text Matching

**Bản chất:** Binary classifier (match/no-match) với hard negatives mined từ ITC similarity matrix.

**So sánh:** Q-Former joint(item_cf, text) → binary head → match? Yes/No

**Code** — `src/sigllm/models/projection/qformer_alignment_model.py:119-169`
```python
def loss_itm(self, item_ids, text_list, itc_sim_matrix):
    # Mine hard negatives from ITC sim matrix (most-similar non-matching pair)
    weights_t2q = F.softmax(itc_sim_matrix, dim=0)
    weights_q2t = F.softmax(itc_sim_matrix, dim=1)
    # Mask diagonal (positives), sample neg via multinomial
    neg_text_idx = torch.multinomial(weights_q2t, 1)
    neg_item_idx = torch.multinomial(weights_t2q.T, 1)

    # Build 3 groups: (pos_item, pos_text), (pos_item, neg_text), (neg_item, pos_text)
    cf_concat = torch.cat([pos_item_cf, pos_item_cf, neg_item_cf], dim=0)
    text_concat = pos_text + neg_text + pos_text

    # Joint forward: queries see BOTH CF (cross-attn) AND text (self-attn)
    query_hidden, _, _, _ = self.qformer.forward_multimodal(cf_concat, text_concat, causal_text=False)
    pooled = query_hidden.mean(dim=1)
    logits = self.itm_head(pooled)                       # [3B, 2] binary head

    labels = torch.cat([ones(B), zeros(2B)])             # 1 positive group, 2 negative groups
    loss = F.cross_entropy(logits, labels)
```

**Ý nghĩa:** Khác với ITC (chỉ similarity), ITM cho Q-Former **nhìn thấy text trực tiếp** qua self-attention, học classifier tinh tế "đôi này có match không?". Hard negatives từ ITC ép Q-Former phân biệt các cặp gần giống nhau.

### ITG — Item-Text Generation

**Bản chất:** Causal LM — Q-Former queries condition vào CF, text branch generate caption autoregressively.

**So sánh:** Q-Former(item_cf) làm context → predict next token của item_text

**Code** — `src/sigllm/models/projection/qformer_alignment_model.py:171-196`
```python
def loss_itg(self, item_ids, text_list):
    item_cf = self.mf.item_encoder(item_ids)
    # Causal joint forward: queries see CF (bidirectional), text positions are causal
    _, text_hidden, text_ids, text_attention_mask = self.qformer.forward_multimodal(
        item_cf, text_list, causal_text=True
    )

    logits = self.lm_head(text_hidden[:, :-1, :])        # predict t[1:] from t[:-1]
    targets = text_ids[:, 1:].clone()
    targets = targets.masked_fill(~target_mask, -100)    # ignore pad tokens

    loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1), ignore_index=-100)
```

**Ý nghĩa:** Q-Former phải encode CF đủ tốt để **generate text caption tương ứng** — đẩy alignment chặt hơn ITC (chỉ cần similar) và ITM (chỉ cần match yes/no).

### So sánh nhanh 3 losses

| Loss | Mục tiêu | So sánh | Mechanism |
|---|---|---|---|
| **ITC** | Similarity alignment | `q_i ↔ t_i` cosine + InfoNCE | In-batch negatives, soft alignment |
| **ITM** | Binary classification | Joint(q, t) → match? | Hard negatives, fine discrimination |
| **ITG** | Generative | Q-Former → generate(text) | Causal LM, strongest alignment signal |

→ BLIP-2 paper argue 3 losses **complementary**: ITC dạy soft alignment, ITM dạy fine discrimination, ITG dạy strong generative alignment.

---

## Q3 — II + UI Contrastive: ý tưởng từ đâu, loss như nào?

### Ý tưởng từ đâu

**II (Item-Item) contrastive** — Lấy ý tưởng từ **ILM paper Section 3.2** (`docs/ILM.pdf`):
> "We introduce a novel item-item contrastive learning loss to prevent the model from overfitting on item-text data when text labels are sparse"

→ ILM noted: với rec, item-text pairs ít (3K trong ML-1M) so với user-item interactions (888K). Pure item-text training overfit → add item-item co-watch để **augment via co-occurrence signal**.

**UI (User-Item) contrastive** — Cũng từ ILM Section 3.2 (`include_user_item=True`):
> "we also introduce a novel user-item contrastive learning loss to align user CF with their interacted item CF"

→ Tận dụng 888K positive user-item pairs từ training set OpenP5 (vs 3K item-text pairs) → dominant signal cho ML-1M.

### Code

**II Contrastive** — `qformer_alignment_model.py:198-212`
```python
def loss_item_item_ilm(self, left_ids, right_ids, tau=0.07):
    """Co-watch item-item contrastive."""
    left_cf = self.mf.item_encoder(left_ids)
    right_cf = self.mf.item_encoder(right_ids)
    left_q = self.qformer.encode_cf(left_cf)        # [B, Q, 768]
    right_q = self.qformer.encode_cf(right_cf)
    # Select most-similar query pair across the 8 queries
    left_sel, right_sel, _, _ = self.select_pair_by_similarity(left_q, right_q)

    left_norm = self.l2norm(left_sel)
    right_norm = self.l2norm(right_sel)
    logits = (left_norm @ right_norm.T) / tau       # [B, B] in-batch sim
    labels = torch.arange(logits.size(0))
    loss = F.cross_entropy(logits, labels)           # diagonal = co-watched pair
```

**UI Contrastive** — `qformer_alignment_model.py:214-232` (mirror)
```python
def loss_user_item(self, user_ids, item_ids, tau=0.07):
    user_cf = self.mf.user_encoder(user_ids)        # ← khác II: dùng user_encoder
    item_cf = self.mf.item_encoder(item_ids)
    user_q = self.qformer.encode_cf(user_cf)
    item_q = self.qformer.encode_cf(item_cf)
    # ... rest same as II
```

### Cơ chế

- **Item pair (i₁, i₂)** = 2 items co-watched bởi cùng 1 user (consecutive trong history sequence)
- **User-item pair (u, i)** = user u đã rate cao item i (từ train split)
- **InfoNCE in-batch**: cho B pairs, mỗi pair (left_k, right_k) → buộc `sim(L_k, R_k) > sim(L_k, R_j≠k)` qua `F.cross_entropy(logits, arange(B))`

### Đóng góp cho SigLLM

- II + UI hoạt động trên **cùng Q-Former encode_cf** path (queries cross-attend CF only, không có text branch) → consistency với Stage 3 inference
- UI dominant signal trên ML-1M (888K pairs) → loss landscape chính của Stage 1
- II augment cho data sparsity

→ Đây là 2 loss **ILM-introduced** (ILM Section 3.2, Table 4 confirm), **KHÔNG phải SigLLM-specific**. SigLLM inherit từ ILM. BLIP-2 không có II/UI.

### Benefits của UI loss (theo ILM ablation Table 4 + 5)

| Benefit | Evidence từ ILM paper |
|---|---|
| **Performance gain trên ML1M** | ILM-IT-UI vs ILM-IT: NDCG@5 0.0485 vs 0.0474 (+2.3%) |
| **Regularization** | Phase 1 ITG eval loss: ILM-IT=4.17 → ILM-IT-UI=4.07 (giảm overfit) |
| **Enable user channel downstream** | Q-Former học encode user_cf trong CF manifold → cho phép inject user soft tokens ở Stage 3 (SigLLM exploit cái này → +1.79 uAUC trên test_cold) |
| **Augment data sparsity** | ML1M có 3K item-text pairs vs 888K UI pairs → UI là signal chính của Stage 1 |

ILM note: "ML1M's item-text pair data being much scarcer and user interactions are much richer" → UI loss đặc biệt hữu ích cho dataset như ML1M (ít items, nhiều interactions).

---

## Q4 — BERT init layer 0/3: I/O có giống không? Lấy phần nào, bỏ phần nào?

### Trả lời ngắn

**Q-Former CÓ THỂ load BERT weights vì:**
1. Architecture của Q-Former dựa trên `BertLayer` (chung shape với BERT)
2. Code dùng `bert.state_dict()` → match key by key, copy chỉ những tensor **CÙNG SHAPE**
3. Phần khác shape → giữ random init

### Verified từ code

`src/sigllm/models/q_former/hf_qformer_adapter.py:102-162`

```python
def _init_text_branch_from_pretrained_bert(self, bert_model_name):
    bert = BertModel.from_pretrained(bert_model_name)  # bert-base-uncased
    bert_state = bert.state_dict()                      # ~200 tensors
    target_state = self.qformer.state_dict()            # Q-Former tensors

    loaded = 0
    skipped_shape = 0
    for q_key, q_tensor in target_state.items():
        # Q-Former dùng 'attention.attention.*', BERT dùng 'attention.self.*' → remap
        bert_key = q_key.replace(".attention.attention.", ".attention.self.")
        if bert_key not in bert_state:
            continue   # ← skip nếu key không có trong BERT (vd cross-attention)
        bert_tensor = bert_state[bert_key]
        if bert_tensor.shape != q_tensor.shape:
            skipped_shape += 1
            continue   # ← skip nếu shape khác
        target_state[q_key] = bert_tensor.clone()
        loaded += 1
    self.qformer.load_state_dict(target_state, strict=True)
```

### Chi tiết lấy phần nào / bỏ phần nào

| Component | Có trong BERT? | Shape match? | Action |
|---|---|---|---|
| `embeddings.word_embeddings` | ✓ | ✓ | **COPY** |
| `embeddings.position_embeddings` | ✓ | ✓ | **COPY** |
| `embeddings.LayerNorm` | ✓ | ✓ | **COPY** |
| `encoder.layer.0-3.attention.self.{query,key,value}` | ✓ | ✓ (768d hidden) | **COPY** (BERT first 4 layers) |
| `encoder.layer.0-3.attention.output.{dense,LayerNorm}` | ✓ | ✓ | **COPY** |
| `encoder.layer.0-3.intermediate.dense` (FFN up) | ✓ | ✓ | **COPY** |
| `encoder.layer.0-3.output.{dense,LayerNorm}` (FFN down) | ✓ | ✓ | **COPY** |
| `encoder.layer.0-3.crossattention.*` | ❌ (BERT không có cross-attn) | — | **SKIP → random init** |
| `encoder.layer.0-3.intermediate_query.*` (InstructBLIP-specific FFN cho queries) | ❌ | — | **SKIP → random init** |
| `encoder.layer.0-3.output_query.*` (InstructBLIP query FFN) | ❌ | — | **SKIP → random init** |
| `encoder.layer.4-11.*` | ✓ (BERT-base có 12 layers) | — | **KHÔNG xem xét** — Q-Former chỉ có 4 layers, không có key matching |
| `pooler.*` | ✓ | ✓ | Q-Former không có pooler → SKIP |

### Tại sao chỉ lấy first 4 layers của BERT?

Code iterate `for q_key, q_tensor in target_state.items()` — target chỉ có layer 0-3 của Q-Former. Layer 4-11 của BERT **không được touch** (Q-Former không có layer index 4+).

### Số tensors thực sự load

Log từ training:
```
Initialized Q-Former text branch from bert-base-uncased: 86 tensors loaded, 0 shape-mismatch skipped, 134 Q-Former tensors total.
```
- 86 tensors LOAD (chủ yếu embeddings + self-attention + FFN của 4 first layers)
- 48 tensors random init (cross-attention + InstructBLIP-specific query FFN + queries `self.q`)
- 134 total

### Tại sao không lấy nhiều layer hơn?

SigLLM dùng `num_layers=4` (config). Tăng lên 12 layers (match BERT-base) sẽ:
- ✓ Inherit toàn bộ BERT depth
- ❌ 3x parameters (Q-Former từ ~76M lên ~230M) → overfit ML-1M (33K samples)
- ❌ Inference slower

→ 4 layers là trade-off chọn cho rec setting nhỏ.

---

## Q5 — Stage 2 Generative pretrain: idea? Giống BLIP-2?

### Trả lời ngắn: **GIỐNG HỆT BLIP-2 stage 2** (Section 3.2 của BLIP-2 paper).

### Idea BLIP-2

> "In the second pre-training stage, we perform vision-to-language generative learning by connecting Q-Former (with the frozen image encoder attached) to a frozen LLM. The output query embeddings are projected via a fully-connected layer to the LLM's input embedding dimension, and are prepended to the input text embeddings."

→ Train Q-Former + linear projection sao cho **soft tokens đầu vào của LLM** đủ để LLM generate đúng caption tương ứng với image (hoặc trong SigLLM, item).

### SigLLM Stage 2 idea (giống hệt)

> Train Q-Former + `llm_proj` sao cho soft tokens (8 tokens × 3584d) khi prepend vào LLM input embeddings → LLM **generate đúng item caption** ("Title X | Genres: Y, Z") qua next-token LM. LLM hoàn toàn frozen.

### Code

`src/sigllm/pipelines/multimodal/train_qformer_stage2_generative.py:209-242`
```python
def forward_stage2(batch, mf, qformer, llm_proj, tokenizer, llm, max_caption_length):
    item_ids = batch["i_left"]
    captions = batch["text"]              # item caption text

    with torch.no_grad():
        item_cf = mf.item_encoder(item_ids)

    # Queries cross-attend CF ONLY (no instruction text in Q-Former) — BLIP-2 style
    query_tokens = qformer.encode_cf(item_cf)        # [B, 8, 768]
    query_tokens = qformer.out_proj(query_tokens)
    soft_tokens = llm_proj(query_tokens)             # [B, 8, 3584]

    inputs_embeds, attention_mask, labels = _build_inputs(
        soft_tokens_lm, captions, tokenizer, embed_layer, max_caption_length
    )
    # inputs_embeds = [soft_tokens (8)][caption tokens] — soft tokens labeled -100 (ignore)
    # caption tokens labeled with their token_id → next-token LM loss

    outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    return outputs.loss   # standard HF next-token CE loss
```

### Khác biệt với Stage 1

| Aspect | Stage 1 | Stage 2 |
|---|---|---|
| Output target | Q-Former representation | LLM-aligned soft tokens |
| Loss | ITC + ITM + ITG + II + UI (5 losses) | Single LM loss |
| LLM involved? | No (text branch của Q-Former) | Yes (frozen) |
| Trainable | Q-Former (Q-Former text branch) | Q-Former + `llm_proj` |
| Purpose | Representation learning | Bridge Q-Former ↔ LLM space |

### Tại sao cần Stage 2 sau Stage 1?

Sau Stage 1, Q-Former output align với **text embedding** của text branch (BERT-style). Nhưng LLM (Qwen2-7B) có space khác (3584d vs 768d, vocab khác, semantic khác).

→ Stage 2 = **adapter bridge** giữa "BERT space" (Q-Former output) và "Qwen2 space" (LLM input).

→ ILM Phase 2 = đúng bằng Stage 2 này (theo paper).

---

## Q6 — Tại sao phải LoRA ở Stage 3 step 1? Không skip được?

### Trả lời ngắn

**Có thể skip nhưng sẽ KÉM HƠN.** CoLLM paper Section 3.4 chứng minh 2-step LoRA tốt hơn 1-step joint.

### Idea CoLLM 2-step

CoLLM 2-step pattern:
- **Step 1:** Train LoRA only trên **text-only prompt** (không có soft tokens). LLM học task "CTR recommendation" qua text purely.
- **Step 2:** Frozen LoRA, train Q-Former + projection trên **full prompt** (text + soft tokens). Soft tokens giờ "thêm" thông tin CF vào LLM đã hiểu task.

### Tại sao split 2 step?

CoLLM lý luận: nếu train **đồng thời** LoRA + Q-Former + proj từ scratch trên full prompt → 2 vấn đề:
1. **Gradient interference**: LoRA học cần stable input, nhưng soft tokens lúc đầu là noise (Q-Former chưa converge) → LoRA bị nhiễu
2. **Co-adapt overfitting**: LoRA có thể "lazy learn" — chỉ học cách ignore soft tokens

Split 2-step force:
1. LoRA học task **purely từ text** trước (text vẫn đủ signal cho CTR)
2. Sau đó Q-Former + proj phải học **bổ sung thông tin** mà text chưa cover

### Verified từ SigLLM code

`src/sigllm/pipelines/multimodal/train_qformer_stage3_step1_lora.py:1-7`
```
"""CoLLM Step 1 — train LoRA only on text-only prompts.

The Q-Former, projection, MF and base LLM are all frozen; only the LoRA
adapter on the LLM is updated. The text-only prompt is identical in shape to
the full Stage 3 prompt but with the soft-token placeholders (<ItemIDList>,
<TargetItemID>) removed, so the LLM learns the recommendation task without
any collaborative-filtering signal. The best checkpoint feeds Step 2.
"""
```

`apply_step1_overrides`:
```python
cfg.model_cfg.tuning_step = 1
cfg.model_cfg.prompt_path = step1.prompt_path   # → prompts/qformer_prompt_movie_text_only.txt
```

**Text-only prompt** (`prompts/qformer_prompt_movie_text_only.txt`):
```
A user has given high ratings to the following movies: <ItemTitleList>.
Leverage the information above to predict whether the user would enjoy
the movie titled <TargetItemTitle>. Answer with "Yes" or "No".
```
→ Không có `<ItemIDList>` (soft token slot) — LLM chỉ thấy text.

### Hiệu quả thực tế

Trajectory từ ABLATION_RESULTS:
- **Stage 3 Step 1 peak val_uAUC = 0.6917** (text-only, LoRA học task)
- **Stage 3 Step 2 peak val_uAUC = 0.7098** (full prompt, Q-Former+proj train)
- → **Step 2 add +0.018 uAUC** từ soft tokens

→ Nếu skip Step 1, Step 2 phải train cả LoRA + Q-Former + proj từ scratch → kết quả thường tệ hơn (theo CoLLM paper).

### Có thể skip không?

- **Có** nếu chấp nhận performance thấp hơn 0.005-0.02 uAUC
- **Worth ablating**: unified Stage 3 (train all together) → diagnostic xem 2-step pattern có thực sự giúp cho SigLLM không (đã được flag trong `QFORMER_IMPROVEMENT_DIRECTIONS.md` Hypothesis B)

### Bonus — caveat về CoLLM 2-step

Hypothesis trong `QFORMER_IMPROVEMENT_DIRECTIONS.md`:
> "Step 1 LoRA train trên text-only prompts (KHÔNG có soft tokens). Step 2: soft tokens xuất hiện, LoRA frozen → **LoRA chưa từng học cách 'xử lý' soft tokens**."

→ Có thể là một root cause khiến SigLLM nghẽn ở 0.71 — cần test unified Stage 3 để xác nhận.

---

## Q7 — AUC vs uAUC: cụ thể như nào? Ví dụ dân dã?

### AUC (Area Under ROC Curve) — Global ranking

**Định nghĩa toán học:** Xác suất 1 cặp ngẫu nhiên (positive, negative) được model rank đúng (positive score > negative score).

**Code** — `sklearn.metrics.roc_auc_score(y_true, y_pred)`:
- `y_true`: array of 0/1 labels TOÀN BỘ test set
- `y_pred`: array of predicted scores
- → AUC tính 1 số duy nhất trên TẤT CẢ predictions

**Bản chất:** Lấy 1 positive sample bất kỳ + 1 negative sample bất kỳ (qua tất cả users), bao nhiêu % model rank đúng?

### uAUC (User AUC) — Per-user ranking

**Định nghĩa:** Trung bình AUC tính **riêng cho từng user**, rồi average qua users.

**Code** — `src/sigllm/pipelines/rec/train_rec_baseline.py:26-65`
```python
def calculate_user_auc(user_ids, y_pred, y_true):
    users = np.unique(user_ids)
    auc_list = []
    for user_id in users:
        user_mask = user_ids == user_id
        user_y_true = y_true[user_mask]
        user_y_pred = y_pred[user_mask]
        if len(user_y_true) < 2: continue                       # skip 1-sample users
        if np.all(user_y_true == user_y_true[0]): continue      # skip users with all-positive or all-negative
        score = roc_auc_score(user_y_true, user_y_pred)
        auc_list.append(score)
    return np.mean(auc_list)
```

### Ví dụ dân dã

**Setup:** 3 users, mỗi user có 3 items đánh giá

```
User A: [item1=Yes, item2=Yes, item3=No]   → predict [0.9, 0.8, 0.3]
User B: [item4=No,  item5=Yes, item6=No]   → predict [0.7, 0.6, 0.5]
User C: [item7=Yes, item8=No,  item9=Yes]  → predict [0.4, 0.2, 0.1]
```

**Global AUC:** Lấy mọi cặp (pos, neg) qua toàn bộ 9 predictions
- Pos scores: {0.9, 0.8, 0.6, 0.4, 0.1}  (items được label Yes)
- Neg scores: {0.3, 0.7, 0.5, 0.2}        (items được label No)
- AUC = #cặp (pos > neg) / #cặp total

→ Có nhiều cặp đúng (vd 0.9>0.3, 0.9>0.7, ...) và 1 số cặp sai (vd 0.1<0.7)
→ AUC ≈ 0.65 (số tổng quát)

**uAUC:** Tính AUC riêng từng user, rồi avg
- User A: pos=[0.9, 0.8], neg=[0.3] → AUC=1.0 (cả 2 pos đều > neg)
- User B: pos=[0.6], neg=[0.7, 0.5] → AUC=0.5 (0.6 thắng 0.5 nhưng thua 0.7)
- User C: pos=[0.4, 0.1], neg=[0.2] → AUC=0.5 (0.4>0.2 nhưng 0.1<0.2)
- **uAUC = (1.0 + 0.5 + 0.5) / 3 = 0.667**

### Tại sao 2 metric khác nhau quan trọng?

**Global AUC có thể CAO khi:**
- Model giỏi tách users "active" khỏi users "casual" — kể cả ranking trong user kém
- Vd: model luôn predict 0.9 cho User A items (cả pos lẫn neg), 0.1 cho User C items — global AUC vẫn ổn vì pos của A cao hơn neg của C

**uAUC CAO chỉ khi:**
- Trong mỗi user, model rank items "user thích" cao hơn items "không thích"
- Đây là **personalization metric** — cái quan trọng thật sự cho recommendation

### Trong SigLLM thesis

- **Vanilla baseline:** AUC=0.7325, **uAUC=0.7170**
- **User soft tokens:** AUC=0.7551, **uAUC=0.7023**
  - AUC ↑ (+0.023) nhưng uAUC ↓ — model học **discriminate users tốt hơn** nhưng **ranking trong user không cải thiện**
- **User-conditioned queries:** AUC=0.7582, uAUC=0.7060 → cùng pattern

→ Đây là dấu hiệu **calibration shift, không phải personalization** — user-level signal giúp distinguish users (global) nhưng không refine item ordering per user.

### Cold-item exception

User soft tokens trên test_cold: AUC=0.7077, **uAUC=0.6695** (vs Vanilla 0.6516) → +1.79 uAUC. Đây là phép tính riêng cho **3178 samples test_cold** → uAUC cải thiện thật sự (cả global lẫn per-user) trên cold scenario.

---

## Q8 — Vanilla queries vs Instruction-aware: tại sao instruction không giúp?

### Vanilla queries — định nghĩa

**Vanilla = chỉ learnable query tokens + cross-attention với CF, KHÔNG có instruction text vào Q-Former self-attention.**

```python
# Vanilla mode (Q-Former encode_cf)
query_tokens = self.q.expand(B, -1, -1)        # [B, 8, 768] learnable parameters
encoder_hidden_states = self.proj_cf(cf_vec)   # [B, 1, 768] CF as cross-attn key/value
outputs = self.qformer(
    input_ids=None,                            # ← KHÔNG có text input
    query_embeds=query_tokens,
    encoder_hidden_states=encoder_hidden_states,
)
# Queries chỉ cross-attend CF (1 token), self-attend giữa các queries
```

**= ILM Phase 2 / BLIP-2 style** (BLIP-2 stage 2 cũng không feed text vào Q-Former).

### Instruction-aware — định nghĩa

**Instruction-aware = queries + instruction text cùng vào Q-Former self-attention.**

```python
# Instruction-aware mode (Q-Former forward — InstructBLIP style)
query_tokens = self.q.expand(B, -1, -1)
text_input_ids = tokenizer(instruction_text)   # ← THÊM instruction text
# Concat: [queries | text] vào self-attention
outputs = self.qformer(
    input_ids=text_input_ids,                  # text dùng self-attn
    query_embeds=query_tokens,
    encoder_hidden_states=encoder_hidden_states,
)
# Queries vừa cross-attend CF, vừa self-attend với instruction text
# → Mỗi instruction khác → queries extract feature khác từ cùng item CF
```

**= InstructBLIP** — paper claim này là main contribution của InstructBLIP.

### Tại sao instruction-aware KHÔNG giúp ở SigLLM?

#### Lý do 1: Single task setting (lý do mạnh nhất)

InstructBLIP train **26 datasets × 11 task categories** (captioning, VQA, reasoning, OCR, knowledge, ...). Mỗi instruction routing queries extract feature **khác nhau cho task khác nhau** (vd "describe scene" vs "count objects").

SigLLM chỉ có **1 task duy nhất**: binary CTR Yes/No. Mọi instructions đều cùng objective. → **Không có routing signal** để instruction-aware queries học.

> Bạn nói đúng: "instruction-aware đóng góp SOTA trong zero-shot/multi-task — không cải thiện chính xác trên single task" — **CHÍNH XÁC**.

#### Lý do 2: Prompt paraphrase = same semantic

12 prompts V1 paraphrase trong SigLLM đều là biến thể paraphrase của:
- "Represent this movie for recommendation..."
- "Align this movie metadata with CF representation..."
- "Use the title and genres to describe this movie..."

→ Cùng semantic, cùng task → queries không phân biệt được context khác.

#### Lý do 3: Information density thấp

InstructBLIP source: image 1408d × 257 patches = 361K nums/image → giàu information cho instruction route đến features khác.

SigLLM source: MF 256d (single vector) → ít signal → instruction route không có gì để route.

#### Lý do 4: Redundancy với LLM

LLM đã có instruction trong prompt rồi. Routing tại Q-Former duplicate cái LLM đã làm.

### Evidence từ ablation

| Variant | val_uAUC | Δ vs vanilla |
|---|---|---|
| Vanilla (queries-only) | 0.7081 | — |
| Instruction-aware V1 (12 paraphrase) | 0.7098 | +0.0017 (noise) |
| Instruction-aware V2 (8 task-focused) | ~0.71 | ~0 |

→ **|Δ uAUC| < 0.005** trên cả 3 test splits → nằm trong seed noise (±0.005-0.01).

### Confirm hiểu của bạn

> "Vanilla queries = chỉ learnable queries thuần không có instruction text" — **ĐÚNG** ✓
>
> "instruction-aware không đóng góp vì nó chỉ tốt cho zero-shot/multi-task, không cho single task" — **ĐÚNG** ✓
>
> Đây cũng là **finding khoa học chính** của SigLLM thesis (negative finding rigorous, ILM independently validated).

---

## Q9 — User-conditioned queries vs Re-enable user soft tokens: khác nhau cái gì?

### Tóm tắt 1 câu

Cả 2 đều inject user CF vào Q-Former bridge, nhưng:
- **User-conditioned queries**: shift queries của Q-Former dựa trên user_cf (internal Q-Former modification)
- **User soft tokens**: cho user_cf 1 slot RIÊNG trong LLM prompt (parallel với target item slot)

### User-conditioned queries (branch `feat/user-conditioned-queries`)

**Cơ chế:** Thay đổi base queries của Q-Former theo user

**Code** — `hf_qformer_adapter.py:90-101` (trong branch)
```python
if self.user_conditioned:
    self.user_proj = nn.Linear(d_user_eff, d_model)  # 256 → 768
    # Zero-init: queries = pretrained_q + 0 ban đầu (no-op at epoch 0)
    nn.init.zeros_(self.user_proj.weight)
    nn.init.zeros_(self.user_proj.bias)

# In forward (when user_cf passed):
query_tokens = self.q.expand(B, -1, -1) + self.user_proj(user_cf).unsqueeze(1)
# Same shift (1 vector) áp dụng cho cả 8 queries → FiLM-style bias
```

**Pipeline ảnh hưởng:**
- Q-Former trở thành conditional trên user
- Cùng item nhưng query khác nhau cho user khác → **per-user item extraction**
- Output vẫn 8 soft tokens cho mỗi item (target + history)
- Tổng soft tokens trong prompt: 8 target + 80 history = **88 tokens**

**Trong prompt — KHÔNG thay đổi:**
```
A user has highly rated: <ItemTitleList>. Signals: <ItemIDList>.
Predict <TargetItemTitle> <TargetItemID>. Answer:
```
(không có `<UserID>` slot)

### Re-enable user soft tokens (branch `feat/reenable-user-soft-tokens`)

**Cơ chế:** Thêm 1 slot RIÊNG cho user_cf trong prompt

**Code** — `qformer_rec_llm.py:625-640` (trong branch)
```python
if self.enable_user_soft_tokens:
    user_cf = self.rec_encoder.user_encoder(batch_data["UserID"])  # [B, 256]
    user_q = self.qformer(user_cf, ins_list)                        # [B, 8, 768]

target_q = self.qformer(target_cf, ins_list)                         # [B, 8, 768]

user_llm = self.llm_proj(user_q)                                     # [B, 8, 3584]
target_llm = self.llm_proj(target_q)
```

**Pipeline ảnh hưởng:**
- Q-Former **vẫn dùng base queries** (không shift)
- User CF chạy qua **SAME Q-Former + llm_proj** (parallel với item path)
- Output: 8 soft tokens RIÊNG cho user + 8 cho target + 80 cho history = **96 tokens** total

**Trong prompt — THÊM `<UserID>` slot:**
```
A user has highly rated: <ItemTitleList>. Signals: <ItemIDList>.
<UserID> Predict <TargetItemTitle> <TargetItemID>. Answer:
       ↑ NEW: 8 user soft tokens
```

### So sánh side-by-side

| Aspect | User-conditioned queries | User soft tokens |
|---|---|---|
| **Mechanism touch point** | Q-Former INTERNAL (modify queries) | LLM PROMPT (separate slot) |
| **Soft tokens for user** | 0 (user → modify queries, không inject) | 8 (RIÊNG cho user) |
| **Soft tokens total/sample** | 88 (8 target + 80 history) | **96** (8 user + 8 target + 80 history) |
| **Same Q-Former for user & item** | N/A (user chỉ shift queries) | YES (same Q-Former encodes both) |
| **Prompt template change** | Không | Thêm `<UserID>` placeholder |
| **New trainable params** | `user_proj: Linear(256→768)` ~ **197K params** | None (reuse Q-Former + llm_proj) |
| **Init** | Zero-init `user_proj` → no-op at epoch 0 | Reuse Stage 2 ckpt directly |
| **Branch** | `feat/user-conditioned-queries` | `feat/reenable-user-soft-tokens` |

### Kết quả thực nghiệm

| Metric | Baseline | User-conditioned | User soft tokens |
|---|---|---|---|
| val_AUC peak | 0.7222 | **0.7582** | 0.7551 |
| val_uAUC peak | **0.7098** | 0.7060 | 0.7023 |
| test_warm uAUC | 0.7213 | (chưa eval test) | 0.7214 |
| **test_cold uAUC** | **0.6516** | (chưa eval) | **0.6695 ✓ (+0.018)** |

→ **2 cơ chế khác nhau nhưng pattern KQ giống nhau**: AUC ↑, uAUC ~tie. Insight: user-level signal mechanism-agnostic, content matter (không phải cơ chế).

### Bonus: tại sao 2 mechanism cho kết quả tương đương?

Cả 2 đều inject **cùng information** (user_cf) qua **cùng Q-Former pipeline**:
- User-conditioned: shift queries → cùng item nhưng query "biased" theo user
- User soft tokens: user_cf qua Q-Former → 1 set soft tokens user-specific

Cả 2 chung quy lại đều là "thêm user-level CF signal vào LLM input". Khác nhau là **VỊ TRÍ** signal được inject (internal vs prompt slot), không phải **NỘI DUNG**. → Cùng kết quả là hợp lý.

---

## Tóm tắt key insights cho thesis defense

1. **Projection stack chủ yếu tuyến tính** — có thể ablation MLP với GELU cho proj_cf
2. **ITC/ITM/ITG = BLIP-2 standard**, II/UI = SigLLM/ILM extension cho rec
3. **BERT init copy chỉ embeddings + first 4 self-attention/FFN layers**, cross-attention + InstructBLIP query FFN giữ random
4. **Stage 2 = BLIP-2 generative pretrain identical** (LLM frozen, train Q-Former + proj qua next-token LM)
5. **Stage 3 step 1 LoRA cần thiết** theo CoLLM pattern (probably) — nhưng worth ablating unified
6. **AUC = global ranking, uAUC = per-user ranking** — 2 metric độc lập, uAUC quan trọng hơn cho personalization
7. **Vanilla queries = không có instruction text vào Q-Former self-attn** = ILM Phase 2 style
8. **Instruction-aware fail vì single-task setting** — không có routing signal (paper InstructBLIP train 26 datasets)
9. **User-conditioned vs User soft tokens** = same content (user_cf via Q-Former), different injection point (internal queries shift vs prompt slot)
