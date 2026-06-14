# SigLLM Codebase Audit — Comprehensive Report

**Date:** 2026-06-02
**Branch:** `feat/reenable-user-soft-tokens` (parent: `feat/best-0.72-warm`)
**Source:** Static read of `src/sigllm/`, `configs/config.yaml`, `prompts/`
**Mục đích:** Trả lời 13 mục audit về kiến trúc Q-Former/LLM recommender

---

## 1. Tổng quan kiến trúc

### Model chính

- **Tên:** `QRecLLM` (alias `mini_gpt4rec_v2` qua registry)
- **File:** `src/sigllm/models/multimodal/qformer_rec_llm.py:51`
- **Base class:** `Rec2Base` (`src/sigllm/models/multimodal/base/rec_base_model.py`)

```python
@registry.register_model("mini_gpt4rec_v2")
class QRecLLM(Rec2Base):
    """QFormer + InstructBLIP for recommendation."""
    PRETRAINED_MODEL_CONFIG_DICT = {"pretrain_vicuna": "configs/models/minigpt4rec.yaml"}
```

### Các module chính

| Module | Class | File | Role |
|---|---|---|---|
| `self.rec_encoder` | `MatrixFactorization` | `src/sigllm/models/rec/matrix_factorization.py:4` | MF baseline cung cấp user_cf, item_cf |
| `self.qformer` | `HFQFormerAdapter` | `src/sigllm/models/q_former/hf_qformer_adapter.py:24` | Q-Former bridge (CF → soft tokens) |
| `self.llm_proj` | `nn.Sequential(Linear, LayerNorm)` | `qformer_rec_llm.py:429` | Q-Former hidden 768 → LLM hidden 3584 |
| `self.llm_model` | HF CausalLM + (optional PEFT LoRA) | `qformer_rec_llm.py:209` | Qwen2-7B-Base |
| `self.llm_tokenizer` | `AutoTokenizer` | `qformer_rec_llm.py:203` | Tokenize prompt |

### Luồng forward (Stage 3 — `forward_v2` → `forward`)

`qformer_rec_llm.py:1009-1024`

```
batch_data (dict)
    │
    ├─ UserID, TargetItemID, InteractedItemIDs_pad, InteractedItemTitles, TargetItemTitle, label
    ▼
self._sample_prompt()                                       # random 1 of 3 paraphrase prompts
    ▼
build_llm_inputs_from_prompt_v2(prompt, batch_data)
    │
    ├─ get_placeholder_order(prompt)                        # tìm <UserID>, <ItemIDList>, <TargetItemID>
    ├─ _build_qformer_instructions(B)                       # random pick 1 of 12 Q-Former instructions
    ├─ encode_rec_features_to_llm_v2(batch_data, order, instruction_list)
    │     │
    │     ├─ target_cf = rec_encoder.item_encoder(TargetItemID)            # [B, 256]
    │     ├─ user_cf = rec_encoder.user_encoder(UserID)  (nếu enable)      # [B, 256]
    │     ├─ inter_cf = rec_encoder.item_encoder(InteractedItemIDs_pad)    # [B, L, 256], L=10
    │     ├─ target_q = qformer(target_cf, ins_list)                       # [B, 8, 768]
    │     ├─ user_q = qformer(user_cf, ins_list)                           # [B, 8, 768]
    │     ├─ inter_q_flat = qformer(inter_cf_flat, inter_ins_list)         # [B*L, 8, 768]
    │     ├─ target_llm = llm_proj(target_q)                               # [B, 8, 3584]
    │     ├─ user_llm = llm_proj(user_q)                                   # [B, 8, 3584]
    │     ├─ inter_llm_flat = llm_proj(inter_q_flat).reshape(B,L*8,3584)   # [B, 80, 3584]
    │     └─ merged_embs = concat per feature_order, mask padded items     # [N_valid, 3584]
    │
    └─ wrap_prompt_with_soft_tokens_v2(rec_embeds, batch_data, prompt)
          │
          ├─ replace <UserID>/<ItemIDList>/<TargetItemID> với " ".join([unk]*Q)
          ├─ tokenize prompt với history-length per sample
          ├─ embed_layer(input_ids)
          └─ scatter merged_embs vào positions có unk_token_id
                ▼
            inputs_embeds: [B, T, 3584]
            attention_mask: [B, T]
    ▼
build_llm_outputs_from_labels(batch_data)
    │
    ├─ text_labels = ["Yes"|"No" theo batch_data["label"]]
    ├─ label_tokens = tokenize(text_labels)
    └─ label_embeds = embed_layer(label_tokens.input_ids)                  # [B, L_label, 3584]
    ▼
assemble_llm_sequences(input_embeds, ..., label_embeds, ...)
    │
    └─ full_embeds = concat([input_embeds, label_embeds], dim=1)           # [B, T+L_label, 3584]
    ▼
llm_model(inputs_embeds=full_embeds, attention_mask=full_atts)             # frozen base + LoRA
    │
    └─ outputs.logits: [B, T+L_label, V]
    ▼
calculate_recommendation_loss / recommendation_scores
    │
    ├─ prediction_logits = outputs.logits[:, -(label_seq_len + 1), :]
    │                       # logit ngay TRƯỚC token "Yes"/"No" đầu tiên
    ├─ binary_logits = stack([logits[:, neg_id], logits[:, pos_id]])
    ├─ loss = CE(binary_logits, label)
    └─ prob_yes = softmax(binary_logits)[:, 1]
    ▼
return {"loss": loss, "logits": prob_yes}
```

### Input fields trong batch (training)

`movie_ood_dataset.py:119-127`

| Field | Shape / Type | Source |
|---|---|---|
| `UserID` | `torch.long [B]` | MF user index |
| `TargetItemID` | `torch.long [B]` | MF item index target |
| `TargetItemTitle` | `list[str] [B]` (quoted) | dataset text |
| `InteractedItemIDs_pad` | `torch.long [B, max_history_len=10]` (padded với 0) | history ids |
| `InteractedItemTitles` | `list[str] [B]` (joined với `, `) | history titles |
| `InteractedNum` | `int` per sample | số items valid |
| `label` | `torch.long [B]` (0/1) | CTR target |
| `prompt_flag` | optional, dùng cho warm/cold | from `not_cold` |

### Output

- **Training:** `{"loss": Tensor scalar, "logits": Tensor [B] (prob_yes)}` — `qformer_rec_llm.py:969`
- **Eval (`generate_for_samples`):** cùng dict, có thêm `outputs` raw nếu `return_all=True` — line 966

---

## 2. Behavioral / collaborative vector

### Nguồn vector hành vi

`src/sigllm/models/rec/matrix_factorization.py:4-25`

```python
class MatrixFactorization(nn.Module):
    def __init__(self, config):
        self.padding_index = 0
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=0)

    def user_encoder(self, user_ids): return self.user_embedding(user_ids)
    def item_encoder(self, item_ids): return self.item_embedding(item_ids)
```

### Dimensions

| Embedding | Shape | Config key |
|---|---|---|
| User embedding | `[user_num=839, 256]` | `rec_config.user_num`, `rec_config.embedding_size` |
| Item embedding | `[item_num=3256, 256]` | `rec_config.item_num`, `rec_config.embedding_size` |
| Behavioral vector / interaction vec | KHÔNG TỒN TẠI riêng (chỉ lookup từ Embedding) | — |

→ **Behavioral vector = trực tiếp output của `nn.Embedding` lookup**, không qua history encoder hay LightGCN. **256-d MF embeddings**.

### Bridge source

- **MF only** (`rec_model: "MF"` trong `configs/config.yaml:46`). LightGCN/SASRec/DIN không được sử dụng (mặc dù có placeholder trong ROADMAP).

### Frozen hay trainable?

**FROZEN** trong toàn bộ Stage 1/2/3. Verified:

`qformer_rec_llm.py:190-195` (init):
```python
if freeze_rec and self.rec_encoder is not None:
    for name, param in self.rec_encoder.named_parameters():
        param.requires_grad = False
    self.rec_encoder = self.rec_encoder.eval()
    self.rec_encoder.train = disabled_train
```

`configs/config.yaml:11`: `freeze_rec: True`

→ MF được pretrain riêng ở Stage 0 (`train_rec_baseline.py`), sau đó load + freeze suốt 3 stages.

### LayerNorm / normalization trước Q-Former

**KHÔNG có**. MF embedding (256-d) được pass thẳng vào `proj_cf`:

`hf_qformer_adapter.py:217-224`
```python
def _project_cf(self, cf_vec):
    encoder_hidden_states = self.proj_cf(cf_vec).unsqueeze(1)
    encoder_attention_mask = torch.ones(cf_vec.size(0), 1, dtype=torch.long, device=cf_vec.device)
    return encoder_hidden_states, encoder_attention_mask
```

`self.proj_cf = nn.Linear(d_cf=256, d_model=768)` — tuyến tính thuần, không có activation/norm.

---

## 3. Q-Former architecture

### File định nghĩa

- **Adapter wrapper:** `src/sigllm/models/q_former/hf_qformer_adapter.py:24` (class `HFQFormerAdapter`)
- **Inner model:** `transformers.InstructBlipQFormerModel` (HuggingFace InstructBLIP Q-Former, hosted bên trong adapter)

### Hyperparameters (từ `configs/config.yaml:50-60`)

| Param | Value | Code reference |
|---|---|---|
| `num_queries` | **8** | `qformer_config.num_queries` |
| `qformer_d_model` (hidden_size) | **768** | `qformer_config.qformer_d_model` |
| `qformer_output_dim` | 768 (= d_model → `out_proj = nn.Identity()`) | `qformer_config.qformer_output_dim` |
| `num_layers` | **4** | `qformer_config.num_layers` |
| `num_heads` | **8** | `qformer_config.num_heads` |
| `cross_attention_frequency` | **2** (default in `HFQFormerAdapter.__init__`) | `hf_qformer_adapter.py:56` |
| `intermediate_size` | `4 * d_model = 3072` (default) | `hf_qformer_adapter.py:90` |
| `max_instruction_length` | 48 | `qformer_config.max_instruction_length` |
| `qformer_text_model_name` | "bert-base-uncased" | for tokenizer + BERT init |

### Tổng tham số

Từ training log:
```
Trainable parameter counts | rec_encoder=0, qformer=76014336, llm_proj=2763264, llm_model=0, llm_lora=0
```
→ Q-Former: **~76M trainable params**.

### Cross-attention

**CÓ.** Cross-attention layers đặt mỗi 2 layers (`cross_attention_frequency=2`):
- Với `num_layers=4` → cross-attn ở layers 0, 2
- Self-attn ở tất cả 4 layers

Tensor mà cross-attention attend vào:

`hf_qformer_adapter.py:267-275` (encode_cf forward):
```python
encoder_hidden_states, encoder_attention_mask = self._project_cf(cf_vec)
# encoder_hidden_states: [B, 1, 768] — CHỈ 1 token = projected CF vec

outputs = self.qformer(
    query_embeds=query_tokens,            # [B, 8, 768]
    encoder_hidden_states=encoder_hidden_states,   # [B, 1, 768]  ← cross-attn key/value
    encoder_attention_mask=encoder_attention_mask,
)
```

→ **Cross-attention attend vào CHÍNH XÁC 1 token** (projected CF vector). Khác BLIP-2 (257 image patch tokens) — bandwidth siêu hẹp tại điểm này.

### Instruction / text conditioning

**CÓ.** Q-Former forward chính (`forward`) là instruction-aware:

`hf_qformer_adapter.py:357-365`
```python
def forward(self, cf_vec, instruction):
    query_hidden, _, _, _ = self.forward_multimodal(cf_vec, instruction, causal_text=False)
    return self.out_proj(query_hidden)
```

`forward_multimodal` (line 343-350): concat queries + instruction text trong self-attention, queries cross-attend CF:
```python
outputs = self.qformer(
    input_ids=text_ids,                          # instruction tokens
    attention_mask=joint_attention_mask,         # queries + text mask
    query_embeds=query_tokens,
    encoder_hidden_states=encoder_hidden_states, # CF token
)
```

Stage 2 ngược lại — dùng `encode_cf` (KHÔNG instruction, BLIP-2 style):
`train_qformer_stage2_generative.py:219` → `query_tokens = qformer.encode_cf(item_cf)`

### Query tokens init

`hf_qformer_adapter.py:80`
```python
self.q = Parameter(torch.randn(1, num_queries=8, d_model=768))
```

→ **Random Gaussian** (standard normal). Không có cluster init / pretrained init.

### Q-Former pretrained weights

**Hỗn hợp:**

1. **BERT init** ở Stage 1 (mặc định `init_from_pretrained_text=True`):
   - `hf_qformer_adapter.py:99-100`
   - Copy embeddings + first 4 self-attention/FFN layers từ `bert-base-uncased`
   - Cross-attention layers + query tokens + InstructBLIP-specific query FFN → **random init**

2. **Stage 1 ckpt** ở Stage 2 (`init_from_pretrained_text=False`):
   - `qformer_rec_llm.py:374-380` (`pretrained_qformer` arg)
   - Load full state dict từ Stage 1 best

3. **Stage 2 ckpt** ở Stage 3:
   - Same path, load Stage 2 final

→ Stage 1 dùng BERT-initialized, Stage 2 dùng Stage 1 ckpt, Stage 3 dùng Stage 2 ckpt.

---

## 4. Projection vào LLM

### Module

`qformer_rec_llm.py:429-432`
```python
self.llm_proj = nn.Sequential(
    nn.Linear(d_q=768, H=3584),   # Linear
    nn.LayerNorm(H=3584),          # LayerNorm
)
```

- **Type:** `nn.Sequential` — chỉ 2 layers (Linear + LayerNorm). **KHÔNG có activation phi tuyến** (GELU/ReLU).
- **Trainable params:** `~2.76M` (Linear: 768×3584 + 3584 bias + LayerNorm: 3584×2)

### Init

`qformer_rec_llm.py:433-436`
```python
nn.init.normal_(self.llm_proj[0].weight, std=0.02)
nn.init.zeros_(self.llm_proj[0].bias)
nn.init.constant_(self.llm_proj[1].weight, H ** -0.5)   # LayerNorm scale
nn.init.zeros_(self.llm_proj[1].bias)                    # LayerNorm shift
```

### Dimensions

| Stage | Input | Output |
|---|---|---|
| Q-Former `out_proj` (Identity) | `[B, 8, 768]` | `[B, 8, 768]` |
| `llm_proj` | `[B, 8, 768]` | `[B, 8, 3584]` |

Tính qua mỗi item:
- Target item: 8 soft tokens
- User: 8 soft tokens (nếu `enable_user_soft_tokens=True`)
- History: L × 8 = 80 soft tokens (L=10)

### Vị trí concat vào prompt

`qformer_rec_llm.py:744-803` (`wrap_prompt_with_soft_tokens_v2`)

Cơ chế:
1. Replace placeholders trong prompt template với chuỗi `" ".join([unk_seq] * Q)`:
   - `<UserID>` → 8 unk tokens (nếu enabled)
   - `<TargetItemID>` → 8 unk tokens
   - `<ItemIDList>` → `valid_items × 8` unk tokens
2. Tokenize prompt → input_ids
3. Embed input_ids qua `llm_model.get_input_embeddings()`
4. **Scatter** soft tokens vào positions có `unk_token_id`:

```python
replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
inputs_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = rec_embeds['merged_embs']
```

→ Soft tokens **thay thế in-place** các unk slots.

### Soft token placeholder ID

`_resolve_soft_token_placeholder` (`qformer_rec_llm.py:230-274`):
- Ưu tiên: `tokenizer.unk_token_id`
- Nếu không có (Qwen2 case), thử hardcoded list `<|im_start|>`, `<|extra_0|>`, v.v.
- Log thực tế: `soft_token_id=151644 ('<|im_start|>')` cho Qwen2

### Format input cuối cùng vào LLM

```
[bos][prompt text với soft tokens scattered]Answer:[label text]
       │                                              │
       └─ input_embeds [B, T_input, 3584]            └─ label_embeds [B, L_label, 3584]
                                                       (concat trong full_embeds)
```

→ Format = standard causal LM input, soft tokens nằm INLINE trong text embeddings (không có separator).

### LLM embedding layer

**FROZEN** (toàn bộ base model frozen):

`qformer_rec_llm.py:216-217`
```python
for name, param in self.llm_model.named_parameters():
    param.requires_grad = False
```

→ Embed layer dùng để lookup nhưng KHÔNG update. Soft tokens thay thế embedding của unk slots tại runtime — không thay đổi embedding table.

---

## 5. Prompt / instruction

### Có 2 SET prompts TÁCH BIỆT trong codebase

#### A. Prompts vào **LLM** (`prompts/qformer_prompt_movie.txt`)

```
1. A user has given high ratings to the following movies: <ItemTitleList>. The interaction-history signals are: <ItemIDList>. Leverage the information above to predict whether the user would enjoy the movie titled <TargetItemTitle>. <TargetItemID> Answer with "Yes" or "No".
2. A user has highly rated these movies: <ItemTitleList>. The interaction-history signals are: <ItemIDList>. Based on this preference history, predict whether the user would like the movie <TargetItemTitle>. <TargetItemID> Answer with "Yes" or "No".
3. A user previously gave strong ratings to: <ItemTitleList>. The interaction-history signals are: <ItemIDList>. Use this information to decide whether the user would enjoy <TargetItemTitle>. <TargetItemID> Answer with "Yes" or "No".
4-6. # DISABLED_USER_CF: ... (skip variants với <UserID>, hiện disabled)
```

- **Số templates active:** 3 (template với `# DISABLED_USER_CF:` bị filter bởi `_init_prompts` line 488)
- **Variant với user:** `prompts/qformer_prompt_movie_with_user.txt` (3 templates, mỗi cái có `<UserID>` slot)
- **Variant text-only:** `prompts/qformer_prompt_movie_text_only.txt` (3 templates, không có soft-token slots — dùng cho Stage 3 Step 1)
- **Selection:** random per training step (`_sample_prompt` → `random.choices` với weights `[5,5,1]` ưu tiên đầu)
- **Eval:** dùng deterministic template đầu (`prompt_list[0]`)
- **Sample-dependent fields:** `<ItemTitleList>`, `<TargetItemTitle>` thay text title; `<ItemIDList>`, `<TargetItemID>`, `<UserID>` thay soft tokens
- **Trailing template:** `prompt_template: '{} Answer:'` (`config.yaml:9`) — append `" Answer:"` vào mọi prompt

#### B. Instructions vào **Q-Former** (`qformer_rec_llm.py:71-84`)

```python
QFORMER_ITEM_INSTRUCTIONS = [
    "Represent this movie for recommendation using its title and genres.",
    "Align this movie metadata with its collaborative filtering representation.",
    # ... 12 templates total (V1 paraphrase)
]
```

- **Số:** 12 templates
- **Selection:** random per row per training step (`random.choices(...)` line 921)
- **Eval:** deterministic — `QFORMER_ITEM_INSTRUCTIONS[0]` lặp lại (line 922)
- **Sample-dependent:** KHÔNG — instruction string giống nhau cho mọi item (1 instruction per sample, không phụ thuộc item content)

### Q-Former vs LLM prompt — KHÁC NHAU HOÀN TOÀN

| Aspect | Q-Former instruction | LLM prompt |
|---|---|---|
| Số templates | 12 paraphrase | 3 paraphrase |
| Content | "Represent this movie for recommendation..." | "A user has highly rated... Predict whether..." |
| Vào đâu | Q-Former text branch self-attention | LLM input embeddings |
| Tokenizer | bert-base-uncased | Qwen2 tokenizer |
| Sample-dependent? | No | Yes (titles + history) |

### Truncation

| Encoder | Max length | Source |
|---|---|---|
| Q-Former tokenizer | `max_instruction_length=48` | `config.yaml:57` |
| LLM tokenizer | `max_txt_len=1024` | `config.yaml:40` |

`qformer_rec_llm.py:761-766`:
```python
prompts_tokens = self.llm_tokenizer(
    prompt_list, return_tensors="pt", padding="longest",
    truncation=True, max_length=self.max_txt_len, add_special_tokens=False
)
```

→ Cả 2 đều truncate. LLM 1024 đủ dài cho prompt thực tế (~few hundred tokens).

---

## 6. Training objective / loss

### Stage 1 — 5 losses (BLIP-2 + ILM)

`src/sigllm/pipelines/multimodal/train_qformer_stage1_representation.py:142-222`

| Loss | Type | Weight | Temperature | Code |
|---|---|---|---|---|
| **ITC** (item-text contrastive) | InfoNCE symmetric | `w_itc=1.0` | `tau_itc=0.07` | `loss_itc` in `qformer_alignment_model.py:87` |
| **ITM** (item-text matching) | Binary CE với hard negatives | `w_itm=1.0` | — | `loss_itm` line 119 |
| **ITG** (item-text generation) | Causal LM CE | `w_itg=1.0` | — | `loss_itg` line 171 |
| **II** (item-item contrastive) | InfoNCE | `w_ii=1.0` | `tau_ii=0.05` | `loss_item_item_ilm` line 198 |
| **UI** (user-item contrastive) | InfoNCE | `w_ui=1.0` | `tau_ui=0.07` | `loss_user_item` line 214 |

Total: `loss = w_itc*L_itc + w_itm*L_itm + w_itg*L_itg + w_ii*L_ii + w_ui*L_ui`

### Stage 2 — 1 loss (BLIP-2 generative)

`train_qformer_stage2_generative.py:209-242`

```python
# Next-token LM trên item caption, queries inject soft tokens vào LLM
outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
return outputs.loss   # HuggingFace standard CE loss
```

- LLM frozen, chỉ train Q-Former + `llm_proj`.

### Stage 3 — 1 loss (CTR binary CE)

`qformer_rec_llm.py:865-879`
```python
def calculate_recommendation_loss(self, outputs, label_tokens, batch_data, ans_map):
    pos_id = self.llm_tokenizer(ans_map[1]).input_ids[0]   # "Yes" token id
    neg_id = self.llm_tokenizer(ans_map[0]).input_ids[0]   # "No" token id

    prediction_logits = outputs.logits[:, -(label_seq_len + 1), :]   # logit position của token đầu của label
    binary_logits = torch.stack([prediction_logits[:, neg_id], prediction_logits[:, pos_id]], dim=1)
    loss = F.cross_entropy(binary_logits, batch_data['label'].long())
```

→ **Binary CE trên logits của 2 tokens "Yes" và "No"** (không phải full vocab) — efficient + interpretable.

### Positive / negative samples

**Stage 3 (CTR):**
- Pos/neg = pre-labeled trong dataset (`label=1/0` từ user ratings)
- KHÔNG có negative sampling tại training time (dataset đã có sẵn)
- KHÔNG có hard negative mining

**Stage 1 (contrastive):**
- ITC/II/UI: in-batch negatives qua InfoNCE (B-1 negatives mỗi anchor)
- ITM: hard negatives mined từ ITC similarity matrix (`loss_itm:135-142`)

### Loss → Update mapping

| Loss | Stage | Updates |
|---|---|---|
| 5 losses (ITC+ITM+ITG+II+UI) | Stage 1 | Q-Former (full) + Stage 1-specific heads (`itm_head`, `lm_head`) |
| Generative LM | Stage 2 | Q-Former + `llm_proj` |
| Binary CE | Stage 3 Step 1 | LoRA only |
| Binary CE | Stage 3 Step 2 | Q-Former + `llm_proj` (LoRA frozen) |

---

## 7. Trainable / frozen parameters

### Configuration policy

`_apply_tuning_step_policy` (`qformer_rec_llm.py:298-341`):

```python
if step == 1:
    # Step 1: LoRA only
    for p in self.qformer.parameters(): p.requires_grad = False
    for p in self.llm_proj.parameters(): p.requires_grad = False
elif step == 2:
    # Step 2: Q-Former + projection, freeze LoRA
    for n, p in self.llm_model.named_parameters():
        if "lora_" in n: p.requires_grad = False
    for p in self.qformer.parameters(): p.requires_grad = True
    for p in self.llm_proj.parameters(): p.requires_grad = True
```

### Tổng kết Trainable / Frozen

| Module | Stage 1 | Stage 2 | Stage 3 Step 1 | Stage 3 Step 2 |
|---|---|---|---|---|
| `rec_encoder` (MF) | ✗ Frozen | ✗ Frozen | ✗ Frozen | ✗ Frozen |
| `qformer` (Q-Former) | ✓ Train | ✓ Train | ✗ Frozen | ✓ Train |
| `llm_proj` | N/A | ✓ Train | ✗ Frozen | ✓ Train |
| `llm_model` base | N/A | ✗ Frozen | ✗ Frozen | ✗ Frozen |
| LoRA adapter | N/A | N/A | ✓ Train | ✗ Frozen |

### Tổng số trainable params (Stage 3 Step 2 — from training log)

```
Trainable parameter counts | rec_encoder=0, qformer=76014336, llm_proj=2763264, llm_model=0, llm_lora=0
```

- **Q-Former:** 76,014,336 (~76M)
- **llm_proj:** 2,763,264 (~2.8M)
- **Total trainable Step 2:** ~78.8M

Step 1:
- **LoRA:** 2,523,136 (~2.5M, r=8 [q_proj, v_proj])
- **Total Step 1:** ~2.5M

### LoRA setup

`config.yaml:28-38`
```yaml
lora_config:
  use_lora: True
  r: 8
  alpha: 16
  target_modules: [q_proj, v_proj]
  dropout: 0.05
```

Attached qua `peft.get_peft_model` (line 292).

---

## 8. Dataset

### Dataset đang dùng

- **Name:** MovieLens-1M (OOD2 split from CoLLM/BinLLM)
- **Class:** `MovieOODDataset` (`src/sigllm/datasets/movie/movie_ood_dataset.py:22`)
- **Builder:** `MovieOODBuilder` (`movie_ood_builder.py`)
- **Storage:** `data/processed/ml-1m/`

### Số liệu

| Field | Value |
|---|---|
| Users (`user_num`) | **839** |
| Items (`item_num`) | **3,256** |
| Train interactions | 33,891 |
| Val interactions | 10,401 |
| Test (overall) | 7,331 |
| Test warm | 3,522 |
| Test cold | 3,178 |
| Max history length | **10** (cap, `movie_ood_dataset.py:81`) |

### Train/val/test split

`config.yaml:214-216`
```yaml
train_splits: ["train"]
valid_splits: ["valid"]
test_splits: ["test", "test_warm", "test_cold"]
```

Loaded from pre-processed `.pkl` files:
- `train_ood2.pkl`, `valid_ood2.pkl`, `test_ood2.pkl`, `test_warm_cold_ood2.pkl`

### Warm / cold split

**CÓ** — qua field `not_cold` / `warm`:

`movie_ood_dataset.py:38-42`
```python
if subset == "warm":
    self.annotation = df[df['warm'].isin([1])].copy()
if subset == "cold":
    self.annotation = df[df['not_cold'].isin([0])].copy()
```

→ `warm=1` (items đã thấy trong training), `not_cold=0` (cold items).

### Một sample format (training)

```python
{
    "UserID": int,                            # MF user index
    "InteractedItemIDs_pad": np.array[10],    # left-padded history IDs (pad=0)
    "InteractedItemTitles": str,              # ", ".join các titles, format: '"Title1", "Title2", ...'
    "TargetItemID": int,                      # MF item index
    "TargetItemTitle": str,                   # f'"{title}"' (quoted)
    "InteractedNum": int,                     # số items valid (non-pad)
    "label": int (0 or 1),                    # CTR label
    "prompt_flag": int (optional),            # warm/cold marker
}
```

### Collator

`src/sigllm/runners/utils/dataloader_builder.py:147` dùng `getattr(dataset, "collater", None)`. Nếu `None` → torch default collate.

→ Movie dataset không define custom collater → default `torch.utils.data.dataloader.default_collate` (tự stack tensors, list strings được giữ list).

### Item text / title

**CÓ.** Field `TargetItemTitle` + `InteractedItemTitles` đều là text từ MovieLens.

### History limit

**10 items** (hardcoded cap line 81). Left-pad với 0 nếu < 10. Truncate keep `last 10` nếu > 10.

---

## 9. Training config

`configs/config.yaml`

### Batch size

| Stage | Train BS | Eval BS |
|---|---|---|
| Stage 1 (Q-Former pretrain) | 256 (line 103) | N/A |
| Stage 2 (generative) | 32 (line 145) | N/A |
| Stage 3 (CTR) | **16** train / **64** eval (line 206-207) | — |

### Learning rate

| Stage | LR | Source |
|---|---|---|
| Stage 1 | `5e-5` | `qformer_stage1.lr` line 119 |
| Stage 2 | `1e-4` | `qformer_stage2.lr` line 163 |
| Stage 3 Step 1 | `2e-4` | `qformer_stage3_step1.init_lr` line 181 |
| Stage 3 Step 2 | `3e-5` | `qformer_stage3_step2.init_lr` line 192 |

Top-level defaults (`run` block line 197-200):
- `init_lr: 1e-4`
- `min_lr: 8e-5`
- `warmup_lr: 1e-5`

### Optimizer

- Stage 1: `Adam` (`train_qformer_stage1_representation.py:93`)
- Stage 2: `Adam` (similar pattern)
- Stage 3: built qua `OptimizerBuilder` (`src/sigllm/runners/utils/optimizer_builder.py`) — likely AdamW

### Weight decay

- Stage 1: `1e-3` (`qformer_stage1.weight_decay`)
- Stage 2: `1e-3`
- Stage 3: `1e-3` (`run.weight_decay`)

### Scheduler

`config.yaml:196`
```yaml
lr_sched: "linear_warmup_cosine_lr"
warmup_steps: 200
```

### Epochs / max steps

| Stage | Max epoch | Iters/epoch |
|---|---|---|
| Stage 1 | 100 | (loops full train_loader) |
| Stage 2 | 20 | (loops full train_loader) |
| Stage 3 Step 1 | 5 | 400 |
| Stage 3 Step 2 | 200 | 400 (top-level default) |

### Gradient accumulation

Default `accum_grad_iters` (not in config explicitly) — looks like default = 1 (no accumulation). Code path: `tasks/base/rec_base_task.py:124-134`.

### Mixed precision

**CÓ** — `amp: True` (line 201). Stage 2/3 dùng `torch.amp.autocast("cuda", dtype=torch.float16)`.

### Early stopping

| Stage | Metric | Mode | Patience |
|---|---|---|---|
| Stage 1 | `val_loss` | min | 10 (line 132) |
| Stage 2 | `val_loss` | min | 5 (line 169) |
| Stage 3 | uses checkpoint best by `agg_metrics` (AUC) — `rec_base_task.py:197`. Patience không config explicitly. |

---

## 10. Evaluation

### Metrics

| Metric | Source | Where |
|---|---|---|
| **AUC** (global) | `sklearn.metrics.roc_auc_score` | `train_rec_baseline.py:9` + `logger.py:300` + `rec_base_task.py:251` |
| **uAUC** (per-user avg) | `calculate_user_auc` | `train_rec_baseline.py:26-67` |
| **ACC@0.5** | (prob_yes ≥ 0.5) accuracy | `rec_base_task.py:166` |
| **Score gap / pos_score_mean / neg_score_mean** | Manual compute on softmax probs | `rec_base_task._compute_metrics` |
| **pos_rate / pred_pos_rate@0.5** | Positive distribution stats | `rec_base_task.py:181-182` |

### Eval script

- **Main:** `src/sigllm/tasks/base/rec_base_task.py:152-202` (`evaluate` method)
- **Per-sample eval:** `qformer_rec_llm.py:948` (`generate_for_samples`)
- **Triggered by:** `run.evaluate=True` flag → `Stage 3 Step 2` pipeline script

### Eval method — scoring NOT generation

`qformer_rec_llm.py:881-891` (`recommendation_scores`):
```python
prediction_logits = outputs.logits[:, -(label_seq_len + 1), :]
binary_logits = torch.stack([prediction_logits[:, neg_id], prediction_logits[:, pos_id]], dim=1)
return torch.softmax(binary_logits, dim=1)[:, 1]
```

→ **KHÔNG generate**, chỉ lấy logits của 2 tokens "Yes"/"No" tại vị trí ngay trước label, softmax → prob_yes.

→ **Efficient** (1 forward pass, không autoregressive sampling) + **deterministic** (không có randomness).

### Candidate items khi evaluate

`MovieOODDataset.__getitem__` — mỗi sample đã có pre-determined `TargetItemID` và `label`. Đây là **pre-built test set** với pre-labeled (target, label) pairs, KHÔNG có ranking over candidates.

→ Bản chất là **point-wise CTR evaluation**, không phải Top-K ranking như NDCG/HR style.

### Same negative set giữa các models?

**CÓ** — tất cả models eval trên cùng `test_ood2.pkl` (pre-processed split). Negative items được fix từ dataset construction.

### Positive vs Negative answer

`qformer_rec_llm.py:546-550` (`_init_prompts`):
```python
if self.ans_type == "v2":
    self.pos_ans = ['Yes']
    self.neg_ans = ['No']
```

Eval comparison sample (`_maybe_log_predictions` line 988):
```python
pred = ans_map[1] if probs[i] >= 0.5 else ans_map[0]
```

→ Threshold **0.5** dùng cho ACC@0.5. AUC/uAUC dựa trên scores raw.

---

## 11. Log kết quả

### Best checkpoint location

| Stage | Path | Best metric peak |
|---|---|---|
| Stage 1 | `/content/SigLLM/ckpt/qformer_stage1/qformer_stage1_best_qformer.pth` | val_loss min |
| Stage 2 | `/content/SigLLM/ckpt/qformer_stage2_qwen2/qformer_stage2_best_qformer.pth` + `qformer_stage2_best_proj.pth` | val_loss min |
| Stage 3 Step 1 | `/content/SigLLM/ckpt/qformer_stage3_step1_lora_qwen2/qwen2-7b-base/checkpoint_best.pth` | val_uAUC peak ~0.6917 (epoch 4) |
| Stage 3 Step 2 (Vanilla) | `/content/SigLLM/ckpt/qformer_stage3_step2_cie_qwen2/qwen2-7b-base/checkpoint_best.pth` | val_uAUC 0.7081 (epoch 5) |
| Stage 3 Step 2 (User soft tokens, NEW) | `/content/SigLLM/ckpt/qformer_stage3_step2_cie_userslot_qwen2/...` | val_uAUC 0.7023 (epoch 4) |

### Trajectory — Vanilla Stage 3 Step 2 (from `docs/ABLATION_RESULTS.md:144-157`)

| Epoch | AUC | uAUC | val_loss | Score gap |
|---|---|---|---|---|
| 0 | 0.7204 | 0.6954 | 0.6129 | 0.1561 |
| 1 | 0.7209 | 0.6893 | 0.6126 | 0.1673 |
| 2 | 0.7223 | 0.6979 | 0.6142 | 0.1814 |
| 3 | 0.7212 | 0.7008 | 0.6143 | 0.1749 |
| 4 | 0.7215 | 0.7057 | 0.6144 | 0.1774 |
| **5** | 0.7207 | **0.7081** (peak) | 0.6169 | 0.1826 |
| 6 | 0.7218 | 0.7061 | 0.6130 | 0.1773 |
| 7 | 0.7216 | 0.7052 | 0.6192 | 0.1854 |

### Trajectory — User Soft Tokens (NEW, from training log gần đây)

| Epoch | val_AUC | val_uAUC | val_loss | score_gap |
|---|---|---|---|---|
| 0 | 0.7299 | 0.6930 | 0.6062 | 0.151 |
| 1 | 0.7476 | 0.6953 | 0.5982 | 0.206 |
| 2 | 0.7551 | 0.7016 | 0.5851 | 0.206 |
| 3 | 0.7547 | 0.7005 | 0.5890 | 0.205 |
| **4** | 0.7543 | **0.7023** (peak) | 0.5890 | 0.208 |
| 5 | 0.7451 | 0.6922 (drop) | 0.5986 | 0.204 |

### Test results — User Soft Tokens

| Split | AUC | uAUC | vs Baseline Vanilla |
|---|---|---|---|
| test | 0.7336 | 0.7145 | tie (Δ −0.0025) |
| test_warm | 0.7462 | 0.7214 | tie (Δ +0.0001) |
| **test_cold** | 0.7077 | **0.6695** | **+0.0179** ✓ |

### Overfit / stagnation patterns

- **Stage 3 Step 2:** val_uAUC peak epoch 3-5, sau đó drift xuống → **early stop pattern**, không phải classic overfit (val_loss không tăng mạnh)
- **AUC plateau:** vanilla ~0.72, user soft tokens ~0.755 → score calibration shift mạnh
- **uAUC nghẽn:** ~0.70-0.71 across mọi variants (4 architectural extensions tested all NEGATIVE)

### Per-epoch training loss

KHÔNG được dump per-epoch trong codebase mặc định. Chỉ có `metric_logger.global_avg()` log mỗi epoch (`rec_base_task.py:148`).

---

## 12. Ablation đã có

Source: `docs/ABLATION_RESULTS.md`

| Ablation | Status | Result |
|---|---|---|
| **Text-only** (CF zeroed via `ablate_soft_tokens=True`) | ✓ Done | uAUC drops −5.15 pts test_warm — confirm CF contribution |
| **CF-only** (no text in prompt) | ✗ Not tested | — |
| **Q-Former vs no Q-Former** (MLP bridge baseline) | ✗ Not tested | Listed as TODO in `QFORMER_IMPROVEMENT_DIRECTIONS.md` #8 |
| **Số query tokens** | ✗ Not tested (giữ Q=8) | — |
| **Prompt / instruction redesign** | ✓ V1 paraphrase + Task-focused both tested | NEGATIVE (Δ < 0.005) |
| **Freeze/unfreeze patterns** | Partially — `tuning_step` controls Q-Former vs LoRA freeze | α2 experiment mentioned in code comment (qformer freeze plateau ~0.694) |
| **Instruction-aware vs Vanilla** | ✓ Done | NEGATIVE — Vanilla edges out slightly |
| **Interaction-aware** | ✓ Done | NEGATIVE — −0.006 uAUC |
| **User-conditioned queries** | ✓ Done | NEGATIVE on uAUC (0.7060), but AUC win (0.7582) |
| **User soft tokens (re-enable)** | ✓ Done | Tied on warm, **WIN +1.79 uAUC on test_cold** |

### Ablate switch hiện có

`config.yaml:20`
```yaml
ablate_soft_tokens: False   # When True, zero out soft tokens at inference
```

`qformer_rec_llm.py:124-131` — Logs khi flag active.

---

## 13. Vấn đề nghi ngờ — bottlenecks tiềm năng

### A. Bottleneck: Cross-attention chỉ 1 token

`hf_qformer_adapter.py:217-224`
```python
encoder_hidden_states = self.proj_cf(cf_vec).unsqueeze(1)  # [B, 1, 768]
```

→ Cross-attention attend vào **chỉ 1 token** (projected CF). Q-Former có 4 layers × 8 queries, nhưng tất cả phải extract info từ 1 single 768-d vector. **Bandwidth siêu hẹp** so với BLIP-2 (257 image patches).

**Probability bottleneck:** HIGH. Worth ablating with multi-token CF (e.g., expand 256-d MF vec thành k > 1 tokens via per-dim projection).

### B. Bottleneck: `llm_proj` là 2-layer Linear + LayerNorm, no GELU

`qformer_rec_llm.py:429-432`
```python
self.llm_proj = nn.Sequential(nn.Linear(768, 3584), nn.LayerNorm(3584))
```

CoLLM dùng 2-layer MLP với intermediate_size=10x input. SigLLM mạch đơn → có thể underutilize 768→3584 expansion.

**Probability bottleneck:** MEDIUM. Worth ablating with `Linear → GELU → Linear → LayerNorm`.

### C. Mismatch giữa Stage 1 và Stage 2

- **Stage 1:** Q-Former forward dùng instruction-aware (5 losses pass text instructions vào self-attn). `loss_itc` dùng `encode_item_queries` (vanilla, no instruction). `loss_itm`/`loss_itg` dùng `forward_multimodal` với instruction.
- **Stage 2:** Q-Former forward chỉ `encode_cf` (vanilla, NO instruction). 

→ Stage 1 ITC + Stage 2 generative **không nhận instruction**, nhưng ITM + ITG **có nhận**. Stage 3 lại dùng `forward()` (instruction-aware) với prompt khác nhau.

**Worry:** Q-Former weights tối ưu cho cả 2 chế độ (with/without instruction) → có thể compromise.

**Probability bottleneck:** MEDIUM. Đáng test `instruction_aware=False` ở Stage 3 (đã tested, kết quả ~same → not the primary issue, but consistency story is fragile).

### D. Constant Q-Former instructions per batch trong eval

`qformer_rec_llm.py:920-922`
```python
if self.training:
    return random.choices(self.QFORMER_ITEM_INSTRUCTIONS, k=batch_size)
return [self.QFORMER_ITEM_INSTRUCTIONS[0]] * batch_size
```

→ **Eval time:** mọi sample có chung instruction `QFORMER_ITEM_INSTRUCTIONS[0]`. Train/eval distribution mismatch.

**Probability bottleneck:** LOW (đã được test). Nhưng vẫn là code smell — eval không reflect train distribution.

### E. Soft token placeholder = `<|im_start|>` cho Qwen2

`qformer_rec_llm.py:267-274` (fallback log):
```python
log_step("Soft-token fallback",
    "no unk_token and no safe single-token candidate; using eos_token as soft-slot placeholder. "
    "Soft slots will COLLIDE with padding if pad_token == eos_token — Step 2 may corrupt embeddings silently.")
```

Verified log: `soft_token_id=151644 ('<|im_start|>')`. → Special token được dùng làm placeholder.

**Risk:** Nếu prompt thực sự chứa `<|im_start|>` literal (Qwen2 chat template token) → collision. Hiện tại prompts không có nên OK, nhưng nếu đổi LLM/prompt cần check lại.

**Probability bottleneck:** LOW (current setup safe). High risk khi đổi LLM.

### F. History padding với index 0 → MF lookup trả về padding_idx embedding

`movie_ood_dataset.py:108-109` left-pads với 0:
```python
padded_history_ids = ([0] * pad_size) + list(history_item_ids)
```

`matrix_factorization.py:9` định nghĩa `padding_idx=0` → embedding[0] sẽ là **all-zero** (PyTorch convention).

Phía forward (`qformer_rec_llm.py:642-643`):
```python
item_mask = (ids != self.rec_encoder.padding_index).long()
item_mask_q = item_mask.unsqueeze(-1).repeat(1, 1, Q).reshape(B, L * Q)
```

→ Mask được apply để skip padded positions khi merge. Nhưng Q-Former vẫn forward TẤT CẢ history items (kể cả padded), sau đó mask trong concat. 

**Compute waste**: Q-Former forward `B * 10 = B*L` times mỗi sample, kể cả padded items. Trên 33K samples × max 10 items → 330K Q-Former forward passes/epoch chỉ để compute padding.

**Probability bottleneck:** Performance (compute waste, không phải accuracy). Worth optimizing với mask trước forward.

### G. Stage 3 Step 1 LoRA chưa từng thấy soft tokens

Prompt Step 1 là **text-only** (no `<UserID>`/`<ItemIDList>`/`<TargetItemID>` placeholders).

`qformer_rec_llm.py:925`:
```python
feature_order = self.get_placeholder_order(prompt_template) if prompt_template else None
if not feature_order:
    rec_embeds = {None values...}  # Q-Former output discarded
```

→ Step 2 inject soft tokens vào prompt, LoRA frozen — **LoRA chưa từng "thấy"** soft tokens trong khi train ở Step 1. 

→ Đây là **CoLLM convention** nhưng có thể suboptimal. Worth testing unified Stage 3 (xem `QFORMER_IMPROVEMENT_DIRECTIONS.md` Hypothesis B).

**Probability bottleneck:** MEDIUM-HIGH. Đã được flag trong `QFORMER_IMPROVEMENT_DIRECTIONS.md` Hypothesis B.

### H. Detached gradients — không thấy mismatch

Tìm kỹ code:
- `forward_stage2:213` có `with torch.no_grad(): item_cf = mf.item_encoder(item_ids)` — correct, MF frozen
- `_log_information_flow` (qformer_rec_llm) dùng `.detach()` chỉ cho logging
- ITM hard negatives dùng `.detach()` trên sim matrix — correct (sim matrix là từ ITC, không cần gradient lại)

→ **Không phát hiện gradient leak hay mismatch.**

### I. Module tưởng train nhưng frozen

Verified per Stage 3 Step 2 log:
```
Trainable parameter counts | rec_encoder=0, qformer=76014336, llm_proj=2763264, llm_model=0, llm_lora=0
```

- `rec_encoder=0` ✓ expected (MF frozen)
- `qformer=76M` ✓ trainable
- `llm_proj=2.8M` ✓ trainable
- `llm_model=0` ✓ base frozen
- `llm_lora=0` ✓ LoRA frozen ở Step 2

→ **Consistent với config**. Không có misalignment.

### J. Tensor shape bất thường

| Tensor | Expected | Actual (training log) | Status |
|---|---|---|---|
| `target_q` | `[B, 8, 768]` | `[16, 8, 768]` | ✓ |
| `target_llm` | `[B, 8, 3584]` | `[16, 8, 3584]` | ✓ |
| `merged_embs` | `[B*total_slots, 3584]` | `[1168, 3584]` (B=16, ~73 slots/sample) | ✓ (depends valid_history) |
| `user_q` | `[B, 8, 768]` | `[16, 8, 768]` (khi user enabled) | ✓ |

→ **Không có anomaly.**

### K. Prompt template static cho từng training step

`forward_v2` line 1010:
```python
prompt = self._sample_prompt()   # random pick 1 of 3, reweighted
```

→ Mỗi training step **1 prompt cho cả batch** (B=16). Không phải per-sample prompt sampling.

**Risk:** Mỗi training step LLM thấy chỉ 1 prompt structure → có thể bias toward style đó. Worth checking: per-sample prompt sampling thay vì per-step.

**Probability bottleneck:** LOW. Nhưng đáng note.

---

## Summary — Top bottleneck candidates ranked

| Rank | Bottleneck | Probability | Worth testing? |
|---|---|---|---|
| 1 | A. Single-token cross-attention bandwidth | HIGH | Yes (multi-token CF projection) |
| 2 | G. LoRA-soft-token mismatch (Step 1) | MEDIUM-HIGH | Yes (unified Stage 3) |
| 3 | B. llm_proj không có activation phi tuyến | MEDIUM | Yes (add GELU between Linear layers) |
| 4 | F. History compute waste (forward padded items) | Performance only | Easy fix (mask before forward) |
| 5 | C. Stage 1/2 instruction inconsistency | MEDIUM | Partial test done — vanilla vs instruction-aware similar |
| 6 | K. Per-step prompt sampling | LOW | Easy fix (per-sample) |
| 7 | D. Constant Q-Former instruction at eval | LOW | Already tested |

---

## File reference summary

| Topic | File | Key lines |
|---|---|---|
| Main model | `src/sigllm/models/multimodal/qformer_rec_llm.py` | 51 (class), 1009 (forward), 555 (encode_rec_features), 744 (wrap_prompt) |
| Q-Former adapter | `src/sigllm/models/q_former/hf_qformer_adapter.py` | 24 (class), 80 (queries init), 102 (BERT init), 357 (forward) |
| Stage 1 losses | `src/sigllm/models/projection/qformer_alignment_model.py` | 87 (ITC), 119 (ITM), 171 (ITG), 198 (II), 214 (UI) |
| Stage 1 train | `src/sigllm/pipelines/multimodal/train_qformer_stage1_representation.py` | 142 (train_step), 305 (train loop) |
| Stage 2 train | `src/sigllm/pipelines/multimodal/train_qformer_stage2_generative.py` | 209 (forward), 265 (loop) |
| Stage 3 step 1 | `src/sigllm/pipelines/multimodal/train_qformer_stage3_step1_lora.py` | 47 (overrides) |
| Stage 3 step 2 | `src/sigllm/pipelines/multimodal/train_qformer_stage3_step2_cie.py` | (similar pattern) |
| MF baseline | `src/sigllm/models/rec/matrix_factorization.py` | full file |
| MF train | `src/sigllm/pipelines/rec/train_rec_baseline.py` | 26 (uAUC) |
| Dataset | `src/sigllm/datasets/movie/movie_ood_dataset.py` | 22 (class), 84 (__getitem__) |
| Eval | `src/sigllm/tasks/base/rec_base_task.py` | 152 (evaluate), 251 (uAUC compute) |
| Config | `configs/config.yaml` | full file |
| Prompts | `prompts/qformer_prompt_movie*.txt` | 3 paraphrase variants per file |
