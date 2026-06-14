# Q-Former Initialization Improvements — Proposal

**Date:** 2026-05-31
**Author:** Thesis discussion (Claude + Bùi Nguyên Quốc Bảo)
**Branch target:** `feat/smart-qformer-init` (chưa tạo)
**Goal:** Cải thiện chất lượng khởi tạo Q-Former để (a) Stage 1 converge nhanh hơn và (b) potentially unlock baseline performance ceiling chưa đạt được do init kém.

---

## Bối cảnh — tại sao xem xét init

Sau **4 negative findings** từ các architectural extensions (instruction-aware, task-focused prompts, interaction-aware, user-conditioned queries), audit code phát hiện thêm các vấn đề ở **starting point** của Q-Former mà có thể là root cause yếu của baseline:

1. **Query tokens random Gaussian init** — 8 queries không có "prior knowledge" về item space
2. **`proj_cf: Linear(256→768)` random init** — Stage 1 phải học từ scratch cách map MF embedding sang Q-Former hidden space
3. **BERT init chỉ lấy 4/12 layers** — bỏ qua 8 layer chứa abstract semantic của BERT-base
4. **BERT-base (2018) khá outdated** — sentence-transformer community đã chứng minh MPNet (2020) outperform đáng kể cho semantic tasks

So sánh với BLIP-2 (nguồn cảm hứng):

| Aspect | BLIP-2 | SigLLM hiện tại | Implication |
|---|---|---|---|
| Số queries | 32 | **8** | Bottleneck SigLLM chặt hơn 4x |
| Số layers Q-Former | 12 | **4** | Chỉ inherit 1/3 BERT depth |
| Source size | 257 × 1024 (image patches) | **1 × 256** (MF vector) | Source nghèo 1000x |
| Query init | random | random | Same |
| Cross-attn init | random | random | Same |
| Text branch init | BERT | BERT | Same |

SigLLM **giữ nguyên init convention của BLIP-2 nhưng scale nhỏ hơn nhiều**. Với scale nhỏ, init quality matter hơn vì có ít gradient steps để recover.

## Hiện trạng init — verified từ code

| Stage | `init_from_pretrained_text` | Behavior |
|---|---|---|
| **Stage 1** | `True` (default in adapter) | **BERT init thực sự xảy ra** (`_init_text_branch_from_pretrained_bert`) |
| Stage 2 | `False` (explicit) | Load Stage 1 ckpt (đã có BERT weights) |
| Stage 3 | `False` (explicit) | Load Stage 2 ckpt (kế thừa) |

Code Stage 1 mặc định pull BERT-base weights vào:
- `embeddings.word_embeddings` ✓
- `embeddings.position_embeddings` ✓
- `embeddings.layernorm` ✓
- `encoder.layer.0-3.attention.self.*` (self-attention) ✓
- `encoder.layer.0-3.intermediate.dense` + `.output.dense` (FFN) ✓
- `encoder.layer.0-3.output.LayerNorm` ✓

KHÔNG init (random — đúng theo BLIP-2 convention):
- Cross-attention layers (BERT có không)
- `self.q` (8 learnable queries)
- `proj_cf` (CF → Q-Former dim projection)
- `out_proj` (Q-Former → output dim, Identity nếu cùng dim)

→ **BERT init đã đúng convention BLIP-2.** Nhưng có dư địa cải tiến.

---

## 4 cải tiến đề xuất

### 🥇 Improvement #1 — Cluster-based query initialization

#### Vấn đề
```python
# Hiện tại — hf_qformer_adapter.py:80
self.q = Parameter(torch.randn(1, num_queries, d_model))  # [1, 8, 768]
```

8 query tokens random Gaussian, **không có semantic prior**. Stage 1 phải tự khám phá ra rằng "query 0 đại diện cho action movies", "query 1 đại diện cho romance", v.v. từ scratch.

#### Đề xuất
Init 8 queries từ centroid của 8 KMeans clusters trên item text embeddings:

```python
import torch
from sklearn.cluster import KMeans
from transformers import BertModel, BertTokenizer

def compute_query_init(num_queries=8, d_model=768, item_texts=None):
    """
    item_texts: list of strings, e.g. [f"{title} {genres}" for each movie]
    """
    bert = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    embeddings = []
    bert.eval()
    with torch.no_grad():
        for text in item_texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
            out = bert(**ids)
            cls_emb = out.last_hidden_state[:, 0]  # [1, 768]
            embeddings.append(cls_emb.squeeze(0))

    embeddings = torch.stack(embeddings)  # [N_items, 768]
    embeddings_np = embeddings.cpu().numpy()

    km = KMeans(n_clusters=num_queries, random_state=42, n_init=10)
    km.fit(embeddings_np)
    centroids = torch.tensor(km.cluster_centers_, dtype=torch.float32)  # [num_queries, 768]

    return centroids.unsqueeze(0)  # [1, num_queries, 768]


# Trong HFQFormerAdapter.__init__ (thêm option)
if cluster_init_queries:
    query_init = compute_query_init(num_queries, d_model, all_item_texts)
    self.q = Parameter(query_init)
else:
    self.q = Parameter(torch.randn(1, num_queries, d_model))
```

#### Ý nghĩa
Mỗi query bắt đầu với "prior" về 1 cluster items. Stage 1 ITC/ITM losses chỉ cần fine-tune từ vị trí có sẵn semantic meaning thay vì discover từ noise.

#### Effort
- 2 giờ code (compute centroids 1 lần, lưu ra `.pth`)
- Retrain Stage 1 + 2 + 3

#### Expected impact
- Stage 1 train loss converge nhanh hơn 30-50%
- Có thể peak val_uAUC ở Stage 3 cao hơn (avoid local minima do init kém)
- Risk: cluster có thể không align với task → giữ nguyên hoặc tệ hơn

#### Defensibility
Đây là **novel contribution** — chưa thấy paper Q-Former rec nào dùng cluster init cho queries. Defensible cho thesis section "Initialization choices".

---

### 🥈 Improvement #2 — OLS-based `proj_cf` warm-start

#### Vấn đề
```python
# Hiện tại — hf_qformer_adapter.py:81
self.proj_cf = nn.Linear(d_cf, d_model)  # Linear(256, 768), random init
```

Stage 1 ITC loss yêu cầu `proj_cf(MF[i])` ≈ "text embedding của item i" trong Q-Former hidden space. Với random init, mapping ban đầu là noise — Stage 1 phải học từ đầu.

#### Đề xuất
Init `proj_cf` qua closed-form OLS regression: tìm W sao cho `W · MF[i]` ≈ `BERT(item_text[i])` cho mọi item i.

```python
def warm_start_proj_cf(mf_embeddings, item_text_bert_embeddings):
    """
    mf_embeddings: torch.Tensor [N, d_cf=256]
    item_text_bert_embeddings: torch.Tensor [N, d_model=768]
    Returns: (W, b) for nn.Linear(d_cf, d_model)
    """
    X = mf_embeddings                            # [N, 256]
    Y = item_text_bert_embeddings                # [N, 768]

    # Add bias column
    X_aug = torch.cat([X, torch.ones(X.size(0), 1)], dim=1)  # [N, 257]

    # Closed-form OLS
    Wb = torch.linalg.lstsq(X_aug, Y).solution   # [257, 768]
    W = Wb[:-1, :].T                             # [768, 256]
    b = Wb[-1, :]                                # [768]

    return W, b


# Trong HFQFormerAdapter.__init__ (thêm option)
self.proj_cf = nn.Linear(d_cf, d_model)
if warm_start_proj_cf_from is not None:
    mf, bert_emb = warm_start_proj_cf_from
    W, b = warm_start_proj_cf(mf, bert_emb)
    self.proj_cf.weight.data = W
    self.proj_cf.bias.data = b
```

#### Ý nghĩa
Tại epoch 0 của Stage 1, `proj_cf(MF[i])` đã ≈ `BERT(item_text[i])`. ITC contrastive loss chỉ cần fine-tune nhẹ. Skip 3-5 epoch học mapping từ scratch.

#### Effort
- 1 giờ code (compute W, b 1 lần, lưu ra `.pth`)
- Retrain Stage 1 + 2 + 3

#### Expected impact
- Stage 1 epoch 1-2 đã có alignment hợp lý
- Tổng training time Stage 1 giảm ~20-30%
- Có thể peak Stage 3 cao hơn nếu init kém là root cause

#### Defensibility
**Standard technique** trong representation learning (warm-start). Defensible easily, không phải novel claim mạnh nhưng đóng góp engineering quality.

---

### 🥉 Improvement #3 — Đổi text base BERT-base → MPNet

#### Vấn đề
`config.yaml`:
```yaml
qformer_text_model_name: "bert-base-uncased"
```

BERT-base (2018) train chủ yếu với MLM trên Wikipedia + BookCorpus. Sentence-transformer community đã chứng minh các model mới hơn outperform đáng kể cho semantic similarity tasks:

| Model | MTEB avg | Đặc tính |
|---|---|---|
| bert-base-uncased | ~38 | MLM only |
| all-mpnet-base-v2 | ~57 | MLM + permuted LM + sentence-level fine-tune |
| **beeformer/Llama-movielens-mpnet** | — | MPNet **+ trained on ML interactions** |

#### Đề xuất
Đổi config:
```yaml
# Option A — pure semantic upgrade
qformer_text_model_name: "sentence-transformers/all-mpnet-base-v2"

# Option B — connect với beeFormer discussion (CF-aware text encoder)
qformer_text_model_name: "beeformer/Llama-movielens-mpnet"
```

Verify shape match:
- BERT-base: hidden_size=768 ✓ (match SigLLM)
- MPNet-base: hidden_size=768 ✓ (match SigLLM)
- Tokenizer: cả 2 đều subword nhưng vocabulary khác → cần re-tokenize toàn bộ Stage 1 dataset

#### Code change
Code đã support `qformer_text_model_name` param. `_init_text_branch_from_pretrained_bert` dùng `BertModel.from_pretrained(name)` — cần đổi sang `AutoModel.from_pretrained(name)` để handle MPNet.

```python
# hf_qformer_adapter.py:124
# from transformers import BertModel
from transformers import AutoModel

# Line 124
bert = AutoModel.from_pretrained(bert_model_name)
```

#### Effort
- 30 phút code (AutoModel swap + verify tokenizer)
- Retrain Stage 1 + 2 + 3 (text encoder distribution khác nhau)

#### Expected impact
- Stage 1 ITC/ITM/ITG losses converge tới điểm thấp hơn (text encoder mạnh hơn)
- Q-Former text branch quality cao hơn → bottleneck tốt hơn
- Option B (beeFormer-MPNet) thêm CF-awareness vào text init

#### Defensibility
Option A: standard upgrade, low novelty.
Option B: kết hợp 2 paradigm (beeFormer + SigLLM), **high novelty**. Có thể publishable contribution.

---

### Improvement #4 — Take LAST 4 BERT layers thay vì FIRST 4

#### Vấn đề
```python
# Hiện tại — hf_qformer_adapter.py:140-141
for q_key, q_tensor in target_state.items():
    bert_key = q_key.replace(".attention.attention.", ".attention.self.")
    # ... copies bert.encoder.layer.0-3 vào q-former.layer.0-3
```

Code copy **first 4 layers của BERT-base** (layer 0-3). Layer đầu BERT chứa **low-level features** (syntactic, surface patterns). Layer cuối (8-11) chứa **abstract semantic, task-relevant features**.

Với 8-query tight bottleneck của SigLLM, abstract features có thể useful hơn surface features.

#### Đề xuất
Cherry-pick 4 layers cuối thay vì đầu:

```python
def _init_text_branch_from_pretrained_bert(self, bert_model_name, prefer_last_layers=True):
    bert = AutoModel.from_pretrained(bert_model_name)
    bert_state = bert.state_dict()
    target_state = self.qformer.state_dict()

    num_q_layers = len([k for k in target_state if "encoder.layer." in k and ".query.weight" in k])
    num_bert_layers = len([k for k in bert_state if "encoder.layer." in k and ".query.weight" in k])

    # NEW: layer index offset
    if prefer_last_layers and num_q_layers < num_bert_layers:
        layer_offset = num_bert_layers - num_q_layers  # e.g. 12 - 4 = 8
    else:
        layer_offset = 0

    loaded = 0
    for q_key, q_tensor in target_state.items():
        # Remap layer index in BERT key
        if "encoder.layer." in q_key:
            # extract local layer idx
            local_idx = int(q_key.split("encoder.layer.")[1].split(".")[0])
            bert_layer_idx = local_idx + layer_offset
            bert_key = q_key.replace(
                f"encoder.layer.{local_idx}.",
                f"encoder.layer.{bert_layer_idx}.",
            )
        else:
            bert_key = q_key

        bert_key = bert_key.replace(".attention.attention.", ".attention.self.")
        if bert_key not in bert_state:
            continue
        bert_tensor = bert_state[bert_key]
        if bert_tensor.shape != q_tensor.shape:
            continue
        target_state[q_key] = bert_tensor.clone()
        loaded += 1

    self.qformer.load_state_dict(target_state, strict=True)
```

#### Effort
- 1 giờ code (modify `_init_text_branch_from_pretrained_bert`)
- Retrain Stage 1 + 2 + 3

#### Expected impact
- Uncertain — depends on task fit
- Có thể giúp nhẹ (abstract features useful cho rec)
- Hoặc neutral (rec task không cần BERT's deepest features)

#### Defensibility
Diagnostic experiment — kết quả either way đều informative cho thesis (chứng minh init details matter / không matter).

---

## Tổng hợp & recommendation

### Combo experiment đề xuất

| Combo | Components | Effort | Expected ROI |
|---|---|---|---|
| **A** (recommended) | #1 + #2 | 1 ngày code + 2 ngày train | High — addresses 2 root issues, độc lập |
| B | #1 + #2 + #3 (Option A) | 1.5 ngày + 2 ngày | Medium-high — thêm MPNet upgrade |
| C (ambitious) | #1 + #2 + #3 (Option B) | 2 ngày + 2 ngày | **Very high** — kết hợp beeFormer paradigm, novel contribution mạnh |
| D | #4 only | 1 giờ + 2 ngày | Low — diagnostic, không hứa hẹn lớn |

### Decision matrix

| Time available | Recommended |
|---|---|
| < 3 ngày | Skip, focus viết thesis với current negative findings |
| 3-7 ngày | **Combo A** (#1 + #2) |
| > 1 tuần | **Combo C** (#1 + #2 + beeFormer-MPNet) |

### Order of operations cho Combo A (recommended)

1. **Day 1 morning** — Implement #1 (cluster init):
   - Compute item-text BERT embeddings, KMeans → centroids
   - Save to `ckpt/qformer_init/query_centroids.pth`
   - Modify `HFQFormerAdapter.__init__` to accept `query_init` param

2. **Day 1 afternoon** — Implement #2 (proj_cf warm-start):
   - Compute OLS W, b from MF + BERT embeddings
   - Save to `ckpt/qformer_init/proj_cf_warmstart.pth`
   - Modify `HFQFormerAdapter.__init__` to accept `proj_cf_init` param

3. **Day 2 morning** — Stage 1 retrain (~4-6 hours):
   - With both inits enabled
   - Monitor Stage 1 train loss vs baseline trajectory

4. **Day 2 afternoon** — Stage 2 retrain (~2-3 hours):
   - Load Stage 1 ckpt với inits đã được "consumed" qua Stage 1 training

5. **Day 3** — Stage 3 step 1 + step 2 retrain (~6 hours)

6. **Day 3 evening** — Eval on test/test_warm/test_cold + compare với baseline

### Hypothesis & verdict matrix

| Outcome | Verdict | Thesis implication |
|---|---|---|
| Stage 1 loss converge faster + Stage 3 uAUC > 0.715 | **Win** — init was the bottleneck | New positive contribution: "smart init unlocks Q-Former" |
| Stage 1 faster but Stage 3 uAUC same ~0.71 | **Engineering win** | Reduces compute, marginal thesis value |
| Stage 1 same + Stage 3 same | **Negative** | Confirms 4 root causes — init không phải bottleneck |
| Stage 3 uAUC < baseline | **Regression** | Init choice broke alignment — informative negative |

Cả 4 outcomes đều có defensible story cho thesis.

---

## Reproducibility

### Files cần tạo/modify

| File | Change |
|---|---|
| `src/sigllm/models/q_former/hf_qformer_adapter.py` | Add `query_init`, `proj_cf_init`, `prefer_last_layers` params |
| `scripts/compute_qformer_init.py` (new) | Script offline tính centroids + OLS warm-start, save `.pth` |
| `configs/config.yaml` | Add `qformer_config.use_cluster_query_init: True`, `use_proj_cf_warmstart: True` |
| `docs/QFORMER_INIT_IMPROVEMENTS.md` | This file |

### Branch

`feat/smart-qformer-init` — branched từ `feat/best-0.72-warm` (clean baseline).

### Training command (sau khi implement)

```bash
# 1. Pre-compute inits (1 lần)
python scripts/compute_qformer_init.py \
    --mf-ckpt /content/SigLLM/ckpt/mf/mf_model.pth \
    --item-meta data/processed/ml-1m/movies.csv \
    --bert-model bert-base-uncased \
    --num-clusters 8 \
    --output-dir ckpt/qformer_init/

# 2. Stage 1 với smart init
python -m sigllm.pipelines.multimodal.train_qformer_stage1_representation \
    --cfg-path configs/config.yaml \
    --options \
        model.qformer_config.use_cluster_query_init=True \
        model.qformer_config.use_proj_cf_warmstart=True

# 3. Stage 2 + 3 standard (inherit từ Stage 1 ckpt)
python -m sigllm.pipelines.multimodal.train_qformer_stage2_generative --cfg-path configs/config.yaml
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step1_lora --cfg-path configs/config.yaml
python -m sigllm.pipelines.multimodal.train_qformer_stage3_step2_cie --cfg-path configs/config.yaml
```

---

## References

- BLIP-2 paper: Section 3.1 "Bootstrapping Vision-Language Representation Learning from a Frozen Image Encoder" — Q-Former architecture description, 188M params, BERT init convention
- beeFormer paper (`docs/beeFormer.pdf`): MPNet trained on interaction data as candidate for Improvement #3 Option B
- ABLATION_RESULTS.md: Current negative findings that motivate this investigation
- Code locations:
  - `src/sigllm/models/q_former/hf_qformer_adapter.py:46-118` — `__init__` + `_init_text_branch_from_pretrained_bert`
  - `src/sigllm/pipelines/multimodal/train_qformer_stage1_representation.py:77-86` — Stage 1 adapter creation (no `init_from_pretrained_text` arg → defaults to `True`)
