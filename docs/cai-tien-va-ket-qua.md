# Cải tiến mô hình SigLLM (MF + Q-Former + LLM) và Kết quả Thực nghiệm

> Tài liệu tổng hợp phục vụ luận văn. Gồm: (1) tổng quan kiến trúc, (2) bốn cải
> tiến A/B/C/F kèm giải thích kỹ thuật, (3) kết quả thực nghiệm lần chạy này
> (số liệu lấy trực tiếp từ log huấn luyện/đánh giá), (4) bảng so sánh với các
> công trình liên quan, (5) phân tích – hạn chế – hướng phát triển.
>
> **Nhánh:** `feat/multi-token-uauc-ranking-loss` (gốc `feat/multi-token-cf`).
> **Dataset:** MovieLens-1M (ML-1M), bài toán CTR nhị phân (Yes/No).
> **Backbone LLM:** Qwen2-7B-Instruct (đông cứng) + LoRA + CoRA injector.

---

## 1. Tổng quan kiến trúc

Mô hình gồm ba khối nối tiếp:

```
                         lịch sử + user + target (ID)
                                   │
                  ┌────────────────▼─────────────────┐
                  │  MF (Matrix Factorization)        │  ← tín hiệu Collaborative Filtering
                  │  → e_u, e_i  (đã train & đông cứng)│
                  └────────────────┬─────────────────┘
                                   │  user_cf, target_cf, history_cf
                  ┌────────────────▼─────────────────┐
                  │  Q-Former (InstructBLIP)          │  ← nén CF + align với text
                  │  → cf_q : [B, Q=16, d_model=768]  │
                  └────────────────┬─────────────────┘
              soft token │                    │ weight delta (CoRA)
                  ┌───────▼────────────────────▼──────┐
                  │  LLM Qwen2-7B-Instruct (frozen)    │  ← suy luận, trả lời "Yes/No"
                  │  + LoRA + CoRA injector            │
                  └────────────────────────────────────┘
```

- **MF** cung cấp tín hiệu collaborative (ai thích phim nào), được huấn luyện riêng
  rồi **đông cứng**.
- **Q-Former** biến embedding CF thành một khối *soft token* mà LLM đọc được, đồng
  thời được pretrain để **align CF ↔ ngữ nghĩa văn bản** (title/genre) qua các loss
  ITC/ITM/ITG (Stage 1) và generative (Stage 2).
- **LLM đông cứng**: chỉ LoRA + CoRA injector + Q-Former + projection được huấn
  luyện. Nhãn giám sát chỉ là **một token Yes/No** (Stage 3).

Quy trình huấn luyện 3 giai đoạn:
1. **Stage 1** — Representation pretraining cho Q-Former (alignment/contrastive).
2. **Stage 2** — Generative pretraining (Q-Former + projector).
3. **Stage 3** — Fine-tune Yes/No với LLM đông cứng. Chia hai bước:
   - *Step 1*: baseline LoRA, prompt **chỉ-text** (không có tín hiệu CF).
   - *Step 2*: mô hình đầy đủ, **bơm tín hiệu CF** (soft token và/hoặc CoRA).

---

## 2. Bốn cải tiến

Tóm tắt nhanh:

| Mã | Tên | Mục tiêu | File bị tác động |
|----|-----|----------|------------------|
| **A** | CoRA – tiêm CF vào *trọng số* LLM | Tín hiệu collaborative ngấm sâu, tránh nhiễu input-space | `models/multimodal/qformer_rec_llm.py`, **mới** `models/projection/collaborative_lora_injector.py`, `configs/config.yaml` |
| **B** | Tách projector user / item | Map đúng hình học user-factor vs item-factor trong không gian MF | `models/q_former/hf_qformer_adapter.py` |
| **C** | Backbone Instruct + prompt ChatML | Tận dụng prior instruction-following cho bài toán Yes/No | `models/multimodal/qformer_rec_llm.py`, `configs/config.yaml` |
| **F** | Cân bằng contrastive user-item ở Stage 1 | Không để user-item áp đảo, cứu align item-text (ITC/ITM/ITG) | `configs/config.yaml` (phải rebuild pkl) |
| (kèm) | Script `eval_test.py` | Đánh giá trên test / test_warm / test_cold | **mới** `pipelines/multimodal/eval_test.py` |

---

### 2.A. CoRA — tiêm tín hiệu collaborative vào *trọng số* LLM

**Vấn đề.** Cách cũ chỉ nhét soft token CF vào *input* của prompt (slot `<CFTokens>`).
Khi đó tín hiệu ID (collaborative) và ngữ nghĩa text **chen lấn** trong cùng không
gian đầu vào; attention phải tự gỡ rối, và ảnh hưởng của soft token tới các tầng
sâu chỉ là gián tiếp → tín hiệu CF nhạt dần.

**Ý tưởng (theo CoRA, arXiv:2408.10645).** Thay vì chỉ ghép vào input, biến các
query collaborative của Q-Former `Z = cf_q : [B, Q, d_model]` thành một **weight
delta low-rank riêng cho từng sample**, cộng thẳng vào các lớp attention
`q_proj`/`v_proj` của LLM:

```
A = down(Z)      # [B, Q, in]
B = up(Z)        # [B, Q, out]
delta = scaling · gate · ((x @ Aᵀ) @ B)
output = output + delta
```

**Điểm khác LoRA thường (chữ "Co" = Collaborative).** LoRA chuẩn học **một** delta
cố định, dùng chung cho mọi input (thích nghi *tác vụ*). CoRA ở đây sinh delta **từ
query của từng sample** → **mỗi user/phim một delta riêng** (cá nhân hoá). Generator
`(down, up)` được **chia sẻ qua mọi tầng**, chỉ **tách theo shape trọng số** (một bộ
cho mọi `q_proj`, một bộ cho mọi `v_proj`) → giữ số tham số thấp, chống overfit trên
~33k mẫu; biến thiên theo sample đến hoàn toàn từ `Z`.

**Bốn kỹ thuật chống sập (tác giả đã gặp lỗi thật, ghi trong code).** Đây là phần
quan trọng nhất vì "ghép một đường mới lên LLM đông cứng" rất dễ phân kỳ:

1. **Zero-init tanh gate (kiểu Flamingo):** mỗi shape có `gate` khởi tạo 0 →
   `tanh(0)=0` → delta **bắt đầu đúng bằng 0**, LLM giữ nguyên hành vi gốc rồi mở
   dần, bị `tanh` chặn trong (−1, 1).
   *Bản không gate (query L2≈20 × scaling 1.0 qua 28 tầng) đã **phân kỳ và làm AUC
   sập về 0.5**.*
2. **Không zero-init `up` (chỉ gate mới zero):** nếu cả gate lẫn up đều 0 thì
   gradient của cả hai bằng 0 → "đường chết" không bao giờ bootstrap. Công thức
   Flamingo: gate zero-init, transform init thường (std=0.02).
3. **Chuẩn hoá query về unit-L2** trước generator → tách độ lớn delta khỏi norm
   query lớn (~20) vốn gây phân kỳ.
4. **Tính trong fp32 + xử lý đa device** (`device_map="auto"`) và **bỏ qua khi batch
   lệch** lúc decode incremental, tránh làm hỏng output.

**Cấu hình (`configs/config.yaml`).**
```yaml
cf_injection_mode: "both"     # soft_token | lora_weight | both
cora_alpha: 4                 # scaling = cora_alpha / num_queries
cora_target_modules: ${model.lora_config.target_modules}   # q_proj, v_proj
```
- `soft_token`: như cũ. `lora_weight`: **chỉ** weight delta → phải dùng prompt
  **chỉ-text** (bỏ `<CFTokens>`). `both` (đang dùng): cả hai đường.
- Injector gắn **SAU** LoRA → bọc lên module đã PEFT-wrap; LoRA và CoRA **cùng tồn
  tại**, không thay thế nhau.

**Đánh đổi.** Thêm forward-hook + 2 matmul trên *mỗi* q_proj/v_proj *mỗi* forward
→ chậm và tốn bộ nhớ hơn; độ ổn định nhạy cảm (phải nhờ gate + normalize + alpha
thấp). Comment dặn để `cora_alpha` thấp, tăng từ từ và canh val AUC.

---

### 2.B. Tách projector user / item

**Vấn đề.** Trước đây dùng **một** lớp `proj_cf` chung để chiếu cả embedding user
lẫn item từ không gian MF sang không gian Q-Former. Nhưng hình học của vector
*user-factor* và *item-factor* trong MF khác nhau → ép chung một phép chiếu là
gượng ép.

**Giải pháp.** Tách thành `proj_user` (slot user) và `proj_item` (target + history,
đều là item). Checkpoint cũ chỉ có `proj_cf` được **warm-load vào cả hai** nên không
vỡ tương thích.

**Lợi ích.** Mỗi loại embedding được map theo đặc trưng riêng → biểu diễn CF đưa vào
Q-Former sạch hơn.

---

### 2.C. Backbone Qwen2-7B-**Instruct** + prompt ChatML

**Bối cảnh.** Trước dùng Qwen2-7B-**Base** (không instruction-tuned) nên prompt bỏ
định dạng chat, chỉ thêm đuôi `Answer:`. **Lưu ý:** đây *không phải lỗi* — Base +
LoRA vẫn chạy ra ~0.71 val. Đây là một **lựa chọn thiết kế**.

**Cải tiến.** Chuyển sang bản **Instruct** (giữ nguyên LoRA — `use_lora: True`),
khôi phục định dạng **ChatML** của Qwen2:
```
<|im_start|>system
You are a helpful movie recommendation assistant. Answer only with "Yes" or "No".<|im_end|>
<|im_start|>user
{prompt ... <CFTokens> ...}<|im_end|>
<|im_start|>assistant
```
Code đổi sang `AutoModelForCausalLM`/`AutoTokenizer`, và `_resolve_soft_token_placeholder`
loại trừ các token điều khiển `<|im_start|>`/`<|im_end|>` để slot soft-token không
va vào chúng.

**Vì sao không chỉ "tăng LoRA cho Base"?** Đã thử và **tệ hơn**: r=16 [q,k,v,o]
(~10M params) **overfit** trên 33k mẫu (val uAUC 0.7048 vs r=8 baseline 0.7077),
train loss giảm nhanh gấp đôi. Kết luận trong code: *"LoRA capacity is NOT the
bottleneck; Q-Former Stage 1 quality is."* Cái thiếu là **prior tốt hơn** — đúng
thứ Instruct cho sẵn, để LoRA bé dồn sức vào việc *dùng tín hiệu collaborative* thay
vì học lại cách làm theo chỉ dẫn.

**Vì sao backbone và prompt đi kèm nhau.** Instruct được train với khuôn ChatML;
cho nó ăn prompt completion sẽ phí prior. Cho Base ăn ChatML thì Base không hiểu
token điều khiển. → Đổi backbone bắt buộc đổi format prompt.

---

### 2.F. Cân bằng contrastive user-item ở Stage 1

**Cơ chế Stage 1.** Builder tạo **một danh sách phẳng** trộn 3 loại mẫu:
`item_text` (phim↔text — *mục tiêu chính*), `item_item`, `user_item` (collaborative).
DataLoader shuffle bình thường; mỗi batch được **tách theo loại** rồi tính loss
riêng. Then chốt: một loss **chỉ chạy khi batch có ≥2 mẫu cùng loại**:
```python
if item_text_idx.numel() >= 2:   # ITC/ITM/ITG mới fire
if w_ui > 0 and user_item_idx.numel() >= 2:   # user-item fire
```

**Vấn đề.** ML-1M có ~**888K** cặp positive user-item nhưng chỉ ~**3K** cặp
item-text. Tỉ lệ item_text ≈ 0.34%; với batch 64, kỳ vọng ≈ 0.22 mẫu/batch → đa số
batch có 0–1 mẫu → điều kiện `≥2` **hiếm đúng** → **ITC/ITM/ITG gần như không train**,
trong khi user-item fire liên tục. Q-Former học align ngôn ngữ kém → soft token mất
nghĩa.

**Giải pháp (2 nút ở 2 tầng).**

| Nút | Tầng | Tác động |
|-----|------|----------|
| `max_user_item_pairs: 20000` | **Data (lúc build pkl)** | Đổi *tần suất* user-item → item_text lên ~13%, batch 64 ⇒ ~8 mẫu ⇒ `≥2` gần như luôn đúng |
| `w_ui: 1.0 → 0.5` | **Loss (lúc train)** | Giảm *cường độ* gradient user-item để điều hoà, không áp đảo |

**Vì sao đổi `max_user_item_pairs` BẮT BUỘC rebuild pkl.** Cap được **nướng cứng
vào file dữ liệu** (cắt `user_item_samples[:20000]` lúc build), còn trainer chỉ load
pkl. Nếu chỉ sửa config mà không chạy lại:
```bash
python -m sigllm.pipelines.multimodal.build_qformer_dataset --cfg-path configs/config.yaml
```
thì pkl cũ (đủ 888K cặp) vẫn được nạp → **config mới vô tác dụng**. Ngược lại `w_ui`
đọc lúc train nên có hiệu lực ngay.

**Caveat.** Cap hiện là **cắt thẳng `[:20000]`** (không lấy mẫu ngẫu nhiên); nếu
`pos_df` sắp theo user/thời gian thì có thể thiên lệch. Nên shuffle trước khi cắt.
Comment gợi ý **sweep** `cap ∈ {10000, 20000, 40000}`, `w_ui ∈ {0.2, 0.5, 1.0}`.

---

## 3. Kết quả thực nghiệm (lần chạy này)

> Nguồn: log Stage-3 (`ckpt/.../log.txt`) và đầu ra `eval_test.py` (chạy
> 2026-06-27). Cấu hình: Qwen2-7B-Instruct (frozen) + LoRA r=8 [q,v] +
> `cf_injection_mode=both`, `cora_alpha=4`, Q=16 soft tokens, d_model=768.
> Số trainable: Q-Former=76,227,840; llm_proj=2,763,264; LLM/LoRA đông cứng ở
> bước eval.

### 3.1. Ablation tín hiệu collaborative (validation)

| Mô hình (Stage 3) | Prompt | CF injection | val AUC | val uAUC |
|-------------------|--------|--------------|:-------:|:--------:|
| **Step 1** – baseline | chỉ-text | ❌ không | 0.7225 | 0.7057 |
| **Step 2** – đầy đủ | có `<CFTokens>` | ✅ `both` (soft+CoRA) | **0.7589** | **0.7088** |
| | | **Δ (cải thiện)** | **+0.0364** | **+0.0031** |

→ Bơm tín hiệu collaborative (đa-token CF + CoRA) nâng val **AUC +3.64 điểm**; uAUC
nhích nhẹ (+0.31 điểm). Cho thấy CF chủ yếu cải thiện khả năng phân biệt toàn cục
(AUC) hơn là xếp hạng nội-user (uAUC) trong cấu hình này.

### 3.2. Kết quả TEST-SET (mô hình Step 2 đầy đủ)

| Split | AUC | uAUC | ACC@0.5 | LogLoss | pos_rate | #user (uAUC) |
|-------|:---:|:----:|:-------:|:-------:|:--------:|:------------:|
| **test** (tổng) | 0.7508 | 0.7041 | 0.6824 | 0.5926 | 0.545 | 224 |
| **test_warm** | **0.7830** | **0.7145** | 0.7065 | 0.5631 | 0.531 | 151 |
| **test_cold** | 0.7039 | 0.6372 | 0.6563 | 0.6246 | 0.571 | 51 |

**Nhận xét.**
- **Warm ≫ cold** (AUC 0.783 vs 0.704; uAUC 0.715 vs 0.637): khoảng cách ~8 điểm
  uAUC cho thấy mô hình **phụ thuộc đáng kể vào lịch sử tương tác** — user/phim ít
  dữ liệu (cold) khó hơn rõ rệt. Đây là hành vi kỳ vọng của hệ CF-augmented.
- **test tổng** nằm giữa warm và cold, lệch về cold do tỉ lệ mẫu khó.
- LogLoss cold cao hơn warm (0.625 vs 0.563) ⇒ mô hình **kém tự tin & sai nhiều
  hơn** ở cold; quan sát định tính trong log: nhiều ca cold dự đoán "Yes" quá tay
  (pred_pos_rate 0.677 > pos_rate 0.571).

### 3.3. Chỉ số phân tách điểm (score separation, test)

| Split | pos_score_mean | neg_score_mean | score_gap |
|-------|:--------------:|:--------------:|:---------:|
| test | 0.687 | 0.474 | 0.213 |
| test_warm | 0.710 | 0.458 | 0.252 |
| test_cold | 0.666 | 0.506 | 0.160 |

Khoảng cách điểm pos–neg ở warm rộng hơn cold ~0.09 → củng cố nhận định warm dễ
phân biệt hơn.

---

## 4. So sánh với các công trình liên quan

> ⚠️ **CẢNH BÁO HỌC THUẬT — PHẢI ĐỌC.** Các ô số liệu của **paper khác bên dưới để
> TRỐNG**. Người viết luận văn **bắt buộc tự điền từ paper gốc** (bảng kết quả ML-1M
> trong chính bài báo) và **trích dẫn đầy đủ**. Tuyệt đối **không** bịa/ước lượng số
> của paper khác — sai lệch trích dẫn là lỗi nghiêm trọng trong luận văn. Chỉ các
> dòng "**Mô hình của luận văn (lần chạy này)**" là số thật, đã verify từ log.
>
> Lưu ý so sánh công bằng: phải khớp **cùng dataset (ML-1M), cùng cách chia
> warm/cold, cùng định nghĩa AUC/uAUC, cùng tiền xử lý nhãn**. Giao thức đánh giá
> (CTR Yes/No + AUC + uAUC + warm/cold) trong repo này tương đồng họ **CoLLM**;
> nên ưu tiên đối chiếu các baseline mà CoLLM dùng.

### 4.1. Bảng so sánh (điền số paper vào ô trống)

| Phương pháp | Nhóm | test AUC | test uAUC | warm AUC | cold AUC | Nguồn |
|-------------|------|:--------:|:---------:|:--------:|:--------:|-------|
| MF | CF thuần | _… điền …_ | _…_ | _…_ | _…_ | [?] |
| LightGCN | CF thuần | _…_ | _…_ | _…_ | _…_ | [?] |
| DIN / DeepFM | CTR truyền thống | _…_ | _…_ | _…_ | _…_ | [?] |
| TALLRec | LLM-Rec (text) | _…_ | _…_ | _…_ | _…_ | [?] |
| PromptRec / P5 | LLM-Rec | _…_ | _…_ | _…_ | _…_ | [?] |
| CoLLM | LLM + CF embed | _…_ | _…_ | _…_ | _…_ | [?] |
| CoRA | LLM + CF weight | _…_ | _…_ | _…_ | _…_ | [arXiv:2408.10645] |
| **Mô hình của luận văn (Step 1, baseline text-only)** | LLM + LoRA | — | — | — | — | val AUC 0.7225 / uAUC 0.7057 |
| **Mô hình của luận văn (Step 2, đầy đủ)** | LLM + Q-Former + CoRA | **0.7508** | **0.7041** | **0.7830** | **0.7039** | log lần chạy này |

### 4.2. Gợi ý nguồn để điền (cần tự kiểm chứng số chính xác)

- **CoLLM** — Zhang et al., *"CoLLM: Integrating Collaborative Embeddings into Large
  Language Models for Recommendation"*, arXiv:2310.19488. (Giao thức ML-1M
  AUC/uAUC + warm/cold sát với repo này nhất.)
- **CoRA** — *"Collaborative Information Perception by LLM's Weights"*,
  arXiv:2408.10645. (Nguồn ý tưởng Cải tiến A.)
- **TALLRec** — Bao et al., RecSys 2023, arXiv:2305.00447.
- **P5** — Geng et al., RecSys 2022. **DIN** — Zhou et al., KDD 2018.
  **LightGCN** — He et al., SIGIR 2020.

> Cách lấy số đúng chuẩn: mở bảng kết quả ML-1M trong từng paper, copy đúng cột
> AUC/uAUC tương ứng split (overall/warm/cold), ghi rõ số trang/bảng vào chú thích.

---

## 5. Phân tích & Thảo luận

1. **CF injection có tác dụng:** chênh lệch Step 1 → Step 2 (+3.64 AUC) là bằng
   chứng trực tiếp rằng đưa tín hiệu collaborative vào LLM (đa-token CF + CoRA) tốt
   hơn LLM thuần-text trên cùng backbone/cấu hình.
2. **Nút thắt nằm ở chất lượng Q-Former (Stage 1), không phải dung lượng LoRA:**
   thí nghiệm tăng LoRA gây overfit; trong khi cải thiện Stage 1 (Cải tiến F: cân
   bằng contrastive) mới là đòn bẩy. Đây là luận điểm xuyên suốt nối A↔C↔F.
3. **Khoảng cách warm–cold lớn** chỉ ra hướng cải thiện rõ nhất: tăng cường biểu
   diễn cho user/item ít dữ liệu (cold-start) — ví dụ khai thác mạnh hơn nhánh
   item-text alignment, hoặc meta-learning cho cold.
4. **Quan hệ A–C:** Instruct (C) cấp prior để LoRA bé tập trung vào việc dùng CF;
   CoRA (A) là kênh đưa CF vào sâu. Hai cái bổ trợ.

---

## 6. Hạn chế

- **Chưa có ablation tách riêng từng cải tiến** (A/B/C/F bật-tắt độc lập). Hiện chỉ
  có baseline (Step 1) vs đầy đủ (Step 2). Nên bổ sung lưới ablation để định lượng
  đóng góp từng phần (xem §7).
- **Một seed, một lần chạy** — chưa có khoảng tin cậy/độ lệch chuẩn.
- **`best_epoch` của Step 2 ở epoch 5** với val AUC dao động 0.72–0.76 giữa các
  epoch ⇒ huấn luyện còn nhiễu; nên báo cáo trung bình nhiều seed.
- **Cap user-item là cắt thẳng** (không random) — rủi ro thiên lệch dữ liệu Stage 1.
- **Số baseline paper chưa điền** (xem §4 — phải tự bổ sung).

---

## 7. Hướng phát triển / thí nghiệm đề xuất

- **Ablation Cải tiến A:** `soft_token` → `lora_weight` (prompt chỉ-text) → `both`;
  sweep `cora_alpha ∈ {2, 4, 8}`. Đo CoRA đóng góp bao nhiêu so với soft token thuần.
- **Ablation Cải tiến C:** chạy lại đúng cấu hình này nhưng `llm_model=qwen2-7b-base`
  + prompt template cũ, để đo Instruct hơn Base bao nhiêu.
- **Ablation Cải tiến F:** sweep `max_user_item_pairs ∈ {10000, 20000, 40000}` ×
  `w_ui ∈ {0.2, 0.5, 1.0}` (nhớ rebuild pkl mỗi lần đổi cap).
- **Cold-start:** tăng trọng số/độ phủ item-text alignment; thử augmentation cho
  user/item thưa.
- **Robustness:** ≥3 seed, báo cáo mean ± std.

---

## 8. Phụ lục — Thay đổi mã nguồn đã áp dụng

So với nhánh nền `feat/multi-token-cf`, đã merge vào nhánh hiện tại
(`feat/multi-token-uauc-ranking-loss`, giữ nguyên phần uAUC ranking loss đang có):

| File | Loại | Cải tiến |
|------|------|----------|
| `src/sigllm/models/projection/collaborative_lora_injector.py` | **mới** (233 dòng) | A |
| `src/sigllm/pipelines/multimodal/eval_test.py` | **mới** (153 dòng) | (đánh giá test) |
| `src/sigllm/models/multimodal/qformer_rec_llm.py` | sửa (merge 3-chiều) | A + C |
| `src/sigllm/models/q_former/hf_qformer_adapter.py` | sửa | B |
| `configs/config.yaml` | sửa (merge 3-chiều) | A + C + F |

**Lệnh chạy đánh giá test (tham khảo từ docstring `eval_test.py`):**
```bash
# Step 2 (mô hình đầy đủ soft-token / both)
python -m sigllm.pipelines.multimodal.eval_test --cfg-path configs/config.yaml \
    --step 2 \
    --model-ckpt   ckpt/qformer_stage3_step1_lora_qwen2/qwen2-7b-instruct/checkpoint_best.pth \
    --overlay-ckpt ckpt/qformer_stage3_step2_cie_qwen2/qwen2-7b-instruct/checkpoint_best.pth

# Step 1 (baseline text-only)
python -m sigllm.pipelines.multimodal.eval_test --cfg-path configs/config.yaml \
    --step 1 \
    --model-ckpt   ckpt/qformer_stage3_step1_lora_qwen2/qwen2-7b-instruct/checkpoint_best.pth
```

---

*Tài liệu này tổng hợp tự động từ phân tích mã nguồn + log thực nghiệm. Số liệu mô
hình của luận văn lấy trực tiếp từ log ngày 2026-06-27; số liệu paper khác cần tự
điền và trích dẫn (xem §4).*
