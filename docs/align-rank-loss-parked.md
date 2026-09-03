# Hàm mất mát bảo toàn thứ hạng (align-rank loss) — ĐÃ GỠ KHỎI LUẬN VĂN (parked)

**Ngày gỡ:** 2026-09-03 · **Nhánh:** feat/plain-qformer-sota · **Lý do:** đóng góp đo được quá
mỏng để tính là một kỹ thuật/đóng góp — chỉ +0,0104 uAUC trên **một** tập (Amazon-Book), **một
seed**; ML-1M KHÔNG bật. Người dùng yêu cầu gỡ hoàn toàn, giữ nguyên khả năng apply lại.

> Đây là bản lưu trữ để **khôi phục nguyên trạng** khi cần. Code + config KHÔNG bị đụng (chỉ gỡ
> phần văn/bảng trong `thesis/`).

---

## 1. Đóng góp thực đo được (toàn bộ bằng chứng)

| Tập | Cấu hình | AUC | uAUC |
|---|---|---|---|
| Amazon-Book | Run-1 (chỉ + điều kiện hóa user, align-rank **OFF**) | 0,8147 | 0,5958 |
| Amazon-Book | Run-2 (+ align-rank **ON**, λ=0,2) | 0,8132 | **0,6062** |
| MovieLens-1M | align-rank **OFF** (uAUC đã bão hòa ~0,70) | — | — |

→ Tác động align-rank = **uAUC +0,0104, AUC −0,0015**, chỉ Book, 1 seed. So sánh: CoRA-MF uAUC 0,6262.
Kết quả headline của luận văn (ML-1M 0,7475/0,6968; Book Run-1 0,8147/0,5958 — đều vượt CoLLM/BinLLM)
**KHÔNG phụ thuộc** align-rank.

## 2. Config (để bật lại)

`configs/config.yaml` và `configs/config_book.yaml`:
- `model.align_rank_loss`: weight `0.2` (Run-2 book; đặt `0.0` để tắt). Per-user BPR trên **soft token
  cộng tác đã căn chỉnh** (khác `ranking_loss` vốn áp trên logit Yes/No của LLM).
- Yêu cầu `run.user_grouped_batch.enabled: true` (batch chứa nhiều mẫu cùng user để có cặp pos/neg).
- τ (temperature) trong công thức là siêu tham số của loss.

Code: `src/coqllm/models/multimodal/qformer_rec_llm.py` (align_rank head, zero-init),
`src/coqllm/common/user_grouped_sampler.py`, `src/coqllm/runners/utils/dataloader_builder.py`.

## 3. Văn bản LaTeX đã gỡ (copy lại nguyên văn khi apply)

### 3a. ch3 — §3.3.5 (cả subsection + eq), đặt lại NGAY TRƯỚC `\section{Biểu diễn cộng tác đa token}`

```latex
\subsection{Hàm mất mát bảo toàn thứ hạng theo người dùng}
\label{ssec:align-rank}

Hàm mất mát mặc định trong giai đoạn tinh chỉnh CTR là entropy chéo nhị phân (BCE) trên nhãn
``Yes''/``No''. BCE tối ưu khả năng phân loại từng mẫu nhưng không trực tiếp tối ưu thứ tự
tương đối giữa các sản phẩm của cùng một người dùng, trong khi uAUC đo chính đặc tính xếp hạng
nội bộ này.

Để bổ sung tín hiệu xếp hạng, luận văn sử dụng một mục tiêu BPR theo cặp trong phạm vi từng
người dùng (\textit{align-rank loss}), áp trực tiếp lên soft token cộng tác đã căn chỉnh
(trước khi vào LLM) thông qua một đầu ra phụ cũng khởi tạo bằng không:

\begin{equation} \label{eq:align-rank}
\mathcal{L} = \mathcal{L}_{\text{BCE}}
  + \lambda \sum_{u} \sum_{(i^{+},\, i^{-}) \in \mathcal{P}_u}
    -\log \sigma\!\Big( \tfrac{1}{\tau}\big( s_u(i^{+}) - s_u(i^{-}) \big) \Big),
\end{equation}

trong đó $s_u(\cdot)$ là điểm xếp hạng được sinh từ soft token cộng tác đã căn chỉnh,
$\mathcal{P}_u$ là tập các cặp dương--âm của cùng người dùng $u$, $\sigma$ là hàm sigmoid,
$\tau$ là nhiệt độ và $\lambda$ là trọng số của thành phần xếp hạng.

Mục tiêu phụ được áp trực tiếp lên biểu diễn cộng tác sau căn chỉnh, thay vì chỉ lên logit
``Yes/No'' của LLM, nhằm buộc bản thân cầu nối giữ đúng thứ tự giữa sản phẩm dương và âm trong
từng người dùng. Để tạo đủ cặp dương--âm trong một batch, quá trình huấn luyện sử dụng lấy mẫu
theo nhóm người dùng (\textit{user-grouped batch}) và mục tiêu được kích hoạt ở Bước 2 của Giai
đoạn 3 (nơi Q-Former và lớp chiếu được huấn luyện). Do đầu ra phụ khởi tạo bằng không, các
epoch đầu gần như trung tính và tác động của mục tiêu xếp hạng hiện dần về sau. Đây là một siêu
tham số phụ thuộc tập dữ liệu: nó cải thiện uAUC trên tập còn nhiều dư địa xếp hạng trong từng
người dùng (Amazon-Book) và gần như trung tính khi uAUC đã bão hòa (MovieLens-1M).
```

### 3b. ch4 — hàng Run-2 trong `tab:ket-qua-book` (đặt lại sau hàng Run-1)

```latex
\textbf{CoQLLM (+ bảo toàn thứ tự, Run-2)} & \textbf{0{,}8132} & \textbf{0{,}6062} \\
\hline
```
Caption gốc có: "Run-1 = + điều kiện hóa; Run-2 = thêm bảo toàn thứ tự."

### 3c. ch4 — hàng Run-2 trong `tab:ablation`

```latex
Amazon-Book & + bảo toàn thứ tự (Run-2) & 0{,}8132 & \textbf{0{,}6062} \\
\hline
```

### 3d. ch4 — `tab:warm-cold` GỐC (Book warm/cold lấy từ Run-2; caption: "Amazon-Book dùng Run-2 (+ bảo toàn thứ tự)")

```latex
MovieLens-1M & toàn bộ & $7{.}331$ & 0{,}7475 & 0{,}6968 \\
MovieLens-1M & warm & $4{.}153$ & \textbf{0{,}7787} & \textbf{0{,}7254} \\
MovieLens-1M & cold & $3{.}178$ & 0{,}7030 & 0{,}5969 \\
Amazon-Book & toàn bộ & $2{.}486$ & 0{,}8132 & 0{,}6062 \\
Amazon-Book & warm & $1{.}648$ & \textbf{0{,}8190} & \textbf{0{,}6238} \\
Amazon-Book & cold & $630$ & 0{,}7928 & 0{,}5258 \\
```
⚠️ Khi gỡ align-rank, **các hàng Amazon-Book warm/cold bị bỏ** (chúng đo trên checkpoint Run-2).
Muốn khôi phục mà KHÔNG bật align-rank thì phải chạy lại warm/cold trên checkpoint Run-1 (chưa có số).

### 3e. Các cụm câu "đóng góp/kỹ thuật" đã gỡ (thêm lại "mục tiêu bảo toàn thứ hạng trong từng người dùng")

- **ch1**: (i) đoạn cuối §khoảng-trống "…điều kiện hóa truy vấn theo người dùng, **mục tiêu bảo toàn
  thứ hạng trong từng người dùng** và biểu diễn cộng tác đa token."; (ii) Mục tiêu 1; (iii) Đóng góp 1.
- **ch3**: §3.3.3 "…hai kỹ thuật huấn luyện — điều kiện hóa (Mục ssec:user-conditioned) **và mục tiêu
  bảo toàn thứ tự (Mục ssec:align-rank)** — nhằm cải thiện lần lượt AUC và uAUC."; §3.5.3 Bước 2 "…**và,
  khi cấu hình cho phép, mục tiêu bảo toàn thứ hạng (Mục ssec:align-rank)**."; §3.6 tổng kết.
- **ch4**: §4.1 dataset "…làm nổi bật vai trò của **mục tiêu bảo toàn thứ tự** (Mục ssec:align-rank)";
  §4.1.5 cài đặt "**hàm mất mát bảo toàn thứ tự (λ=0,2, kèm lấy mẫu theo nhóm người dùng) bật trên
  Amazon-Book, tắt trên MovieLens-1M**"; §4.2 Amazon-Book "Bổ sung hàm mất mát bảo toàn thứ tự (Run-2)
  nâng uAUC lên 0,6062 (+0,0104)… tiệm cận CoRA-MF"; §4.4 ablation (đoạn "hai kỹ thuật tác động hai độ
  đo") + mục "ablation chưa thực hiện (iii) align-rank trên ML-1M".
- **ch5**: §5.1 Đóng góp 1 "…**và mục tiêu bảo toàn thứ tự trong từng người dùng**"; §5.2 Kết quả chính
  (Book Run-2 0,6062, "hai kỹ thuật tác động hai độ đo"); §5.3 hạn chế (ablation align-rank ML-1M);
  §5.4 tương lai "Mở rộng mục tiêu bảo toàn thứ tự (đã hiệu quả trên Amazon-Book) sang miền khác + khảo
  sát λ".

## 4. Hệ quả của việc gỡ (đã chấp nhận)

- Amazon-Book chỉ còn **Run-1: 0,8147 / 0,5958** (vẫn vượt CoLLM-MF & BinLLM re-run trên cả hai độ đo).
- **Book warm/cold biến mất** → finding "trần uAUC = MF cold" giờ tựa **chỉ trên ML-1M** (warm 0,7254
  vs cold 0,5969). Bằng chứng mạnh nhất (Book cold 0,5258 ≈ ngẫu nhiên) tạm mất — apply lại nếu cần.
- Luận văn còn **một** kỹ thuật huấn luyện: điều kiện hóa truy vấn theo người dùng (đòn bẩy AUC).
