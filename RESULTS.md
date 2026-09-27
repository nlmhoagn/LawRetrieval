# DSC 2026 — Task 1: LegalIR — Kết quả

Số liệu từ lần chạy thật, kèm cỡ mẫu và điều kiện đo. Chỗ nào là ước lượng đều ghi rõ.

Cập nhật: 2026-08-08

---

## 1. Điểm public leaderboard

| # | Cấu hình | Public | Delta | Validation | Lệch |
|---|---|---|---|---|---|
| 1 | BM25 + dense zero-shot, pool 1000, α=0.7, top3 | **0.8637** | — | 0.8633 | +0.000 |
| 2 | + fine-tune bi-encoder | **0.8855** | +0.0218 | 0.8927 | −0.007 |
| 3 | + pool 2000, top2 mean | **0.8897** | +0.0042 | 0.9002 | −0.011 |
| 4 | + cross-encoder β=0.5, ndocs=50, mchunks=2 | **0.9005** | +0.0108 | 0.9192 | −0.019 |

Top-1 leaderboard: 0.94.

### Validation đang lạc quan dần

Cột "Lệch" là điều đáng chú ý nhất bảng. Ban đầu validation khớp public gần như tuyệt đối
(+0.0004), giờ đã lệch −0.019. Nguyên nhân: **cùng một tập 1.000 câu bị dùng để chọn tham
số nhiều lần**, mỗi lần chọn lại làm nó lạc quan thêm.

Từ giờ ước public = validation − 0.02, đừng lấy nguyên.

---

## 2. Pipeline hiện tại

```
8.532 văn bản → chunker 3 tầng → 487.194 chunk
        │
   BM25 (k1=2.5, b=0.9) → pool 2.000 chunk
        │
   + dense fine-tune, hoà linear α=0.7 (70% dense / 30% BM25)
        │
   → 50 văn bản × 2 chunk = 100 ứng viên
        │
   + cross-encoder AITeamVN/Vietnamese_Reranker, trộn β=0.5
        │
   → gộp top-2 mean → top-5 văn bản
```

Thời gian sinh bài nộp: **25 phút** (cross-encoder chiếm gần hết).

---

## 3. Từng thành phần đóng góp bao nhiêu

### 3.1 Chunking 3 tầng

```
8.532 văn bản → 487.194 chunk    1m31s, 592 MB
median 165 từ · p99 336 · max 407   (từ max gốc 1.242.409)
0/8.532 văn bản bị mất · 0/3.105 đáp án gold thiếu chunk
```

| Tầng | Điều kiện | Chunk |
|---|---|---|
| 1 | cắt theo `Điều N` | 122.520 |
| 2 | Điều > 350 từ → `Khoản` → cửa sổ cố định | 333.416 |
| 3 | không có `Điều` → cửa sổ 256 từ, overlap 25% | 31.238 |
| 0 | `passage` rỗng → tiêu đề từ slug `link` | 20 |

### 3.2 Gộp chunk → document

Đo 2 lần, kết luận đổi sau khi có fine-tune:

| Cách gộp | BM25 thuần (200 câu) | Sau fine-tune (1.000 câu) |
|---|---|---|
| max | 0.7950 | 0.8785 |
| top2 mean | — | **0.9002** |
| top3 mean | **0.8250** | 0.8952 |
| top5 mean | — | 0.8762 |
| top3 / log(số chunk) | 0.1706 | — |

`top3` đúng cho BM25 thuần nhưng **sai sau khi có dense** — chunk đúng nổi bật hơn nên gộp
3 là pha loãng. `top2` thắng ở cả 24/24 ô lưới pool × α.

`avg` toàn bộ hỏng vì văn bản `68843` có 6.514 chunk: một câu chỉ khớp 2–3 chunk, phần còn
lại điểm 0 kéo trung bình về sát 0.

### 3.3 Fine-tune bi-encoder

5.999 cặp đào từ 6.000 câu train (1.000 câu để validation, không huấn luyện).

- **Positive**: chunk điểm hybrid cao nhất trong văn bản gold (giám sát yếu — nhãn ở mức
  document, model học ở mức chunk)
- **Negative**: chunk thuộc văn bản không phải gold, hạng 5–80. Bỏ 5 hạng đầu vì corpus có
  5.427 văn bản chưa từng là đáp án, rất dễ là false negative
- Loss: `CachedMultipleNegativesRankingLoss` (GradCache) trong `MatryoshkaLoss`
- 93 bước, 11.4 phút, VRAM đỉnh 5.58/6.4 GB, loss 5.12 → 3.34

| α | gốc | fine-tune | delta |
|---|---|---|---|
| 0.0 (BM25 thuần) | 0.8112 | 0.8112 | +0.0000 |
| 0.7 | 0.8633 | **0.8927** | +0.0293 |
| 1.0 (dense thuần) | 0.8307 | 0.8780 | +0.0473 |

`α=0.0` cho delta đúng bằng 0 — phép kiểm tra tự thân, xác nhận không có rò rỉ trong đường đo.

Dense thuần +0.047 nhưng hybrid chỉ +0.029: một phần thứ model học được là thứ BM25 vốn đã
biết. α tối ưu **không dịch** (0.8 → 0.7, coi như hoà) — dự đoán của tôi rằng nó sẽ nhảy lên
0.8–0.9 là sai.

### 3.4 Cross-encoder

| ndocs | mchunks | β | nửa đầu | nửa sau | cả 1000 | delta |
|---|---|---|---|---|---|---|
| 20 | 2 | 0.0 | 0.9057 | 0.8947 | 0.9002 | — |
| 20 | 3 | 0.5 | 0.9243 | 0.9083 | 0.9163 | +0.0162 |
| **50** | **2** | **0.5** | 0.9200 | **0.9183** | 0.9192 | **+0.0190** |
| 50 | 3 | 0.5 | 0.9233 | 0.9153 | 0.9193 | +0.0192 |
| 50 | 2 | 1.0 | 0.8420 | 0.8503 | 0.8462 | −0.0540 |

Bootstrap 95% CI: **[+0.0033, +0.0290]**, thắng 99.4% — tham số duy nhất trong dự án có
khoảng tin cậy không chứa 0.

**Lần thử đầu tôi kết luận sai** ("reranker vô dụng") vì hai lỗi cộng lại:
- `β=1.0` — để cross-encoder **đè** thay vì **trộn**. β=1.0 luôn tệ (−0.023 đến −0.054)
- `ndocs=20` khi trần lúc đó mới 0.9177, gần sát mức đã đạt → +0.03 bị triệt tiêu bởi −0.02

Sau fine-tune, trần `ndocs=20` lên 0.9603 và `ndocs=50` lên 0.9782, nên reranker mới có đất.

---

## 4. Trần hệ thống

Recall ở mức văn bản, pipeline hiện tại (1.000 câu validation):

| top-N | Recall@N | |
|---|---|---|
| 10 | 0.9347 | |
| 20 | 0.9603 | |
| 50 | 0.9782 | ← ndocs đang dùng |
| 120 | 0.9843 | |

Thực tế đạt 0.9192. Dư địa còn **+0.059** chỉ bằng xếp hạng lại, không cần tìm thêm.

---

## 5. Bài học đo đạc

Ở cỡ mẫu 200–500 câu, **chênh lệch dưới ~0.01 không đọc được**. Đã bốn lần suýt chốt nhầm:

| Tham số | Bẫy | Cách phát hiện |
|---|---|---|
| `α=0.8` | đỉnh 0.9061 seed 777 → 0.8445 seed 2024 | đo lại seed khác |
| `k1/b` | 3 lần đo cho 3 kết quả khác nhau | 2.000 câu + bootstrap → CI chứa 0 |
| `top3_log` | lý thuyết hợp lý, thực tế 0.1706 | cứ đo |
| cấu hình B | +0.0075, CI [−0.001, +0.016] | bootstrap → chấp nhận có rủi ro |

Quy trình đã chốt:
1. Quét lưới trên nửa đầu tập validation
2. **Kiểm chứng trên nửa sau** — chưa từng nhìn lúc chọn
3. Bootstrap trên hiệu số **từng câu**, xem CI có chứa 0 không
4. Chỉ tin **delta**, không tin giá trị tuyệt đối
5. Bằng chứng từ public test thắng mọi phép đo trên train

---

## 6. Phát hiện về dữ liệu

| Phát hiện | Con số | Hệ quả |
|---|---|---|
| Văn bản khổng lồ | median 4.813 từ, max 1.242.409, 99.6% vượt 512 token | bắt buộc chunk |
| Tín hiệu lexical loãng | 0.944 (văn bản đúng) vs 0.683 (ngẫu nhiên) | BM25 trên nguyên văn bản sẽ hỏng |
| Văn bản rỗng | 20 văn bản, **6 là gold** | cần fallback tầng 0 |
| Thiếu `name` | 1.125/8.532 | phải `.get()` |
| Lệch popularity | 36.4% corpus từng là đáp án; `245154` xuất hiện 109 lần | đừng prune 5.427 văn bản còn lại |
| Viết tắt | 44/1000 câu có viết tắt, Recall 0.9091 vs 0.8998 câu thường | **không phải vấn đề — đã đo, xem 6.1** |

### 6.1 Viết tắt: giả thuyết bị bác bỏ

Tôi từng ghi `cccd` như bằng chứng rằng viết tắt lái kết quả sai. Kiểm lại: **`CCCD`
xuất hiện 0/1.000 câu public và 1/7.000 câu train** — ví dụ đó là câu tôi tự gõ, không
phải dữ liệu thật.

Đo trên 1.000 câu validation:

| | Số câu | Recall@5 |
|---|---|---|
| Có viết tắt | 44 | 0.9091 |
| Không viết tắt | 956 | 0.8998 |

Câu có viết tắt chạy **tốt hơn**. Ngay cả khi sửa toàn bộ 44 câu lên 100% thì tổng chỉ
tăng **+0.0040**.

Nguyên nhân: văn bản pháp luật tự định nghĩa viết tắt ngay Điều 1 rồi dùng cả hai dạng,
nên chunk chứa cả hai. Câu hỏi cũng thường có cả hai (*"bảo hiểm y tế (BHYT)"*). Và viết
tắt hiếm (`UKVFTA` df 69, `RCEP` df 53) là **tín hiệu mạnh** — mở rộng sẽ làm loãng.

Cùng lý do đó, **mở rộng đồng nghĩa bằng LLM cũng không đáng làm**: đo trần pool cho thấy
BM25 chỉ để lọt 0.79% gold ở pool 2000, và bổ sung dense global toàn cục chỉ vớt thêm
+0.0012. Nút thắt là xếp hạng, không phải tìm kiếm.

---

## 7. Nợ kỹ thuật

**`BM25Index.read_chunk()` mở lại file 592 MB cho mỗi chunk.** Lần sinh bài nộp vừa rồi
gọi 100.000 lần → GPU chỉ đạt 54–92% thay vì bám 100%. Giữ sẵn file handle sẽ rút 25 phút
xuống còn ~10–15. Nên sửa trước khi chạy private test.

---

## 8. Việc tiếp theo

Cách top-1 (0.94) khoảng 0.04.

1. **Fine-tune cross-encoder** trên cùng 6.000 cặp. β tối ưu là 0.5 chứ không phải 1.0 —
   đó là chữ ký của lệch miền, và fine-tune chữa đúng chỗ đó. Lực đòn bẩy lớn nhất còn lại.
2. **Bi-encoder 2 epoch** — loss vẫn đang giảm ở bước cuối (3.3547 → 3.3374). 11 phút train
   + 36 phút encode.
3. **Word segmentation + bigram cho BM25** — "mức lương cơ sở" đang bị xé thành 4 âm tiết rời.
4. **Sửa `read_chunk()`** — không tăng điểm nhưng rút ngắn mọi vòng lặp thử nghiệm.

~~Từ điển viết tắt~~ và ~~mở rộng đồng nghĩa bằng LLM~~ đã bị loại sau khi đo (xem 6.1).
