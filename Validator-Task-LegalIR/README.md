# Validator — DSC 2026 Task 1: Legal Information Retrieval (LegalIR)

Công cụ kiểm tra file nộp **ở máy local** trước khi upload lên hệ thống chấm điểm.
Chỉ dùng thư viện chuẩn của Python — **không cần cài gì thêm** (Python 3.7+).

## Cách dùng

```bash
# Kiểm tra cơ bản
python validate_submission.py submission.zip

# Khuyến nghị: đối chiếu với đề bài để bắt lỗi thiếu/thừa câu hỏi
python validate_submission.py submission.zip --ref public-official.json

# Đầy đủ: kiểm tra thêm document_id có thật trong kho văn bản
python validate_submission.py submission.zip \
    --ref public-official.json \
    --corpus selected-contexts/
```

Exit code: `0` = hợp lệ · `1` = không hợp lệ · `2` = lỗi đường dẫn đầu vào.

## Định dạng bài nộp

`submission.zip` chứa **duy nhất** `submission.json` nằm ở **thư mục gốc** của zip:

```
submission.zip
└── submission.json      ✅ đúng

submission.zip
└── my_team/
    └── submission.json  ❌ sai — hệ thống sẽ không tìm thấy file
```

```json
{
  "147194": { "answer": ["177504", "740"] },
  "38096":  { "answer": ["102345"] }
}
```

Quy tắc:

- Key ngoài cùng là `question_id`, phải khớp **chính xác** tập câu hỏi trong đề bài — không thiếu, không thừa.
- `answer` phải là **list các chuỗi**, tối đa **5** `document_id`, không trùng lặp.
- ⚠️ **`document_id` bắt buộc là chuỗi.** Trường `id` trong `context_*.json` là **số nguyên**, nên nhớ ép kiểu `str()` trước khi ghi JSON:

  ```python
  json.dump({qid: {"answer": [str(d) for d in docs]} for qid, docs in preds.items()}, f)
  ```

## Bảng mã lỗi

| Mã | Ý nghĩa |
|---|---|
| E001 | Không tìm thấy `submission.json` ở gốc zip (hoặc file không phải zip hợp lệ) |
| E002 | Zip chứa file thừa ngoài `submission.json` |
| E003 | JSON không parse được, hoặc file không phải UTF-8 |
| E004 | Cấp ngoài cùng không phải JSON object, hoặc file rỗng |
| E005 | Thiếu `question_id` so với đề bài |
| E006 | Có `question_id` không tồn tại trong đề bài |
| E007 | Giá trị của một câu không phải object `{"answer": [...]}` |
| E008 | Thiếu key `"answer"` |
| E009 | `"answer"` sai kiểu — là `null`, chuỗi, hoặc số thay vì list |
| E010 | `document_id` không phải chuỗi (lỗi phổ biến nhất: để nguyên kiểu `int`) |
| E011 | `document_id` rỗng hoặc toàn khoảng trắng |
| E012 | `document_id` bị lặp trong cùng một câu |
| E013 | `question_id` bị lặp trong `submission.json` |
| E014 | `document_id` không tồn tại trong kho văn bản *(chỉ khi dùng `--corpus`)* |

| Mã | Cảnh báo |
|---|---|
| W001 | Câu trả về danh sách rỗng → Recall = Precision = 0 cho câu đó |
| W002 | Câu trả về hơn 5 `document_id` → Recall = Precision = 0 cho câu đó |
| W003 | Zip chứa file rác của hệ điều hành (`__MACOSX/`, `.DS_Store`) — bỏ qua khi chấm |
| W004 | Đang kiểm tra file `.json` trực tiếp; hệ thống chỉ nhận `.zip` |
| W005 | Không truyền `--ref` nên chưa kiểm tra được thiếu/thừa `question_id` |

W001 và W002 **không** làm bài nộp bị từ chối, nhưng làm mất điểm của những câu liên quan.

## Cách chấm điểm

Độ đo chính là **Recall**, độ đo phụ là **Precision** (dùng để phân hạng khi bằng Recall).
Cả hai được tính trung bình trên **toàn bộ** câu hỏi, bao gồm cả những câu bị 0 điểm do vi phạm
ràng buộc 5 `document_id`.

## Ghi chú

Thông báo của validator viết bằng tiếng Việt **không dấu** để hiển thị đúng trên mọi terminal
(cmd.exe, PowerShell, macOS, Linux) mà không bị lỗi font.
