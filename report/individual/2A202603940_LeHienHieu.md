# Báo cáo Day 7

**MSSV:** 2A202603940 · **Họ tên:** Le Hien Hieu

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** Sách *Workplace Software and Skills*, OpenStax, CC BY-NC-SA 4.0; [PDF nguồn](https://assets.openstax.org/oscms-prodcms/media/documents/Workplace_Software_and_Skills_-_WEB_IlfJtcP.pdf). Corpus về Word/Google Docs gồm 64 trang PDF được chọn trong khoảng 148–218, 317 chunks theo đoạn, tối đa 500 ký tự, không overlap. Số trang tính theo từng trang PDF được giữ lại.
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL/pgvector Lab 6, collection `day6_nemotron3_1b_v1`, strategy B; `nvidia/nemotron-3-embed-1b:free`, 2.048 chiều, query/passage prefix và chuẩn hóa L2. Query embedding gọi API OpenRouter; không sử dụng cache query của Lab 6.
- **Agent model, cách chạy và phân chia dev/test:** GPT-4o qua OpenAI API, phiên bản trả về `gpt-4o-2024-08-06`, temperature 0. Benchmark có 24 câu: dev 15 có đáp án + 2 thiếu evidence; test 5 có đáp án + 2 thiếu evidence.
- **Cách xác định evidence và tiêu chí chấm:** Gold đúng nguồn/trang và nằm trong chunk; câu thiếu evidence có lý do kiểm tra toàn corpus. Gold không truyền vào model. Context và câu trả lời chấm 0/1/2; citation phải có trong context và hỗ trợ khẳng định. Câu thiếu evidence không tính vào mẫu số retrieval.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense; top-3; filter theo câu hỏi; không rerank; 4.000 ký tự | Mốc đối chiếu | dev | 14/15 — 93,33% | 15/15 — 100% | 15/15 — 100% | 606,35 giây / 0,076580 USD* |
| Phương pháp 2 / Cải thiện 1 | Dense + TF-IDF; 20 candidates/nhánh; RRF 60; top-3; cùng filter/budget | Lexical hỗ trợ tên lệnh và menu | dev | 13/15 — 86,67% | 15/15 — 100% | 15/15 — 100% | 638,07 giây / 0,077540 USD* |
| Baseline | Cấu hình Dense giữ nguyên | Đối chiếu baseline | test | 5/5 — 100% | 5/5 — 100% | 5/5 — 100% | 133,39 giây / 0,028448 USD* |
| Hybrid | Cấu hình Hybrid giữ nguyên | Đối chiếu sau khi chạy lại lỗi dịch vụ | test | 5/5 — 100% | 5/5 — 100% | 5/5 — 100% | 219,12 giây / 0,034135 USD |

*Chi phí là phần ước tính có usage; tổng chi phí giữ `null` khi còn request thiếu usage. Thời gian gồm API và retry. Sáu câu Hybrid test từng lỗi HTTP 429 đã chạy lại thành công với cấu hình giữ nguyên; lịch sử lỗi được lưu trong `previous_attempts`.*

- **Lỗi dev và kết quả trước/sau:** `q15` được đưa từ hạng 2 lên hạng 1; ngược lại `q01`, `q08` từ hạng 1 xuống hạng 2. Hybrid giảm Hit@1 nhưng giữ nguyên đủ evidence trong top-3.
- **Lỗi câu trả lời:** `q06` có context đúng nhưng cả hai phương pháp đều thêm trạng thái comment chưa được nguồn xác nhận; answer chấm 1. Thay retrieval chưa khắc phục lỗi sinh câu trả lời.
- **Tiến độ và lựa chọn:** Đã thử 2/5 phương pháp; Hybrid đồng thời là cải thiện 1 nhưng chưa tăng chất lượng tổng thể trên dev. Chưa thực hiện cải thiện 2 và chưa chọn cấu hình cuối. Các bước chọn/chỉnh tiếp theo dùng dev; nếu dùng test để chỉnh thì cần test mới.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, context và lỗi:** Một lượt `search` bằng câu hỏi gốc, sau đó `finish_answer` hoặc `abstain`. Context loại trùng và giữ nguyên chunk trong budget; cho phép sửa citation một lần. Lỗi dịch vụ ghi `error`, tách khỏi chấm câu trả lời. Hybrid tái sử dụng TF-IDF trong `starter/baseline.py`; trace lưu ranking từng nhánh và RRF.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Dense / dev | 17/17 | 1 search/câu | 0/1/14; 1,93/2 | 15/16 | 15/15 | 2/2 | 0/15 |
| Hybrid / dev | 17/17 | 1 search/câu | 0/1/14; 1,93/2 | 16/17 | 15/15 | 2/2 | 0/15 |
| Dense / test | 7/7 | 1 search/câu | 0/0/5; 2,00/2 | 5/5 | 5/5 | 2/2 | 0/5 |
| Hybrid / test | 7/7 | 1 search/câu | 0/0/5; 2,00/2 | 5/5 | 5/5 | 2/2 | 0/5 |

Context mức 2: cả Dense và Hybrid đều có 15/15 câu dev và 5/5 câu test. Mỗi phương pháp đã có kết quả và nhãn cho đủ 24 câu.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q15` | Dev có đáp án | PDF 203: Dense hạng 2, Hybrid hạng 1 | Editing > Suggesting; citation đúng | Context 2, answer 2; Hybrid cải thiện ranking |
| `q01` | Dev có đáp án | PDF 151: Dense hạng 1, Hybrid hạng 2 | Outline cần headings cho section/subsection | Context 2, answer 2; Hybrid giảm Hit@1 |
| `q06` | Dev có đáp án | Cả hai tìm đúng PDF 171 | So sánh xóa/resolve đúng nhưng thêm chi tiết chưa được hỗ trợ | Context 2, answer 1; citation chưa hỗ trợ toàn bộ giải thích |
| `q21` | Dev thiếu evidence | Có hướng dẫn Track Changes, không có mã VBA | Từ chối, không bổ sung code | Abstention đúng ở cả hai phương pháp |
| `q17–q20` | Test có đáp án | Hybrid tìm đủ evidence ở hạng 1 | Trả lời đúng, citation hỗ trợ nguồn | Context 2, answer 2 cho cả bốn câu |
| `q23–q24` | Test thiếu evidence | Không có code Apps Script hoặc quota API cần hỏi | Từ chối, không bổ sung code hoặc số liệu | Abstention đúng ở cả hai phương pháp |

- **Trace, nhãn, thời gian và chi phí:** [Dense](../../results/dense_baseline/summary.json) và [Hybrid](../../results/hybrid/summary.json); mỗi thư mục có `traces.json`, `retrieval.json`, `reviews.json`, `summary.json`. Nhãn được tổng hợp bằng công cụ trong `starter/agent_metrics.py`.
- **Lệnh chạy:** `python -X utf8 -B -m src.benchmark --method hybrid --split all --output results/hybrid`. Thêm `--retry-errors` để chạy riêng lỗi dịch vụ và giữ lịch sử; dữ liệu hiện tại đã hoàn tất.
- **Đánh đổi và giới hạn:** Hybrid giữ nguyên chất lượng trả lời và evidence top-3, nhưng giảm Hit@1 trên dev. Hai phương pháp cùng đạt 5/5 câu có đáp án và từ chối đúng 2/2 câu thiếu evidence trên test; benchmark hiện chủ yếu là câu cần một evidence.

## Thử thách nâng cao (nếu thực hiện)

Chưa thực hiện; phạm vi hiện tại là phần cơ bản với corpus 64 trang.
