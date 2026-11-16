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
| Baseline | Dense; top-3; filter theo câu hỏi; 4.000 ký tự | Mốc đối chiếu | dev | 14/15 — 93,33% | 15/15 — 100% | 15/15 — 100% | 606,35 giây / 0,076580 USD* |
| Phương pháp 2 / Cải thiện 1 | Dense + TF-IDF; 20 candidates/nhánh; RRF 60; cùng top-3/filter/budget | Lexical hỗ trợ tên lệnh và menu | dev | 13/15 — 86,67% | 15/15 — 100% | 15/15 — 100% | 638,07 giây / 0,077540 USD* |
| Phương pháp 3 / Cải thiện 2 | Dense top-20 + GPT-4o rerank; cùng top-3/filter/budget | Ưu tiên đoạn trả lời đúng ý hỏi | dev | 15/15 — 100% | 15/15 — 100% | 15/15 — 100% | 622,67 giây / 0,395848 USD* |
| Baseline | Dense giữ nguyên | Đối chiếu baseline | test | 5/5 — 100% | 5/5 — 100% | 5/5 — 100% | 133,39 giây / 0,028448 USD* |
| Hybrid | Hybrid giữ nguyên | Đối chiếu cấu hình cố định | test | 5/5 — 100% | 5/5 — 100% | 5/5 — 100% | 219,12 giây / 0,034135 USD |
| Rerank | Rerank giữ nguyên | Đối chiếu cấu hình cố định | test | 5/5 — 100% | 5/5 — 100% | 5/5 — 100% | 130,37 giây / 0,154178 USD |

*Chi phí là phần ước tính có usage; tổng giữ `null` khi còn request thiếu usage. Rerank tính cả model xếp hạng và agent. Thời gian gồm API và retry; các lần lỗi được giữ trong trace.*

- **Lỗi dev và kết quả trước/sau:** `q15`: Dense hạng 2, Hybrid/rerank hạng 1. `q01`, `q08`: Hybrid hạng 2, rerank đưa về hạng 1. Rerank tăng Hit@1 dev từ 14/15 lên 15/15 so với Dense; top-3 vẫn đủ evidence.
- **Lỗi câu trả lời:** `q06` có context đúng nhưng cả ba phương pháp đều thêm chi tiết chưa được nguồn xác nhận, nên answer chấm 1. Với rerank, chi tiết vượt evidence là việc deletion xóa hoàn toàn khỏi lịch sử tài liệu.
- **Tiến độ và lựa chọn:** Đã thử 3 phương pháp (Dense, Hybrid, Rerank), trong đó Hybrid và Rerank là hai cấu hình cải thiện. Chọn **Rerank** chỉ dựa trên dev, theo thứ tự tiêu chí: điểm câu trả lời, context evidence, Evidence Hit@3, abstention, Hit@1. Ba cấu hình bằng nhau ở bốn tiêu chí đầu, Rerank thắng ở Hit@1 (15/15) nhưng chi phí cao hơn khoảng 5 lần. Sau khi chốt, cả ba cấu hình được chạy một lần trên test để đối chiếu; test không dùng để chọn hay chỉnh. Chưa thử query transformation và multi-stage.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, context và xử lý lỗi:** Một `search` bằng câu hỏi gốc, rồi `finish_answer` hoặc `abstain`; context loại trùng, giữ nguyên chunk trong budget. Cho phép sửa citation một lần. Rerank thêm một lượt GPT-4o xếp 20 candidates, không nhận gold/đáp án. Output phải có đủ ID, không trùng; `q14` từng trả 19/20 ID và đã chạy lại thành công với cấu hình giữ nguyên.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Dense / dev | 17/17 | 1 search/câu | 0/1/14; 1,93/2 | 15/16 | 15/15 | 2/2 | 0/15 |
| Hybrid / dev | 17/17 | 1 search/câu | 0/1/14; 1,93/2 | 16/17 | 15/15 | 2/2 | 0/15 |
| Rerank / dev | 17/17 | 1 search + 1 rerank/câu | 0/1/14; 1,93/2 | 16/17 | 15/15 | 2/2 | 0/15 |
| Dense / test | 7/7 | 1 search/câu | 0/0/5; 2,00/2 | 5/5 | 5/5 | 2/2 | 0/5 |
| Hybrid / test | 7/7 | 1 search/câu | 0/0/5; 2,00/2 | 5/5 | 5/5 | 2/2 | 0/5 |
| Rerank / test | 7/7 | 1 search + 1 rerank/câu | 0/0/5; 2,00/2 | 5/5 | 5/5 | 2/2 | 0/5 |

Context mức 2: cả ba phương pháp có 15/15 câu dev và 5/5 câu test. Mỗi phương pháp đã có kết quả và nhãn cho đủ 24 câu; không còn lỗi hệ thống ở kết quả cuối.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q15` | Dev có đáp án | PDF 203: Dense hạng 2, Hybrid/rerank hạng 1 | Editing > Suggesting; citation đúng | Context 2, answer 2; cải thiện ranking |
| `q01, q08` | Dev có đáp án | Rerank đưa evidence từ hạng 2 của Hybrid về hạng 1 | Đủ điều kiện headings/styles; citation hỗ trợ | Context 2, answer 2 |
| `q06` | Dev có đáp án | Cả ba tìm đúng PDF 171 | So sánh xóa/resolve đúng nhưng thêm chi tiết thiếu nguồn | Context 2, answer 1; citation chưa hỗ trợ toàn bộ giải thích |
| `q17–q20` | Test có đáp án | Cả ba tìm đủ evidence ở hạng 1 | Trả lời đúng, citation hỗ trợ nguồn | Context 2, answer 2 cho cả bốn câu |
| `q23–q24` | Test thiếu evidence | Không có code Apps Script hoặc quota API cần hỏi | Từ chối, không bổ sung code/số liệu | Abstention đúng ở cả ba phương pháp |

- **Trace, nhãn, thời gian và chi phí:** [Dense](../../results/dense_baseline/summary.json), [Hybrid](../../results/hybrid/summary.json), [Rerank](../../results/rerank/summary.json). Mỗi thư mục có `traces.json`, `retrieval.json`, `reviews.json`, `summary.json`; nhãn được tổng hợp bằng `starter/agent_metrics.py`. Trace rerank lưu candidates trước/sau và request/usage/chi phí của model xếp hạng.
- **Lệnh chạy:** `python -X utf8 -B -m src.benchmark --method rerank --split all --output results/rerank`. Dùng `--method dense` hoặc `hybrid` với thư mục output riêng; `--retry-errors` giữ lịch sử lỗi.
- **Đánh đổi và giới hạn:** Rerank tăng Hit@1 dev nhưng thêm chi phí model (~5 lần); điểm câu trả lời chưa tăng. Cả ba đạt 5/5 câu có đáp án và từ chối đúng 2/2 câu thiếu evidence trên test, nên test nhỏ (7 câu) không phân biệt được các phương pháp. Benchmark chủ yếu là câu cần một evidence.

## Thử thách nâng cao (nếu thực hiện)

Chưa thực hiện; phạm vi hiện tại là phần cơ bản với corpus 64 trang.
