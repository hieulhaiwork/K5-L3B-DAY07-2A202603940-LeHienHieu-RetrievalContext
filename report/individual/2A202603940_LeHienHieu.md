# Báo cáo Day 7 — Retrieval và Context Engineering

**MSSV:** 2A202603940 · **Họ tên:** Le Hien Hieu

## Dữ liệu và benchmark

Corpus gồm các mục hướng dẫn tạo, định dạng, cộng tác và quản lý phiên bản tài liệu Microsoft Word và Google Docs trong sách *Workplace Software and Skills* của OpenStax. PDF được tải ngày 14/11/2026 từ [nguồn OpenStax](https://assets.openstax.org/oscms-prodcms/media/documents/Workplace_Software_and_Skills_-_WEB_IlfJtcP.pdf). Quyền sử dụng ghi trên tài liệu là CC BY-NC-SA 4.0.

Phạm vi sử dụng là trang PDF 148–218, loại các trang 191, 193, 202, 206, 210, 215, 218 do chủ yếu là ảnh, còn **64 trang và 317 chunks**. Số trang corpus được tính theo từng trang PDF được giữ lại; số trang nguồn giữ nguyên sau khi loại trang. Chunks được chia theo đoạn, tối đa 500 ký tự, không overlap. Dữ liệu và ID được tái sử dụng từ Lab 6, chuyển sang schema Lab 7 tại [data/chunks.json](../../data/chunks.json).

Vector store là PostgreSQL/pgvector của Lab 6, kết nối qua `DATABASE_URL` trong `.env`. Collection `day6_nemotron3_1b_v1`, strategy B chứa 317 vectors, đã đối chiếu khớp ID, nội dung, nguồn, trang và metadata với corpus. Embedding dùng `nvidia/nemotron-3-embed-1b:free` qua OpenRouter, 2.048 chiều, tiền tố `passage: ` cho tài liệu và `query: ` cho câu hỏi, chuẩn hóa L2 và tìm bằng cosine.

Agent dùng OpenAI `gpt-4o`; phiên bản API trả về là `gpt-4o-2024-08-06`. Embedding và model điều phối sử dụng hai key riêng trong `.env`.

Bộ [data/questions.json](../../data/questions.json) gồm:

| Loại câu | Dev | Test | Tổng |
|---|---:|---:|---:|
| Có đáp án | 15 | 5 | 20 |
| Thiếu bằng chứng | 2 | 2 | 4 |
| **Tổng** | **17** | **7** | **24** |

Câu có đáp án chứa nguồn, trang, evidence và đáp án chuẩn. Evidence đã được kiểm tra nằm trọn trong chunk đúng nguồn/trang; `q20` có hai evidence thay thế hợp lệ. Bốn câu thiếu bằng chứng hỏi về VBA, thời hạn giữ phiên bản OneDrive, Apps Script và quota Google Docs API; cách kiểm tra trên toàn corpus được ghi trong từng câu. Gold và đáp án chuẩn chỉ dùng để chấm, không truyền vào model hoặc retrieval.

Baseline được chạy trên cả dev và test với cùng cấu hình đã cố định. Các phương pháp tiếp theo sẽ được phát triển và chọn bằng dev. Kết quả test hiện tại là kết quả baseline, chưa phải so sánh cuối giữa baseline và cấu hình được chọn. Nếu dùng kết quả test này để điều chỉnh hệ thống, cần chuẩn bị test mới cho đánh giá độc lập.

## Retrieval và thử nghiệm

Dense baseline dùng câu hỏi gốc, top-k 3, một lượt tìm kiếm, không rerank. Sáu câu có filter `application_scope` xác định từ tên ứng dụng ghi rõ trong câu hỏi; các câu còn lại không filter. Filter được áp dụng trong SQL trên cùng index B thông qua alias C.

Context được chọn theo thứ tự retrieval, loại trùng ID và giữ nguyên từng chunk, với budget **4.000 ký tự**, tính cả JSON và thông tin nguồn. Trong lượt chạy này, cả ba candidates của mỗi câu đều được giữ lại.

Hit@1 và Evidence Hit@3 được tính bằng evidence hợp lệ đúng nguồn/trang trong top-1 hoặc top-3, chia cho số câu có đáp án. Context Evidence Rate được tính bằng số câu có đủ bằng chứng trong context cuối, chia cho số câu có đáp án. Câu thiếu bằng chứng không tham gia mẫu số của ba chỉ số này.

| Cấu hình | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian toàn pipeline | Chi phí chat có usage |
|---|---|---:|---:|---:|---:|---:|
| Dense baseline | Dev | 14/15 — 93,33% | 15/15 — 100% | 15/15 — 100% | 606,35 giây | 0,076580 USD |
| Dense baseline | Test | 5/5 — 100% | 5/5 — 100% | 5/5 — 100% | 133,39 giây | 0,028448 USD |

Thời gian trên gồm các lần thử lỗi và chạy lại. Tổng thời gian retrieval, gồm query embedding và truy vấn DB, là 120,71 giây trên dev và 30,88 giây trên test. Chi phí embedding được provider báo là 0 USD. Chi phí chat đã xác định theo usage và bảng giá cấu hình là khoảng **0,105028 USD**; năm request lỗi mạng không có usage nên chi phí toàn bộ lượt chạy được giữ `null`, không coi phần chưa xác định là 0 USD.

Hai trường hợp dev cần cải thiện:

| Query | Kết quả và nguyên nhân | Hướng thử nghiệm |
|---|---|---|
| `q15` | Evidence đầy đủ về Editing > Suggesting ở hạng 2. Hạng 1 chỉ hướng dẫn chia sẻ ở chế độ Commenter, nên Hit@1 thất bại dù top-3 đủ evidence. | Thử rerank để ưu tiên đoạn trả lời đúng thao tác được hỏi; so sánh ranking trước/sau trên dev. |
| `q06` | Retrieval và context đúng, nhưng câu trả lời thêm nhận định rằng comment đã resolve không hiện trong main document view. Các đoạn context không xác nhận chi tiết đó. | Thử giới hạn câu trả lời vào các khẳng định có đoạn hỗ trợ và kiểm tra từng điều kiện; đo lại grounded answer và citation. |

Ngoài ra, `q03` và `q10` gặp lỗi kết nối OpenAI ở lượt đầu. Hai câu được chạy lại với cấu hình giữ nguyên; kết quả lỗi trước đó được lưu trong `previous_attempts`. `q23` có một request lỗi mạng được client thử lại trong cùng lượt chạy. Sau xử lý, đủ 24 câu có kết quả cuối; không có lỗi hệ thống còn tồn đọng.

Hiện mới hoàn tất **1/5 phương pháp dự kiến**. Hybrid, retrieval + rerank, query transformation và multi-stage chưa được benchmark. Chưa có hai thí nghiệm cải thiện đã đo trước/sau và chưa chọn cấu hình cuối. Việc chọn sẽ dựa trên chất lượng câu trả lời, context, citation và abstention trên dev, đồng thời cân nhắc thời gian và chi phí.

## Agent và đánh giá

Model gọi `search`, sau đó chọn `finish_answer` hoặc `abstain`. Câu trả lời phải có citation dạng `[chunk-id]`, các ID phải xuất hiện trong context. Nếu citation sai định dạng, cho phép một lượt sửa bằng context cũ. Thiếu bằng chứng hoặc citation vẫn không hợp lệ dẫn đến từ chối; lỗi API/DB được ghi riêng với trạng thái `error`. Nội dung tài liệu được xử lý như dữ liệu tham khảo.

Câu trả lời được chấm 0 = sai hoặc thiếu evidence, 1 = đúng một phần, 2 = đúng và đủ. Citation chỉ hợp lệ khi ID nằm trong context và nguồn hỗ trợ các khẳng định gắn với citation. Vì vậy, citation của `q06` không được tính hợp lệ cho toàn bộ phần giải thích, dù nội dung so sánh chính đúng.

| Cấu hình / split | Tool use | Retrieval mỗi câu | Grounded answer 0 / 1 / 2 | Điểm trung bình /2 | Citation hợp lệ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Dense / dev | 17/17 — 100% | 1 | 0 / 1 / 14 | 1,93 | 15/16 — 93,75% | 15/15 — 100% | 2/2 — 100% | 0/15 — 0% |
| Dense / test | 7/7 — 100% | 1 | 0 / 0 / 5 | 2,00 | 5/5 — 100% | 5/5 — 100% | 2/2 — 100% | 0/5 — 0% |

Context được đọc đối chiếu cho cả 20 câu có đáp án: dev có 15 câu mức 2, test có 5 câu mức 2; không có câu mức 0 hoặc 1. Trong các lượt hoàn tất, dev có 17 câu dùng hai request model; test có sáu câu dùng hai request và một câu dùng ba request do retry mạng. Tính cả lịch sử chạy lỗi, có 38 request OpenAI trên dev và 15 trên test. Usage nhận được là 21.845 tokens trên dev và 8.508 tokens trên test; các request không có usage được ghi riêng.

Các trường hợp đối chiếu:

| Query ID | Loại câu | Evidence / kết quả tìm | Câu trả lời hoặc hành vi | Nhãn và lý do |
|---|---|---|---|---|
| `q01` | Dev có đáp án | PDF 151, hạng 1 | Navigation pane có outline khi đã áp dụng headings cho section/subsection | Context 2; answer 2; citation hỗ trợ điều kiện |
| `q06` | Dev có đáp án | PDF 171, hạng 1 | So sánh đúng nhưng thêm chi tiết về hiển thị comment | Context 2; answer 1; citation chưa hỗ trợ toàn bộ lời giải thích |
| `q15` | Dev có đáp án | PDF 203, hạng 2 | Editing dropdown > Suggesting | Context 2; answer 2; ranking hạng 1 còn hạn chế |
| `q16` | Test có đáp án | PDF 207, hạng 1 | File > Version history > See version history | Context 2; answer 2; citation hợp lệ |
| `q17` | Test có đáp án | PDF 209, hạng 1 | Hover trên đoạn thay đổi để thấy tên người sửa | Context 2; answer 2; citation hợp lệ |
| `q18` | Test có đáp án | PDF 212, hạng 1 | Bản copy không mang version history, comments, suggestions | Context 2; answer 2; đủ ba thành phần |
| `q19` | Test có đáp án | PDF 213, hạng 1 | Nhấn tam giác bên trái version để xem micro changes | Context 2; answer 2; citation hợp lệ |
| `q20` | Test có đáp án | PDF 214, hạng 1 | File phải lưu trong OneDrive hoặc SharePoint | Context 2; answer 2; giữ đủ điều kiện |
| `q23` | Test thiếu evidence | Chỉ tìm được hướng dẫn giao diện Docs | Từ chối cung cấp Apps Script xuất PDF | Abstention đúng; không bổ sung code ngoài corpus |
| `q24` | Test thiếu evidence | Không có quota API trong context | Từ chối đưa ra quota theo phút/người dùng | Abstention đúng; không đưa số liệu thiếu nguồn |

Kết quả được lưu tại:

- [traces.json](../../results/dense_baseline/traces.json): tool call, query, filter, ranking đầy đủ, context, câu trả lời, citation, usage, thời gian và lịch sử lỗi.
- [retrieval.json](../../results/dense_baseline/retrieval.json): ranking từng câu và kết quả khớp evidence.
- [reviews.json](../../results/dense_baseline/reviews.json): nhãn context, câu trả lời, từng citation và abstention, kèm lý do.
- [summary.json](../../results/dense_baseline/summary.json): chỉ số tổng hợp riêng dev/test.

Chạy từ thư mục gốc repo trong môi trường Conda `vin20k-lab6`:

```powershell
python -X utf8 -B -m src.benchmark --split all --output results/dense_baseline
```

Lệnh trên bảo vệ trace đã tồn tại. Muốn chạy một lượt mới, dùng thư mục output khác. Chạy lại riêng lỗi mạng với cấu hình giữ nguyên:

```powershell
python -X utf8 -B -m src.benchmark --split all --output results/dense_baseline --retry-errors
```

Tổng hợp lại từ trace và nhãn, không gọi thêm API:

```powershell
python -X utf8 -B -m src.benchmark --split all --output results/dense_baseline --summarize-only --reviews results/dense_baseline/reviews.json
```

Giới hạn hiện tại là benchmark nhỏ, chủ yếu yêu cầu một evidence; chưa đủ để kết luận chất lượng trên câu multi-hop hoặc tài liệu khác. Kết quả retrieval cao vẫn có thể đi kèm câu trả lời thêm chi tiết không được hỗ trợ, như `q06`.

## Tiến độ

Đã hoàn tất chuẩn hóa corpus/benchmark, đối chiếu index Lab 6, xây Dense adapter và agent GPT-4o, chạy baseline trên 24 câu, lưu trace và chấm kết quả. Các test code kiểm tra adapter, agent và cách tính benchmark được cung cấp trong `tests/`.

Phần còn lại gồm bốn phương pháp retrieval dự kiến, ít nhất hai thí nghiệm cải thiện trên dev, so sánh để chọn cấu hình và đánh giá test của cấu hình đã chọn. Báo cáo sẽ được bổ sung kết quả trước/sau khi các bước này hoàn tất.

## Thử thách nâng cao

Chưa thực hiện. Corpus hiện có 64 trang; phạm vi hiện tại là phần cơ bản của Lab 7.
