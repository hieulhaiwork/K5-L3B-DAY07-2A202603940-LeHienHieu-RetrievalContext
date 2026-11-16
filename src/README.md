# Chạy và tái lập bài nộp cơ bản

Cấu hình được chọn là **Dense + GPT-4o rerank**, ghi tại `src/configs/selected.json`. Kết quả dev của ba phương pháp nằm trong `results/dense_baseline`, `results/hybrid`, `results/rerank`. Mỗi thư mục chứa cả kết quả dev và test; test chỉ chạy sau khi chốt cấu hình trên dev.

## Tái lập trên máy mới

Dùng Python 3.11+ và Docker Desktop. Trong thư mục gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Điền `OPENAI_API_KEY`, `OPENROUTER_API_KEY` trong `.env`. Cấu hình mẫu dùng PostgreSQL riêng ở cổng 55434; nếu đổi port/password, cập nhật cả `DATABASE_URL`. Máy đang có DB Lab 6 có thể giữ nguyên `DATABASE_URL` hiện tại.

```powershell
docker compose up -d --wait
python -X utf8 -B -m src.rebuild_index
python -X utf8 -B -m src.rebuild_index --check-only
python -X utf8 -B -m unittest discover -s tests -v
```

Dựng mới index dùng 317 chunks được chia sẻ tại `data/chunks.json`, gọi embedding tài liệu thật theo batch 32 (10 requests), giữ nguyên ID/nguồn/trang/metadata. Không cần PDF để tái tạo từ snapshot. Index B có sẵn được đối chiếu và giữ nguyên; dữ liệu không khớp sẽ báo lỗi. Query embedding luôn tương thích model 2.048 chiều và prefix của Lab 6.

## Cấu hình chọn và đánh giá cuối

Tiêu chí chọn chỉ dùng dev: grounded answer, Context Evidence Rate, Evidence Hit@3, abstention, rồi Hit@1 khi hòa. Rerank đạt Hit@1 15/15 so với Dense 14/15; chi phí model bổ sung được báo như đánh đổi. Cấu hình được khóa trước khi chạy test.

- `data/questions.json`: benchmark 24 câu (17 dev + 7 test).
- `src/configs/selected.json`: tiêu chí, kết quả dev và snapshot cấu hình chọn/baseline.
- `--selection`: từ chối cấu hình hoặc file benchmark khác với snapshot đã khóa.

Chạy lại test, dùng thư mục output mới để giữ kết quả đã nộp:

```powershell
python -X utf8 -B -m src.benchmark --method dense --questions data/questions.json --split test --selection src/configs/selected.json --output results/reproduction/test_dense
python -X utf8 -B -m src.benchmark --method rerank --questions data/questions.json --split test --selection src/configs/selected.json --output results/reproduction/test_rerank
```

Dev của cả ba phương pháp có thể tái chạy bằng `--questions data/questions.json --split dev --method dense|hybrid|rerank`, mỗi phương pháp dùng thư mục output riêng, bỏ `--selection` cho lượt thí nghiệm dev. Lỗi API được lưu để chạy lại bằng `--retry-errors`; giữ nguyên cấu hình. Hạn mức OpenRouter được tính cho cả dựng index và query embedding.

Demo agent mặc định dùng cấu hình được chọn; có thể chỉ định `--method dense`, `hybrid` hoặc `rerank`:

```powershell
python -X utf8 -B -m src.agent --question "How do collaborators review changes in a document?" --output results/demo.json
```

Không đưa gold vào model. Sau khi đối chiếu từng câu/citation với context, điền `reviews.json` cho đúng lượt trace, rồi dùng `--summarize-only --reviews <file>` để tổng hợp không gọi thêm API. Các thư mục kết quả đã nộp có đủ trace, nhãn và summary; có thể đọc mà không cần API key.

---

# Code

Đặt code tại đây hoặc mở rộng [starter](../starter/README.md). Agent trong `src/agent.py` đã kết nối model điều phối GPT-4o và Dense retriever thật.

## Dense retrieval

`dense.py` nối API query embedding với collection B của Lab 6, trả về schema chunk Lab 7. Không tạo bảng hay ghi lại index. Filter `application_scope` và `section` dùng alias C trên cùng dữ liệu B; đây không phải phương pháp retrieval riêng.

Chạy từ thư mục gốc repo, trong môi trường Python đã cài `psycopg[binary]` và `python-dotenv`. Cần PostgreSQL của Lab 6 đang chạy; `.env` chứa `DATABASE_URL`, `OPENROUTER_API_KEY`, `EMBEDDING_MODEL`. Cấu hình embedding nằm tại `src/configs/embedding.json`.

Chỉ xác minh kết nối, profile embedding và toàn bộ snapshot B với `data/chunks.json`, không gọi API:

```powershell
python -X utf8 -B -m src.dense --check-only
```

Xác minh index rồi thử retrieval thật với một query ngoài benchmark:

```powershell
python -X utf8 -B -m src.dense --query "How do collaborators review changes in a document?" --top-k 3
```

CLI xuất kết quả, score cosine và usage/chi phí API; cache query của lần kiểm tra được tạo trong thư mục tạm và tự xóa. Lệnh kiểm tra không phải kết quả benchmark dev/test.

Khi nối với agent, dùng `from_project(cache_dir=...)`, gọi `verify_index(Path("data/chunks.json"))` trước khi đánh giá, rồi gọi `search(query, filters=None, top_k=3)`. Chọn thư mục cache phù hợp khi chạy nhiều query. Chỉ truyền câu hỏi và filter vào retrieval; không truyền gold hoặc đáp án chuẩn.

Chạy các test offline; adapter giả trong test chỉ dùng kiểm tra hành vi, không đánh giá chất lượng retrieval:

```powershell
python -X utf8 -B -m unittest discover -s tests -v
```

Trên Windows, `-X utf8` tránh lỗi đọc dữ liệu tiếng Việt trong các test mẫu.

## Agent GPT-4o với Dense

`agent.py` dùng `chat.py` gọi OpenAI Chat Completions với function tools. `OPENAI_API_KEY` được đọc từ `.env`; embedding vẫn dùng `OPENROUTER_API_KEY`. Cấu hình chung nằm trong `src/configs/agent.json`.

Cấu hình ban đầu: model `gpt-4o`, temperature 0, top-k 3, tối đa một lượt retrieval, context budget 4.000 ký tự gồm cả JSON và nguồn, tối đa 1.024 output tokens mỗi lượt model. Đây là lựa chọn baseline, không phải giới hạn bắt buộc của đề.

Model gọi `search` bằng câu hỏi gốc, sau đó chọn `finish_answer` hoặc `abstain`. Context được loại trùng, giữ nguyên từng chunk theo thứ tự retrieval. Không đưa gold hoặc đáp án chuẩn vào model. Khi citation sai định dạng, cho phép một lượt sửa bằng context cũ, không tìm thêm. Citation vẫn sai sẽ dẫn tới abstention. Lỗi API/DB được ghi là `error`, không tính thành từ chối đúng.

Chạy một câu ngoài benchmark và lưu trace mới:

```powershell
python -X utf8 -B -m src.agent --method dense --question "In Microsoft Word, how do I add a comment to selected text?" --output results/agent_manual.json
```

File output là JSON array để dùng với công cụ tổng hợp của starter. File có sẵn sẽ không bị ghi đè. CLI kiểm tra index trước khi gọi model và dùng cache embedding tạm thời. Các lần chạy nhiều câu nên dùng lại retriever/cache và gọi `run_agent` để tránh gọi embedding trùng.

Trace có câu hỏi, cấu hình, model yêu cầu/thực trả, request và response tool, query/filter/top-k, kết quả retrieval, context cuối, câu trả lời, citation, token usage, thời gian, chi phí và lỗi. Không ghi API key hoặc chuỗi kết nối database. Chi phí OpenAI là ước tính từ token usage và bảng giá trong cấu hình; usage không xác định được giữ `null`. Chi phí embedding lưu theo số liệu provider trả về.


Tham khảo: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [GPT-4o và bảng giá](https://developers.openai.com/api/docs/models/gpt-4o).

## Benchmark Dense

Bộ `data/questions.json` có 24 câu: dev 17 và test 7. Chỉ dùng dev để phát triển/chọn cấu hình; chạy test sau khi chốt. Kết quả baseline hiện tại đã bao gồm cả hai split.

Chạy benchmark với thư mục kết quả mới:

```powershell
python -X utf8 -B -m src.benchmark --split all --output results/dense_baseline
```

Có thể chọn `--split dev` hoặc `--split test`. Runner chỉ truyền question và filter vào agent; gold được dùng sau đó để tính retrieval metrics. Trace được lưu sau từng câu. File trace có sẵn được bảo vệ khỏi chạy đè.

Chạy lại lỗi mạng/provider hoặc output reranker không hợp lệ, giữ nguyên cấu hình và lưu lịch sử vào `previous_attempts`:

```powershell
python -X utf8 -B -m src.benchmark --split all --output results/dense_baseline --retry-errors
```

Sau khi đối chiếu context, câu trả lời và citation, lưu nhãn theo schema `results/dense_baseline/reviews.json`, rồi tổng hợp mà không gọi API:

```powershell
python -X utf8 -B -m src.benchmark --split all --output results/dense_baseline --summarize-only --reviews results/dense_baseline/reviews.json
```

Kết quả gồm `traces.json`, `retrieval.json`, `reviews.json` và `summary.json`. Nhãn câu trả lời/citation được tổng hợp bằng `starter/agent_metrics.py`. Câu thiếu evidence không tham gia mẫu số retrieval. `context_exact_evidence` ghi khớp trích dẫn nguyên văn; Context Evidence Rate dùng nhãn đủ bằng chứng của context cuối. ID citation hợp lệ vẫn cần được đối chiếu với các khẳng định trong câu trả lời.

Thời gian và usage có tính cả lịch sử retry. Chi phí chat được ước tính từ usage đã nhận; nếu request lỗi không có usage, tổng chi phí giữ `null` và báo riêng phần đã xác định cùng số request chưa xác định.

## Hybrid retrieval

`src/hybrid.py` tái sử dụng `TfidfIndex` từ `starter/baseline.py`, kết hợp với Dense bằng [Reciprocal Rank Fusion](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf): mỗi nhánh đóng góp `1 / (60 + rank)`. Cấu hình `src/configs/hybrid.json` cố định 20 candidates mỗi nhánh; kết quả hợp nhất trả top-3 theo cấu hình agent.

Cả hai nhánh dùng cùng corpus và filter. Nhánh TF-IDF lọc metadata trước khi cắt candidates; IDF được tính trên toàn corpus. Context, model GPT-4o và quy tắc citation/abstention giữ cùng cấu hình Dense. Trace lưu thêm `retrieval_details.component_rankings`, còn kết quả top-3 có rank và score từng nhánh.

```powershell
python -X utf8 -B -m src.benchmark --method hybrid --split all --output results/hybrid
python -X utf8 -B -m src.benchmark --method hybrid --split all --output results/hybrid --retry-errors
python -X utf8 -B -m src.benchmark --method hybrid --split all --output results/hybrid --summarize-only --reviews results/hybrid/reviews.json
```

Các phương pháp được lưu ở thư mục kết quả riêng; kết quả Dense có sẵn được giữ nguyên. Dùng thư mục output mới nếu muốn chạy lại toàn bộ benchmark.

Khi có lỗi dịch vụ, metrics chất lượng chỉ tính các câu có kết quả để đánh giá; `retrieval_coverage` và `context_evaluation_coverage` ghi số câu đã đánh giá trên tổng số câu có đáp án. Không kết luận kết quả test đầy đủ khi coverage chưa đạt 100%. HTTP 429 `free-models-per-day` là hết hạn mức ngày; giữ trace lỗi và chạy lại khi hạn mức được khôi phục, không dùng cache query của Lab 6.

## Dense + rerank

`src/rerank.py` lấy 20 candidates từ Dense, áp dụng cùng filter, rồi gọi GPT-4o xếp hạng toàn bộ candidates theo mức hỗ trợ câu hỏi. Cấu hình ở `src/configs/rerank.json`: temperature 0, tối đa 2.048 output tokens. Agent sau đó dùng top-3 với cùng context budget và quy tắc citation/abstention.

Reranker chỉ nhận query, đoạn văn và nguồn/metadata; không nhận gold hoặc đáp án. Kết quả phải là hoán vị đầy đủ của các ID đầu vào. Lỗi xếp hạng được ghi riêng, không thay bằng ranking Dense. `score = 1 / rerank_rank` là điểm thứ tự, không phải xác suất.

Trace có `reranker_calls` với request/response, model, token usage, thời gian và chi phí. `retrieval_details` lưu candidates Dense và ranking sau rerank; mỗi kết quả có `dense_rank`, `dense_score`, `rerank_rank`. Tổng chi phí và usage bao gồm cả reranker và agent.

```powershell
python -X utf8 -B -m src.benchmark --method rerank --split all --output results/rerank
python -X utf8 -B -m src.benchmark --method rerank --split all --output results/rerank --retry-errors
python -X utf8 -B -m src.benchmark --method rerank --split all --output results/rerank --summarize-only --reviews results/rerank/reviews.json
```
