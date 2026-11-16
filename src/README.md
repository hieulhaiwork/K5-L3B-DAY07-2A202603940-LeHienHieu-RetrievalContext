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
python -X utf8 -B -m src.agent --question "In Microsoft Word, how do I add a comment to selected text?" --output results/agent_manual.json
```

File output là JSON array để dùng với công cụ tổng hợp của starter. File có sẵn sẽ không bị ghi đè. CLI kiểm tra index trước khi gọi model và dùng cache embedding tạm thời. Các lần chạy nhiều câu nên dùng lại retriever/cache và gọi `run_agent` để tránh gọi embedding trùng.

Trace có câu hỏi, cấu hình, model yêu cầu/thực trả, request và response tool, query/filter/top-k, kết quả retrieval, context cuối, câu trả lời, citation, token usage, thời gian, chi phí và lỗi. Không ghi API key hoặc chuỗi kết nối database. Chi phí OpenAI là ước tính từ token usage và bảng giá trong cấu hình; usage không xác định được giữ `null`. Chi phí embedding lưu theo số liệu provider trả về.

Các lượt thử ngoài benchmark được lưu trong `results/agent_dense_smoke.json`, gồm cả lần bị chặn do thiếu citation trong nội dung trả lời. Không dùng chúng làm kết quả dev/test. Kiểm tra ID citation chỉ bảo đảm ID có trong context và xuất hiện trong câu trả lời; chất lượng hỗ trợ từng khẳng định vẫn cần đọc đối chiếu evidence khi đánh giá.

Tham khảo: [OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling), [GPT-4o và bảng giá](https://developers.openai.com/api/docs/models/gpt-4o).
