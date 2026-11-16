# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Dữ liệu dùng trong bài nộp

- `chunks.json`: 317 chunks trên 64 trang PDF được chọn từ *Workplace Software and Skills* của OpenStax, giấy phép CC BY-NC-SA 4.0. [PDF nguồn](https://assets.openstax.org/oscms-prodcms/media/documents/Workplace_Software_and_Skills_-_WEB_IlfJtcP.pdf). Attribution và quyền sử dụng được giữ trong metadata từng chunk.
- `chunks_B.jsonl`: snapshot đầu vào từ Lab 6; chuyển sang schema Lab 7 bằng `src/convert_data/convert_chunks.py`, giữ nguyên giá trị ID và nội dung.
- `questions.json`: 24 câu, gồm dev (15 có đáp án + 2 thiếu evidence) và test (5 có đáp án + 2 thiếu evidence), đều có nguồn/nhãn hoặc cách kiểm tra thiếu thông tin.
- ID nguồn và số trang PDF giữ nguyên từ Lab 6; đường dẫn PDF trong chunk là provenance của nguồn cũ. Index có thể dựng lại từ `chunks.json` mà không cần tải PDF.
