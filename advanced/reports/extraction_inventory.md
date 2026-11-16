# Kiểm kê tài liệu (bước parse)

Nguồn: [NVIDIA 2026 Annual Report](https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-Annual-Report-Web.pdf). Tạo lại bằng `python advanced/src/parse_pdf.py`. Số trang trích dẫn là **trang PDF vật lý (1–175)**, không phải số in trên trang.

## Thống kê

| Mục | Giá trị |
|---|---|
| Số trang PDF | 175 (PDF không có page label) |
| Ký tự text trích được | ~636 nghìn |
| Trang gần như không có text | 1 (bìa), 175 (bìa sau) |
| Bảng do `find_tables` phát hiện | 38 |
| Cụm hình (raster / vector) | 22 / 61 |
| Trang lỗi font khi trích text | 128 (bảng % doanh thu: text bị mã hóa sai, cần vision/OCR) |

**Cấu trúc:** trang 1–11 là Annual Review (khổ ngang, ảnh nền kèm text) → 12–20 thư CEO và tổng quan → 21–~100 Proxy Statement → ~101–174 Form 10-K.

## Nhận xét về trích xuất

- Bảng tài chính dùng dạng tô nền xen kẽ, không có đường kẻ, nên `find_tables` bỏ sót nhiều. Phần lớn "vector figure" thực chất là bảng (ví dụ trang 139–143, 149–169). Bước enrich cần **phân loại lại**: cụm vector chủ yếu là số thì xếp vào bảng.
- Biểu đồ trong tài liệu là **ảnh raster**. Các nhãn số trong biểu đồ không có trong text layer, nên phải dùng vision.
- Một số sơ đồ có nhãn nằm trong ảnh (trang 3, 4, 14).

## Hình có thông tin (ứng viên benchmark)

| Trang | Element | Loại | Nội dung |
|---|---|---|---|
| 3 | p003-f1 | Sơ đồ | "AI Is a Five-Layer Cake": Energy → Chips → Infrastructure → Models → Applications |
| 4 | p004-f1 | Sơ đồ | AI factory DSX: Rubin DSX, DSX Max-Q/Flex/Exchange/Sim, Power, Liquid Cooling |
| 14 | p014-f1 | Ảnh/sơ đồ | Slide "Inference At-Scale is Extreme Computing" (throughput và tokens/s) |
| 23 | p023-f1 | Ảnh | 4 market platforms kèm doanh thu và tăng trưởng (dạng bảng-ảnh) |
| 24 | p024-f1/f2 | Biểu đồ | TSR 1/3/5 năm; Return of Capital (dividends và buybacks) |
| 46 | p046-f1 | Sơ đồ | Risk Oversight: Board → AC/CC/NCGC → Management → Internal Audit |
| 47 | p047-f1 | Sơ đồ | ERM Process |
| 57 | p057-f1 | Biểu đồ | CEO / Other NEO Fiscal 2026 Target Pay Mix |
| 60 | p060-f1 | Biểu đồ | Peer group: Revenue và Market Cap (75th, median, 25th percentile) |
| 76 | p076-f1 | Biểu đồ | NEO CAP vs TSR; NEO CAP vs Net Income và Non-GAAP Operating Income |
| 123 | p123-f1 | Biểu đồ | 5-year cumulative total return: NVIDIA vs S&P 500 vs Nasdaq 100 |

Tổng cộng có ~9 biểu đồ trên 5 trang và ~6 sơ đồ hoặc ảnh có thông tin, **đủ** cho 5+5 câu test và các câu dev. Chưa xem hết các ảnh ở trang 2, 5, 7–12, 17, 18 (chủ yếu là ảnh chân dung hoặc minh họa); cần lọc ảnh trang trí.

## Bảng chính (ứng viên câu hỏi bảng và multi-hop)

Gồm: Income statement (139), Comprehensive income (140), Balance sheet (141), Equity (142), Cash flow (143), Stock-based comp (149), EPS (151), Intangibles (152), Fair value (153–154), Inventory/PP&E (155), Accrued liabilities (156), Notes payable (158), Income tax (160–164), Segment (165–166), Revenue theo market và vùng (166–167), Summary MD&A (125), Income statement theo % doanh thu (128, lỗi font) và Segment revenue/operating income (128), Compensation tables (proxy).
