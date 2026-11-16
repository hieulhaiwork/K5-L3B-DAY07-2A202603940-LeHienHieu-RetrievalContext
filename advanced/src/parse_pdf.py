"""Step 1-2: parse the annual report into pages and candidate elements (text, table, figure).

Outputs (all reproducible from the source PDF):
  artifacts/pages/p###.png        rendered page images
  artifacts/crops/<id>.png        crops of tables and figures
  artifacts/parsed/pages.jsonl    per-page text, label and text blocks with bbox
  artifacts/parsed/elements.jsonl candidate tables/figures with bbox and crop path
  artifacts/parsed/stats.json     document statistics
"""
import argparse
import json
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "data" / "2026-Annual-Report-Web.pdf"
SOURCE_URL = "https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-Annual-Report-Web.pdf"


def rect_list(rect):
    return [round(v, 1) for v in (rect.x0, rect.y0, rect.x1, rect.y1)]


def overlap_ratio(a, b):
    """Share of rect a covered by rect b."""
    inter = fitz.Rect(a) & fitz.Rect(b)
    area = fitz.Rect(a).get_area()
    return inter.get_area() / area if area and not inter.is_empty else 0.0


def merge_rects(rects, gap=12):
    """Merge rects that touch or lie within `gap` points of each other."""
    rects = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed, merged = False, []
        for r in rects:
            for i, m in enumerate(merged):
                if (m + (-gap, -gap, gap, gap)).intersects(r):
                    merged[i] = m | r
                    changed = True
                    break
            else:
                merged.append(r)
        rects = merged
    return rects


def page_tables(page):
    try:
        found = page.find_tables()
    except Exception as error:  # table detection failures are recorded, not hidden
        return [], f"{type(error).__name__}: {error}"
    tables = []
    for t in found.tables:
        rows = t.extract()
        if len(rows) < 2 or max((len(r) for r in rows), default=0) < 2:
            continue
        tables.append({"bbox": list(t.bbox), "rows": rows, "markdown": t.to_markdown()})
    return tables, None


def page_figures(page, table_boxes, min_area_ratio, min_side):
    """Raster images plus clusters of vector drawings (charts are often vector graphics)."""
    page_area = page.rect.get_area()
    candidates = []
    for info in page.get_image_info(xrefs=True):
        r = fitz.Rect(info["bbox"]) & page.rect
        if not r.is_empty:
            candidates.append(("raster", r))
    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    for r in clusters:
        r = fitz.Rect(r) & page.rect
        if not r.is_empty:
            candidates.append(("vector", r))
    figures = []
    for kind in ("raster", "vector"):
        rects = merge_rects([r for k, r in candidates if k == kind])
        for r in rects:
            if r.width < min_side or r.height < min_side or r.get_area() / page_area < min_area_ratio:
                continue
            if any(overlap_ratio(r, t) > 0.6 for t in table_boxes):
                continue  # table rules/shading, already captured as table
            if r.get_area() / page_area > 0.97 and kind == "vector":
                continue  # full-page background
            figures.append({"kind": kind, "bbox": rect_list(r)})
    return figures


def crop(page, bbox, path, dpi):
    clip = fitz.Rect(bbox) + (-4, -4, 4, 4)
    page.get_pixmap(dpi=dpi, clip=clip & page.rect).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--page-dpi", type=int, default=110)
    parser.add_argument("--crop-dpi", type=int, default=200)
    parser.add_argument("--min-area-ratio", type=float, default=0.02)
    parser.add_argument("--min-side", type=float, default=60)
    parser.add_argument("--no-render", action="store_true", help="skip page/crop PNG rendering")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}. Download it from {SOURCE_URL}")
    for sub in ("pages", "crops", "parsed"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    pages_out, elements_out, errors = [], [], []
    for index, page in enumerate(doc):
        number = index + 1  # 1-based physical PDF page, used for all citations
        text = page.get_text("text")
        blocks = [{"bbox": [round(v, 1) for v in b[:4]], "text": b[4].strip()}
                  for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        tables, table_error = page_tables(page)
        if table_error:
            errors.append({"page": number, "stage": "find_tables", "error": table_error})
        figures = page_figures(page, [t["bbox"] for t in tables], args.min_area_ratio, args.min_side)
        if not args.no_render:
            page.get_pixmap(dpi=args.page_dpi).save(args.out / "pages" / f"p{number:03d}.png")

        for k, t in enumerate(tables, 1):
            eid = f"p{number:03d}-t{k}"
            path = args.out / "crops" / f"{eid}.png"
            if not args.no_render:
                crop(page, t["bbox"], path, args.crop_dpi)
            elements_out.append({"id": eid, "page": number, "type": "table", "bbox": rect_list(fitz.Rect(t["bbox"])),
                                 "n_rows": len(t["rows"]), "n_cols": max(len(r) for r in t["rows"]),
                                 "rows": t["rows"], "markdown": t["markdown"],
                                 "crop": path.relative_to(ROOT).as_posix()})
        for k, f in enumerate(figures, 1):
            eid = f"p{number:03d}-f{k}"
            path = args.out / "crops" / f"{eid}.png"
            if not args.no_render:
                crop(page, f["bbox"], path, args.crop_dpi)
            inside = " ".join(b["text"] for b in blocks if overlap_ratio(b["bbox"], f["bbox"]) > 0.5)
            elements_out.append({"id": eid, "page": number, "type": "figure", "kind": f["kind"], "bbox": f["bbox"],
                                 "text_inside": inside[:1500], "crop": path.relative_to(ROOT).as_posix()})

        pages_out.append({"page": number, "label": page.get_label() or None,
                          "width": round(page.rect.width, 1), "height": round(page.rect.height, 1),
                          "chars": len(text.strip()), "n_tables": len(tables), "n_figures": len(figures),
                          "text": text, "blocks": blocks})
        print(f"page {number}/{len(doc)}: {len(text.strip())} chars, {len(tables)} tables, {len(figures)} figures", flush=True)

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(args.out / "parsed" / "pages.jsonl", pages_out)
    write_jsonl(args.out / "parsed" / "elements.jsonl", elements_out)
    stats = {
        "source_url": SOURCE_URL, "pdf_pages": len(doc), "metadata": doc.metadata,
        "pages_low_text": [p["page"] for p in pages_out if p["chars"] < 200],
        "tables": sum(e["type"] == "table" for e in elements_out),
        "figures_raster": sum(e["type"] == "figure" and e["kind"] == "raster" for e in elements_out),
        "figures_vector": sum(e["type"] == "figure" and e["kind"] == "vector" for e in elements_out),
        "pages_with_figures": sorted({e["page"] for e in elements_out if e["type"] == "figure"}),
        "errors": errors,
        "settings": {k: v for k, v in vars(args).items() if k not in {"pdf", "out"}},
    }
    (args.out / "parsed" / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "metadata"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
