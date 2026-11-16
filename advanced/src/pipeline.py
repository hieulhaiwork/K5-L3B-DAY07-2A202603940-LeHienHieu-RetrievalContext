"""Automatic local PDF extraction, with GPT-4o recovery for failed text/tables and visuals.

Examples (from repository root):
  python advanced/src/pipeline.py --pages 7,46,128,139 --no-vision
  python advanced/src/pipeline.py --pages all
  python advanced/src/pipeline.py --pages all --device cpu --no-vision

No page-specific rules or manual overrides are read. Output manifest records
partial coverage, extraction errors and unprocessed visual regions explicitly.
"""
import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import threading
import time

import fitz
from local_ocr import LocalParser, MODELS
from table_recovery import recover_table, table_fragments
from quality import TableReader, corrupt, normalize_rows, table_checks, table_text, content_checks, deduplicate_text, link_short_text_context, finalize_index_selection

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 6
LOCAL_CACHE_SCHEMA = 2

def digest(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)

class GPUMonitor:
    def __init__(self):
        self.maximum, self.stop_event = None, threading.Event()
        self.thread = threading.Thread(target=self.watch, daemon=True)
    def watch(self):
        while not self.stop_event.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                value = int(r.stdout.splitlines()[0])
                self.maximum = max(self.maximum or 0, value)
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                pass
            self.stop_event.wait(1)
    def __enter__(self):
        self.thread.start()
        return self
    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=6)

def choose_pages(value, count):
    selected = list(range(1, count + 1)) if value == "all" else sorted({int(x) for x in value.split(",")})
    if not selected or min(selected) < 1 or max(selected) > count:
        raise ValueError(f"Page numbers must be within 1..{count}")
    return selected

def make_records(page, raw, image, output, page_num):
    sx, sy = image.width / page.rect.width, image.height / page.rect.height
    native_page_bad = corrupt(page.get_text())
    records = []
    fragments = table_fragments(raw.get('parsing_res_list',[]))
    for i, block in enumerate(raw.get("parsing_res_list", []), 1):
        bounds = block.get("block_bbox")
        if not bounds or len(bounds) != 4:
            raise ValueError("Parser returned a block without a valid bounding box")
        rect = fitz.Rect(bounds[0] / sx, bounds[1] / sy, bounds[2] / sx, bounds[3] / sy) & page.rect
        if rect.is_empty:
            continue
        label = block.get("block_label", "unknown")
        content = block.get("block_content", "") or ""
        native = page.get_text("text", clip=rect, sort=True).strip()
        visual = label in {"image", "chart", "figure", "header_image", "footer_image"}
        kind = "table" if label == "table" else ("visual" if visual else "text")
        record_id = f"p{page_num:03d}-b{i:03d}"
        crop = output / "crops" / (record_id + ".png")
        crop.parent.mkdir(parents=True, exist_ok=True)
        margin = (4, 48, 4, 18) if kind == "table" else (3, 3, 3, 3)
        crop_rect = (rect + (-margin[0], -margin[1], margin[2], margin[3])) & page.rect
        page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=crop_rect, alpha=False).save(crop)
        flags = []
        record = {
            "id": record_id, "page": page_num, "bbox": [round(v, 2) for v in rect],
            "content_type": kind, "layout_label": label,
            "crop_bbox": [round(v, 2) for v in crop_rect],
            "crop": crop.relative_to(output).as_posix(),
            "reading_order": block.get("block_order", i),
            "extraction_method": "paddleocr_layout_ocr_and_table_v2",
            "verification": "automated_checks_only",
            "quality_flags": flags,
        }
        if kind == "table":
            reader = TableReader()
            try:
                reader.feed(content)
                rows = normalize_rows(reader.rows)
            except (ValueError, TypeError):
                rows = []
            record["table_html"], record["table_rows"] = content, rows
            text = table_text(rows)
            if native and not native_page_bad and not corrupt(native):
                reference, method = native, "pdf_native"
            else:
                reference_lines = []
                ocr_data = raw.get("overall_ocr_res", {})
                for line, bounds in zip(ocr_data.get("rec_texts", []), ocr_data.get("rec_boxes", [])):
                    cx, cy = (bounds[0]+bounds[2])/2/sx, (bounds[1]+bounds[3])/2/sy
                    if rect.contains(fitz.Point(cx,cy)):
                        reference_lines.append(line)
                reference, method = "\n".join(reference_lines), "paddle_ocr"
            if corrupt(reference):
                reference, method = "", "unavailable"
            record["table_source_text"] = reference
            record["source_reference"] = {"text": reference, "method": method}
            found_flags, metrics = table_checks(rows, reference, method == "pdf_native")
            if found_flags and i-1 not in fragments:
                candidate,diagnostic = recover_table(page,raw,block,sx,sy,reference,method=="pdf_native")
                diagnostic["original_quality_flags"] = list(found_flags)
                record["table_structure_recovery"] = diagnostic
                if candidate is not None:
                    rows = candidate["rows"]
                    record["table_rows"],record["table_html"] = rows,candidate["html"]
                    text = candidate["text"]
                    reference = candidate["reference"]
                    record["source_reference"] = {"text":reference,"method":method,
                        "header_context_extension":candidate["header_reference_extension"]}
                    record["table_source_text"] = reference
                    record["table_context"] = candidate["context"]
                    record["table_units"] = [c for c in candidate["context"] if re.search(r"\b(?:thousands|millions|billions)\b",c,re.I)]
                    record["extraction_method"] = ("pdf_native_coordinate_table" if method=="pdf_native"
                                                   else "paddle_ocr_coordinate_table")
                    found_flags,metrics = table_checks(rows,reference,method=="pdf_native",
                                                      "\n".join(candidate["context"]))
            flags.extend(found_flags)
            record.update(metrics)
            # A text-dominated grid may actually be a process/organization
            # diagram. Queue visual classification, never silently certify it.
            ambiguous = (len(text)>300 and sum(c.isdigit() for c in text)/max(1,len(text))<.01) or "possible_icon_ocr_artifact" in flags
            if ambiguous:
                record["table_candidate_text"] = text
                record["content_type"] = "visual"
                record["ocr_context"] = reference
                flags.append("table_or_diagram_classification_uncertain")

            flags[:] = sorted(set(flags))

            record["embedding_text"] = text
            record["status"] = "needs_review" if flags else "extracted"
            if ambiguous:
                record["embedding_text"] = ""
                record["status"] = "pending_visual_description"
        elif kind == "visual":
            record["embedding_text"] = ""
            record["ocr_context"] = content + ("\n" + native if not native_page_bad and not corrupt(native) else "")
            record["status"] = "pending_visual_description"
        else:
            use_native = bool(native) and not native_page_bad and not corrupt(native)
            text = native if use_native else re.sub("<[^>]+>", "", content)
            record["extraction_method"] = "pdf_native_with_model_layout" if use_native else "paddleocr"
            record["embedding_text"] = text
            record["source_reference"] = {
                "text": text if not corrupt(text) else "",
                "method": "pdf_native" if use_native else "paddle_ocr"}
            found_flags, _ = content_checks(text)
            flags.extend(found_flags)
            record["status"] = "needs_review" if flags else "extracted"
        record["embedding_ready"] = record["status"] == "extracted" and bool(record["embedding_text"].strip())
        if i-1 in fragments:
            record.update(status="excluded_table_fragment",embedding_ready=False,
                          table_fragment_of=f"p{page_num:03d}-b{fragments[i-1]+1:03d}",
                          exclusion_reason="numeric_slice_of_same_table")
        records.append(record)

    # Recover good native text blocks missed by the learned layout. This is
    # geometry-based coverage, never a list of manually selected source pages.
    if not native_page_bad:
        for j, source in enumerate(page.get_text("blocks"), 1):
            if source[6] != 0 or not source[4].strip():
                continue
            rect, text = fitz.Rect(source[:4]) & page.rect, source[4].strip()
            if rect.is_empty:
                continue
            covered = any(
                (r["content_type"] != "visual" or "table_candidate_text" in r) and
                (rect & fitz.Rect(r["bbox"])).get_area()/max(1,rect.get_area())>.8
                for r in records)
            if covered:
                continue
            record_id = f"p{page_num:03d}-native{j:03d}"
            crop = output / "crops" / (record_id + ".png")
            page.get_pixmap(matrix=fitz.Matrix(2,2),clip=rect,alpha=False).save(crop)
            records.append({
                "id":record_id,"page":page_num,"bbox":[round(v,2) for v in rect],
                "content_type":"text","layout_label":"native_coverage_fallback",
                "crop":crop.relative_to(output).as_posix(),"reading_order":None,
                "embedding_text":text,"extraction_method":"pdf_native_coverage_fallback",
                "verification":"automated_checks_only","quality_flags":[],
                "source_reference":{"text":text,"method":"pdf_native"},
                "status":"extracted","embedding_ready":True})
    for record in records:
        text = record["embedding_text"].strip()
        if record["content_type"]=="text" and (re.fullmatch(r"\d{1,4}",text) or record["layout_label"]=="number") and record["bbox"][1]>.92*page.rect.height:
            record["status"],record["embedding_ready"]="excluded_navigation",False

    if not records:
        raise ValueError("No layout blocks returned; do not mark page processed")
    return records

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/2026-Annual-Report-Web.pdf")
    parser.add_argument("--pages", default="all")
    parser.add_argument("--device", choices=["gpu:0", "cpu"], default="gpu:0")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/extraction")
    parser.add_argument("--local-cache-dir",type=Path,
                        help="Read an existing local parse cache; cache misses are written only under --output")
    parser.add_argument("--no-vision", action="store_true", help="Disable all GPT-4o calls, including text/table recovery")
    parser.add_argument("--vision-limit", type=int, default=20, help="Shared new-call budget for visuals and text/table recovery; cache hits do not count")
    parser.add_argument("--vision-token-budget",type=int,default=50000,
                        help="Per-run new-request input+output token budget; conservative reservation, then provider usage")
    parser.add_argument("--max-table-fallback-ratio",type=float,default=.85,
                        help="Block new GPT calls if local table fallback ratio exceeds this value; 1 disables this guard")
    parser.add_argument("--max-text-fallback-ratio",type=float,default=.5,
                        help="Block new GPT calls if local text fallback ratio exceeds this value; 1 disables this guard")
    parser.add_argument("--fallback-min-records",type=int,default=10,
                        help="Minimum records per type before applying a fallback-ratio guard")
    parser.add_argument("--retry-vision-failures",action="store_true",
                        help="Allow a fresh paid call for a saved failed reply; otherwise revalidate it locally")
    parser.add_argument("--force", action="store_true", help="Rerun local extraction; visual cache remains reusable")
    args = parser.parse_args()
    if not 110 <= args.dpi <= 300 or args.vision_limit < 0 or args.vision_token_budget < 0:
        parser.error("DPI must be 110..300; vision call/token budgets must be nonnegative")
    if not 0 <= args.max_table_fallback_ratio <= 1 or not 0 <= args.max_text_fallback_ratio <= 1 or args.fallback_min_records < 1:
        parser.error("Fallback ratios must be 0..1 and fallback-min-records must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_digest = digest(args.pdf.read_bytes())
    versions = {name: version(name) for name in ["paddlepaddle-gpu", "paddleocr", "paddlex", "pymupdf"]}
    signature = digest(json.dumps({"models": MODELS, "dpi": args.dpi, "schema": LOCAL_CACHE_SCHEMA, "versions": versions}, sort_keys=True).encode())
    started = time.perf_counter()
    manifest = {
        "schema_version": SCHEMA, "document": {"id": source_digest[:16], "file": str(args.pdf.resolve()),
        "source_url": "https://s201.q4cdn.com/141608511/files/doc_financials/2026/ar/2026-Annual-Report-Web.pdf",
        "version": "2026 Annual Report", "sha256": source_digest,
        "page_numbering": "physical_pdf_1_based", "rights": "Public access; redistribution license not established"},
        "config": {"versions": versions, "models": MODELS, "device": args.device, "dpi": args.dpi, "batch_size": 1},
        "pages": [], "records": [],
    }
    with GPUMonitor() as monitor, fitz.open(args.pdf) as pdf:
        selected = choose_pages(args.pages, len(pdf))
        manifest["document"]["page_count"] = len(pdf)
        manifest["selected_pages"] = selected
        manifest["full_document_run"] = len(selected) == len(pdf)
        local = None
        for n in selected:
            page = pdf[n - 1]
            entry = {"page": n, "status": "pending", "native_text_corrupt": corrupt(page.get_text())}
            raw_path = output / "cache/local" / f"{source_digest[:16]}-{signature[:12]}-p{n:03d}.json"
            try:
                image_path = output / "pages" / f"p{n:03d}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = page.get_pixmap(dpi=args.dpi, alpha=False)
                image.save(image_path)
                external_cache = args.local_cache_dir / raw_path.name if args.local_cache_dir else None
                cache_input = raw_path if raw_path.is_file() else external_cache
                if cache_input is not None and cache_input.is_file() and not args.force:
                    raw = json.loads(cache_input.read_text(encoding="utf-8"))
                    entry["cache_hit"] = True
                else:
                    if local is None:
                        local = LocalParser(args.device)
                    raw = local.parse(image_path)
                    write_json(raw_path, raw)
                    entry["cache_hit"] = False
                records = make_records(page, raw, image, output, n)
                for record in records:
                    record["document_id"] = source_digest[:16]
                manifest["records"].extend(records)
                scores = raw.get("overall_ocr_res", {}).get("rec_scores", [])
                entry.update(status="extracted", records=len(records),
                             ocr_mean_score=round(sum(scores) / len(scores), 4) if scores else None)
                if scores and entry["ocr_mean_score"] < .85:
                    entry["quality_warning"] = "low_mean_ocr_confidence"
                    for record in records:
                        if record["status"] in {"excluded_navigation", "excluded_decorative"}:
                            continue
                        if not record["extraction_method"].startswith("pdf_native"):
                            record["quality_flags"].append("low_page_ocr_confidence")
                            record["status"] = "needs_review" if record["content_type"] != "visual" else record["status"]
                            record["embedding_ready"] = False
            except Exception as error:
                entry.update(status="error", error_type=type(error).__name__, error=str(error)[:300])
            manifest["pages"].append(entry)
            print(f"PDF {n}: {entry['status']}; records={entry.get('records', 0)}", flush=True)
            write_json(output / "manifest.json", manifest)
        # Release local predictors before optional network requests.
        del local
    manifest["deduplication"] = {"excluded_before_vision":deduplicate_text(manifest["records"])}
    if not args.no_vision:
        from vision import enrich_records
        manifest["vision"] = enrich_records(
            manifest["records"], output, args.vision_limit, ROOT.parent / ".env",
            on_update=lambda: write_json(output / "manifest.json", manifest),
            token_budget=args.vision_token_budget,table_ratio=args.max_table_fallback_ratio,
            text_ratio=args.max_text_fallback_ratio,min_records=args.fallback_min_records,
            retry_failures=args.retry_vision_failures)
    manifest["deduplication"]["excluded_after_vision"] = deduplicate_text(manifest["records"])
    manifest["index_selection"] = finalize_index_selection(manifest["records"])
    if "vision" in manifest:
        manifest["vision"]["excluded_unresolved_records"] = manifest["index_selection"]["excluded_uncertain_records"]
    manifest["config"]["vision_limits"] = {
        "priority":["figure","failed_table","failed_text"],
        "new_calls":args.vision_limit,"token_budget":args.vision_token_budget,
        "max_table_fallback_ratio":args.max_table_fallback_ratio,
        "max_text_fallback_ratio":args.max_text_fallback_ratio,
        "fallback_min_records":args.fallback_min_records}
    manifest["summary"] = {
        "pages_requested": len(manifest["pages"]),
        "pages_extracted": sum(p["status"] == "extracted" for p in manifest["pages"]),
        "pages_failed": sum(p["status"] == "error" for p in manifest["pages"]),
        "records_by_type": dict(Counter(r["content_type"] for r in manifest["records"])),
        "records_by_status": dict(Counter(r["status"] for r in manifest["records"])),
        "embedding_ready_records": sum(r["embedding_ready"] for r in manifest["records"]),
        "gpu_device_peak_used_mib": monitor.maximum,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "content_verified_against_original": False,
    }
    manifest["ready_for_complete_index"] = (
        manifest["full_document_run"] and not manifest["summary"]["pages_failed"]
        and all(r["status"] in {"extracted", "excluded_decorative", "excluded_navigation", "excluded_duplicate", "excluded_table_fragment"} for r in manifest["records"]))
    manifest["ready_for_partial_index"] = bool(manifest["index_selection"]["record_ids"])
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["summary"], indent=2), flush=True)
    print(f"Index selection: {manifest['index_selection']['selected_records']} retained; "
          f"{manifest['index_selection']['excluded_uncertain_records']} uncertain records excluded.",flush=True)
    if manifest["summary"]["pages_failed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
