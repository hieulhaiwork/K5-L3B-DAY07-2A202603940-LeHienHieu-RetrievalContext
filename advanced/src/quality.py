"""Shared automated checks for local and GPT-4o extraction.

These detect common failures; passing them is not a claim of human verification.
"""
from collections import Counter
from decimal import Decimal
from html import escape
from html.parser import HTMLParser
import re

def corrupt(text):
    controls = sum(ord(c) < 32 and c not in "\n\r\t" for c in text)
    return (controls + text.count("\ufffd")) / max(1, len(text)) > .01

class TableReader(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self.row, self.cell, self.parts = [], None, None, []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = {"header": tag == "th", "rowspan": int(attrs.get("rowspan", 1)),
                         "colspan": int(attrs.get("colspan", 1))}
            self.parts = []
        elif tag == "br" and self.cell is not None:
            self.parts.append(" ")
    def handle_data(self, data):
        if self.cell is not None:
            self.parts.append(data)
    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cell is not None:
            self.cell["text"] = " ".join("".join(self.parts).split())
            if self.row is not None:
                self.row.append(self.cell)
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None

def normalize_rows(rows):
    if not isinstance(rows, list) or not rows or len(rows) > 1000:
        raise ValueError("Missing or invalid table rows")
    normalized = []
    for row in rows:
        if not isinstance(row, list) or not row or len(row) > 128:
            raise ValueError("Invalid table row")
        cells = []
        for cell in row:
            if isinstance(cell, str):
                cell = {"text": cell}
            if not isinstance(cell, dict) or not isinstance(cell.get("text"), str):
                raise ValueError("Table cells must contain strings")
            parsed = {"text": cell["text"].strip(), "header": cell.get("header", False),
                      "rowspan": cell.get("rowspan", 1), "colspan": cell.get("colspan", 1)}
            if type(parsed["header"]) is not bool:
                raise ValueError("Cell header must be boolean")
            if any(type(parsed[k]) is not int or not 1 <= parsed[k] <= 128 for k in ["rowspan", "colspan"]):
                raise ValueError("Invalid cell span")
            cells.append(parsed)
        normalized.append(cells)
    return normalized

def table_text(rows):
    return "\n".join(" | ".join(c["text"] for c in row) for row in rows)

def table_html(rows):
    result = ["<table>"]
    for row in rows:
        result.append("<tr>")
        for cell in row:
            tag = "th" if cell["header"] else "td"
            result.append(f'<{tag} rowspan="{cell["rowspan"]}" colspan="{cell["colspan"]}">{escape(cell["text"])}</{tag}>')
        result.append("</tr>")
    return "".join(result) + "</table>"

def number_tokens(value):
    pattern = r"(?<![A-Za-z0-9])(?:[-\u2212][ \t]*|\([ \t]*)?\d[\d,]*(?:\.\d+)?(?:[ \t]*%|[ \t]*\))?"
    tokens = []
    for match in re.findall(pattern, value):
        item = re.sub(r"[\s,]", "", match).replace("\u2212", "-")
        percent = "%" if "%" in item else ""
        negative = item.startswith(("-", "("))
        digits = item.replace("%", "").strip("()-")
        token = format(Decimal(digits).normalize(), "f")
        tokens.append(("-" if negative and Decimal(digits) != 0 else "") + token + percent)
    return Counter(tokens)

def content_checks(text, reference="", reliable=False):
    flags, metrics = [], {}
    if not text.strip() or corrupt(text):
        flags.append("empty_or_corrupt_text")
    expected, actual = number_tokens(reference), number_tokens(text)
    if sum(expected.values()) >= 3:
        recall = sum((expected & actual).values()) / sum(expected.values())
        metrics["numeric_token_recall"] = round(recall, 4)
        if recall < (1.0 if reliable else .98):
            flags.append("numeric_tokens_missing_or_changed")
    if reliable and actual - expected:
        flags.append("numeric_tokens_not_in_source_reference")
    for unit in ["thousands", "millions", "billions"]:
        if re.search(r"\b" + unit + r"\b", reference, re.I) and not re.search(r"\b" + unit + r"\b", text, re.I):
            flags.append("source_unit_missing")
    if "$" in reference and not re.search(r"\$|\bUSD\b|\bdollars?\b", text, re.I):
        flags.append("source_currency_missing")
    flags.extend(text_noise_flags(text))
    return sorted(set(flags)), metrics

def table_checks(rows, reference="", reliable=False, context=""):
    text = table_text(rows)
    flags, metrics = content_checks(text + "\n" + context, reference, reliable)
    if not rows or not text.strip():
        flags.append("table_structure_or_cells_missing")
    if possible_icon_table(rows,reference,reliable):
        flags.append("possible_icon_ocr_artifact")
    numeric_rows = [row for row in rows if sum(
        bool(number_tokens(cell["text"])) and not re.search(r"[A-Za-z]",cell["text"])
        for cell in row)>=2]
    labelled = [row for row in numeric_rows if row and re.search(r"[A-Za-z]{2,}",row[0]["text"])
                and not re.match(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",row[0]["text"])]
    numeric_identity = bool(rows and re.fullmatch(r"(?:fiscal )?year|period|quarter|id|rank|date",
                                                  rows[0][0]["text"].strip(),re.I))
    if len(numeric_rows)>=2 and not labelled and not numeric_identity:
        flags.append("table_row_labels_missing")
    occupied, widths = {}, []
    for ri, row in enumerate(rows):
        cursor = 0
        for cell in row:
            while occupied.get(cursor, -1) >= ri:
                cursor += 1
            for ci in range(cursor, cursor + cell["colspan"]):
                if occupied.get(ci, -1) >= ri:
                    flags.append("overlapping_table_cells")
                occupied[ci] = ri + cell["rowspan"] - 1
            cursor += cell["colspan"]
            value = cell["text"]
            if len(re.findall(r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}", value)) > 1:
                flags.append("multiple_dates_merged_in_one_cell")
            if not re.search(r"[A-Za-z]", value) and len(re.findall(r"\d[\d,]*(?:\.\d+)?", value)) > 1:
                flags.append("multiple_numeric_values_in_one_cell")
        current = [ci for ci, until in occupied.items() if until >= ri]
        widths.append(max(current, default=-1) + 1)
    if len(set(widths)) > 1:
        flags.append("inconsistent_table_column_count")
    if any(until >= len(rows) for until in occupied.values()):
        flags.append("rowspan_outside_table")
    return sorted(set(flags)), metrics

def text_noise_flags(text):
    """Conservative glyph-noise signals; short labels are never rejected alone."""
    visible = [c for c in text if not c.isspace()]
    punctuation = set(".,;:!?%$()[]{}-/+*=<>_&'\"#@|\\")
    odd = sum(not c.isalnum() and c not in punctuation for c in visible)
    if len(visible) >= 24 and odd >= 3 and odd / len(visible) > .15:
        return ["possible_glyph_ocr_noise"]
    if re.search(r"\b([A-Za-z])\1{7,}\b", text):
        return ["possible_glyph_ocr_noise"]
    return []

def possible_icon_table(rows, reference="", reliable=False):
    cells = [cell["text"].strip() for row in rows for cell in row if cell["text"].strip()]
    descriptive = sum(len(re.findall(r"[A-Za-z]{2,}", value)) >= 3 for value in cells)
    if descriptive < 3:
        return False
    text = " ".join(cells)
    if sum(c.isdigit() for c in text) / max(1,len(text)) >= .12:
        return False
    glyphs = sum(len(value) <= 6 and (
        not any(c.isalnum() for c in value) or
        (any(c.isdigit() for c in value) and len(re.findall(r"[A-Za-z]{2,}",value)) == 0)
    ) for value in cells)
    unexpected = bool(number_tokens(text) - number_tokens(reference)) if reliable else False
    return glyphs >= 2 and (unexpected or sum(not any(c.isalnum() for c in v) for v in cells) >= 2)

def _area(box):
    return max(0,box[2]-box[0]) * max(0,box[3]-box[1])

def _intersection(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))

def deduplicate_text(records):
    """Retain evidence records; exclude exact overlapping copies from embedding."""
    groups = {}
    for index, record in enumerate(records):
        if record.get("content_type") != "text" or record.get("status","").startswith("excluded_"):
            continue
        text = " ".join(record.get("embedding_text","").split())
        if text:
            groups.setdefault((record["page"],text), []).append((index,record))
    for items in groups.values():
        ordered = sorted(items,key=lambda item:(
            not item[1].get("embedding_ready",False),
            not item[1].get("extraction_method","").startswith("pdf_native"),
            -_area(item[1]["bbox"]),item[0]))
        keep = []
        for _, record in ordered:
            duplicate = next((other for other in keep if
                _intersection(record["bbox"],other["bbox"]) /
                max(1,min(_area(record["bbox"]),_area(other["bbox"]))) > .8),None)
            if duplicate is None:
                keep.append(record)
            else:
                record.update(status="excluded_duplicate", embedding_ready=False,
                    duplicate_of=duplicate["id"],
                    exclusion_reason="same_text_same_page_overlapping_region")
    return sum(r.get("status")=="excluded_duplicate" for r in records)

def link_short_text_context(records):
    by_page = {}
    for record in records:
        if not record.get("status","").startswith("excluded_"):
            by_page.setdefault(record["page"],[]).append(record)
    for record in records:
        if record.get("content_type")!="text" or record.get("status","").startswith("excluded_"):
            continue
        if len(record.get("embedding_text","").split()) > 3:
            continue
        record["needs_context"] = True
        box = record["bbox"]
        nearby = []
        for other in by_page.get(record["page"],[]):
            if other is record or other["content_type"] not in {"text","table"}:
                continue
            if other.get("status")!="extracted" or not other.get("embedding_ready"):
                continue
            if other["content_type"]=="text" and len(other.get("embedding_text","").split())<=3:
                continue
            candidate = other["bbox"]
            horizontal = max(0,min(box[2],candidate[2])-max(box[0],candidate[0]))
            gap = max(0,max(box[1],candidate[1])-min(box[3],candidate[3]))
            if gap<=32 and horizontal/max(1,min(box[2]-box[0],candidate[2]-candidate[0])) >= .3:
                nearby.append((gap,other["id"]))
        record["context_record_ids"] = [item[1] for item in sorted(nearby)[:2]]

def finalize_index_selection(records):
    """Keep uncertain evidence for audit, but never expose it to embedding."""
    allowed = {"text","table","diagram","chart","informative_image"}
    def exclude(record,reason):
        record.setdefault("status_before_exclusion",record.get("status"))
        record.update(status="excluded_uncertain",embedding_ready=False,
                      exclusion_reason=reason)
    for record in records:
        if record.get("status","").startswith("excluded_"):
            record["embedding_ready"]=False
            continue
        if record.get("status")!="extracted":
            exclude(record,"extraction_not_completed_or_not_validated")
        elif record.get("quality_flags") or record.get("vision_result",{}).get("uncertainties"):
            exclude(record,"quality_checks_not_passed")
        elif record.get("content_type") not in allowed:
            exclude(record,"unsupported_or_unclassified_content")
        elif not record.get("embedding_text","").strip() or corrupt(record["embedding_text"]):
            exclude(record,"empty_or_corrupt_text")
        else:
            record["embedding_ready"]=True
    link_short_text_context(records)
    retained={r["id"]:r for r in records if r.get("embedding_ready")}
    for record in records:
        if not record.get("embedding_ready") or record.get("content_type")!="text" or len(record.get("embedding_text","").split())>3:
            continue
        context=[identifier for identifier in record.get("context_record_ids",[])
                 if identifier in retained and (
                     retained[identifier]["content_type"]=="table" or
                     len(retained[identifier].get("embedding_text","").split())>3)]
        record["context_record_ids"]=context
        if not context:
            exclude(record,"short_text_without_retained_context")
    identifiers=[r["id"] for r in records if r.get("embedding_ready")]
    return {
        "policy":"exclude_uncertain",
        "record_ids":identifiers,
        "selected_records":len(identifiers),
        "excluded_uncertain_records":sum(r.get("status")=="excluded_uncertain" for r in records),
        "excluded_uncertain_by_type":dict(Counter(
            r["content_type"] for r in records if r.get("status")=="excluded_uncertain")),
        "excluded_uncertain_by_reason":dict(Counter(
            r.get("exclusion_reason") for r in records if r.get("status")=="excluded_uncertain")),
    }
