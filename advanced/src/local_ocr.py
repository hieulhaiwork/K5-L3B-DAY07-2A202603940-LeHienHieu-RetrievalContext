"""Compose PaddleOCR layout, English OCR and table pipelines on a 4 GB GPU.

Using separate public components avoids PP-StructureV3's unconditional loading
of the large chart model in PaddleX 3.3.13. Charts are described separately.
"""
import os
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

MODELS = {
    "layout": "PP-DocLayout-L",
    "text_detection": "PP-OCRv5_mobile_det",
    "text_recognition": "en_PP-OCRv5_mobile_rec",
    "wired_table_structure": "SLANeXt_wired",
    "wireless_table_structure": "SLANeXt_wireless",
}


def intersection(a, b):
    return max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))

def area(a):
    return max(1,(a[2]-a[0])*(a[3]-a[1]))

def clean_layout(boxes):
    """Generic geometry checks, without document/page-specific decisions."""
    boxes = sorted(boxes, key=lambda b:float(b["score"]), reverse=True)
    unique = []
    for box in boxes:
        rect = box["coordinate"]
        if any(intersection(rect,b["coordinate"]) / max(area(rect),area(b["coordinate"])) > .9 for b in unique):
            continue
        unique.append(box)
    tables = [b for b in unique if b["label"]=="table"]
    containers = set()
    for outer in tables:
        children = [b for b in tables if b is not outer and area(b["coordinate"])<.85*area(outer["coordinate"])
                    and intersection(b["coordinate"],outer["coordinate"])/area(b["coordinate"])>.95]
        if any(intersection(a["coordinate"],b["coordinate"])/min(area(a["coordinate"]),area(b["coordinate"]))<.1
               for i,a in enumerate(children) for b in children[i+1:]):
            containers.add(id(outer))
    unique = [b for b in unique if id(b) not in containers]
    return [b for b in unique if not any(
        b is not outer and b["label"]==outer["label"] and
        area(b["coordinate"])<.9*area(outer["coordinate"]) and
        intersection(b["coordinate"],outer["coordinate"])/area(b["coordinate"])>.97
        for outer in unique)]

class LocalParser:
    def __init__(self, device="gpu:0"):
        from paddleocr import LayoutDetection, PaddleOCR
        self.device = device
        options = dict(device=device, enable_hpi=False, use_tensorrt=False, enable_mkldnn=False)
        self.layout = LayoutDetection(model_name=MODELS["layout"], **options)
        self.ocr = PaddleOCR(
            text_detection_model_name=MODELS["text_detection"],
            text_recognition_model_name=MODELS["text_recognition"],
            text_recognition_batch_size=1,
            text_det_limit_type="max", text_det_limit_side_len=1600,
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False, **options)
        self.tables = None

    def table_pipeline(self):
        if self.tables is None:
            from paddleocr import TableRecognitionPipelineV2
            self.tables = TableRecognitionPipelineV2(
                device=self.device, use_layout_detection=False, use_ocr_model=False,
                use_doc_orientation_classify=False, use_doc_unwarping=False,
                wired_table_structure_recognition_model_name=MODELS["wired_table_structure"],
                wireless_table_structure_recognition_model_name=MODELS["wireless_table_structure"],
                enable_hpi=False, use_tensorrt=False, enable_mkldnn=False)
        return self.tables

    def parse(self, image):
        layout = list(self.layout.predict(str(image), threshold=.35))[0]
        ocr = list(self.ocr.predict(str(image)))[0]
        boxes = clean_layout(layout["boxes"])
        layout["boxes"] = boxes
        table_boxes = [b for b in boxes if b["label"] == "table"]
        html_by_box = {}
        if table_boxes:
            result = self.table_pipeline().predict(
                str(image), overall_ocr_res=ocr, layout_det_res=layout,
                use_table_orientation_classify=False,
                use_e2e_wired_table_rec_model=False,
                use_e2e_wireless_table_rec_model=False,
                use_ocr_results_with_table_cells=False)[0]
            tables = result.json["res"]["table_res_list"]
            if len(tables) != len(table_boxes):
                raise ValueError("Table count differs from detected table regions")
            html_by_box = {tuple(b["coordinate"]): t["pred_html"] for b, t in zip(table_boxes, tables)}
        # Assign each OCR line to one layout block, preserving otherwise missed lines.
        grouped = [[] for _ in boxes]
        unassigned = []
        texts, rects = ocr["rec_texts"], ocr["rec_boxes"]
        for text, rect in zip(texts, rects):
            x0, y0, x1, y1 = [float(v) for v in rect]
            line_area = max(1, (x1-x0)*(y1-y0))
            overlaps = []
            for box in boxes:
                a,b,c,d = box["coordinate"]
                overlaps.append(max(0,min(c,x1)-max(a,x0))*max(0,min(d,y1)-max(b,y0))/line_area)
            best = max(range(len(boxes)), key=lambda i:(round(overlaps[i],2), boxes[i]["label"] not in {"image","chart"}, -area(boxes[i]["coordinate"]))) if boxes else None
            item = (y0, x0, text)
            if best is not None and overlaps[best] >= .5:
                grouped[best].append(item)
            else:
                unassigned.append({"block_label":"text","block_content":text,
                                   "block_bbox":[x0,y0,x1,y1], "layout_score":None})
        blocks = []
        for box, lines in zip(boxes, grouped):
            bbox = [float(v) for v in box["coordinate"]]
            label = box["label"]
            content = html_by_box[tuple(box["coordinate"])] if label == "table" else "\n".join(t[2] for t in sorted(lines))
            blocks.append({"block_label":label,"block_content":content,"block_bbox":bbox,
                           "layout_score":float(box["score"])})
        blocks.extend(unassigned)
        blocks.sort(key=lambda b:(b["block_bbox"][1],b["block_bbox"][0]))
        for i, block in enumerate(blocks):
            block["block_order"] = i
        return {"parsing_res_list":blocks, "overall_ocr_res":ocr.json["res"],
                "layout_det_res":layout.json["res"]}
