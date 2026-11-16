"""Recover conservative numeric tables from source coordinates, without API calls.

No source-page rules. Ambiguous layouts are rejected, not forced into a grid.
Raw model HTML remains available as the local candidate in the manifest.
"""
import re
from statistics import median
import fitz
from quality import corrupt, normalize_rows, table_checks, table_html, table_text

DATE = re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b")
MONTH = re.compile(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*$")
VALUE = re.compile(r"^(?:[$€£]\s*)?[-\u2212]?(?:\(\s*)?\d[\d,]*(?:\.\d+)?\s*\)?\s*%?$")
MARKERS = {"$","€","£","%"}
DASHES = {"-","—","–","N/A","n/a"}

def area(box):
    return max(0,box[2]-box[0])*max(0,box[3]-box[1])

def overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))

def table_fragments(blocks):
    """A near-contained, same-height numeric slice must not be a second table."""
    tables=[(i,b["block_bbox"]) for i,b in enumerate(blocks) if b["block_label"]=="table"]
    result={}
    for i,box in tables:
        parents=[]
        for j,outer in tables:
            if i==j or area(box)>=.85*area(outer):
                continue
            height=min(box[3]-box[1],outer[3]-outer[1])/max(box[3]-box[1],outer[3]-outer[1],1)
            if overlap(box,outer)/max(1,area(box))>=.94 and height>=.85 and box[0]>outer[0]+.15*(outer[2]-outer[0]):
                parents.append((area(outer),j))
        if parents:
            result[i]=max(parents)[1]
    return result

def cluster_rows(items):
    if not items:
        return [],1
    height=median(item["box"][3]-item["box"][1] for item in items)
    rows=[]
    for item in sorted(items,key=lambda t:((t["box"][1]+t["box"][3])/2,t["box"][0])):
        cy=(item["box"][1]+item["box"][3])/2
        found=next((row for row in reversed(rows[-3:]) if abs(row["cy"]-cy)<=height*.42),None)
        if found is None:
            rows.append({"cy":cy,"items":[item]})
        else:
            found["items"].append(item)
            found["cy"]=median((t["box"][1]+t["box"][3])/2 for t in found["items"])
    for row in rows:
        row["items"].sort(key=lambda t:t["box"][0])
        # Native PDF words can be merged into a date without estimating glyph positions.
        merged=[];i=0
        while i<len(row["items"]):
            group=row["items"][i:i+3]
            if len(group)==3 and MONTH.fullmatch(group[0]["text"]) and re.fullmatch(r"\d{1,2},?",group[1]["text"]) and re.fullmatch(r"\d{4}",group[2]["text"]):
                merged.append({"text":" ".join(t["text"] for t in group),
                    "box":[min(t["box"][0] for t in group),min(t["box"][1] for t in group),
                           max(t["box"][2] for t in group),max(t["box"][3] for t in group)]})
                i+=3
            else:
                merged.append(row["items"][i]);i+=1
        row["items"]=merged
    return rows,height

def source_items(page,raw,rect,sx,sy,top):
    expanded=fitz.Rect(rect.x0,top,rect.x1,rect.y1+3)&page.rect
    if not corrupt(page.get_text()):
        items=[{"text":word[4],"box":list(word[:4])} for word in page.get_text("words")
               if expanded.contains(fitz.Point((word[0]+word[2])/2,(word[1]+word[3])/2))]
        return items,"pdf_native_words"
    ocr=raw.get("overall_ocr_res",{})
    items=[]
    for text,b in zip(ocr.get("rec_texts",[]),ocr.get("rec_boxes",[])):
        box=[b[0]/sx,b[1]/sy,b[2]/sx,b[3]/sy]
        if expanded.contains(fitz.Point((box[0]+box[2])/2,(box[1]+box[3])/2)):
            items.append({"text":text.strip(),"box":box})
    return items,"paddle_ocr_boxes"

def recover_table(page,raw,block,sx,sy,reference,reliable):
    bounds=block["block_bbox"]
    rect=fitz.Rect(bounds[0]/sx,bounds[1]/sy,bounds[2]/sx,bounds[3]/sy)&page.rect
    # Do not import the last row of a preceding table as this table's header.
    fragments=table_fragments(raw.get("parsing_res_list",[]))
    previous=[b["block_bbox"][3]/sy for i,b in enumerate(raw.get("parsing_res_list",[]))
              if b["block_label"]=="table" and i not in fragments and b is not block
              and b["block_bbox"][3]/sy<rect.y0]
    top=max(0,rect.y0-48,max(previous,default=0)+2)
    items,method=source_items(page,raw,rect,sx,sy,top)
    rows,height=cluster_rows(items)
    diagnostic={"method":method,"accepted":False}
    def reject(reason):
        diagnostic["reason"]=reason
        return None,diagnostic
    if not rows:
        return reject("no_source_tokens")
    numeric=[]
    for ri,row in enumerate(rows):
        if row["cy"]<rect.y0-height*.3:
            continue
        for item in row["items"]:
            if VALUE.fullmatch(item["text"]) and item["box"][0]>rect.x0+.25*rect.width:
                numeric.append((item,ri))
    epsilon=max(height*.85,rect.width*.008)
    clusters=[]
    for item,ri in sorted(numeric,key=lambda p:p[0]["box"][2]):
        x=item["box"][2]
        cluster=next((c for c in clusters if abs(median(t[0]["box"][2] for t in c)-x)<=epsilon),None)
        if cluster is None:
            clusters.append([(item,ri)])
        else:
            cluster.append((item,ri))
    clusters=[c for c in clusters if len({ri for _,ri in c})>=2]
    anchors=[median(item["box"][2] for item,_ in c) for c in clusters]
    if not 2<=len(anchors)<=8 or any(b-a<epsilon*2 for a,b in zip(anchors,anchors[1:])):
        return reject("numeric_column_axes_ambiguous")
    boundary=min(item["box"][0] for item,_ in clusters[0])-height*.5
    n=len(anchors)
    def column(item):
        return min(range(n),key=lambda i:abs(anchors[i]-item["box"][2]))
    bodies=[]
    for ri,row in enumerate(rows):
        values=[t for t in row["items"] if VALUE.fullmatch(t["text"]) and t["box"][0]>=boundary
                and abs(t["box"][2]-anchors[column(t)])<=epsilon*1.6]
        labels=[t for t in row["items"] if (t["box"][0]+t["box"][2])/2<boundary and t["text"] not in MARKERS]
        if len({column(t) for t in values})>=2 and any(re.search(r"[A-Za-z]{2,}",t["text"]) for t in labels):
            bodies.append(ri)
    if len(bodies)<2:
        return reject("row_labels_or_repeated_data_rows_missing")
    first,last=min(bodies),max(bodies)
    # All real numeric rows up to the detected lower boundary must be retained.
    last=max([last]+[ri for ri,row in enumerate(rows) if first<=ri and row["cy"]<=rect.y1+height*.2
                    and any(VALUE.fullmatch(t["text"]) for t in row["items"])])
    columns=[""]*n;label_header="";context=[];header_extension=[]
    for row in rows[:first]:
        joined=" ".join(t["text"] for t in row["items"])
        if re.search(r"\b(?:thousands|millions|billions)\b",joined,re.I) or re.fullmatch(r"Year Ended",joined,re.I):
            context.append(joined)
            if row["cy"]<rect.y0:header_extension.append(joined)
            continue
        # A complete visible header row can be ordered without splitting OCR boxes.
        header_parts=[]
        for item in row["items"]:
            dates=DATE.findall(item["text"])
            if dates:
                header_parts.extend(dates)
            elif re.fullmatch(r"Change|\d{4}",item["text"],re.I):
                header_parts.append(item["text"])
        if len(header_parts)==n:
            if row["cy"]<rect.y0:header_extension.append(joined)
            for ci,text in enumerate(header_parts):
                columns[ci]=(columns[ci]+" "+text).strip()
            continue
        markers=[t for t in row["items"] if t["text"] in MARKERS]
        if markers and len(markers)==len(row["items"]):
            for t in markers:
                # Markers are centered above a data column, not right-aligned.
                cx=(t["box"][0]+t["box"][2])/2
                ci=min(range(n),key=lambda i:abs((anchors[i]-(height*2))-cx))
                columns[ci]=(columns[ci]+" "+t["text"]).strip()
            continue
        if joined.strip() and (row["cy"]>=rect.y0 or not re.search(r"\d",joined)):
            context.append(joined)
    if any(not name.strip() for name in columns):
        return reject("column_headers_missing_or_ambiguous")
    result=[[{"text":label_header,"header":True}]+[{"text":c,"header":True} for c in columns]]
    data_rows=0
    for row in rows[first:last+1]:
        cells=[""]*(n+1);values=[];left=[];markers=[];other=[]
        for item in row["items"]:
            text=item["text"]
            if text in MARKERS:
                markers.append(item)
            elif (item["box"][0]+item["box"][2])/2<boundary:
                left.append(item)
            elif VALUE.fullmatch(text) or text in DASHES:
                ci=column(item)
                if abs(item["box"][2]-anchors[ci])>epsilon*1.6:
                    return reject("value_does_not_align_with_column")
                if cells[ci+1]:
                    return reject("multiple_values_at_same_row_and_column")
                cells[ci+1]=text;values.append((item,ci))
            else:
                other.append(item)
        if not values:
            # Preserve section labels as their own row; do not attach them to a
            # preceding financial metric or manufacture numeric cells.
            joined=" ".join(t["text"] for t in row["items"])
            if joined.strip():
                result.append([joined]+[""]*n)
            continue
        if other:
            return reject("unresolved_text_inside_numeric_columns")
        cells[0]=" ".join(t["text"] for t in sorted(left,key=lambda t:t["box"][0]))
        if not re.search(r"[A-Za-z]{2,}",cells[0]):
            return reject("data_row_label_missing")
        for marker in markers:
            # A separate currency symbol belongs to the next visible value;
            # a percent marker belongs to the preceding visible value.
            if marker["text"]=="%":
                candidates=[v for v in values if v[0]["box"][0]<marker["box"][0]]
                chosen=max(candidates,key=lambda v:v[0]["box"][0],default=None)
                if chosen:cells[chosen[1]+1]+=" %"
            else:
                candidates=[v for v in values if v[0]["box"][2]>marker["box"][2]]
                chosen=min(candidates,key=lambda v:v[0]["box"][0],default=None)
                if chosen:cells[chosen[1]+1]=marker["text"]+" "+cells[chosen[1]+1]
            if chosen is None:
                return reject("currency_or_percent_marker_unassigned")
        result.append(cells);data_rows+=1
    normalized=normalize_rows(result)
    context_text="\n".join(context)
    expanded_reference=reference+"\n"+"\n".join(header_extension)
    flags,metrics=table_checks(normalized,expanded_reference,reliable,context_text)
    diagnostic.update(column_right_edges=[round(x,2) for x in anchors],data_rows=data_rows,
                      flags=flags,**metrics)
    if flags:
        return reject("reconstructed_grid_failed_quality_checks")
    diagnostic["accepted"]=True
    return {"rows":normalized,"html":table_html(normalized),"context":context,
            "text":context_text+"\n"+table_text(normalized),"reference":expanded_reference,
            "header_reference_extension":header_extension},diagnostic
