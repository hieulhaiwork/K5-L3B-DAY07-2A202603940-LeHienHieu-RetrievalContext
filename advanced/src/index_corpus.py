"""Chunk the selected advanced corpus, embed it, and publish an isolated DB snapshot.

  python advanced/src/index_corpus.py --prepare-only
  python advanced/src/index_corpus.py
Only manifest.index_selection.record_ids are eligible. Excluded evidence is not indexed.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

from dotenv import dotenv_values
import tiktoken
from embedding_api import OpenAIEmbeddings,MODEL,DIMENSIONS
from index_store import AdvancedStore

ROOT=Path(__file__).resolve().parents[1]
PROFILE="same-page-layout-text700-overlap80-atomic-table-visual-v1"
ENCODING=tiktoken.get_encoding("cl100k_base")

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
def token_count(text):
    return len(ENCODING.encode(text,disallowed_special=()))
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)
def split_text(text,maximum=700,overlap=80):
    start=0
    while start<len(text):
        if token_count(text[start:])<=maximum:
            yield text[start:],start,len(text);break
        low,high=start+1,len(text)
        while low<high:
            middle=(low+high+1)//2
            if token_count(text[start:middle])<=maximum:low=middle
            else:high=middle-1
        end=low
        boundary=text.rfind(" ",start+int((end-start)*.8),end)
        if boundary>start:end=boundary
        yield text[start:end],start,end
        # Find a character-safe tail for overlap; never split a Unicode character.
        low,high=start+1,end
        while low<high:
            middle=(low+high)//2
            if token_count(text[middle:end])<=overlap:high=middle
            else:low=middle+1
        start=max(start+1,low)

def build_chunks(manifest,artifact_base):
    selected=manifest.get("index_selection",{}).get("record_ids")
    if not selected or len(selected)!=len(set(selected)):
        raise ValueError("Missing/duplicate index whitelist")
    all_records=manifest["records"]
    if len({r["id"] for r in all_records})!=len(all_records):
        raise ValueError("Duplicate source record IDs")
    lookup={r["id"]:r for r in all_records}
    for identifier in selected:
        r=lookup[identifier]
        if not r.get("embedding_ready") or r.get("status")!="extracted" or r.get("quality_flags"):
            raise ValueError("Whitelist contains an unsafe source record: "+identifier)
    retained={identifier:lookup[identifier] for identifier in selected}
    by_page=defaultdict(list)
    for r in retained.values():by_page[r["page"]].append(r)
    chunks=[]
    def emit(parts,kind,page):
        members=list(dict.fromkeys(part["id"] for part in parts))
        extra=[]
        body="\n\n".join(part["text"].strip() for part in parts if part["text"].strip())
        if token_count(body)<40:
            for identifier in members:
                for context_id in retained[identifier].get("context_record_ids",[]):
                    if context_id not in retained:raise ValueError("Context points outside whitelist")
                    if context_id not in members and context_id not in extra:extra.append(context_id)
            for context_id in extra[:1]:
                context=retained[context_id]["embedding_text"]
                snippet=next(split_text(context,maximum=300,overlap=0))[0]
                body="Context from same PDF page:\n"+snippet+"\n\n"+body
            extra=extra[:1]
        text=f"NVIDIA 2026 Annual Report | PDF page {page} | {kind}\n"+body
        if not body.strip() or token_count(text)>8191:
            raise ValueError("Empty or oversized chunk; no silent truncation")
        ids=members+extra
        evidence=[{"record_id":identifier,"page":retained[identifier]["page"],
                   "bbox":retained[identifier]["bbox"],
                   "crop":(artifact_base/retained[identifier]["crop"]).as_posix(),
                   "content_type":retained[identifier]["content_type"],
                   "extraction_method":retained[identifier].get("extraction_method"),
                   "verification":retained[identifier].get("verification")} for identifier in ids]
        chunk_id="adv-"+digest([manifest["document"]["id"],parts,text,evidence])[:24]
        chunks.append({"chunk_id":chunk_id,"page":page,"content_type":kind,"text":text,
            "tokens":token_count(text),"metadata":{
                "document_id":manifest["document"]["id"],"source_url":manifest["document"]["source_url"],
                "source_record_ids":ids,"source_records":evidence,
                "source_parts":[{"record_id":p["id"],"char_start":p["start"],"char_end":p["end"]} for p in parts],
                "chunking_profile":PROFILE}})
    for page,records in sorted(by_page.items()):
        flows=[]
        def flush(flow):
            emit(flow["parts"],"text",page)
        for r in sorted(records,key=lambda r:(r["bbox"][1],r["bbox"][0],r["id"])):
            if r["content_type"]!="text":
                for flow in flows:flush(flow)
                flows=[]
                emit([{"id":r["id"],"text":r["embedding_text"],"start":0,"end":len(r["embedding_text"])}],r["content_type"],page)
                continue
            for text,start,end in split_text(r["embedding_text"]):
                box=r["bbox"];compatible=[]
                for flow in flows:
                    previous=flow["box"]
                    horizontal=max(0,min(box[2],previous[2])-max(box[0],previous[0]))
                    if horizontal/max(1,max(box[2]-box[0],previous[2]-previous[0]))>=.5 and box[1]-previous[3]<=100:
                        compatible.append(flow)
                flow=compatible[-1] if compatible else None
                part={"id":r["id"],"text":text,"start":start,"end":end}
                if flow and token_count("\n\n".join([p["text"] for p in flow["parts"]]+[text]))>700:
                    flush(flow);flows.remove(flow);flow=None
                if flow is None:
                    flows.append({"parts":[part],"box":box})
                else:
                    flow["parts"].append(part);flow["box"]=box
        for flow in flows:flush(flow)
    covered={identifier for c in chunks for identifier in c["metadata"]["source_record_ids"]}
    if covered!=set(selected):
        raise ValueError("Chunk coverage differs from selected source records")
    if len({c["chunk_id"] for c in chunks})!=len(chunks):
        raise ValueError("Duplicate chunk IDs")
    return chunks

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=ROOT/"artifacts/extraction/manifest.json")
    parser.add_argument("--output",type=Path,default=ROOT/"artifacts/index")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--token-budget",type=int,default=250000,help="Maximum total embedding input tokens")
    args=parser.parse_args()
    started=time.perf_counter()
    manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    env=dotenv_values(ROOT.parent/".env")
    if env.get("ADVANCED_EMBEDDING_MODEL",MODEL)!=MODEL:
        raise ValueError("Set ADVANCED_EMBEDDING_MODEL to the implemented text-embedding-3-small profile")
    try:artifact_base=args.manifest.resolve().parent.relative_to(ROOT.parent)
    except ValueError:raise ValueError("Source manifest must be inside this repository")
    chunks=build_chunks(manifest,artifact_base)
    total_tokens=sum(c["tokens"] for c in chunks)
    if total_tokens>args.token_budget:
        raise ValueError(f"Embedding token budget exceeded: {total_tokens}>{args.token_budget}")
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    snapshot=digest([MODEL,DIMENSIONS,PROFILE,[c["chunk_id"] for c in chunks]])
    collection="nvidia2026_"+snapshot[:16]
    profile={"chunking":PROFILE,"selection_hash":digest(sorted(manifest["index_selection"]["record_ids"])),
             "snapshot_hash":snapshot,"preprocessing":"raw-text-l2-v1"}
    summary={"stage":"prepared","collection":collection,"model":MODEL,"dimensions":DIMENSIONS,
             "source_records":len(manifest["index_selection"]["record_ids"]),"chunks":len(chunks),
             "chunks_by_type":dict(Counter(c["content_type"] for c in chunks)),
             "pages_represented":len({c["page"] for c in chunks}),"embedding_input_tokens":total_tokens,
             "estimated_embedding_cost_usd":round(total_tokens*.02/1000000,6),
             "profile":profile,"excluded_uncertain_records":manifest["index_selection"]["excluded_uncertain_records"]}
    with (output/"chunks.jsonl").open("w",encoding="utf-8") as stream:
        for chunk in chunks:stream.write(json.dumps(chunk,ensure_ascii=False)+"\n")
    write_json(output/"index_stats.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if args.prepare_only:return
    store=AdvancedStore(env.get("ADVANCED_DATABASE_URL"),env.get("DATABASE_URL"))
    # Verify DB connectivity and pgvector before buying embeddings.
    store.create_schema()
    embedder=OpenAIEmbeddings(env.get("OPENAI_API_KEY"),output/"cache/embeddings")
    try:
        vectors=embedder.embed([c["text"] for c in chunks])
        count=store.save(collection,manifest["document"]["id"],chunks,vectors,profile)
        probe=max(range(len(chunks)),key=lambda i:chunks[i]["tokens"])
        hits=store.search(collection,vectors[probe],top_k=5)
        if not any(hit["chunk_id"]==chunks[probe]["chunk_id"] and hit["score"]>.999 for hit in hits):
            raise ValueError("Database self-retrieval validation failed")
        summary.update(stage="indexed",database_rows=count,cache_hits=embedder.cache_hits,
            new_embedding_calls=sum("inputs" in e for e in embedder.events),
            actual_prompt_tokens=sum(e.get("usage",{}).get("prompt_tokens",0) for e in embedder.events),
            self_retrieval_verified=True,elapsed_seconds=round(time.perf_counter()-started,2))
    except Exception:
        summary.update(stage="failed",cache_hits=embedder.cache_hits,events=embedder.events)
        write_json(output/"index_stats.json",summary)
        raise
    summary["events"]=embedder.events
    write_json(output/"index_stats.json",summary)
    print(json.dumps({k:v for k,v in summary.items() if k!="events"},ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":
    main()
