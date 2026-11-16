"""OpenAI embeddings with validated vectors and a model-specific resumable cache."""
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.error
import urllib.request

MODEL = "text-embedding-3-small"
DIMENSIONS = 1536

def checked_vector(vector):
    if not isinstance(vector,list) or len(vector)!=DIMENSIONS or any(
        type(x) not in {int,float} or not math.isfinite(x) for x in vector):
        raise ValueError("Invalid embedding dimensions or values")
    norm=math.sqrt(sum(x*x for x in vector))
    if norm<=0 or not math.isfinite(norm):
        raise ValueError("Zero/non-finite embedding")
    return [x/norm for x in vector]

class OpenAIEmbeddings:
    model_name=MODEL
    dimensions=DIMENSIONS
    def __init__(self,key,cache_dir):
        if not key:
            raise ValueError("OPENAI_API_KEY missing")
        self._key=key
        self.cache_dir=Path(cache_dir)
        self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.events=[]
        self.cache_hits=0
    def _key_for(self,text):
        return hashlib.sha256(json.dumps([MODEL,DIMENSIONS,"raw-text-l2-v1",text],ensure_ascii=False).encode()).hexdigest()
    def embed(self,texts):
        if any(not isinstance(t,str) or not t.strip() for t in texts):
            raise ValueError("Empty embedding input")
        keys=[self._key_for(text) for text in texts]
        vectors={};pending={}
        for key,text in zip(keys,texts):
            if key in vectors or key in pending:
                continue
            path=self.cache_dir/(key+".json")
            if path.is_file():
                cached=json.loads(path.read_text(encoding="utf-8"))
                if cached.get("model")!=MODEL or cached.get("key")!=key:
                    raise ValueError("Embedding cache profile mismatch")
                vectors[key]=checked_vector(cached["vector"]);self.cache_hits+=1
            else:
                pending[key]=text
        items=list(pending.items())
        for start in range(0,len(items),32):
            batch=items[start:start+32]
            payload={"model":MODEL,"input":[t for _,t in batch],"dimensions":DIMENSIONS,"encoding_format":"float"}
            for attempt in range(1,4):
                request=urllib.request.Request("https://api.openai.com/v1/embeddings",
                    data=json.dumps(payload).encode(),
                    headers={"Content-Type":"application/json","Authorization":"Bearer "+self._key})
                before=time.perf_counter()
                try:
                    with urllib.request.urlopen(request,timeout=90) as response:
                        reply=json.load(response)
                    data=reply.get("data",[])
                    if len(data)!=len(batch) or sorted(d.get("index",-1) for d in data)!=list(range(len(batch))):
                        raise ValueError("Embedding result count/index mismatch")
                    returned=[checked_vector(d["embedding"]) for d in sorted(data,key=lambda d:d["index"])]
                    if reply.get("model")!=MODEL:
                        raise ValueError("Unexpected embedding model")
                    event={"inputs":len(batch),"model":reply["model"],"usage":reply.get("usage",{}),
                           "elapsed_ms":round((time.perf_counter()-before)*1000,2),"attempt":attempt}
                    self.events.append(event)
                    for (key,_),vector in zip(batch,returned):
                        vectors[key]=vector
                        path=self.cache_dir/(key+".json");temporary=path.with_suffix(".tmp")
                        temporary.write_text(json.dumps({"key":key,"model":MODEL,"dimensions":DIMENSIONS,"vector":vector}),encoding="utf-8")
                        temporary.replace(path)
                    break
                except urllib.error.HTTPError as error:
                    code=None
                    try:code=json.load(error).get("error",{}).get("code")
                    except (ValueError,AttributeError):pass
                    self.events.append({"http_status":error.code,"attempt":attempt,"error_code":code if code in {
                        "insufficient_quota","invalid_api_key","rate_limit_exceeded"} else None})
                    if attempt==3 or error.code not in {429,500,502,503,504} or code=="insufficient_quota":
                        raise RuntimeError(f"OpenAI embeddings HTTP {error.code}; check quota/access. Cached progress preserved.") from None
                except (urllib.error.URLError,TimeoutError):
                    if attempt==3:
                        raise RuntimeError("Embedding connection failed; cached progress preserved.") from None
                time.sleep(2**attempt)
            print(f"Embedded {min(start+32,len(items))}/{len(items)} new chunks",flush=True)
        return [vectors[key] for key in keys]
