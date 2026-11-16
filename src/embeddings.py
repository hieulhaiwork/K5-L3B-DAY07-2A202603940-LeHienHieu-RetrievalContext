"""Real OpenRouter embeddings with validated vectors and resumable disk cache."""
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol

MODEL = "nvidia/nemotron-3-embed-1b:free"
PREPROCESSING = "nemotron-query-passage-prefix-l2-v1"


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, question: str) -> list[float]: ...


def checked_vector(vector, dimensions, normalize=False):
    if len(vector) != dimensions or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in vector):
        raise ValueError("Embedding has wrong dimensions or non-finite values")
    norm = math.sqrt(sum(v * v for v in vector))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding has zero/invalid norm")
    return [v / norm for v in vector] if normalize else list(vector)


class OpenRouterEmbedder:
    dimensions = 2048

    def __init__(self, api_key: str, cache_dir: Path, model_name: str = MODEL,
                 batch_size: int = 32, min_interval: float = 3.2):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY missing")
        if model_name != MODEL:
            raise ValueError("Adapter supports the verified free Nemotron model only")
        if not 1 <= batch_size <= 32 or min_interval < 0:
            raise ValueError("Invalid batch size or request interval")
        self.model_name, self._key = model_name, api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size, self.min_interval, self._last_call = batch_size, min_interval, 0.0
        self.events, self.cache_hits = [], 0

    def _cache_key(self, text, kind):
        return hashlib.sha256(json.dumps([self.model_name, self.dimensions, PREPROCESSING, kind, text], ensure_ascii=False).encode()).hexdigest()

    def _request(self, inputs):
        payload = json.dumps({"model": self.model_name, "input": inputs, "encoding_format": "float"}).encode()
        for attempt in range(3):
            time.sleep(max(0, self.min_interval - (time.monotonic() - self._last_call)))
            request = urllib.request.Request("https://openrouter.ai/api/v1/embeddings", data=payload,
                headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"})
            self._last_call = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    result = json.load(response)
                rows = result.get("data", [])
                if len(rows) != len(inputs) or sorted(row.get("index", -1) for row in rows) != list(range(len(inputs))):
                    raise ValueError("Embedding response count/index mismatch")
                vectors = [checked_vector(row["embedding"], self.dimensions, normalize=True)
                           for row in sorted(rows, key=lambda row: row["index"])]
                usage = result.get("usage", {})
                event = {"event": "embedding_batch", "inputs": len(inputs), "attempt": attempt + 1,
                         "model_requested": self.model_name, "model_returned": result.get("model"),
                         "prompt_tokens": usage.get("prompt_tokens"), "cost_usd": usage.get("cost"),
                         "request_id": result.get("id")}
                self.events.append(event)
                if usage.get("cost") is not None and usage["cost"] > 0:
                    raise ValueError("Expected zero-cost free endpoint; stopping after nonzero reported charge")
                return vectors, event
            except urllib.error.HTTPError as error:
                self.events.append({"event": "provider_error", "http_status": error.code, "attempt": attempt + 1})
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise RuntimeError(f"OpenRouter HTTP {error.code}; check account/quota/provider. No fallback used.") from None
                time.sleep(2 ** (attempt + 1))
            except (urllib.error.URLError, TimeoutError) as error:
                self.events.append({"event": "network_error", "error_type": type(error).__name__, "attempt": attempt + 1})
                if isinstance(getattr(error, "reason", None), PermissionError) or attempt == 2:
                    raise RuntimeError("OpenRouter network connection failed; no fallback used.") from None
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("Embedding retries exhausted")

    def _embed(self, texts, kind):
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding input must be nonempty text")
        keys = [self._cache_key(text, kind) for text in texts]
        vectors, missing = {}, {}
        for key, text in zip(keys, texts):
            if key in vectors or key in missing:
                continue
            path = self.cache_dir / f"{key}.json"
            if path.exists():
                cached = json.loads(path.read_text(encoding="utf-8"))
                if cached.get("cache_key") != key or cached.get("model") != self.model_name or cached.get("preprocessing") != PREPROCESSING:
                    raise ValueError("Cache fingerprint mismatch")
                vectors[key] = checked_vector(cached["vector"], self.dimensions)
                self.cache_hits += 1
            else:
                missing[key] = text
        pending = list(missing.items())
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start:start + self.batch_size]
            prefix = "query: " if kind == "query" else "passage: "
            returned, event = self._request([prefix + text for _, text in batch])
            for (key, _), vector in zip(batch, returned):
                vectors[key] = vector
                record = {"cache_key": key, "model": self.model_name, "dimensions": self.dimensions,
                          "preprocessing": PREPROCESSING, "kind": kind, "vector": vector,
                          "response_model": event["model_returned"]}
                path = self.cache_dir / f"{key}.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record), encoding="utf-8")
                temporary.replace(path)
            print(f"Embedded {min(start + self.batch_size, len(pending))}/{len(pending)} uncached {kind} texts", flush=True)
        return [vectors[key] for key in keys]

    def embed_documents(self, texts):
        return self._embed(texts, "document")

    def embed_query(self, question):
        return self._embed([question], "query")[0]

    def embed_queries(self, questions):
        return self._embed(questions, "query")


MyEmbedder = OpenRouterEmbedder
