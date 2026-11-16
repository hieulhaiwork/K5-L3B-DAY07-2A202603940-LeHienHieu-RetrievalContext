"""Dense + existing starter TF-IDF retrieval fused by reciprocal ranks."""
from starter.baseline import TfidfIndex


class HybridRetriever:
    def __init__(self, dense, chunks, *, candidate_k=20, rrf_k=60, lexical="tfidf"):
        if type(candidate_k) is not int or not 1 <= candidate_k <= 100:
            raise ValueError("candidate_k must be between 1 and 100")
        if type(rrf_k) is not int or rrf_k < 1 or lexical != "tfidf":
            raise ValueError("Invalid RRF/lexical configuration")
        self.dense, self.embedder = dense, dense.embedder
        self.index = TfidfIndex(chunks)
        self.candidate_k, self.rrf_k = candidate_k, rrf_k
        self.last_search = None

    def verify_index(self, chunks_path):
        return self.dense.verify_index(chunks_path)

    @staticmethod
    def matches(chunk, filters):
        for key, value in (filters or {}).items():
            if key == "application_scope":
                if value not in chunk["metadata"].get(key, []):
                    return False
            elif chunk["metadata"].get(key) != value:
                return False
        return True

    def search(self, query, *, filters=None, top_k=3):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be nonempty text")
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("Invalid top_k")
        self.last_search = None
        pool = max(self.candidate_k, top_k)
        dense = self.dense.search(query, filters=filters, top_k=pool)
        # Score all local chunks first so filtering happens before the candidate cutoff.
        lexical = [c for c in self.index.search(query, top_k=len(self.index.chunks))
                   if self.matches(c, filters)][:pool]
        fused = {}
        branches = {}
        for name, ranking in (("dense", dense), ("lexical", lexical)):
            branches[name] = []
            seen = set()
            for rank, chunk in enumerate(ranking, 1):
                identifier = chunk["id"]
                if identifier in seen:
                    raise ValueError("Duplicate chunk in component ranking")
                seen.add(identifier)
                item = fused.setdefault(identifier, {**chunk, "score": 0.0, "strategy": "hybrid", "component_ranks": {},
                                                      "component_scores": {}})
                item["score"] += 1 / (self.rrf_k + rank)
                item["component_ranks"][name] = rank
                item["component_scores"][name] = chunk["score"]
                branches[name].append({"id": identifier, "rank": rank, "score": chunk["score"],
                                       "source_id": chunk["source_id"], "page": chunk["page"]})
        self.last_search = {"method": "hybrid", "lexical": "tfidf", "fusion": "rrf",
                            "rrf_k": self.rrf_k, "candidate_k": pool,
                            "component_rankings": branches}
        return sorted(fused.values(), key=lambda c: (-c["score"], c["id"]))[:top_k]
