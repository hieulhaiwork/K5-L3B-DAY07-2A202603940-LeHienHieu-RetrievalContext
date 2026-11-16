"""Dense retrieval over the existing Lab 6 pgvector B collection."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import psycopg
from dotenv import dotenv_values

from .embeddings import MODEL, PREPROCESSING, OpenRouterEmbedder
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]


class DenseRetriever:
    def __init__(self, embedder, store):
        if (embedder.model_name, embedder.dimensions) != (store.model_name, store.dimensions):
            raise ValueError("Query embedder and vector store profiles differ")
        self.embedder, self.store = embedder, store

    def verify_index(self, chunks_path: Path) -> dict:
        """Read-only comparison of the DB profile and full B snapshot with local chunks."""
        chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("Local chunks must be a nonempty JSON array")
        local = {c["id"]: c for c in chunks}
        if len(local) != len(chunks):
            raise ValueError("Duplicate local chunk IDs")
        with self.store._connect() as connection:
            connection.execute("SET TRANSACTION READ ONLY")
            profile = connection.execute(
                "SELECT model_name,dimensions,preprocessing FROM lab6_embedding_collections WHERE collection=%s",
                (self.store.collection,),
            ).fetchone()
            if profile != (self.embedder.model_name, self.embedder.dimensions, PREPROCESSING):
                raise ValueError("Database embedding profile differs from query preprocessing/model/dimensions")
            rows = connection.execute(
                "SELECT chunk_id,document_id,source,page,text_content,metadata,dimensions,vector_dims(embedding) "
                "FROM lab6_vector_chunks WHERE collection=%s AND strategy='B'",
                (self.store.collection,),
            ).fetchall()
        if {row[0] for row in rows} != set(local):
            raise ValueError("Database B chunk IDs differ from local chunks")
        for row in rows:
            c = local[row[0]]
            if tuple(row[1:6]) != (c["source_id"], c["source"], c["page"], c["text"], c["metadata"]):
                raise ValueError("Database B content/source/metadata differ from local chunks")
            if row[6] != self.embedder.dimensions or row[7] != self.embedder.dimensions:
                raise ValueError("Database contains incompatible vector dimensions")
        return {"collection": self.store.collection, "strategy": "B", "chunks": len(rows),
                "model": self.embedder.model_name, "dimensions": self.embedder.dimensions,
                "preprocessing": PREPROCESSING, "local_snapshot_matches": True}

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        """Match the Lab 7 Retriever interface; filters use the Lab 6 C alias of B."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be nonempty text")
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer between 1 and 100")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("Filters must be an object or None")
        for key, value in (filters or {}).items():
            if key not in {"application_scope", "section"} or not isinstance(value, str) or not value.strip():
                raise ValueError("Unsupported or empty metadata filter")
        vector = self.embedder.embed_query(query)
        rows = self.store.search("C" if filters else "B", vector, top_k=top_k, filters=filters)
        return [{"id": row["chunk_id"], "source_id": row["document_id"],
                 "source": row["source"], "page": row["page"], "text": row["text"],
                 "metadata": row["metadata"], "score": float(row["score"])} for row in rows]


def from_project(*, cache_dir: Path, root: Path = ROOT) -> DenseRetriever:
    """Load explicit project configuration without changing process environment."""
    env = dotenv_values(root / ".env")
    config = json.loads((root / "src/configs/embedding.json").read_text(encoding="utf-8"))
    expected = {"model": MODEL, "dimensions": 2048, "document_prefix": "passage: ",
                "query_prefix": "query: ", "normalize_l2": True, "metric": "cosine"}
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError("Embedding config does not match the copied Lab 6 adapter")
    if env.get("EMBEDDING_MODEL") != config["model"]:
        raise ValueError("EMBEDDING_MODEL in .env differs from embedding.json")
    if not env.get("DATABASE_URL"):
        raise ValueError("DATABASE_URL missing from .env")
    embedder = OpenRouterEmbedder(env.get("OPENROUTER_API_KEY"), cache_dir,
                                 model_name=config["model"], batch_size=config["batch_size"],
                                 min_interval=config["min_request_interval_seconds"])
    store = PgVectorStore(env["DATABASE_URL"], config["dimensions"],
                          collection=config["collection"], model_name=config["model"])
    return DenseRetriever(embedder, store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="How do collaborators review changes in a document?")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--filters", type=json.loads, help='JSON object, e.g. {"application_scope":"word"}')
    parser.add_argument("--check-only", action="store_true", help="Check DB snapshot without calling the embedding API")
    args = parser.parse_args()
    # The smoke check leaves no embedding cache in the submission tree.
    with TemporaryDirectory(prefix="lab7-dense-") as cache:
        retriever = from_project(cache_dir=Path(cache))
        try:
            print(json.dumps(retriever.verify_index(ROOT / "data/chunks.json"), ensure_ascii=False))
            if not args.check_only:
                results = retriever.search(args.query, filters=args.filters, top_k=args.top_k)
                print(json.dumps({"query": args.query, "filters": args.filters, "results": results,
                                  "embedding_events": retriever.embedder.events}, ensure_ascii=False, indent=2))
        except psycopg.Error as error:
            raise SystemExit(f"PostgreSQL check/search failed ({type(error).__name__}); check Docker, port and DATABASE_URL.") from None


if __name__ == "__main__":
    main()
