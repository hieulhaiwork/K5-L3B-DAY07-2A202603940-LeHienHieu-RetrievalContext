"""Reproduce the B index from shared chunks using real document embeddings."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import psycopg

from .dense import ROOT, from_project
from .models import Chunk


def load_snapshot(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Chunks must be a nonempty JSON array")
    chunks, seen = [], set()
    for row in rows:
        for key in ("id", "source_id", "source", "text"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError("Invalid chunk field: " + key)
        if row["id"] in seen or type(row.get("page")) is not int or row["page"] < 1:
            raise ValueError("Duplicate chunk ID or invalid page")
        if not isinstance(row.get("metadata"), dict):
            raise ValueError("Missing chunk metadata")
        seen.add(row["id"])
        chunks.append(Chunk(chunk_id=row["id"], document_id=row["source_id"],
                            source=row["source"], page=row["page"], text=row["text"], metadata=row["metadata"]))
    return chunks


def rebuild_index(retriever, chunks_path):
    chunks = load_snapshot(chunks_path)
    retriever.store.create_schema()
    if retriever.store.count("B"):
        # Preserve existing data; mismatches fail instead of replacing an unknown index.
        return {"action": "verified_existing", **retriever.verify_index(chunks_path)}
    vectors = retriever.embedder.embed_documents([c.text for c in chunks])
    retriever.store.upsert("B", chunks, vectors)
    return {"action": "created", **retriever.verify_index(chunks_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/chunks.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="lab7-index-") as cache:
        retriever = from_project(cache_dir=Path(cache))
        try:
            result = (retriever.verify_index(args.chunks) if args.check_only
                      else rebuild_index(retriever, args.chunks))
        except psycopg.Error as error:
            raise SystemExit(f"Database operation failed ({type(error).__name__}); check Docker and DATABASE_URL.") from None
        result["embedding_events"] = retriever.embedder.events
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
