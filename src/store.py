"""Persistent pgvector storage; exact cosine search with SQL metadata filtering."""
import json
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .embeddings import MODEL, PREPROCESSING, checked_vector
from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, *, collection="day6_nemotron3_1b_v1", model_name=MODEL):
        if not database_url or not 1 <= dimensions <= 16000:
            raise ValueError("Database URL/dimensions invalid")
        self.database_url, self.dimensions = database_url, dimensions
        self.collection, self.model_name = collection, model_name

    def _connect(self):
        return psycopg.connect(self.database_url, connect_timeout=5)

    def create_schema(self):
        with self._connect() as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute("""CREATE TABLE IF NOT EXISTS lab6_embedding_collections (
                collection text PRIMARY KEY, model_name text NOT NULL, dimensions integer NOT NULL,
                preprocessing text NOT NULL)""")
            connection.execute("""CREATE TABLE IF NOT EXISTS lab6_vector_chunks (
                collection text NOT NULL REFERENCES lab6_embedding_collections(collection),
                strategy text NOT NULL CHECK (strategy IN ('A','B')), chunk_id text NOT NULL,
                document_id text NOT NULL, source text NOT NULL, page integer NOT NULL,
                text_content text NOT NULL, metadata jsonb NOT NULL,
                dimensions integer NOT NULL, embedding vector NOT NULL,
                PRIMARY KEY (collection,strategy,chunk_id), CHECK(vector_dims(embedding)=dimensions))""")
            connection.execute("INSERT INTO lab6_embedding_collections VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                               (self.collection, self.model_name, self.dimensions, PREPROCESSING))
            row = connection.execute("SELECT model_name,dimensions,preprocessing FROM lab6_embedding_collections WHERE collection=%s",
                                     (self.collection,)).fetchone()
            if row != (self.model_name, self.dimensions, PREPROCESSING):
                raise ValueError("Collection embedding profile mismatch; use a new collection")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]):
        """Replace a complete snapshot of supplied documents in one transaction.

        Never pass a partial chunk batch: omitted old IDs of these documents are deleted.
        C is a search alias for B and owns no duplicate physical vector rows.
        """
        if strategy not in {"A", "B"} or not chunks or len(chunks) != len(vectors):
            raise ValueError("Supply a complete nonempty A/B snapshot and matching vectors")
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate IDs in snapshot")
        values = []
        for chunk, vector in zip(chunks, vectors):
            checked_vector(vector, self.dimensions)
            values.append((self.collection, strategy, chunk.chunk_id, chunk.document_id, chunk.source,
                           chunk.page, chunk.text, Jsonb(chunk.metadata), self.dimensions, json.dumps(vector)))
        with self._connect() as connection:
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (self.collection + strategy,))
            with connection.cursor() as cursor:
                cursor.executemany("""INSERT INTO lab6_vector_chunks
                    (collection,strategy,chunk_id,document_id,source,page,text_content,metadata,dimensions,embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    ON CONFLICT (collection,strategy,chunk_id) DO UPDATE SET
                    document_id=EXCLUDED.document_id, source=EXCLUDED.source, page=EXCLUDED.page,
                    text_content=EXCLUDED.text_content, metadata=EXCLUDED.metadata,
                    dimensions=EXCLUDED.dimensions, embedding=EXCLUDED.embedding""", values)
            connection.execute("""DELETE FROM lab6_vector_chunks WHERE collection=%s AND strategy=%s
                AND document_id=ANY(%s) AND NOT (chunk_id=ANY(%s))""",
                (self.collection, strategy, sorted({chunk.document_id for chunk in chunks}), ids))

    def count(self, strategy):
        physical = "B" if strategy == "C" else strategy
        with self._connect() as connection:
            return connection.execute("SELECT count(*) FROM lab6_vector_chunks WHERE collection=%s AND strategy=%s",
                                      (self.collection, physical)).fetchone()[0]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if strategy not in {"A", "B", "C"} or not 1 <= top_k <= 100:
            raise ValueError("Invalid strategy/top_k")
        checked_vector(query_vector, self.dimensions)
        if filters and strategy != "C":
            raise ValueError("Filters belong to C only; A/B are unfiltered baselines")
        clauses, arguments = ["collection=%s", "strategy=%s"], [self.collection, "B" if strategy == "C" else strategy]
        for key, value in (filters or {}).items():
            if key not in {"application_scope", "section"} or not isinstance(value, str):
                raise ValueError("Unsupported metadata filter")
            clauses.append("(metadata->'application_scope') ? %s" if key == "application_scope" else "metadata->>'section'=%s")
            arguments.append(value)
        query = """SELECT chunk_id,document_id,source,page,text_content,metadata,
            1-(embedding <=> %s::vector) AS score FROM lab6_vector_chunks WHERE """ + " AND ".join(clauses) + " ORDER BY embedding <=> %s::vector,chunk_id LIMIT %s"
        vector = json.dumps(query_vector)
        with self._connect() as connection:
            rows = connection.execute(query, [vector, *arguments, vector, top_k]).fetchall()
        return [{"chunk_id": row[0], "document_id": row[1], "source": row[2], "page": row[3],
                 "text": row[4], "metadata": row[5], "score": row[6]} for row in rows]
