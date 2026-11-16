"""Separate advanced pgvector collections; basic database is never modified."""
import json
from urllib.parse import urlsplit
import psycopg
from psycopg.types.json import Jsonb
from embedding_api import MODEL,DIMENSIONS,checked_vector

class AdvancedStore:
    def __init__(self,url,basic_url=None):
        if not url:
            raise ValueError("ADVANCED_DATABASE_URL missing")
        if not urlsplit(url).path.strip("/") or urlsplit(url).path==urlsplit(basic_url or "").path:
            raise ValueError("Advanced and basic must use separate databases")
        self._url=url
    def connect(self):
        return psycopg.connect(self._url,connect_timeout=5)
    def create_schema(self):
        with self.connect() as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            c.execute("""CREATE TABLE IF NOT EXISTS advanced_collections(
                collection text PRIMARY KEY,document_id text NOT NULL,model text NOT NULL,
                dimensions integer NOT NULL,profile jsonb NOT NULL,chunk_count integer NOT NULL)""")
            c.execute("""CREATE TABLE IF NOT EXISTS advanced_chunks(
                collection text NOT NULL REFERENCES advanced_collections(collection),
                chunk_id text NOT NULL,text_content text NOT NULL,page integer NOT NULL,
                content_type text NOT NULL,metadata jsonb NOT NULL,
                dimensions integer NOT NULL,embedding vector NOT NULL,
                search_terms tsvector GENERATED ALWAYS AS (to_tsvector('english',text_content)) STORED,
                PRIMARY KEY(collection,chunk_id),CHECK(vector_dims(embedding)=dimensions))""")
            c.execute("CREATE INDEX IF NOT EXISTS advanced_chunks_fts ON advanced_chunks USING gin(search_terms)")
    def save(self,collection,document_id,chunks,vectors,profile):
        if not chunks or len(chunks)!=len(vectors) or len({c['chunk_id'] for c in chunks})!=len(chunks):
            raise ValueError("Invalid complete chunk/vector snapshot")
        rows=[(collection,chunk["chunk_id"],chunk["text"],chunk["page"],chunk["content_type"],
               Jsonb(chunk["metadata"]),DIMENSIONS,json.dumps(checked_vector(vector))) for chunk,vector in zip(chunks,vectors)]
        with self.connect() as c:
            c.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",(collection,))
            c.execute("INSERT INTO advanced_collections VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (collection,document_id,MODEL,DIMENSIONS,Jsonb(profile),len(chunks)))
            existing=c.execute("SELECT model,dimensions,profile,chunk_count FROM advanced_collections WHERE collection=%s",(collection,)).fetchone()
            if existing!=(MODEL,DIMENSIONS,profile,len(chunks)):
                raise ValueError("Collection profile mismatch; refusing overwrite")
            with c.cursor() as cursor:
                cursor.executemany("""INSERT INTO advanced_chunks
                    (collection,chunk_id,text_content,page,content_type,metadata,dimensions,embedding)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)
                    ON CONFLICT(collection,chunk_id) DO UPDATE SET embedding=EXCLUDED.embedding""",rows)
            count=c.execute("SELECT count(*) FROM advanced_chunks WHERE collection=%s",(collection,)).fetchone()[0]
            if count!=len(chunks):
                raise ValueError("Database snapshot count mismatch")
        return count
    def search(self,collection,vector,top_k=5):
        if not 1<=top_k<=50:
            raise ValueError("Invalid top_k")
        value=json.dumps(checked_vector(vector))
        with self.connect() as c:
            rows=c.execute("""SELECT chunk_id,text_content,page,content_type,metadata,
                1-(embedding <=> %s::vector) AS score FROM advanced_chunks
                WHERE collection=%s ORDER BY embedding <=> %s::vector,chunk_id LIMIT %s""",
                (value,collection,value,top_k)).fetchall()
        return [dict(zip(["chunk_id","text","page","content_type","metadata","score"],row)) for row in rows]
