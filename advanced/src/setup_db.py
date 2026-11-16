"""Create the separate advanced database (with pgvector) on the existing Postgres server.

Reads DATABASE_URL from the repo .env only to reach the server; the Lab 6 database is not touched.
Writes ADVANCED_DATABASE_URL to .env if missing. Passwords are never printed.
"""
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from dotenv import dotenv_values
from psycopg import sql

REPO = Path(__file__).resolve().parents[2]
DB_NAME = "day7_advanced"


def main():
    env_path = REPO / ".env"
    env = dotenv_values(env_path)
    base = env.get("DATABASE_URL")
    if not base:
        raise SystemExit("DATABASE_URL missing in .env")
    parts = urlsplit(base)
    advanced_url = urlunsplit(parts._replace(path="/" + DB_NAME))
    with psycopg.connect(base, autocommit=True, connect_timeout=5) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB_NAME,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
        print("database:", DB_NAME, "existed" if exists else "created")
    with psycopg.connect(advanced_url, autocommit=True, connect_timeout=5) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        version = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
        print("pgvector:", version)
    if not env.get("ADVANCED_DATABASE_URL"):
        text = env_path.read_text(encoding="utf-8")
        sep = "" if text.endswith("\n") else "\n"
        env_path.write_text(text + f"{sep}ADVANCED_DATABASE_URL={advanced_url}\n", encoding="utf-8")
        print("ADVANCED_DATABASE_URL added to .env")
    else:
        print("ADVANCED_DATABASE_URL already in .env")


if __name__ == "__main__":
    main()
