"""Database connection helpers. Set DATABASE_URL to point elsewhere (e.g. the Docker service)."""
import os
from pathlib import Path

import psycopg
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "postgresql://postgres@localhost:5433/retail"


def url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_URL)


def connect() -> psycopg.Connection:
    return psycopg.connect(url())


def engine():
    return create_engine(url().replace("postgresql://", "postgresql+psycopg://", 1))


def run_sql_file(conn: psycopg.Connection, name: str) -> None:
    conn.execute((ROOT / "sql" / name).read_text(encoding="utf-8"))
