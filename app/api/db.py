# app/api/db.py
import os
from functools import lru_cache

from sqlalchemy import create_engine


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set")
    return database_url


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(get_database_url(), pool_pre_ping=True)
