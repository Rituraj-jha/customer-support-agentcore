from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError as import_error:  # pragma: no cover - import-time guard
    PostgresSaver = None
    _POSTGRES_IMPORT_ERROR = import_error
else:
    _POSTGRES_IMPORT_ERROR = None


logger = logging.getLogger(__name__)


def postgres_dsn_details(dsn: str) -> dict[str, str | int | None]:
    parsed = urlsplit(dsn)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    return {
        "host": parsed.hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip("/") or None,
        "sslmode": query.get("sslmode"),
    }


def normalize_postgres_dsn(dsn: str) -> str:
    parsed = urlsplit(dsn)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))

    if not query.get("sslmode"):
        query["sslmode"] = "require"

    query.setdefault("keepalives", "1")
    query.setdefault("keepalives_idle", "30")
    query.setdefault("keepalives_interval", "5")
    query.setdefault("keepalives_count", "5")

    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            parsed.fragment,
        )
    )


@contextmanager
def checkpointer_from_env() -> Iterator[object]:
    dsn = os.getenv("POSTGRES_CHECKPOINT_DSN", "").strip()
    allow_inmemory = os.getenv("ALLOW_INMEMORY_CHECKPOINTER", "false").lower() == "true"

    if not dsn:
        if allow_inmemory:
            logger.info("ALLOW_INMEMORY_CHECKPOINTER=true; using MemorySaver")
            yield MemorySaver()
            return
        raise RuntimeError(
            "POSTGRES_CHECKPOINT_DSN is required for durable checkpoint persistence. "
            "Set ALLOW_INMEMORY_CHECKPOINTER=true only for local testing."
        )

    if PostgresSaver is None:
        raise RuntimeError(
            "langgraph-checkpoint-postgres is not installed. Install requirements first."
        ) from _POSTGRES_IMPORT_ERROR

    normalized_dsn = normalize_postgres_dsn(dsn)
    if normalized_dsn != dsn:
        logger.info("POSTGRES_CHECKPOINT_DSN missing sslmode; appending sslmode=require")

    logger.info("Creating PostgreSQL checkpointer")
    with PostgresSaver.from_conn_string(normalized_dsn) as saver:
        saver.setup()
        yield saver
