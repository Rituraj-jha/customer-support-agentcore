from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Callable
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


@dataclass
class CheckpointerResource:
    checkpointer: object
    close: Callable[[], None]


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


def checkpointer_from_env() -> CheckpointerResource:
    dsn = os.getenv("POSTGRES_CHECKPOINT_DSN", "").strip()
    allow_inmemory = os.getenv("ALLOW_INMEMORY_CHECKPOINTER", "false").lower() == "true"

    if not dsn:
        if allow_inmemory:
            logger.info("ALLOW_INMEMORY_CHECKPOINTER=true; using MemorySaver")
            checkpointer = MemorySaver()
            logger.info("Checkpointer type=%s", type(checkpointer))
            logger.info("Checkpointer created successfully")
            return CheckpointerResource(
                checkpointer=checkpointer,
                close=lambda: None,
            )
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

    logger.info("Creating Postgres checkpointer")
    saver_cm = PostgresSaver.from_conn_string(normalized_dsn)
    checkpointer = saver_cm.__enter__()

    try:
        checkpointer.setup()
    except Exception:
        logger.exception("Postgres checkpointer setup failed")
        saver_cm.__exit__(None, None, None)
        raise

    logger.info("Checkpointer type=%s", type(checkpointer))
    logger.info("Checkpointer created successfully")

    def _close_checkpointer() -> None:
        logger.info("Closing Postgres checkpointer")
        saver_cm.__exit__(None, None, None)
        logger.info("Postgres checkpointer closed")

    return CheckpointerResource(
        checkpointer=checkpointer,
        close=_close_checkpointer,
    )
