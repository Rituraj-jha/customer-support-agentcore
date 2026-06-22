from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError as import_error:  # pragma: no cover - import-time guard
    PostgresSaver = None
    _POSTGRES_IMPORT_ERROR = import_error
else:
    _POSTGRES_IMPORT_ERROR = None


@contextmanager
def checkpointer_from_env() -> Iterator[object]:
    dsn = os.getenv("POSTGRES_CHECKPOINT_DSN", "").strip()
    allow_inmemory = os.getenv("ALLOW_INMEMORY_CHECKPOINTER", "false").lower() == "true"

    if not dsn:
        if allow_inmemory:
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

    with PostgresSaver.from_conn_string(dsn) as saver:
        saver.setup()
        yield saver
