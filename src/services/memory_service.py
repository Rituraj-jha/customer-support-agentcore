from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class InMemoryMemoryService:
    _store: dict[str, dict[str, Any]]

    @classmethod
    def create(cls) -> "InMemoryMemoryService":
        return cls(_store={})

    def read(self, actor_id: str) -> dict[str, Any]:
        return self._store.get(actor_id, {})

    def write(self, actor_id: str, memory_update: dict[str, Any]) -> None:
        previous = self._store.get(actor_id, {})
        merged = {**previous, **memory_update}
        self._store[actor_id] = merged


@dataclass(slots=True)
class AgentCoreMemoryService:
    endpoint: str
    api_key: str | None = None
    timeout_seconds: int = 10

    @classmethod
    def from_env(cls) -> "AgentCoreMemoryService | None":
        endpoint = os.getenv("AGENTCORE_MEMORY_ENDPOINT", "").strip()
        if not endpoint:
            return None
        return cls(endpoint=endpoint.rstrip("/"), api_key=os.getenv("AGENTCORE_MEMORY_API_KEY"))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def read(self, actor_id: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.endpoint}/memories/{actor_id}",
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        return {"items": payload}

    def write(self, actor_id: str, memory_update: dict[str, Any]) -> None:
        response = requests.post(
            f"{self.endpoint}/memories/{actor_id}",
            headers=self._headers(),
            json={"actor_id": actor_id, "memory": memory_update},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()


class MemoryService:
    """Adapter for AgentCore memory with in-memory fallback for local development."""

    def __init__(self) -> None:
        remote = AgentCoreMemoryService.from_env()
        self._service: AgentCoreMemoryService | InMemoryMemoryService
        self._service = remote if remote is not None else InMemoryMemoryService.create()

    def read(self, actor_id: str) -> dict[str, Any]:
        return self._service.read(actor_id)

    def write(self, actor_id: str, memory_update: dict[str, Any]) -> None:
        self._service.write(actor_id, memory_update)
