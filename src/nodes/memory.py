from __future__ import annotations

from typing import Any

from src.state import SupportState


def load_memory_context(state: SupportState, memory_service: Any) -> dict[str, Any]:
    actor_id = state.get("user_id", "anonymous")
    memory = memory_service.read(actor_id)
    return {
        "memory_context": memory,
        "workflow_status": "memory_loaded",
    }


def persist_memory_context(state: SupportState, memory_service: Any) -> dict[str, Any]:
    actor_id = state.get("user_id", "anonymous")
    user_message = state.get("user_message", "")
    memory_update: dict[str, Any] = {
        "last_intent": state.get("intent", "unknown"),
        "last_resolution": state.get("final_response", ""),
    }

    lowered = user_message.lower()
    if "i prefer" in lowered:
        memory_update["communication_preference"] = user_message.strip()

    memory_service.write(actor_id, memory_update)
    merged_memory = {**state.get("memory_context", {}), **memory_update}

    return {
        "memory_context": merged_memory,
        "workflow_status": "memory_updated",
    }
