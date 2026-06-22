from __future__ import annotations

from typing import Any

from src.state import SupportState


def retrieve_knowledge(state: SupportState, kb_service: Any) -> dict[str, Any]:
    query = state.get("user_message", "")
    knowledge = kb_service.retrieve(query, top_k=3)
    return {
        "retrieved_knowledge": knowledge,
        "workflow_status": "knowledge_retrieved",
    }
