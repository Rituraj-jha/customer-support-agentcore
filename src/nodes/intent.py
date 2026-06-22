from __future__ import annotations

from typing import Any

from src.state import SupportState


def classify_intent(state: SupportState, llm_service: Any) -> dict[str, Any]:
    user_message = state.get("user_message", "")
    memory_context = state.get("memory_context", {})
    knowledge = state.get("retrieved_knowledge", [])

    result = llm_service.classify_intent(user_message, memory_context, knowledge)
    intent = result.get("intent", "unknown")
    confidence = float(result.get("confidence", 0.0))

    if intent not in {
        "refund",
        "technical_support",
        "account_access",
        "product_information",
        "unknown",
    }:
        intent = "unknown"

    return {
        "intent": intent,
        "intent_confidence": confidence,
        "workflow_status": "intent_classified",
    }
