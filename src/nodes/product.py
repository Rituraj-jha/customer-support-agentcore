from __future__ import annotations

from typing import Any

from src.state import SupportState


def product_information_node(state: SupportState, llm_service: Any) -> dict[str, Any]:
    message = state.get("user_message", "")
    memory = state.get("memory_context", {})
    knowledge = state.get("retrieved_knowledge", [])

    system_prompt = "You are a product specialist. Provide accurate product information from context."
    user_prompt = (
        f"User request: {message}\n"
        f"Memory context: {memory}\n"
        f"Knowledge: {knowledge}"
    )

    return {
        "final_response": llm_service.answer(system_prompt, user_prompt),
        "workflow_status": "product_information_completed",
    }
