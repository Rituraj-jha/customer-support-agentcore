from __future__ import annotations

from typing import Any

from src.state import SupportState


def technical_support_node(state: SupportState, llm_service: Any) -> dict[str, Any]:
    memory = state.get("memory_context", {})
    knowledge = state.get("retrieved_knowledge", [])
    message = state.get("user_message", "")

    system_prompt = (
        "You are a technical support specialist. Use memory context and knowledge snippets "
        "to provide actionable troubleshooting steps."
    )
    user_prompt = (
        f"User issue: {message}\n"
        f"Memory context: {memory}\n"
        f"Knowledge: {knowledge}"
    )

    return {
        "final_response": llm_service.answer(system_prompt, user_prompt),
        "workflow_status": "technical_support_completed",
    }
