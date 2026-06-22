from __future__ import annotations

from typing import Any

from src.state import SupportState


def account_access_node(state: SupportState, llm_service: Any) -> dict[str, Any]:
    email = state.get("extracted_entities", {}).get("email", "your account")
    memory = state.get("memory_context", {})

    system_prompt = (
        "You are an account recovery assistant. Provide secure and concise recovery steps."
    )
    user_prompt = (
        f"Account identifier: {email}\n"
        f"Memory context: {memory}\n"
        "Include MFA-first recovery instructions and escalation path."
    )

    return {
        "final_response": llm_service.answer(system_prompt, user_prompt),
        "workflow_status": "account_access_completed",
    }
