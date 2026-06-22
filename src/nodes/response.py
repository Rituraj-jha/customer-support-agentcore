from __future__ import annotations

from src.state import SupportState


def clarification_node(state: SupportState) -> dict[str, str]:
    return {
        "final_response": (
            "I want to make sure I route this correctly. Is your request about a refund, "
            "technical issue, account access, or product information?"
        ),
        "workflow_status": "clarification_requested",
    }


def ensure_response_node(state: SupportState) -> dict[str, str]:
    response = state.get("final_response")
    if response:
        return {"workflow_status": "response_ready"}

    return {
        "final_response": "Your request has been processed. A support specialist will follow up shortly.",
        "workflow_status": "response_fallback_generated",
    }
