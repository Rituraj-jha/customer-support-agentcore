from __future__ import annotations

from typing import Any

from src.state import SupportState


def validate_order(state: SupportState) -> dict[str, Any]:
    order_id = state.get("extracted_entities", {}).get("order_id")
    if not order_id:
        return {
            "validation_passed": False,
            "validation_errors": ["Missing required field: order_id"],
            "workflow_status": "refund_order_invalid",
        }

    return {
        "workflow_status": "refund_order_validated",
    }


def check_refund_policy(state: SupportState) -> dict[str, Any]:
    knowledge = state.get("retrieved_knowledge", [])
    policy_summary = next(
        (
            doc.get("content", "")
            for doc in knowledge
            if doc.get("topic") == "refund_policy"
        ),
        "Refund policy context unavailable.",
    )

    return {
        "user_profile": {
            **state.get("user_profile", {}),
            "refund_policy_summary": policy_summary,
        },
        "workflow_status": "refund_policy_checked",
    }


def set_approval_requirement(state: SupportState, threshold: float) -> dict[str, Any]:
    amount = float(state.get("extracted_entities", {}).get("refund_amount", 0.0))
    approval_required = amount > threshold

    return {
        "approval_required": approval_required,
        "approval_status": "pending" if approval_required else "not_required",
        "workflow_status": "approval_checked",
    }


def finalize_refund_response(state: SupportState) -> dict[str, Any]:
    approval_status = state.get("approval_status", "not_required")
    amount = state.get("extracted_entities", {}).get("refund_amount", "requested")
    order_id = state.get("extracted_entities", {}).get("order_id", "unknown")

    if approval_status == "rejected":
        response = (
            f"Refund request for order {order_id} was reviewed and not approved. "
            "A support specialist can help with alternatives."
        )
    else:
        response = (
            f"Refund request for order {order_id} has been accepted for ${amount}. "
            "You will receive confirmation once processing is complete."
        )

    return {
        "final_response": response,
        "workflow_status": "refund_completed",
    }
