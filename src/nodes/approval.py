from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from src.state import SupportState


def human_approval_node(state: SupportState) -> dict[str, Any]:
    entities = state.get("extracted_entities", {})
    amount = entities.get("refund_amount", 0)
    order_id = entities.get("order_id", "unknown")

    decision = interrupt(
        {
            "type": "human_approval_required",
            "reason": "Refund exceeds approval threshold",
            "order_id": order_id,
            "refund_amount": amount,
            "instructions": "Resume workflow with decision: approve or reject",
        }
    )

    if isinstance(decision, dict):
        raw = str(decision.get("decision", "reject")).lower()
    else:
        raw = str(decision).lower()

    status = "approved" if raw in {"approve", "approved", "yes"} else "rejected"

    return {
        "approval_status": status,
        "workflow_status": "approval_received",
    }
