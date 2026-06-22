from __future__ import annotations

import re
from typing import Any

from src.state import SupportState


EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ORDER_ID_PATTERN = re.compile(
    r"\b(?:order[_ -]?id[:\s]*)?([A-Z]*\d[A-Z0-9]{4,})\b",
    re.IGNORECASE,
)
AMOUNT_PATTERN = re.compile(r"\$\s*(\d+(?:\.\d{1,2})?)")


def _extract_entities(message: str, metadata: dict[str, Any]) -> dict[str, Any]:
    entities: dict[str, Any] = {}

    order_match = ORDER_ID_PATTERN.search(message)
    if order_match:
        entities["order_id"] = order_match.group(1)
    elif metadata.get("order_id"):
        entities["order_id"] = metadata["order_id"]

    email_match = EMAIL_PATTERN.search(message)
    if email_match:
        entities["email"] = email_match.group(0)
    elif metadata.get("email"):
        entities["email"] = metadata["email"]

    if metadata.get("product_name"):
        entities["product_name"] = metadata["product_name"]
    else:
        lowered = message.lower()
        if "product" in lowered:
            # Lightweight extraction; real systems should use structured extraction.
            entities["product_name"] = message

    amount_match = AMOUNT_PATTERN.search(message)
    if amount_match:
        entities["refund_amount"] = float(amount_match.group(1))
    elif metadata.get("refund_amount") is not None:
        entities["refund_amount"] = float(metadata["refund_amount"])

    return entities


def validate_request(state: SupportState) -> dict[str, Any]:
    intent = state.get("intent", "unknown")
    message = state.get("user_message", "")
    metadata = state.get("request_metadata", {})

    entities = _extract_entities(message, metadata)
    required_fields_by_intent = {
        "refund": ["order_id"],
        "technical_support": ["product_name"],
        "account_access": ["email"],
        "product_information": [],
        "unknown": [],
    }

    missing = [
        field for field in required_fields_by_intent.get(intent, []) if not entities.get(field)
    ]

    validation_passed = len(missing) == 0
    errors = [f"Missing required field: {field}" for field in missing]

    return {
        "validation_passed": validation_passed,
        "validation_errors": errors,
        "extracted_entities": entities,
        "workflow_status": "validated",
    }


def validation_failure_response(state: SupportState) -> dict[str, Any]:
    errors = state.get("validation_errors", [])
    missing_text = "; ".join(errors) if errors else "Request details are incomplete."

    return {
        "final_response": (
            "I need additional information before I can continue. "
            f"{missing_text}. Please provide the missing details."
        ),
        "workflow_status": "validation_failed",
    }
