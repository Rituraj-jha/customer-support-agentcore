from typing import Any, Literal, NotRequired, TypedDict


IntentType = Literal[
    "refund",
    "technical_support",
    "account_access",
    "product_information",
    "unknown",
]

ApprovalStatus = Literal["not_required", "pending", "approved", "rejected"]


class SupportState(TypedDict):
    session_id: str
    user_id: str

    messages: list[dict[str, Any]]
    user_message: str

    user_profile: dict[str, Any]
    memory_context: dict[str, Any]

    intent: IntentType
    intent_confidence: float

    validation_passed: bool
    validation_errors: list[str]

    request_metadata: dict[str, Any]
    extracted_entities: dict[str, Any]

    approval_required: bool
    approval_status: ApprovalStatus

    retrieved_knowledge: list[dict[str, str]]

    final_response: str

    workflow_status: str

    # Optional runtime fields (added during execution)
    resume_reason: NotRequired[str]
