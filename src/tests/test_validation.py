from src.nodes.validation import validate_request


def _base_state(intent: str, message: str, metadata: dict | None = None) -> dict:
    return {
        "intent": intent,
        "user_message": message,
        "request_metadata": metadata or {},
    }


def test_refund_requires_order_id():
    result = validate_request(_base_state("refund", "I need a refund please"))
    assert result["validation_passed"] is False
    assert "Missing required field: order_id" in result["validation_errors"]


def test_account_access_accepts_email():
    result = validate_request(
        _base_state("account_access", "I am locked out, email me at user@example.com")
    )
    assert result["validation_passed"] is True
    assert result["extracted_entities"]["email"] == "user@example.com"


def test_technical_support_accepts_product_name_from_metadata():
    result = validate_request(
        _base_state(
            "technical_support",
            "The app crashes on startup",
            metadata={"product_name": "Acme Desktop"},
        )
    )
    assert result["validation_passed"] is True
    assert result["extracted_entities"]["product_name"] == "Acme Desktop"
