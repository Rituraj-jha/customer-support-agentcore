import os
from contextlib import contextmanager
from typing import Any

from fastapi.testclient import TestClient

# Force in-memory checkpointer when the app lifespan initializes during tests.
os.environ["ALLOW_INMEMORY_CHECKPOINTER"] = "true"
os.environ["POSTGRES_CHECKPOINT_DSN"] = ""

from src.api.app import app


class _Interrupt:
    def __init__(self, value: dict[str, Any]):
        self.value = value


class _DummyGraph:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def invoke(self, payload, config: dict[str, Any]):
        thread_id = config["configurable"]["thread_id"]

        if isinstance(payload, dict):
            state = dict(payload)
            text = state.get("user_message", "").lower()
            metadata = state.get("request_metadata", {})

            intent = "product_information"
            if "refund" in text:
                intent = "refund"

            state["intent"] = intent
            state["validation_passed"] = True
            state["validation_errors"] = []

            if intent == "refund":
                amount = float(metadata.get("refund_amount", 0))
                state["approval_required"] = amount > 100
                if amount > 100:
                    state["approval_status"] = "pending"
                    state["workflow_status"] = "approval_checked"
                    self._sessions[thread_id] = state
                    return {
                        **state,
                        "__interrupt__": [
                            _Interrupt(
                                {
                                    "type": "human_approval_required",
                                    "decision_options": ["approve", "reject"],
                                    "thread_id": thread_id,
                                }
                            )
                        ],
                    }

                state["approval_status"] = "not_required"
                state["workflow_status"] = "memory_updated"
                state["final_response"] = "Refund accepted for low amount"
                self._sessions[thread_id] = state
                return state

            state["approval_required"] = False
            state["approval_status"] = "not_required"
            state["workflow_status"] = "memory_updated"
            state["final_response"] = "Product info response"
            self._sessions[thread_id] = state
            return state

        decision = getattr(payload, "resume", {}).get("decision", "reject")
        previous = self._sessions.get(thread_id, {})
        approval = "approved" if decision in {"approve", "approved"} else "rejected"
        result = {
            **previous,
            "approval_status": approval,
            "approval_required": True,
            "workflow_status": "memory_updated",
            "final_response": f"Refund {approval}",
            "validation_passed": True,
            "validation_errors": [],
            "intent": "refund",
        }
        self._sessions[thread_id] = result
        return result


@contextmanager
def _client():
    with TestClient(app) as client:
        app.state.graph = _DummyGraph()
        yield client


def test_ping_returns_healthy():
    with _client() as client:
        response = client.get("/ping")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "Healthy"
    assert isinstance(payload["time_of_last_update"], int)


def test_invocations_product_query():
    with _client() as client:
        response = client.post(
            "/invocations",
            json={
                "session_id": "demo-product-1",
                "user_id": "user-1",
                "message": "Tell me about your premium support plan",
                "request_metadata": {},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == "demo-product-1"
    assert payload["thread_id"] == "demo-product-1"
    assert payload["workflow_status"] == "memory_updated"
    assert payload["interrupt"] is None
    assert payload["metadata"]["intent"] == "product_information"


def test_invocations_refund_below_threshold():
    with _client() as client:
        response = client.post(
            "/invocations",
            json={
                "session_id": "demo-refund-low-1",
                "user_id": "user-1",
                "message": "Refund order ORDER123 for $50",
                "request_metadata": {"order_id": "ORDER123", "refund_amount": 50},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workflow_status"] == "memory_updated"
    assert payload["interrupt"] is None
    assert payload["metadata"]["approval_required"] is False
    assert payload["metadata"]["approval_status"] == "not_required"


def test_invocations_refund_above_threshold():
    with _client() as client:
        response = client.post(
            "/invocations",
            json={
                "session_id": "demo-refund-hi-1",
                "user_id": "user-1",
                "message": "Refund order ORDER123 for $150",
                "request_metadata": {"order_id": "ORDER123", "refund_amount": 150},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workflow_status"] == "waiting_for_human_approval"
    assert payload["interrupt"] is not None
    assert payload["metadata"]["approval_required"] is True
    assert payload["metadata"]["approval_status"] == "pending"


def test_resume_approval_through_invocations():
    with _client() as client:
        start_response = client.post(
            "/invocations",
            json={
                "session_id": "demo-refund-hi-2",
                "user_id": "user-2",
                "message": "Refund order ORDER999 for $250",
                "request_metadata": {"order_id": "ORDER999", "refund_amount": 250},
            },
        )
        assert start_response.status_code == 200

        resume_response = client.post(
            "/invocations",
            json={
                "session_id": "demo-refund-hi-2",
                "user_id": "user-2",
                "message": "",
                "thread_id": "demo-refund-hi-2",
                "action": "resume",
                "decision": "approve",
            },
        )

    assert resume_response.status_code == 200
    payload = resume_response.json()
    assert payload["thread_id"] == "demo-refund-hi-2"
    assert payload["workflow_status"] == "memory_updated"
    assert payload["metadata"]["approval_status"] == "approved"
    assert payload["interrupt"] is None


def test_interrupt_payload_structure():
    with _client() as client:
        response = client.post(
            "/invocations",
            json={
                "session_id": "demo-refund-hi-3",
                "user_id": "user-3",
                "message": "Refund order ORDER111 for $500",
                "request_metadata": {"order_id": "ORDER111", "refund_amount": 500},
            },
        )

    assert response.status_code == 200
    payload = response.json()
    interrupt = payload["interrupt"]
    assert isinstance(interrupt, dict)
    assert interrupt.get("type") == "human_approval_required"
    assert interrupt.get("thread_id") == "demo-refund-hi-3"
    assert set(interrupt.get("decision_options", [])) == {"approve", "reject"}


def test_invocations_accepts_prompt_only_shape():
    with _client() as client:
        response = client.post(
            "/invocations",
            json={
                "prompt": "Tell me about your premium support plan",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"].startswith("agentcore-")
    assert payload["workflow_status"] == "memory_updated"
    assert payload["metadata"]["intent"] == "product_information"
