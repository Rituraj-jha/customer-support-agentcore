from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from src.graph import build_support_graph
from src.services.knowledge_base import KnowledgeBaseService
from src.services.memory_service import InMemoryMemoryService


class StubLLM:
    def classify_intent(self, user_message, memory_context, knowledge):
        text = user_message.lower()
        if "refund" in text:
            return {"intent": "refund", "confidence": 0.95}
        if "locked" in text or "password" in text:
            return {"intent": "account_access", "confidence": 0.95}
        return {"intent": "unknown", "confidence": 0.5}

    def answer(self, system_prompt, user_prompt):
        return "stub-response"


def _initial_state(message: str, metadata: dict | None = None):
    return {
        "session_id": "s-1",
        "user_id": "u-1",
        "messages": [{"role": "user", "content": message}],
        "user_message": message,
        "user_profile": {},
        "memory_context": {},
        "intent": "unknown",
        "intent_confidence": 0.0,
        "validation_passed": False,
        "validation_errors": [],
        "request_metadata": metadata or {},
        "extracted_entities": {},
        "approval_required": False,
        "approval_status": "not_required",
        "retrieved_knowledge": [],
        "final_response": "",
        "workflow_status": "started",
    }


def test_refund_flow_interrupts_for_approval():
    graph = build_support_graph(
        llm_service=StubLLM(),
        memory_service=InMemoryMemoryService.create(),
        knowledge_service=KnowledgeBaseService.default(),
        refund_threshold=100,
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "thread-approval"}}
    result = graph.invoke(
        _initial_state(
            "Please refund my order ORDER12345 for $150",
            metadata={"order_id": "ORDER12345", "refund_amount": 150},
        ),
        config=config,
    )

    assert "__interrupt__" in result

    resumed = graph.invoke(Command(resume={"decision": "approve"}), config=config)
    assert resumed["approval_status"] == "approved"
    assert resumed["workflow_status"] in {"memory_updated", "response_ready", "refund_completed"}
