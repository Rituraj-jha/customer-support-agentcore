from __future__ import annotations

import logging
import os
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.nodes.account import account_access_node
from src.nodes.approval import human_approval_node
from src.nodes.intent import classify_intent
from src.nodes.memory import load_memory_context, persist_memory_context
from src.nodes.product import product_information_node
from src.nodes.refund import (
    check_refund_policy,
    finalize_refund_response,
    set_approval_requirement,
    validate_order,
)
from src.nodes.response import clarification_node, ensure_response_node
from src.nodes.retrieval import retrieve_knowledge
from src.nodes.technical import technical_support_node
from src.nodes.validation import validate_request, validation_failure_response
from src.state import SupportState


logger = logging.getLogger(__name__)


def _route_after_validation(state: SupportState) -> Literal[
    "validation_failure",
    "refund",
    "technical_support",
    "account_access",
    "product_information",
    "unknown",
]:
    if not state.get("validation_passed", False):
        return "validation_failure"

    intent = state.get("intent", "unknown")
    if intent in {
        "refund",
        "technical_support",
        "account_access",
        "product_information",
    }:
        return intent
    return "unknown"


def _route_after_approval_gate(state: SupportState) -> Literal["human_approval", "refund_finalize"]:
    if state.get("approval_required", False):
        return "human_approval"
    return "refund_finalize"


def build_support_graph(
    *,
    llm_service: Any,
    memory_service: Any,
    knowledge_service: Any,
    refund_threshold: float,
    checkpointer: Any,
):
    builder = StateGraph(SupportState)

    builder.add_node("load_memory", lambda s: load_memory_context(s, memory_service))
    builder.add_node("retrieve_knowledge", lambda s: retrieve_knowledge(s, knowledge_service))
    builder.add_node("classify_intent", lambda s: classify_intent(s, llm_service))
    builder.add_node("validate_request", validate_request)
    builder.add_node("validation_failure", validation_failure_response)

    builder.add_node("refund_validate_order", validate_order)
    builder.add_node("refund_policy_check", check_refund_policy)
    builder.add_node(
        "refund_approval_gate",
        lambda s: set_approval_requirement(s, threshold=refund_threshold),
    )
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("refund_finalize", finalize_refund_response)

    builder.add_node("technical_support", lambda s: technical_support_node(s, llm_service))
    builder.add_node("account_access", lambda s: account_access_node(s, llm_service))
    builder.add_node("product_information", lambda s: product_information_node(s, llm_service))
    builder.add_node("clarification", clarification_node)

    builder.add_node("ensure_response", ensure_response_node)
    builder.add_node("persist_memory", lambda s: persist_memory_context(s, memory_service))

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "retrieve_knowledge")
    builder.add_edge("retrieve_knowledge", "classify_intent")
    builder.add_edge("classify_intent", "validate_request")

    builder.add_conditional_edges(
        "validate_request",
        _route_after_validation,
        {
            "validation_failure": "validation_failure",
            "refund": "refund_validate_order",
            "technical_support": "technical_support",
            "account_access": "account_access",
            "product_information": "product_information",
            "unknown": "clarification",
        },
    )

    builder.add_edge("refund_validate_order", "refund_policy_check")
    builder.add_edge("refund_policy_check", "refund_approval_gate")

    builder.add_conditional_edges(
        "refund_approval_gate",
        _route_after_approval_gate,
        {
            "human_approval": "human_approval",
            "refund_finalize": "refund_finalize",
        },
    )

    builder.add_edge("human_approval", "refund_finalize")

    builder.add_edge("refund_finalize", "ensure_response")
    builder.add_edge("technical_support", "ensure_response")
    builder.add_edge("account_access", "ensure_response")
    builder.add_edge("product_information", "ensure_response")
    builder.add_edge("clarification", "ensure_response")
    builder.add_edge("validation_failure", "ensure_response")

    builder.add_edge("ensure_response", "persist_memory")
    builder.add_edge("persist_memory", END)

    logger.info("Checkpointer type=%s", type(checkpointer))

    if os.getenv("DISABLE_CHECKPOINTER", "").lower() == "true":
        graph = builder.compile()
    else:
        graph = builder.compile(checkpointer=checkpointer)

    return graph
