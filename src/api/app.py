from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.graph import build_support_graph
from src.services.bedrock import BedrockLLMService
from src.services.knowledge_base import KnowledgeBaseService
from src.services.memory_service import MemoryService
from src.services.postgres import checkpointer_from_env

load_dotenv()


class StartWorkflowRequest(BaseModel):
    session_id: str
    user_id: str
    message: str
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str
    decision: str = Field(description="approve or reject")


class WorkflowResponse(BaseModel):
    thread_id: str
    workflow_status: str
    final_response: str | None = None
    intent: str | None = None
    validation_passed: bool | None = None
    validation_errors: list[str] = Field(default_factory=list)
    approval_required: bool | None = None
    approval_status: str | None = None
    interrupt: Any = None


class InvocationRequest(BaseModel):
    session_id: str
    user_id: str
    message: str
    request_metadata: dict[str, Any] = Field(default_factory=dict)
    thread_id: str | None = None
    action: Literal["resume"] | None = None
    decision: str | None = None


class InvocationResponse(BaseModel):
    session_id: str
    thread_id: str
    workflow_status: str
    final_response: str | None = None
    interrupt: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm_service = BedrockLLMService.from_env()
    memory_service = MemoryService()
    knowledge_service = KnowledgeBaseService.default()
    refund_threshold = float(os.getenv("REFUND_APPROVAL_THRESHOLD", "100"))

    checkpoint_cm = checkpointer_from_env()
    checkpointer = checkpoint_cm.__enter__()

    app.state.graph = build_support_graph(
        llm_service=llm_service,
        memory_service=memory_service,
        knowledge_service=knowledge_service,
        refund_threshold=refund_threshold,
        checkpointer=checkpointer,
    )
    app.state.checkpoint_cm = checkpoint_cm

    try:
        yield
    finally:
        checkpoint_cm.__exit__(None, None, None)


app = FastAPI(
    title="Customer Support Workflow Agent",
    version="1.0.0",
    lifespan=lifespan,
)


def _base_state(payload: StartWorkflowRequest) -> dict[str, Any]:
    return {
        "session_id": payload.session_id,
        "user_id": payload.user_id,
        "messages": [{"role": "user", "content": payload.message}],
        "user_message": payload.message,
        "user_profile": {},
        "memory_context": {},
        "intent": "unknown",
        "intent_confidence": 0.0,
        "validation_passed": False,
        "validation_errors": [],
        "request_metadata": payload.request_metadata,
        "extracted_entities": {},
        "approval_required": False,
        "approval_status": "not_required",
        "retrieved_knowledge": [],
        "final_response": "",
        "workflow_status": "started",
    }


def execute_workflow(
    *,
    thread_id: str,
    base_state: dict[str, Any] | None = None,
    resume_decision: str | None = None,
) -> dict[str, Any]:
    graph = app.state.graph
    config = {"configurable": {"thread_id": thread_id}}

    if resume_decision is not None:
        return graph.invoke(Command(resume={"decision": resume_decision}), config=config)

    if base_state is None:
        raise ValueError("base_state is required when resume_decision is not provided")
    return graph.invoke(base_state, config=config)


def _response_from_result(thread_id: str, result: dict[str, Any]) -> WorkflowResponse:
    interrupts = result.get("__interrupt__")
    return WorkflowResponse(
        thread_id=thread_id,
        workflow_status=result.get("workflow_status", "unknown"),
        final_response=result.get("final_response"),
        intent=result.get("intent"),
        validation_passed=result.get("validation_passed"),
        validation_errors=result.get("validation_errors", []),
        approval_required=result.get("approval_required"),
        approval_status=result.get("approval_status"),
        interrupt=interrupts[0].value if interrupts else None,
    )


def _invocation_response_from_result(
    session_id: str,
    thread_id: str,
    result: dict[str, Any],
) -> InvocationResponse:
    interrupts = result.get("__interrupt__")
    interrupt_payload = interrupts[0].value if interrupts else None
    waiting_for_approval = interrupt_payload is not None

    return InvocationResponse(
        session_id=session_id,
        thread_id=thread_id,
        workflow_status=(
            "waiting_for_human_approval"
            if waiting_for_approval
            else result.get("workflow_status", "unknown")
        ),
        final_response=result.get("final_response"),
        interrupt=interrupt_payload,
        metadata={
            "intent": result.get("intent"),
            "approval_required": result.get("approval_required"),
            "approval_status": result.get("approval_status"),
            "validation_passed": result.get("validation_passed"),
            "validation_errors": result.get("validation_errors", []),
        },
    )


def _normalize_agentcore_payload(body: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(body)

    prompt_value = normalized.get("prompt")
    message_value = normalized.get("message")
    if message_value is None and prompt_value is not None:
        normalized["message"] = str(prompt_value)

    session_id = normalized.get("session_id")
    thread_id = normalized.get("thread_id")
    if not session_id:
        normalized["session_id"] = str(thread_id or f"agentcore-{uuid.uuid4().hex[:12]}")

    if not normalized.get("user_id"):
        normalized["user_id"] = "agentcore-user"

    if normalized.get("request_metadata") is None:
        normalized["request_metadata"] = {}

    return normalized


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ping")
def ping() -> dict[str, Any]:
    return {
        "status": "Healthy",
        "time_of_last_update": int(time.time()),
    }


@app.post("/workflow/start", response_model=WorkflowResponse)
def start_workflow(payload: StartWorkflowRequest) -> WorkflowResponse:
    thread_id = payload.thread_id or payload.session_id
    result = execute_workflow(thread_id=thread_id, base_state=_base_state(payload))

    return _response_from_result(thread_id, result)


@app.post("/workflow/approval", response_model=WorkflowResponse)
def submit_approval(payload: ApprovalRequest) -> WorkflowResponse:
    thread_id = payload.thread_id
    decision = payload.decision.strip().lower()
    if decision not in {"approve", "approved", "reject", "rejected"}:
        raise HTTPException(status_code=400, detail="decision must be approve or reject")

    result = execute_workflow(thread_id=thread_id, resume_decision=decision)

    return _response_from_result(thread_id, result)


@app.post("/invocations", response_model=InvocationResponse)
async def invoke(request: Request) -> InvocationResponse:
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid JSON body") from exc

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="request body must be a JSON object")

    print("===================================")
    print("AGENTCORE REQUEST RECEIVED")
    print(body)
    print("===================================")

    normalized_body = _normalize_agentcore_payload(body)

    try:
        payload = InvocationRequest(**normalized_body)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid invocation payload: {exc}") from exc

    thread_id = payload.thread_id or payload.session_id
    action = (payload.action or "").strip().lower()

    if action == "resume":
        decision = (payload.decision or "").strip().lower()
        if decision not in {"approve", "approved", "reject", "rejected"}:
            raise HTTPException(status_code=400, detail="decision must be approve or reject")

        result = execute_workflow(thread_id=thread_id, resume_decision=decision)
        return _invocation_response_from_result(payload.session_id, thread_id, result)

    start_payload = StartWorkflowRequest(
        session_id=payload.session_id,
        user_id=payload.user_id,
        message=payload.message,
        request_metadata=payload.request_metadata,
        thread_id=thread_id,
    )
    result = execute_workflow(thread_id=thread_id, base_state=_base_state(start_payload))
    return _invocation_response_from_result(payload.session_id, thread_id, result)


@app.get("/workflow/state/{thread_id}")
def workflow_state(thread_id: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread_id}}
    graph = app.state.graph

    try:
        snapshot = graph.get_state(config)
    except Exception as exc:  # pragma: no cover - runtime inspection endpoint
        raise HTTPException(status_code=404, detail=f"No state found for thread_id={thread_id}") from exc

    values = getattr(snapshot, "values", None)
    if values is None:
        return {"thread_id": thread_id, "state": {}}

    return {"thread_id": thread_id, "state": values}
