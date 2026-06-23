from __future__ import annotations

import logging
import os
import time
import traceback
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    logger.info("Starting application")
    logger.info("BEDROCK_MODEL_ID=%s", os.getenv("BEDROCK_MODEL_ID"))
    logger.info(
        "POSTGRES_CHECKPOINT_DSN exists=%s",
        bool(os.getenv("POSTGRES_CHECKPOINT_DSN")),
    )

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


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    try:
        logger.info(
            "REQUEST method=%s path=%s",
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        logger.info(
            "RESPONSE method=%s path=%s status=%s",
            request.method,
            request.url.path,
            response.status_code,
        )

        return response

    except Exception:
        logger.exception(
            "UNHANDLED EXCEPTION method=%s path=%s",
            request.method,
            request.url.path,
        )
        raise


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
    logger.info(
        "execute_workflow thread_id=%s resume=%s",
        thread_id,
        resume_decision,
    )

    try:
        graph = app.state.graph
        config = {"configurable": {"thread_id": thread_id}}

        if resume_decision is not None:
            return graph.invoke(Command(resume={"decision": resume_decision}), config=config)

        if base_state is None:
            raise ValueError("base_state is required when resume_decision is not provided")
        return graph.invoke(base_state, config=config)
    except Exception:
        logger.exception(
            "execute_workflow failed thread_id=%s",
            thread_id,
        )
        raise


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
        try:
            body = await request.json()
        except Exception as exc:
            logger.exception("ERROR INSIDE /invocations: invalid JSON body")
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        if not isinstance(body, dict):
            logger.exception("ERROR INSIDE /invocations: request body must be a JSON object")
            raise HTTPException(status_code=400, detail="request body must be a JSON object")

        normalized_body = _normalize_agentcore_payload(body)

        try:
            payload = InvocationRequest(**normalized_body)
        except Exception as exc:
            logger.exception("ERROR INSIDE /invocations: invalid invocation payload")
            raise HTTPException(status_code=400, detail=f"invalid invocation payload: {exc}") from exc

        logger.info("INVOCATION PAYLOAD: %s", payload.model_dump())

        thread_id = payload.thread_id or payload.session_id
        action = (payload.action or "").strip().lower()

        logger.info(
            "thread_id=%s action=%s",
            thread_id,
            action,
        )

        if action == "resume":
            decision = (payload.decision or "").strip().lower()

            logger.info("resume decision=%s", decision)

            if decision not in {
                "approve",
                "approved",
                "reject",
                "rejected",
            }:
                raise HTTPException(
                    status_code=400,
                    detail="decision must be approve or reject",
                )

            result = execute_workflow(
                thread_id=thread_id,
                resume_decision=decision,
            )

            logger.info("workflow result=%s", result)

            return _invocation_response_from_result(
                payload.session_id,
                thread_id,
                result,
            )

        start_payload = StartWorkflowRequest(
            session_id=payload.session_id,
            user_id=payload.user_id,
            message=payload.message,
            request_metadata=payload.request_metadata,
            thread_id=thread_id,
        )

        logger.info("starting workflow")

        result = execute_workflow(
            thread_id=thread_id,
            base_state=_base_state(start_payload),
        )

        logger.info("workflow result=%s", result)

        return _invocation_response_from_result(
            payload.session_id,
            thread_id,
            result,
        )

    except Exception:
        logger.exception("ERROR INSIDE /invocations")
        raise


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
