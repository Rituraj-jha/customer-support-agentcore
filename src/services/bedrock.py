from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

try:
    from langchain_aws import ChatBedrockConverse
except ImportError as import_error:  # pragma: no cover - import-time guard
    ChatBedrockConverse = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


logger = logging.getLogger(__name__)


class IntentOutput(BaseModel):
    intent: str = Field(description="One of refund, technical_support, account_access, product_information, unknown")
    confidence: float = Field(ge=0.0, le=1.0)


@dataclass(slots=True)
class BedrockLLMService:
    model_id: str
    region: str

    @classmethod
    def from_env(cls) -> "BedrockLLMService":
        return cls(
            model_id=os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-micro-v1:0"),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )

    def _client(self) -> Any:
        if ChatBedrockConverse is None:
            raise RuntimeError(
                "langchain-aws is not installed. Install requirements before running."
            ) from _IMPORT_ERROR
        return ChatBedrockConverse(
            model=self.model_id,
            region_name=self.region,
            temperature=0,
        )

    def classify_intent(
        self,
        user_message: str,
        memory_context: dict[str, Any],
        knowledge: list[dict[str, str]],
    ) -> dict[str, Any]:
        prompt = (
            "You are a support intent classifier. "
            "Classify into exactly one of: refund, technical_support, account_access, product_information, unknown.\n"
            f"User message: {user_message}\n"
            f"Memory context: {json.dumps(memory_context)}\n"
            f"Retrieved knowledge: {json.dumps(knowledge)}"
        )

        try:
            llm = self._client().with_structured_output(IntentOutput)
            parsed = llm.invoke(prompt)
            return {"intent": parsed.intent, "confidence": float(parsed.confidence)}
        except Exception as exc:
            logger.exception("Bedrock intent classification failed for model %s", self.model_id)
            text = user_message.lower()
            if "refund" in text:
                return {"intent": "refund", "confidence": 0.7}
            if any(word in text for word in ["error", "bug", "issue", "crash"]):
                return {"intent": "technical_support", "confidence": 0.68}
            if any(word in text for word in ["login", "password", "locked", "account"]):
                return {"intent": "account_access", "confidence": 0.68}
            if any(word in text for word in ["price", "feature", "plan", "product"]):
                return {"intent": "product_information", "confidence": 0.66}
            return {"intent": "unknown", "confidence": 0.4}

    def _answer_fallback(self, exc: Exception) -> str:
        if os.getenv("BEDROCK_DEBUG_ERRORS", "false").lower() == "true":
            return (
                "I could not reach Bedrock right now. "
                f"Error: {exc.__class__.__name__}: {exc}"
            )
        return "I could not reach Bedrock right now. Please try again shortly."

    def answer(self, system_prompt: str, user_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n{user_prompt}"
        try:
            llm = self._client()
            response = llm.invoke(prompt)
            return getattr(response, "content", str(response))
        except Exception as exc:
            logger.exception("Bedrock response generation failed for model %s", self.model_id)
            return self._answer_fallback(exc)
