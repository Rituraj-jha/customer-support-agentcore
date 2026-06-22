from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class KnowledgeBaseService:
    docs: list[dict[str, str]]

    @classmethod
    def default(cls) -> "KnowledgeBaseService":
        return cls(
            docs=[
                {
                    "topic": "refund_policy",
                    "content": "Refunds are allowed within 30 days. Refunds above $100 require supervisor approval.",
                },
                {
                    "topic": "account_lockout",
                    "content": "Users may reset passwords using MFA and verified email recovery.",
                },
                {
                    "topic": "technical_support",
                    "content": "Gather product name, error details, and reproduction steps before troubleshooting.",
                },
                {
                    "topic": "product_information",
                    "content": "Product details should include version, subscription tier, and compatibility constraints.",
                },
            ]
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, str]]:
        tokens = {token.lower() for token in query.split() if token.strip()}
        if not tokens:
            return self.docs[:top_k]

        scored: list[tuple[int, dict[str, str]]] = []
        for doc in self.docs:
            haystack = f"{doc['topic']} {doc['content']}".lower()
            score = sum(1 for token in tokens if token in haystack)
            scored.append((score, doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        relevant = [doc for score, doc in scored if score > 0]
        if not relevant:
            return self.docs[:top_k]
        return relevant[:top_k]
