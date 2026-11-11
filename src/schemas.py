"""Strict model contracts; IDs are checked against stored evidence after parsing."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Citation(Contract):
    document_id: str = Field(min_length=1, max_length=100)
    chunk_id: str = Field(min_length=1, max_length=100)
    page_number: int | None


class Answer(Contract):
    answer: str = Field(min_length=1, max_length=6000)
    answerable: bool
    sources: list[Citation] = Field(max_length=8)
    needs_human_support: bool
    reason: str | None = Field(max_length=500)


class FAQCandidate(Contract):
    question: str = Field(min_length=5, max_length=500)
    answer: str = Field(min_length=3, max_length=3000)
    category: str = Field(min_length=1, max_length=100)
    question_type: Literal["direct", "paraphrased", "procedural", "follow_up", "troubleshooting",
                           "multi_step", "policy", "numerical", "multilingual"]
    supporting_text: str = Field(min_length=5, max_length=6000)


class FAQBatch(Contract):
    faqs: list[FAQCandidate] = Field(max_length=20)


class EvidenceClaim(Contract):
    claim: str = Field(min_length=1, max_length=6000)
    chunk_id: str = Field(min_length=1, max_length=100)
    supporting_text: str = Field(min_length=1, max_length=6000)
    supported: bool


class Verification(Contract):
    supported: bool
    answerable: bool
    contradicts_policy: bool
    claims: list[EvidenceClaim] = Field(max_length=30)
    reason: str = Field(max_length=500)


class Rewrite(Contract):
    question: str = Field(min_length=1, max_length=500)


class Scope(Contract):
    label: Literal["IN_SCOPE", "OUT_OF_SCOPE"]


class QueryIntent(Contract):
    intent: Literal["KNOWLEDGE", "CONVERSATION", "CLARIFICATION", "ACCOUNT_ACTION", "HUMAN_SUPPORT", "OUT_OF_SCOPE"]
