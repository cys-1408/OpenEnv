from __future__ import annotations

from enum import Enum
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionType(str, Enum):
    EXTRACT = "EXTRACT"
    COMPARE = "COMPARE"
    SUMMARISE = "SUMMARISE"
    ANNOTATE = "ANNOTATE"
    QUERY = "QUERY"
    SUBMIT = "SUBMIT"


class DocumentView(BaseModel):
    doc_id: str
    doc_type: str
    title: str
    content: str
    sections: dict[str, str]
    version: str
    word_count: int


class ActionResult(BaseModel):
    action_type: str
    success: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class Observation(BaseModel):
    task_id: str
    task_name: Literal["ICF_EXTRACTION", "PROTOCOL_ICF_CONSISTENCY", "SAP_AE_RECONCILIATION"]
    step_number: int
    max_steps: int
    documents: dict[str, DocumentView]
    last_action_result: ActionResult | None
    task_instruction: str
    partial_score: float
    done: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractPayload(BaseModel):
    doc_id: str
    fields: list[str]
    section_hint: str | None = None


class ComparePayload(BaseModel):
    doc_id_a: str
    doc_id_b: str
    comparison_fields: list[str]
    section_hint_a: str | None = None
    section_hint_b: str | None = None


class SummarisePayload(BaseModel):
    doc_id: str
    focus_areas: list[str]
    max_words: int = 300


class AnnotatePayload(BaseModel):
    doc_id: str
    section: str
    label: str
    note: str
    severity: Literal["INFO", "WARNING", "ERROR"]


class QueryPayload(BaseModel):
    doc_id: str
    question: str


class SubmitPayload(BaseModel):
    answer: dict[str, Any]
    confidence: float = Field(ge=0.0, le=1.0)


class Action(BaseModel):
    action_type: ActionType
    payload: (
        ExtractPayload
        | ComparePayload
        | SummarisePayload
        | AnnotatePayload
        | QueryPayload
        | SubmitPayload
        | dict[str, Any]
    )

    model_config = ConfigDict(use_enum_values=False)

    @model_validator(mode="after")
    def _validate_payload_type(self) -> "Action":
        payload_map: dict[ActionType, type[BaseModel]] = {
            ActionType.EXTRACT: ExtractPayload,
            ActionType.COMPARE: ComparePayload,
            ActionType.SUMMARISE: SummarisePayload,
            ActionType.ANNOTATE: AnnotatePayload,
            ActionType.QUERY: QueryPayload,
            ActionType.SUBMIT: SubmitPayload,
        }
        expected = payload_map[self.action_type]
        if isinstance(self.payload, expected):
            return self
        if isinstance(self.payload, dict):
            self.payload = cast(
                ExtractPayload
                | ComparePayload
                | SummarisePayload
                | AnnotatePayload
                | QueryPayload
                | SubmitPayload,
                expected.model_validate(self.payload),
            )
            return self
        raise ValueError(f"Payload must match action_type={self.action_type}")


class RewardWeights(BaseModel):
    accuracy: float = 0.40
    completeness: float = 0.30
    regulatory_alignment: float = 0.20
    efficiency: float = 0.10


class Reward(BaseModel):
    total: float
    accuracy: float
    completeness: float
    regulatory_alignment: float
    efficiency: float
    step_penalty: float
    weights: RewardWeights
