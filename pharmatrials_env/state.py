from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import DocumentView


class ConsistencyFlag(BaseModel):
    doc_id_a: str
    doc_id_b: str
    field: str
    value_a: str
    value_b: str
    is_inconsistent: bool
    severity: Literal["INFO", "WARNING", "ERROR"]


class Annotation(BaseModel):
    doc_id: str
    section: str
    label: str
    note: str
    severity: Literal["INFO", "WARNING", "ERROR"]
    step_number: int


class QueryRecord(BaseModel):
    doc_id: str
    question: str
    answer: str
    step_number: int
    consumed: bool = False


class RewardAccumulator(BaseModel):
    accuracy_sum: float = 0.0
    completeness_sum: float = 0.0
    regulatory_alignment_sum: float = 0.0
    efficiency_sum: float = 0.0
    step_penalty_sum: float = 0.0
    episode_reward_sum: float = 0.0
    steps_rewarded: int = 0


class EnvState(BaseModel):
    task_id: str = ""
    task_name: str = "ICF_EXTRACTION"
    seed: int = 0
    step_number: int = 0
    max_steps: int = 0
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    documents: dict[str, DocumentView] = Field(default_factory=dict)
    extraction_outputs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    consistency_flags: list[ConsistencyFlag] = Field(default_factory=list)
    annotations: list[Annotation] = Field(default_factory=list)
    query_history: list[QueryRecord] = Field(default_factory=list)
    reward_components: RewardAccumulator = Field(default_factory=RewardAccumulator)
    done: bool = False
    episode_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    required_items: list[str] = Field(default_factory=list)
    addressed_items: set[str] = Field(default_factory=set)
    invalid_action_streak: int = 0
    action_count_by_fingerprint: dict[str, list[int]] = Field(default_factory=dict)


class StateManager:
    def __init__(self) -> None:
        self._state = EnvState()

    @property
    def state(self) -> EnvState:
        return self._state

    def replace(self, state: EnvState) -> None:
        self._state = state
