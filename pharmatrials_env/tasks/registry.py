from __future__ import annotations

from dataclasses import dataclass

from pharmatrials_env.graders.base import AbstractGrader
from pharmatrials_env.models import ActionType, RewardWeights


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    task_name: str
    instruction_template: str
    max_steps: int
    allowed_actions: set[ActionType]
    reward_weights: RewardWeights
    required_output_items: list[str]
    grader: AbstractGrader
    document_types: list[str]


class TaskRegistry:
    def __init__(self, specs: dict[str, TaskSpec]) -> None:
        self._specs = specs

    @classmethod
    def default(cls) -> "TaskRegistry":
        from .easy import EASY_TASK
        from .hard import HARD_TASK
        from .medium import MEDIUM_TASK

        return cls(
            {
                EASY_TASK.task_id: EASY_TASK,
                MEDIUM_TASK.task_id: MEDIUM_TASK,
                HARD_TASK.task_id: HARD_TASK,
            }
        )

    def get(self, task_id: str | None) -> TaskSpec:
        resolved = task_id or "EASY"
        if resolved not in self._specs:
            valid = ", ".join(sorted(self._specs))
            raise ValueError(f"Unknown task_id '{resolved}'. Valid task IDs: {valid}")
        return self._specs[resolved]

    def summaries(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for spec in self._specs.values():
            rows.append(
                {
                    "task_id": spec.task_id,
                    "task_name": spec.task_name,
                    "max_steps": spec.max_steps,
                    "has_grader": True,
                    "grader": {
                        "name": spec.grader.__class__.__name__,
                        "module": spec.grader.__class__.__module__,
                    },
                    "allowed_actions": [
                        a.value for a in sorted(spec.allowed_actions, key=lambda x: x.value)
                    ],
                }
            )
        return rows
