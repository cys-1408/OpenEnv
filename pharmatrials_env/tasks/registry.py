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
        grader_metrics = {
            "EASY": "field_accuracy",
            "MEDIUM": "inconsistency_f1",
            "HARD": "weighted_reconciliation_score",
        }
        rows: list[dict[str, object]] = []
        for spec in self._specs.values():
            grader_name = spec.grader.__class__.__name__
            grader_module = spec.grader.__class__.__module__
            grader_impl = f"{grader_module}.{grader_name}"
            grader_metric = grader_metrics.get(spec.task_id, "score")
            rows.append(
                {
                    "id": spec.task_id,
                    "task_id": spec.task_id,
                    "name": spec.task_name,
                    "task_name": spec.task_name,
                    "max_steps": spec.max_steps,
                    "has_grader": True,
                    "has_graders": True,
                    "grader_enabled": True,
                    "grader": {
                        "type": "python",
                        "name": grader_name,
                        "module": grader_module,
                        "implementation": grader_impl,
                        "metric": grader_metric,
                        "enabled": True,
                    },
                    "graders": [
                        {
                            "type": "python",
                            "name": grader_name,
                            "module": grader_module,
                            "implementation": grader_impl,
                            "metric": grader_metric,
                            "enabled": True,
                        }
                    ],
                    "allowed_actions": [
                        a.value for a in sorted(spec.allowed_actions, key=lambda x: x.value)
                    ],
                }
            )
        return rows
