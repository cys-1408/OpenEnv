from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from pharmatrials_env.models import Reward, RewardWeights
from pharmatrials_env.state import EnvState
from pharmatrials_env.tasks.registry import TaskSpec


@dataclass(slots=True)
class RewardContext:
    accuracy: float = 0.0
    regulatory_alignment: float = 0.0
    step_penalty: float = 0.0
    is_submit: bool = False


class RewardEngine:
    def compute(self, state: EnvState, task_spec: TaskSpec, ctx: RewardContext) -> Reward:
        completeness = self._completeness(state)
        efficiency = self._efficiency(state) if ctx.is_submit else 0.0
        step_total = (
            task_spec.reward_weights.accuracy * ctx.accuracy
            + task_spec.reward_weights.completeness * completeness
            + task_spec.reward_weights.regulatory_alignment * ctx.regulatory_alignment
            + task_spec.reward_weights.efficiency * efficiency
            - ctx.step_penalty
        )
        state.reward_components.episode_reward_sum += step_total
        normalized_total = state.reward_components.episode_reward_sum / max(1, state.max_steps)
        normalized_total = max(0.0, min(1.0, normalized_total))

        state.reward_components.accuracy_sum += ctx.accuracy
        state.reward_components.completeness_sum += completeness
        state.reward_components.regulatory_alignment_sum += ctx.regulatory_alignment
        state.reward_components.efficiency_sum += efficiency
        state.reward_components.step_penalty_sum += ctx.step_penalty
        state.reward_components.steps_rewarded += 1

        return Reward(
            total=normalized_total,
            accuracy=max(0.0, min(1.0, ctx.accuracy)),
            completeness=max(0.0, min(1.0, completeness)),
            regulatory_alignment=max(0.0, min(1.0, ctx.regulatory_alignment)),
            efficiency=max(0.0, min(1.0, efficiency)),
            step_penalty=ctx.step_penalty,
            weights=RewardWeights.model_validate(task_spec.reward_weights.model_dump()),
        )

    @staticmethod
    def _completeness(state: EnvState) -> float:
        required = set(state.required_items)
        if not required:
            return 0.0
        return len(state.addressed_items & required) / len(required)

    @staticmethod
    def completeness(state: EnvState) -> float:
        return RewardEngine._completeness(state)

    @staticmethod
    def _efficiency(state: EnvState) -> float:
        if state.max_steps <= 0:
            return 0.0
        return max(0.0, (state.max_steps - state.step_number) / state.max_steps)

    @staticmethod
    def regulatory_alignment_from_payload(payload: dict[str, Any]) -> float:
        text = " ".join(str(v) for v in payload.values()).lower()
        keywords = ["ich", "gcp", "meddra", "ctcae", "ind", "eudract", "protocol"]
        hits = sum(1 for k in keywords if k in text)
        return min(1.0, hits / 4)

    @staticmethod
    def extraction_accuracy(extracted: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        if not extracted:
            return 0.0
        scores: list[float] = []
        for key, value in extracted.items():
            truth = ground_truth.get(key)
            if truth is None:
                scores.append(0.0)
                continue
            if isinstance(truth, (int, float)):
                try:
                    val_f = float(value)
                    truth_f = float(truth)
                    if truth_f == 0:
                        scores.append(1.0 if val_f == 0 else 0.0)
                    else:
                        scores.append(max(0.0, 1.0 - abs(val_f - truth_f) / abs(truth_f)))
                except (TypeError, ValueError):
                    scores.append(0.0)
            else:
                scores.append(fuzz.token_sort_ratio(str(value), str(truth)) / 100.0)
        return sum(scores) / len(scores)
