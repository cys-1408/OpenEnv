from __future__ import annotations

from typing import Any, cast

from rapidfuzz import fuzz

from .base import AbstractGrader


class ICFExtractionGrader(AbstractGrader):
    NUMERIC_FIELDS = {
        "dose_mg",
        "num_visits",
        "total_duration_weeks",
        "compensation_amount_usd",
    }

    def generate_ground_truth(
        self, documents: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        return cast(dict[str, Any], metadata["ground_truth"])

    def score(self, submission: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
        answer = submission.get("answer", {})
        per_field: dict[str, float] = {}
        for field, truth in ground_truth.items():
            pred = answer.get(field)
            if pred is None:
                per_field[field] = 0.0
                continue
            if field in self.NUMERIC_FIELDS:
                per_field[field] = self._numeric_score(pred, truth)
            else:
                per_field[field] = fuzz.token_sort_ratio(str(pred), str(truth)) / 100.0
        score = sum(per_field.values()) / max(1, len(per_field))
        return {"score": max(0.0, min(1.0, score)), "per_field": per_field}

    @staticmethod
    def _numeric_score(pred: Any, truth: Any) -> float:
        try:
            pred_f = float(pred)
            truth_f = float(truth)
        except (TypeError, ValueError):
            return 0.0
        if pred_f == truth_f:
            return 1.0
        if truth_f == 0:
            return 1.0 if pred_f == 0 else 0.0
        deviation = abs(pred_f - truth_f) / abs(truth_f)
        if deviation <= 0.1:
            return 0.5
        return max(0.0, 1.0 - deviation)
