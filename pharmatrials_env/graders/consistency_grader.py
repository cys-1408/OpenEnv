from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from .base import AbstractGrader


class ConsistencyGrader(AbstractGrader):
    REQUIRED_SCHEMA_FIELDS = (
        "section_in_protocol",
        "section_in_icf",
        "regulatory_basis",
    )

    def generate_ground_truth(
        self, documents: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        return {"inconsistencies": metadata.get("inconsistencies", [])}

    def score(self, submission: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
        pred_items = submission.get("answer", {}).get("inconsistencies", [])
        true_items = ground_truth.get("inconsistencies", [])
        if not true_items:
            return self._empty_truth_score(pred_items)

        tp, schema_hits = self._accumulate_matches(pred_items, true_items)
        return self._build_score(tp, schema_hits, len(pred_items), len(true_items))

    def _accumulate_matches(
        self, pred_items: list[dict[str, Any]], true_items: list[dict[str, Any]]
    ) -> tuple[float, int]:
        matched_true: set[int] = set()
        tp = 0.0
        schema_hits = 0
        for pred in pred_items:
            best_idx, best_points, qualified_schema_hit = self._best_match(
                pred, true_items, matched_true
            )
            if best_idx >= 0:
                matched_true.add(best_idx)
                tp += best_points
            if qualified_schema_hit:
                schema_hits += 1
        return tp, schema_hits

    def _build_score(
        self, tp: float, schema_hits: int, pred_count: int, truth_count: int
    ) -> dict[str, float]:
        precision, recall, f1 = self._compute_metrics(tp, pred_count, truth_count)
        schema_completeness = schema_hits / pred_count if pred_count else 0.0
        return {
            "score": max(0.0, min(1.0, f1)),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "schema_completeness": schema_completeness,
        }

    def _empty_truth_score(self, pred_items: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "score": 1.0 if not pred_items else 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "schema_completeness": 1.0 if not pred_items else 0.0,
        }

    def _best_match(
        self,
        pred: dict[str, Any],
        true_items: list[dict[str, Any]],
        matched_true: set[int],
    ) -> tuple[int, float, bool]:
        best_idx = -1
        best_points = 0.0
        best_schema_hit = False
        for idx, truth in enumerate(true_items):
            if idx in matched_true:
                continue
            candidate = self._candidate_match(pred, truth)
            if candidate is None:
                continue
            points, schema_hit = candidate
            if points > best_points:
                best_points = points
                best_idx = idx
                best_schema_hit = schema_hit
        return best_idx, best_points, best_schema_hit

    def _candidate_match(
        self, pred: dict[str, Any], truth: dict[str, Any]
    ) -> tuple[float, bool] | None:
        if not self._is_field_match(pred, truth):
            return None
        points = self._base_points(pred, truth)
        if points <= 0.0:
            return None
        return self._apply_schema_adjustment(points, pred, truth)

    def _is_field_match(self, pred: dict[str, Any], truth: dict[str, Any]) -> bool:
        return bool(
            fuzz.token_sort_ratio(str(pred.get("field", "")), str(truth.get("field", ""))) >= 80
        )

    def _base_points(self, pred: dict[str, Any], truth: dict[str, Any]) -> float:
        severity_pred = str(pred.get("severity", "")).upper()
        severity_true = str(truth.get("severity", "")).upper()
        value_match = self._value_match(pred, truth)
        if severity_pred == severity_true and value_match:
            return 1.0
        if severity_pred != severity_true:
            return 0.5
        return 0.0

    def _value_match(self, pred: dict[str, Any], truth: dict[str, Any]) -> bool:
        return bool(
            fuzz.token_sort_ratio(
                str(pred.get("doc_a_value", "")),
                str(truth.get("doc_a_value", "")),
            )
            >= 75
            or fuzz.token_sort_ratio(
                str(pred.get("doc_b_value", "")),
                str(truth.get("doc_b_value", "")),
            )
            >= 75
        )

    def _apply_schema_adjustment(
        self, points: float, pred: dict[str, Any], truth: dict[str, Any]
    ) -> tuple[float, bool]:
        if not self._has_required_schema(pred):
            return points * 0.7, False
        schema_quality = self._schema_quality(pred, truth)
        adjusted = points * (0.8 + (0.2 * schema_quality))
        return adjusted, schema_quality >= (2 / 3)

    def _has_required_schema(self, pred: dict[str, Any]) -> bool:
        return all(str(pred.get(k, "")).strip() for k in self.REQUIRED_SCHEMA_FIELDS)

    def _schema_quality(self, pred: dict[str, Any], truth: dict[str, Any]) -> float:
        section_protocol_match = (
            fuzz.token_sort_ratio(
                str(pred.get("section_in_protocol", "")),
                str(truth.get("section_in_protocol", "")),
            )
            >= 60
        )
        section_icf_match = (
            fuzz.token_sort_ratio(
                str(pred.get("section_in_icf", "")),
                str(truth.get("section_in_icf", "")),
            )
            >= 60
        )
        regulatory_basis_match = (
            fuzz.token_sort_ratio(
                str(pred.get("regulatory_basis", "")),
                str(truth.get("regulatory_basis", "")),
            )
            >= 50
        )
        return float(section_protocol_match + section_icf_match + regulatory_basis_match) / 3

    def _compute_metrics(
        self, tp: float, pred_count: int, truth_count: int
    ) -> tuple[float, float, float]:
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / truth_count if truth_count else 0.0
        if (precision + recall) == 0:
            return precision, recall, 0.0
        return precision, recall, 2 * precision * recall / (precision + recall)
