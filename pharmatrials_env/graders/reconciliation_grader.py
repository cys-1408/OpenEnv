from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz

from .base import AbstractGrader


class ReconciliationGrader(AbstractGrader):
    def generate_ground_truth(
        self, documents: dict[str, Any], metadata: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "sap_ae_analysis_summary": metadata.get("sap_ae_analysis_summary", {}),
            "narrative_extractions": metadata.get("narrative_extractions", []),
            "reconciliation_findings": metadata.get("reconciliation_findings", []),
        }

    def score(self, submission: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, Any]:
        answer = submission.get("answer", {})
        if not answer:
            return {
                "score": 0.0,
                "sap_score": 0.0,
                "narrative_score": 0.0,
                "coding_score": 0.0,
                "findings_f1": 0.0,
                "impact_score": 0.0,
            }

        sap_score = self._sap_score(
            answer.get("sap_ae_analysis_summary", {}),
            ground_truth.get("sap_ae_analysis_summary", {}),
        )
        narrative_score = self._narrative_score(
            answer.get("narrative_extractions", []),
            ground_truth.get("narrative_extractions", []),
        )
        coding_score = self._coding_score(
            answer.get("narrative_extractions", []),
            ground_truth.get("narrative_extractions", []),
        )
        findings_score, impact_score = self._findings_score(
            answer.get("reconciliation_findings", []),
            ground_truth.get("reconciliation_findings", []),
        )

        total = (
            0.20 * sap_score
            + 0.30 * narrative_score
            + 0.20 * coding_score
            + 0.20 * findings_score
            + 0.10 * impact_score
        )
        return {
            "score": max(0.0, min(1.0, total)),
            "sap_score": sap_score,
            "narrative_score": narrative_score,
            "coding_score": coding_score,
            "findings_f1": findings_score,
            "impact_score": impact_score,
        }

    def _sap_score(self, pred: dict[str, Any], truth: dict[str, Any]) -> float:
        fields = [
            "analysis_population",
            "meddra_coding_level",
            "ctcae_version",
            "sae_definition",
        ]
        if not truth:
            return 0.0
        scores: list[float] = []
        for field in fields:
            scores.append(
                fuzz.token_sort_ratio(str(pred.get(field, "")), str(truth.get(field, ""))) / 100.0
            )
        return sum(scores) / len(scores)

    def _narrative_score(
        self, pred_items: list[dict[str, Any]], true_items: list[dict[str, Any]]
    ) -> float:
        if not true_items:
            return 0.0
        by_id_true = {str(item.get("narrative_id")): item for item in true_items}
        score_sum = 0.0
        fields = [
            "ae_term_reported",
            "severity_grade_reported",
            "causality",
            "onset_date",
            "resolution_date",
            "action_taken",
            "outcome",
            "meets_sae_criteria",
        ]
        for pred in pred_items:
            nid = str(pred.get("narrative_id", ""))
            truth = by_id_true.get(nid)
            if not truth:
                continue
            local = 0.0
            for field in fields:
                if isinstance(truth.get(field), (int, bool)):
                    local += 1.0 if pred.get(field) == truth.get(field) else 0.0
                else:
                    local += (
                        fuzz.token_sort_ratio(str(pred.get(field, "")), str(truth.get(field, "")))
                        / 100.0
                    )
            score_sum += local / len(fields)
        return score_sum / len(true_items)

    def _coding_score(
        self, pred_items: list[dict[str, Any]], true_items: list[dict[str, Any]]
    ) -> float:
        if not true_items:
            return 0.0
        by_id_true = {str(item.get("narrative_id")): item for item in true_items}
        score_sum = 0.0
        for pred in pred_items:
            nid = str(pred.get("narrative_id", ""))
            truth = by_id_true.get(nid)
            if not truth:
                continue
            pt = (
                fuzz.token_sort_ratio(
                    str(pred.get("ae_term_coded_pt", "")),
                    str(truth.get("ae_term_coded_pt", "")),
                )
                / 100.0
            )
            soc = (
                fuzz.token_sort_ratio(
                    str(pred.get("ae_term_coded_soc", "")),
                    str(truth.get("ae_term_coded_soc", "")),
                )
                / 100.0
            )
            score_sum += (pt + soc) / 2
        return score_sum / len(true_items)

    def _findings_score(
        self, pred_items: list[dict[str, Any]], true_items: list[dict[str, Any]]
    ) -> tuple[float, float]:
        if not true_items:
            return 0.0, 0.0
        matched: set[int] = set()
        tp = 0
        impact_tp = 0
        for pred in pred_items:
            for i, truth in enumerate(true_items):
                if i in matched:
                    continue
                if str(pred.get("narrative_id")) != str(truth.get("narrative_id")):
                    continue
                if str(pred.get("finding_type")) != str(truth.get("finding_type")):
                    continue
                matched.add(i)
                tp += 1
                if str(pred.get("regulatory_impact")) == str(truth.get("regulatory_impact")):
                    impact_tp += 1
                break
        precision = tp / len(pred_items) if pred_items else 0.0
        recall = tp / len(true_items)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        impact = impact_tp / len(true_items)
        return f1, impact
