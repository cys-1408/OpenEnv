from __future__ import annotations

from pharmatrials_env.graders.consistency_grader import ConsistencyGrader
from pharmatrials_env.graders.icf_grader import ICFExtractionGrader
from pharmatrials_env.graders.reconciliation_grader import ReconciliationGrader


def test_icf_grader_perfect_score() -> None:
    grader = ICFExtractionGrader()
    gt = {"study_title": "Title", "dose_mg": 200}
    submission = {"answer": {"study_title": "Title", "dose_mg": 200}}
    out = grader.score(submission, gt)
    assert out["score"] == 1.0


def test_icf_grader_half_correct_is_partial() -> None:
    grader = ICFExtractionGrader()
    gt = {
        "study_title": "Correct Title",
        "sponsor_name": "Correct Sponsor",
        "dose_mg": 200,
        "num_visits": 12,
    }
    submission = {
        "answer": {
            "study_title": "Correct Title",
            "sponsor_name": "Wrong Sponsor",
            "dose_mg": 200,
            "num_visits": 5,
        }
    }
    out = grader.score(submission, gt)
    assert 0.3 <= out["score"] <= 0.8


def test_consistency_grader_partial() -> None:
    grader = ConsistencyGrader()
    gt = {
        "inconsistencies": [
            {
                "field": "dose_mg",
                "severity": "ERROR",
                "doc_a_value": "200",
                "doc_b_value": "100",
                "section_in_protocol": "Study Design",
                "section_in_icf": "Study Procedures",
                "regulatory_basis": "ICH E6(R3) consistency requirement",
            },
            {
                "field": "num_visits",
                "severity": "ERROR",
                "doc_a_value": "12",
                "doc_b_value": "10",
                "section_in_protocol": "Study Design",
                "section_in_icf": "Study Procedures",
                "regulatory_basis": "ICH E6(R3) consistency requirement",
            },
        ]
    }
    submission = {
        "answer": {
            "inconsistencies": [
                {
                    "field": "dose_mg",
                    "severity": "ERROR",
                    "doc_a_value": "200",
                    "doc_b_value": "100",
                    "section_in_protocol": "Study Design",
                    "section_in_icf": "Study Procedures",
                    "regulatory_basis": "ICH E6(R3) consistency requirement",
                }
            ]
        }
    }
    out = grader.score(submission, gt)
    assert 0.0 < out["score"] < 1.0


def test_consistency_grader_exact_all_is_one() -> None:
    grader = ConsistencyGrader()
    gt = {
        "inconsistencies": [
            {
                "field": "dose_mg",
                "severity": "ERROR",
                "doc_a_value": "200",
                "doc_b_value": "100",
                "section_in_protocol": "Study Design",
                "section_in_icf": "Study Procedures",
                "regulatory_basis": "ICH E6(R3) consistency requirement",
            },
            {
                "field": "num_visits",
                "severity": "ERROR",
                "doc_a_value": "12",
                "doc_b_value": "10",
                "section_in_protocol": "Study Design",
                "section_in_icf": "Study Procedures",
                "regulatory_basis": "ICH E6(R3) consistency requirement",
            },
        ]
    }
    submission = {
        "answer": {
            "inconsistencies": [
                {
                    "field": "dose_mg",
                    "severity": "ERROR",
                    "doc_a_value": "200",
                    "doc_b_value": "100",
                    "section_in_protocol": "Study Design",
                    "section_in_icf": "Study Procedures",
                    "regulatory_basis": "ICH E6(R3) consistency requirement",
                },
                {
                    "field": "num_visits",
                    "severity": "ERROR",
                    "doc_a_value": "12",
                    "doc_b_value": "10",
                    "section_in_protocol": "Study Design",
                    "section_in_icf": "Study Procedures",
                    "regulatory_basis": "ICH E6(R3) consistency requirement",
                },
            ]
        }
    }
    out = grader.score(submission, gt)
    assert out["score"] == 1.0


def test_consistency_grader_penalizes_missing_schema_fields() -> None:
    grader = ConsistencyGrader()
    gt = {
        "inconsistencies": [
            {
                "field": "dose_mg",
                "severity": "ERROR",
                "doc_a_value": "200",
                "doc_b_value": "100",
                "section_in_protocol": "Study Design",
                "section_in_icf": "Study Procedures",
                "regulatory_basis": "ICH E6(R3) consistency requirement",
            }
        ]
    }
    submission_missing_schema = {
        "answer": {
            "inconsistencies": [
                {
                    "field": "dose_mg",
                    "severity": "ERROR",
                    "doc_a_value": "200",
                    "doc_b_value": "100",
                }
            ]
        }
    }
    out = grader.score(submission_missing_schema, gt)
    assert out["score"] < 1.0


def test_reconciliation_grader_range() -> None:
    grader = ReconciliationGrader()
    out = grader.score(
        {"answer": {}},
        {
            "sap_ae_analysis_summary": {},
            "narrative_extractions": [],
            "reconciliation_findings": [],
        },
    )
    assert 0.0 <= out["score"] <= 1.0


def test_all_graders_empty_submission_zero_like() -> None:
    icf = ICFExtractionGrader().score({"answer": {}}, {"study_title": "A"})
    con = ConsistencyGrader().score(
        {"answer": {"inconsistencies": []}},
        {"inconsistencies": [{"field": "x", "severity": "ERROR"}]},
    )
    rec = ReconciliationGrader().score(
        {"answer": {}},
        {
            "sap_ae_analysis_summary": {"analysis_population": "ITT"},
            "narrative_extractions": [{"narrative_id": "ae_001"}],
            "reconciliation_findings": [
                {
                    "narrative_id": "ae_001",
                    "finding_type": "UNDETECTED_SAE",
                    "regulatory_impact": "HIGH",
                }
            ],
        },
    )
    assert icf["score"] == 0.0
    assert con["score"] == 0.0
    assert rec["score"] == 0.0


def test_reconciliation_grader_full_path_scoring() -> None:
    grader = ReconciliationGrader()
    gt = {
        "sap_ae_analysis_summary": {
            "analysis_population": "Intent-to-treat",
            "meddra_coding_level": "PT",
            "ctcae_version": "v5.0",
            "sae_definition": "hospitalization",
        },
        "narrative_extractions": [
            {
                "narrative_id": "ae_001",
                "ae_term_reported": "Headache",
                "ae_term_coded_pt": "Headache",
                "ae_term_coded_soc": "General disorders and administration site conditions",
                "severity_grade_reported": 2,
                "causality": "Related",
                "onset_date": "2025-01-01",
                "resolution_date": "2025-01-05",
                "action_taken": "No change",
                "outcome": "Recovered",
                "meets_sae_criteria": False,
            }
        ],
        "reconciliation_findings": [
            {
                "narrative_id": "ae_001",
                "finding_type": "CTCAE_GRADE_INCONSISTENCY",
                "description": "Mismatch",
                "regulatory_impact": "MEDIUM",
            }
        ],
    }
    submission = {
        "answer": {
            "sap_ae_analysis_summary": gt["sap_ae_analysis_summary"],
            "narrative_extractions": gt["narrative_extractions"],
            "reconciliation_findings": gt["reconciliation_findings"],
        }
    }
    out = grader.score(submission, gt)
    assert 0.8 <= out["score"] <= 1.0
