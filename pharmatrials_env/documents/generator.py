from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
from faker import Faker
from jinja2 import Environment, FileSystemLoader, select_autoescape

from pharmatrials_env.models import DocumentView
from pharmatrials_env.tasks.registry import TaskSpec


class DocumentGenerator:
    def __init__(self, seed: int, task_spec: TaskSpec) -> None:
        self.seed = seed
        self.task_spec = task_spec
        self.rng = np.random.default_rng(seed)
        self.fake = Faker()
        self.fake.seed_instance(seed)

        base = Path(__file__).resolve().parent
        self.templates = Environment(
            loader=FileSystemLoader(str(base / "templates")),
            autoescape=select_autoescape(
                enabled_extensions=("html", "htm", "xml"),
                default_for_string=True,
                default=False,
            ),
        )
        self.meddra_terms = self._load_json(base / "vocabulary" / "meddra_terms.json")
        self.indications = self._load_json(base / "vocabulary" / "nci_thesaurus.json")

    @staticmethod
    def _load_json(path: Path) -> list[str]:
        with path.open("r", encoding="utf-8") as f:
            return cast(list[str], json.load(f))

    def generate(self) -> tuple[dict[str, DocumentView], dict[str, Any]]:
        common = self._common_fields()
        icf_fields = self._icf_fields(common)
        protocol_fields = self._protocol_fields(common, icf_fields)
        sap_fields = self._sap_fields(common, protocol_fields)
        narratives, narrative_truth, findings_truth = self._ae_narratives(sap_fields)

        docs: dict[str, DocumentView] = {}
        metadata: dict[str, Any] = {
            "ground_truth": {
                k: icf_fields[k]
                for k in [
                    "study_title",
                    "sponsor_name",
                    "protocol_number",
                    "compound_name",
                    "dose_mg",
                    "route_of_administration",
                    "dosing_frequency",
                    "num_visits",
                    "total_duration_weeks",
                    "primary_indication",
                    "irb_name",
                    "compensation_amount_usd",
                ]
            },
            "inconsistencies": self._build_inconsistency_truth(protocol_fields, icf_fields),
            "sap_ae_analysis_summary": {
                "analysis_population": sap_fields["analysis_population"],
                "meddra_coding_level": sap_fields["meddra_coding_level"],
                "ctcae_version": sap_fields["ctcae_version"],
                "sae_definition": sap_fields["sae_definition"],
            },
            "narrative_extractions": narrative_truth,
            "reconciliation_findings": findings_truth,
            "protocol_number": common["protocol_number"],
        }

        if "ICF" in self.task_spec.document_types:
            docs["icf_001"] = self._render_doc(
                "icf_001", "ICF", "Informed Consent Form", "icf.j2", icf_fields
            )
        if "PROTOCOL" in self.task_spec.document_types:
            docs["protocol_001"] = self._render_doc(
                "protocol_001",
                "PROTOCOL",
                "Study Protocol",
                "protocol.j2",
                protocol_fields,
            )
        if "SAP" in self.task_spec.document_types:
            docs["sap_001"] = self._render_doc(
                "sap_001", "SAP", "Statistical Analysis Plan", "sap.j2", sap_fields
            )
        if "AE_NARRATIVE" in self.task_spec.document_types:
            docs.update(narratives)

        return docs, metadata

    def _render_doc(
        self,
        doc_id: str,
        doc_type: str,
        title: str,
        template_name: str,
        fields: dict[str, Any],
    ) -> DocumentView:
        content = self.templates.get_template(template_name).render(**fields)
        sections = self._split_sections(content)
        return DocumentView(
            doc_id=doc_id,
            doc_type=doc_type,
            title=title,
            content=content,
            sections=sections,
            version="1.0",
            word_count=len(content.split()),
        )

    def _split_sections(self, content: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        current = "Document"
        bucket: list[str] = []
        for line in content.splitlines():
            if line.strip() and line.strip() == line and ":" not in line and len(line.split()) <= 5:
                if bucket:
                    sections[current] = "\n".join(bucket).strip()
                current = line.strip()
                bucket = []
            else:
                bucket.append(line)
        if bucket:
            sections[current] = "\n".join(bucket).strip()
        return sections

    def _common_fields(self) -> dict[str, Any]:
        indication = str(self.rng.choice(self.indications))
        protocol_number = (
            f"SYN-{int(self.rng.integers(2024, 2028))}-{int(self.rng.integers(1, 999)):03d}-Ph3"
        )
        return {
            "study_title": f"A Phase 3 Study of SYN-{int(self.rng.integers(100, 999))} in {indication}",
            "sponsor_name": f"{self.fake.company()} Biopharma, Inc.",
            "protocol_number": protocol_number,
            "phase": "Phase 3",
            "primary_indication": indication,
            "compound_name": f"SYN-{int(self.rng.integers(100, 999))}",
        }

    def _compose_paragraphs(self, topic: str, count: int, sentence_count: int = 5) -> list[str]:
        fragments = [
            "The study team documents source data and protocol compliance activities.",
            "All trial conduct follows ICH E6(R3) principles and local regulations.",
            "Clinical safety observations are coded using approved terminology standards.",
            "Operational procedures are monitored for quality and consistency at each site.",
            "Regulatory inspection readiness is maintained through traceable records.",
            "Investigators evaluate participant status and adverse event progression.",
            "Data clarification and reconciliation are performed with auditability controls.",
            "Endpoints are assessed under prespecified definitions and analysis conventions.",
            "Protocol deviations are reviewed with corrective and preventive measures.",
            "Confidentiality obligations are maintained through secure data handling practices.",
        ]
        paragraphs: list[str] = []
        for i in range(count):
            selected = self.rng.choice(fragments, size=sentence_count, replace=True).tolist()
            paragraphs.append(f"{topic} Section {i + 1}. " + " ".join(selected))
        return paragraphs

    def _icf_fields(self, common: dict[str, Any]) -> dict[str, Any]:
        num_visits = int(self.rng.integers(10, 16))
        duration = int(self.rng.choice([36, 48, 52, 60]))
        dose = int(self.rng.choice([100, 150, 200, 300]))
        route = str(self.rng.choice(["oral", "intravenous", "subcutaneous"]))
        freq = str(self.rng.choice(["once daily", "once weekly", "twice weekly"]))
        risk_list = list(self.rng.choice(self.meddra_terms, size=6, replace=False))
        procedures_per_visit = {
            f"Visit {i}": [
                "symptom review",
                "vital signs",
                "laboratory panel",
                "drug accountability",
            ]
            for i in range(1, num_visits + 1)
        }

        return {
            **common,
            "study_rationale": "To evaluate safety and efficacy under ICH E6(R3) compliant trial conduct.",
            "mechanism_of_action_summary": "The investigational product modulates inflammatory signaling pathways relevant to disease progression.",
            "primary_endpoint": f"Change from baseline in {common['primary_indication']} severity index",
            "visit_schedule": [f"Visit {i}" for i in range(1, num_visits + 1)],
            "procedures_per_visit": procedures_per_visit,
            "total_duration_weeks": duration,
            "num_visits": num_visits,
            "dose_mg": dose,
            "route_of_administration": route,
            "dosing_frequency": freq,
            "risk_list": risk_list,
            "serious_risk_list": risk_list[:2],
            "risk_probability_qualifiers": ["common", "uncommon", "rare"],
            "data_sharing_entities": ["Sponsor", "CRO", "Regulatory Authority"],
            "hipaa_reference": "HIPAA 45 CFR 160/164",
            "gdpr_reference": "GDPR Article 9",
            "withdrawal_rights_statement": "You may withdraw at any time without penalty.",
            "no_penalty_clause": "No penalty or loss of benefits for withdrawal.",
            "pi_name": self.fake.name(),
            "pi_phone": self.fake.phone_number(),
            "emergency_contact": self.fake.phone_number(),
            "irb_name": "Western Institutional Review Board (WIRB)",
            "irb_phone": self.fake.phone_number(),
            "compensation_amount_usd": int(self.rng.choice([0, 50, 75, 100])),
            "injury_compensation_policy": "Medical treatment for injury will be available per local policy.",
            "num_signature_lines": 3,
            "witness_required": bool(self.rng.choice([True, False])),
            "icf_long_sections": self._compose_paragraphs(
                "ICF Detailed Narrative", count=28, sentence_count=6
            ),
        }

    def _protocol_fields(self, common: dict[str, Any], icf: dict[str, Any]) -> dict[str, Any]:
        fields = self._base_protocol_fields(common, icf)

        if self.task_spec.task_id in {"MEDIUM", "HARD"}:
            self._apply_protocol_inconsistencies(fields, common, icf)
        return fields

    def _base_protocol_fields(self, common: dict[str, Any], icf: dict[str, Any]) -> dict[str, Any]:
        return {
            **common,
            "primary_endpoint": icf["primary_endpoint"],
            "num_visits": icf["num_visits"],
            "total_duration_weeks": icf["total_duration_weeks"],
            "dose_mg": icf["dose_mg"],
            "route_of_administration": icf["route_of_administration"],
            "risk_list": list(icf["risk_list"]),
            "compensation_amount_usd": icf["compensation_amount_usd"],
            "irb_name": icf["irb_name"],
            "irb_phone": icf["irb_phone"],
            "inclusion_criteria": [
                "Adult patient",
                "Signed ICF",
                "Confirmed diagnosis",
            ],
            "exclusion_criteria": [
                "Recent investigational therapy",
                "Uncontrolled infection",
                "Pregnancy",
            ],
            "data_safety_monitoring_plan": "Independent DMC review every 12 weeks.",
            "randomisation_ratio": "1:1",
            "blinding_type": "Double-blind",
            "stratification_factors": ["Region", "Baseline severity"],
            "study_objectives": {
                "primary": "Assess treatment effect on primary disease severity endpoint.",
                "secondary": "Characterize safety profile and key secondary efficacy outcomes.",
                "exploratory": "Investigate biomarker-response relationships.",
            },
            "endpoint_definitions": {
                "primary_endpoint": icf["primary_endpoint"],
                "secondary_endpoints": [
                    "Treatment-emergent adverse events incidence",
                    "Change in quality-of-life score",
                ],
            },
            "amendment_history": [
                "Version 1.0: Initial protocol.",
                "Version 1.1: Clarified eligibility and safety language.",
            ],
            "protocol_long_sections": self._compose_paragraphs(
                "Protocol Operational Guidance", count=120, sentence_count=6
            ),
        }

    def _apply_protocol_inconsistencies(
        self, fields: dict[str, Any], common: dict[str, Any], icf: dict[str, Any]
    ) -> None:
        chosen = self._select_protocol_inconsistency_categories()
        if "visit_count" in chosen:
            fields["num_visits"] = max(2, icf["num_visits"] + int(self.rng.integers(1, 4)))
        if "duration" in chosen:
            fields["total_duration_weeks"] = icf["total_duration_weeks"] + int(
                self.rng.integers(2, 8)
            )
        if "dose" in chosen:
            fields["dose_mg"] = int(icf["dose_mg"] * int(self.rng.choice([2, 3])))
        if "route" in chosen:
            fields["route_of_administration"] = self._alternate_route(
                str(icf["route_of_administration"])
            )
        if "risk_omission" in chosen:
            self._inject_risk_omission(fields, icf)
        if "compensation" in chosen:
            fields["compensation_amount_usd"] = icf["compensation_amount_usd"] + int(
                self.rng.choice([5, 10, 15])
            )
        if "irb_reference" in chosen:
            fields["irb_phone"] = self.fake.phone_number()
        if "endpoint_term" in chosen:
            fields["primary_endpoint"] = (
                f"Percent change from baseline in {common['primary_indication']} severity score"
            )

    def _select_protocol_inconsistency_categories(self) -> set[str]:
        categories = [
            "visit_count",
            "duration",
            "dose",
            "route",
            "risk_omission",
            "compensation",
            "irb_reference",
            "endpoint_term",
        ]
        injected_count = int(self.rng.integers(5, 9))
        return set(self.rng.choice(categories, size=injected_count, replace=False).tolist())

    @staticmethod
    def _alternate_route(route: str) -> str:
        return "intravenous" if route != "intravenous" else "oral"

    def _inject_risk_omission(self, fields: dict[str, Any], icf: dict[str, Any]) -> None:
        extra_terms = [t for t in self.meddra_terms if t not in icf["risk_list"]]
        if extra_terms:
            fields["risk_list"] = list(fields["risk_list"]) + [str(self.rng.choice(extra_terms))]

    def _sap_fields(self, common: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
        return {
            "protocol_number": common["protocol_number"],
            "primary_analysis_method": "MMRM with treatment, visit, and interaction terms",
            "covariate_list": ["baseline score", "region", "age"],
            "multiplicity_adjustment_method": "Hierarchical gatekeeping",
            "sensitivity_analysis_descriptions": [
                "Pattern-mixture model",
                "Tipping point analysis",
            ],
            "missing_data_strategy": "Multiple imputation under MAR",
            "analysis_population": "Intent-to-treat",
            "meddra_coding_level": "PT",
            "ctcae_version": "v5.0",
            "sae_definition": "Any event resulting in death, life-threatening condition, hospitalization, disability, or congenital anomaly.",
            "sap_endpoint_list": [
                protocol["primary_endpoint"],
                "Treatment-emergent adverse events",
            ],
            "sap_long_sections": self._compose_paragraphs(
                "SAP Statistical Methodology", count=72, sentence_count=6
            ),
        }

    def _ae_narratives(
        self, sap_fields: dict[str, Any]
    ) -> tuple[dict[str, DocumentView], list[dict[str, Any]], list[dict[str, Any]]]:
        docs: dict[str, DocumentView] = {}
        truth_rows: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        count = self._ae_narrative_count()
        for i in range(1, count + 1):
            row, meets_sae = self._build_narrative_row(i)
            narrative_id = str(row["narrative_id"])
            view = self._render_doc(
                narrative_id,
                "AE_NARRATIVE",
                f"AE Narrative {i}",
                "ae_narrative.j2",
                row,
            )
            docs[narrative_id] = view
            truth_rows.append(self._narrative_truth_row(row, meets_sae))
            self._append_narrative_findings(findings, row, narrative_id, meets_sae)

        if not findings and truth_rows:
            findings.append(self._default_narrative_finding(truth_rows, sap_fields))

        return docs, truth_rows, findings

    def _ae_narrative_count(self) -> int:
        if self.task_spec.task_id != "HARD":
            return 0
        return int(self.rng.integers(8, 16))

    def _build_narrative_row(self, index: int) -> tuple[dict[str, Any], bool]:
        ae_term = str(self.rng.choice(self.meddra_terms))
        severity = int(self.rng.integers(1, 6))
        onset = date(2025, 1, 1) + timedelta(days=int(self.rng.integers(0, 120)))
        resolution = onset + timedelta(days=int(self.rng.integers(2, 30)))
        meets_sae = severity >= 4
        sae_flag = (
            "No" if (meets_sae and self.rng.random() < 0.35) else ("Yes" if meets_sae else "No")
        )
        row = {
            "narrative_id": f"ae_{index:03d}",
            "patient_id": f"PT-{int(self.rng.integers(1000, 9999))}",
            "ae_term": ae_term,
            "onset_date": onset.isoformat(),
            "resolution_date": resolution.isoformat(),
            "severity_grade": severity,
            "causality_assessment": str(
                self.rng.choice(["Related", "Possibly related", "Unrelated"])
            ),
            "action_taken": str(self.rng.choice(["Dose reduced", "Drug interrupted", "No change"])),
            "outcome": str(self.rng.choice(["Recovered", "Recovering", "Not recovered"])),
            "sae_flag": sae_flag,
            "narrative_long_sections": self._compose_paragraphs(
                "Narrative Clinical Course", count=8, sentence_count=5
            ),
        }
        return row, meets_sae

    def _narrative_truth_row(self, row: dict[str, Any], meets_sae: bool) -> dict[str, Any]:
        return {
            "narrative_id": str(row["narrative_id"]),
            "ae_term_reported": row["ae_term"],
            "ae_term_coded_pt": row["ae_term"],
            "ae_term_coded_soc": "General disorders and administration site conditions",
            "severity_grade_reported": row["severity_grade"],
            "causality": row["causality_assessment"],
            "onset_date": row["onset_date"],
            "resolution_date": row["resolution_date"],
            "action_taken": row["action_taken"],
            "outcome": row["outcome"],
            "meets_sae_criteria": meets_sae,
        }

    def _append_narrative_findings(
        self,
        findings: list[dict[str, Any]],
        row: dict[str, Any],
        narrative_id: str,
        meets_sae: bool,
    ) -> None:
        if meets_sae and row["sae_flag"] == "No":
            findings.append(
                {
                    "narrative_id": narrative_id,
                    "finding_type": "UNDETECTED_SAE",
                    "description": "Narrative meets SAE criteria but SAE flag is not set.",
                    "regulatory_impact": "CRITICAL",
                }
            )
            return
        severity = int(row["severity_grade"])
        if severity == 5 and self.rng.random() < 0.2:
            findings.append(
                {
                    "narrative_id": narrative_id,
                    "finding_type": "CTCAE_GRADE_INCONSISTENCY",
                    "description": "Narrative grade encoding conflicts with SAP CTCAE expectations.",
                    "regulatory_impact": "HIGH",
                }
            )

    def _default_narrative_finding(
        self, truth_rows: list[dict[str, Any]], sap_fields: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "narrative_id": truth_rows[0]["narrative_id"],
            "finding_type": "CTCAE_GRADE_INCONSISTENCY",
            "description": f"Grade representation must follow CTCAE {sap_fields['ctcae_version']}.",
            "regulatory_impact": "MEDIUM",
        }

    def _build_inconsistency_truth(
        self, protocol: dict[str, Any], icf: dict[str, Any]
    ) -> list[dict[str, Any]]:
        truth: list[dict[str, Any]] = []
        for field, severity in self._inconsistency_field_specs():
            left = protocol.get(field)
            right = icf.get(field)
            is_diff, left_repr, right_repr = self._inconsistency_values(field, left, right)

            if is_diff:
                truth.append(
                    {
                        "field": field,
                        "doc_a_value": left_repr,
                        "doc_b_value": right_repr,
                        "section_in_protocol": self._section_in_protocol(field),
                        "section_in_icf": self._section_in_icf(field),
                        "severity": severity,
                        "regulatory_basis": "ICH E6(R3) consistency requirement",
                    }
                )
        return truth

    @staticmethod
    def _inconsistency_field_specs() -> list[tuple[str, str]]:
        return [
            ("num_visits", "ERROR"),
            ("total_duration_weeks", "ERROR"),
            ("dose_mg", "ERROR"),
            ("route_of_administration", "ERROR"),
            ("risk_list", "ERROR"),
            ("compensation_amount_usd", "WARNING"),
            ("irb_phone", "WARNING"),
            ("primary_endpoint", "INFO"),
        ]

    def _inconsistency_values(self, field: str, left: Any, right: Any) -> tuple[bool, str, str]:
        if field == "risk_list":
            is_diff = set(str(x) for x in left or []) != set(str(x) for x in right or [])
            left_repr = ", ".join(str(x) for x in left or [])
            right_repr = ", ".join(str(x) for x in right or [])
            return is_diff, left_repr, right_repr
        left_repr = str(left)
        right_repr = str(right)
        return left_repr != right_repr, left_repr, right_repr

    @staticmethod
    def _section_in_protocol(field: str) -> str:
        if field == "risk_list":
            return "Safety and Monitoring"
        if field in {"irb_phone", "compensation_amount_usd"}:
            return "Compensation and Contacts"
        return "Study Design"

    @staticmethod
    def _section_in_icf(field: str) -> str:
        if field == "risk_list":
            return "Potential Risks"
        if field == "irb_phone":
            return "Contact Information"
        return "Study Procedures"
