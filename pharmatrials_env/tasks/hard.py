from __future__ import annotations

from pharmatrials_env.graders.reconciliation_grader import ReconciliationGrader
from pharmatrials_env.models import ActionType, RewardWeights
from pharmatrials_env.tasks.registry import TaskSpec

HARD_TASK = TaskSpec(
    task_id="HARD",
    task_name="SAP_AE_RECONCILIATION",
    instruction_template=(
        "Reconcile AE narratives with SAP/Protocol for protocol {protocol_number}. "
        "Extract SAP AE analysis rules, normalize MedDRA coding, identify CTCAE and SAE issues, "
        "then submit a structured reconciliation report. Max steps: {max_steps}."
    ),
    max_steps=40,
    allowed_actions={
        ActionType.EXTRACT,
        ActionType.COMPARE,
        ActionType.ANNOTATE,
        ActionType.QUERY,
        ActionType.SUMMARISE,
        ActionType.SUBMIT,
    },
    reward_weights=RewardWeights(
        accuracy=0.40, completeness=0.30, regulatory_alignment=0.20, efficiency=0.10
    ),
    required_output_items=[
        "analysis_population",
        "meddra_coding_level",
        "ctcae_version",
        "sae_definition",
        "narrative_extractions",
        "reconciliation_findings",
    ],
    grader=ReconciliationGrader(),
    document_types=["SAP", "PROTOCOL", "AE_NARRATIVE"],
)
