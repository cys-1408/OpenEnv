from __future__ import annotations

from pharmatrials_env.graders.consistency_grader import ConsistencyGrader
from pharmatrials_env.models import ActionType, RewardWeights
from pharmatrials_env.tasks.registry import TaskSpec

MEDIUM_TASK = TaskSpec(
    task_id="MEDIUM",
    task_name="PROTOCOL_ICF_CONSISTENCY",
    instruction_template=(
        "You are reviewing protocol/ICF consistency for protocol {protocol_number}. "
        "Identify all inconsistencies, assign severity "
        "(INFO/WARNING/ERROR), and cite regulatory basis. "
        "You have {max_steps} steps."
    ),
    max_steps=25,
    allowed_actions={
        ActionType.EXTRACT,
        ActionType.COMPARE,
        ActionType.ANNOTATE,
        ActionType.QUERY,
        ActionType.SUBMIT,
    },
    reward_weights=RewardWeights(
        accuracy=0.40, completeness=0.30, regulatory_alignment=0.20, efficiency=0.10
    ),
    required_output_items=[
        "num_visits",
        "total_duration_weeks",
        "dose_mg",
        "route_of_administration",
        "risk_list",
        "compensation_amount_usd",
        "irb_phone",
        "primary_endpoint",
    ],
    grader=ConsistencyGrader(),
    document_types=["PROTOCOL", "ICF"],
)
