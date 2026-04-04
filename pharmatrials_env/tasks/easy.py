from __future__ import annotations

from pharmatrials_env.graders.icf_grader import ICFExtractionGrader
from pharmatrials_env.models import ActionType, RewardWeights
from pharmatrials_env.tasks.registry import TaskSpec

EASY_REQUIRED_FIELDS = [
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

EASY_TASK = TaskSpec(
    task_id="EASY",
    task_name="ICF_EXTRACTION",
    instruction_template=(
        "You are a clinical-trial regulatory specialist. Extract the 12 required fields "
        "from the ICF and submit as JSON. Protocol number: {protocol_number}. "
        "You have {max_steps} steps available."
    ),
    max_steps=15,
    allowed_actions={ActionType.EXTRACT, ActionType.QUERY, ActionType.SUBMIT},
    reward_weights=RewardWeights(
        accuracy=0.50, completeness=0.30, regulatory_alignment=0.10, efficiency=0.10
    ),
    required_output_items=EASY_REQUIRED_FIELDS,
    grader=ICFExtractionGrader(),
    document_types=["ICF"],
)
