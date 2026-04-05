from __future__ import annotations

import pytest

from pharmatrials_env.models import Action, ActionType, QueryPayload


def test_action_payload_dict_is_coerced_to_model() -> None:
    action = Action.model_validate(
        {
            "action_type": "QUERY",
            "payload": {
                "doc_id": "icf_001",
                "question": "What is the protocol number?",
            },
        }
    )
    assert action.payload.__class__.__name__ == "QueryPayload"


def test_action_payload_wrong_model_type_raises() -> None:
    with pytest.raises(ValueError, match="Payload must match action_type"):
        Action(
            action_type=ActionType.EXTRACT,
            payload=QueryPayload(doc_id="icf_001", question="q"),
        )
