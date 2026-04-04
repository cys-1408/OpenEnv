from __future__ import annotations

import pytest

from pharmatrials_env import PharmaTrialsEnv
from pharmatrials_env.models import Action


def test_reset_returns_observation() -> None:
    env = PharmaTrialsEnv()
    obs = env.reset()
    assert obs.step_number == 0
    assert obs.done is False
    assert obs.documents


def test_reset_seed_is_deterministic() -> None:
    env = PharmaTrialsEnv()
    obs1 = env.reset(task_id="EASY", seed=42)
    obs2 = env.reset(task_id="EASY", seed=42)
    assert obs1.documents["icf_001"].content == obs2.documents["icf_001"].content


def test_extract_action_valid() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    action = Action.model_validate(
        {
            "action_type": "EXTRACT",
            "payload": {
                "doc_id": "icf_001",
                "fields": ["protocol_number"],
                "section_hint": None,
            },
        }
    )
    obs, reward, done, _info = env.step(action)
    assert obs.last_action_result is not None
    assert obs.last_action_result.success is True
    assert reward.total >= 0.0
    assert done is False


def test_submit_ends_episode() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    submit = Action.model_validate(
        {
            "action_type": "SUBMIT",
            "payload": {"answer": {}, "confidence": 0.5},
        }
    )
    _obs, _reward, done, _info = env.step(submit)
    assert done is True


def test_three_invalid_actions_terminate() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    invalid = Action.model_validate(
        {
            "action_type": "COMPARE",
            "payload": {
                "doc_id_a": "icf_001",
                "doc_id_b": "icf_001",
                "comparison_fields": ["dose_mg"],
                "section_hint_a": None,
                "section_hint_b": None,
            },
        }
    )
    for _ in range(2):
        _obs, _reward, done, _info = env.step(invalid)
        assert done is False
    _obs, _reward, done, _info = env.step(invalid)
    assert done is True


def test_state_reflects_changes() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    extract = Action.model_validate(
        {
            "action_type": "EXTRACT",
            "payload": {
                "doc_id": "icf_001",
                "fields": ["protocol_number", "dose_mg"],
                "section_hint": None,
            },
        }
    )
    env.step(extract)
    state = env.state()
    assert state.step_number == 1
    assert state.extraction_outputs


def test_max_steps_termination_without_submit() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    act = Action.model_validate(
        {
            "action_type": "QUERY",
            "payload": {"doc_id": "icf_001", "question": "Protocol number?"},
        }
    )
    done = False
    for _ in range(15):
        _obs, _reward, done, _info = env.step(act)
    assert done is True


def test_reset_invalid_task_raises() -> None:
    env = PharmaTrialsEnv()
    with pytest.raises(ValueError):
        env.reset(task_id="UNKNOWN", seed=1)


def test_medium_compare_and_annotate_paths() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="MEDIUM", seed=42)
    compare = Action.model_validate(
        {
            "action_type": "COMPARE",
            "payload": {
                "doc_id_a": "protocol_001",
                "doc_id_b": "icf_001",
                "comparison_fields": [
                    "num_visits",
                    "dose_mg",
                    "route_of_administration",
                ],
                "section_hint_a": "Study Design",
                "section_hint_b": "Study Procedures",
            },
        }
    )
    obs, _reward, _done, _info = env.step(compare)
    assert obs.last_action_result is not None
    assert obs.last_action_result.success is True
    inconsistencies = obs.last_action_result.output.get("inconsistencies", [])
    for row in inconsistencies:
        assert "section_in_protocol" in row
        assert "section_in_icf" in row
        assert "regulatory_basis" in row

    annotate = Action.model_validate(
        {
            "action_type": "ANNOTATE",
            "payload": {
                "doc_id": "icf_001",
                "section": "Study Procedures",
                "label": "CHECK",
                "note": "Review mismatch",
                "severity": "WARNING",
            },
        }
    )
    _obs2, _reward2, _done2, _info2 = env.step(annotate)
    assert len(env.state().annotations) == 1


def test_hard_summarise_and_submit_paths() -> None:
    env = PharmaTrialsEnv()
    obs = env.reset(task_id="HARD", seed=42)
    sap_id = next(k for k, v in obs.documents.items() if v.doc_type == "SAP")
    summarise = Action.model_validate(
        {
            "action_type": "SUMMARISE",
            "payload": {"doc_id": sap_id, "focus_areas": ["ae"], "max_words": 120},
        }
    )
    obs2, _r1, _d1, _i1 = env.step(summarise)
    assert obs2.last_action_result is not None
    assert obs2.last_action_result.success is True

    submit = Action.model_validate(
        {"action_type": "SUBMIT", "payload": {"answer": {}, "confidence": 0.2}}
    )
    _obs3, _r2, done, _i2 = env.step(submit)
    assert done is True


def test_step_after_done_returns_unsuccessful_action_result() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    submit = Action.model_validate(
        {"action_type": "SUBMIT", "payload": {"answer": {}, "confidence": 0.5}}
    )
    env.step(submit)
    obs, _reward, done, _info = env.step(submit)
    assert done is True
    assert obs.last_action_result is not None
    assert obs.last_action_result.success is False
