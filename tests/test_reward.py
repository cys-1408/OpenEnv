from __future__ import annotations

from pharmatrials_env import PharmaTrialsEnv
from pharmatrials_env.models import Action


def test_reward_total_in_range() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    act = Action.model_validate(
        {
            "action_type": "QUERY",
            "payload": {"doc_id": "icf_001", "question": "Protocol number?"},
        }
    )
    _obs, reward, _done, _info = env.step(act)
    assert 0.0 <= reward.total <= 1.0


def test_efficiency_zero_for_non_submit() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    act = Action.model_validate(
        {
            "action_type": "QUERY",
            "payload": {"doc_id": "icf_001", "question": "Protocol number?"},
        }
    )
    _obs, reward, _done, _info = env.step(act)
    assert reward.efficiency == 0.0


def test_completeness_monotonic() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    a1 = Action.model_validate(
        {
            "action_type": "EXTRACT",
            "payload": {
                "doc_id": "icf_001",
                "fields": ["study_title"],
                "section_hint": None,
            },
        }
    )
    _o1, r1, _d1, _i1 = env.step(a1)
    a2 = Action.model_validate(
        {
            "action_type": "EXTRACT",
            "payload": {
                "doc_id": "icf_001",
                "fields": ["study_title", "protocol_number"],
                "section_hint": None,
            },
        }
    )
    _o2, r2, _d2, _i2 = env.step(a2)
    assert r2.completeness >= r1.completeness


def test_reward_weights_sum_close_to_one() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="MEDIUM", seed=42)
    # Probe with a valid action and inspect returned weights.
    act = Action.model_validate(
        {
            "action_type": "QUERY",
            "payload": {"doc_id": "protocol_001", "question": "What is the dose?"},
        }
    )
    _obs, reward, _done, _info = env.step(act)
    total = (
        reward.weights.accuracy
        + reward.weights.completeness
        + reward.weights.regulatory_alignment
        + reward.weights.efficiency
    )
    assert abs(total - 1.0) < 1e-6


def test_query_penalty_applies_per_unused_query() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    q = Action.model_validate(
        {"action_type": "QUERY", "payload": {"doc_id": "icf_001", "question": "Dose?"}}
    )
    env.step(q)
    env.step(q)
    submit = Action.model_validate(
        {"action_type": "SUBMIT", "payload": {"answer": {}, "confidence": 0.5}}
    )
    _obs, reward, _done, _info = env.step(submit)
    assert reward.step_penalty >= 0.04


def test_repeated_action_penalty_same_doc_within_three_steps() -> None:
    env = PharmaTrialsEnv()
    env.reset(task_id="EASY", seed=42)
    q1 = Action.model_validate(
        {
            "action_type": "QUERY",
            "payload": {"doc_id": "icf_001", "question": "Protocol number?"},
        }
    )
    q2 = Action.model_validate(
        {
            "action_type": "QUERY",
            "payload": {"doc_id": "icf_001", "question": "What sponsor?"},
        }
    )
    _obs1, r1, _done1, _info1 = env.step(q1)
    _obs2, r2, _done2, _info2 = env.step(q2)
    assert r1.step_penalty == 0.0
    assert r2.step_penalty >= 0.03
