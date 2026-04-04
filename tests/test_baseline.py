from __future__ import annotations

from dataclasses import dataclass

from baseline.run_baseline import run_episode


@dataclass
class _FakeObs:
    done: bool = False

    def model_dump_json(self) -> str:
        return "{}"


@dataclass
class _FakeReward:
    total: float


class _FakeEnv:
    def __init__(self) -> None:
        self._step = 0

    def reset(self, task_id: str, seed: int) -> _FakeObs:
        self._step = 0
        return _FakeObs(done=False)

    def step(self, _action: object) -> tuple[_FakeObs, _FakeReward, bool, dict[str, object]]:
        self._step += 1
        totals = {1: 0.10, 2: 0.25, 3: 0.40}
        done = self._step >= 3
        return _FakeObs(done=done), _FakeReward(total=totals[self._step]), done, {}


def test_run_episode_returns_final_normalized_score(monkeypatch) -> None:
    # Action parsing is already tested elsewhere; for this unit test we isolate scoring behavior.
    monkeypatch.setattr(
        "baseline.run_baseline.call_gpt4o",
        lambda *_args, **_kwargs: (
            '{"action_type":"QUERY","payload":{"doc_id":"icf_001","question":"q"}}'
        ),
    )

    class _ActionStub:
        @staticmethod
        def model_validate_json(_raw: str) -> object:
            return object()

    monkeypatch.setattr("baseline.run_baseline.Action", _ActionStub)

    score = run_episode(task_id="EASY", seed=42, env=_FakeEnv())
    assert score == 0.40
