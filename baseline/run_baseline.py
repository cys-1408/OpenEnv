from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol

from openai import OpenAI
from openai import APIError, RateLimitError

from pharmatrials_env import PharmaTrialsEnv
from pharmatrials_env.models import Action
from pharmatrials_env.models import Observation

SYSTEM_PROMPT = """
You are an expert clinical-trial regulatory specialist AI agent operating
within the PharmaTrials-Env environment. At each step, you receive an
Observation JSON object and must output a single Action JSON object.

Available action types: EXTRACT, COMPARE, SUMMARISE, ANNOTATE, QUERY, SUBMIT.
Use SUBMIT only when you are confident in your complete answer.

Think step-by-step before choosing each action.
Format your response EXACTLY as a JSON Action object with no additional text.
""".strip()

BENCHMARK_NAME = "pharmatrials-env"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
API_BASE_URL = os.getenv("API_BASE_URL")

EASY_FIELDS = [
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

MEDIUM_FIELDS = [
    "num_visits",
    "total_duration_weeks",
    "dose_mg",
    "route_of_administration",
    "risk_list",
    "compensation_amount_usd",
    "irb_phone",
    "primary_endpoint",
]


class EnvLike(Protocol):
    def reset(self, task_id: str, seed: int) -> Any: ...

    def step(self, action: Action, /) -> tuple[Any, Any, bool, object]: ...


def _print_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK_NAME} model={MODEL_NAME}")


def _print_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}"
    )


def _print_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}")


def _action_label(action: Any) -> str:
    action_type = str(
        getattr(
            getattr(action, "action_type", None),
            "value",
            getattr(action, "action_type", type(action).__name__),
        )
    )
    return f"{action_type}(...)"


def _fallback_action(
    task_id: str, obs: Observation, easy_cache: dict[str, Any] | None = None
) -> Action:
    if task_id == "EASY":
        easy_batches = [
            ["study_title", "sponsor_name", "protocol_number"],
            ["compound_name", "dose_mg", "route_of_administration"],
            ["dosing_frequency", "num_visits", "total_duration_weeks"],
            ["primary_indication", "irb_name", "compensation_amount_usd"],
        ]

        if obs.step_number < len(easy_batches):
            return Action.model_validate(
                {
                    "action_type": "EXTRACT",
                    "payload": {
                        "doc_id": "icf_001",
                        "fields": easy_batches[obs.step_number],
                        "section_hint": None,
                    },
                }
            )

        extracted: dict[str, Any] = dict(easy_cache or {})
        if obs.last_action_result is not None:
            extracted.update(obs.last_action_result.output.get("extracted_fields", {}))

        # If previous action output is unavailable, still submit required keys with best-effort defaults.
        if not extracted:
            extracted = {
                "protocol_number": "",
                "dose_mg": 0,
                "num_visits": 0,
            }

        return Action.model_validate(
            {
                "action_type": "SUBMIT",
                "payload": {"answer": extracted, "confidence": 0.50},
            }
        )

    if task_id == "MEDIUM":
        if obs.step_number == 0:
            return Action.model_validate(
                {
                    "action_type": "COMPARE",
                    "payload": {
                        "doc_id_a": "protocol_001",
                        "doc_id_b": "icf_001",
                        "comparison_fields": MEDIUM_FIELDS,
                        "section_hint_a": "Study Design",
                        "section_hint_b": "Study Procedures",
                    },
                }
            )

        inconsistencies = []
        if obs.last_action_result is not None:
            inconsistencies = obs.last_action_result.output.get("inconsistencies", [])

        return Action.model_validate(
            {
                "action_type": "SUBMIT",
                "payload": {
                    "answer": {"inconsistencies": inconsistencies},
                    "confidence": 0.45,
                },
            }
        )

    # HARD
    if obs.step_number == 0:
        return Action.model_validate(
            {
                "action_type": "EXTRACT",
                "payload": {
                    "doc_id": "sap_001",
                    "fields": [
                        "analysis_population",
                        "meddra_coding_level",
                        "ctcae_version",
                        "sae_definition",
                    ],
                    "section_hint": None,
                },
            }
        )

    sap_summary = {}
    if obs.last_action_result is not None:
        sap_summary = obs.last_action_result.output.get("extracted_fields", {})

    return Action.model_validate(
        {
            "action_type": "SUBMIT",
            "payload": {
                "answer": {
                    "sap_ae_analysis_summary": sap_summary,
                    "narrative_extractions": [],
                    "reconciliation_findings": [],
                },
                "confidence": 0.35,
            },
        }
    )


def call_gpt4o(observation_json: str, system_prompt: str, max_retries: int = 3) -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("Neither OPENAI_API_KEY nor HF_TOKEN is set.")

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL

    client = OpenAI(**client_kwargs)
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": observation_json},
                ],
            )
            return completion.choices[0].message.content or "{}"
        except RateLimitError as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2**attempt, 8))
        except APIError as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(2**attempt, 8))

    assert last_error is not None
    raise last_error


def _run_episode_with_meta(
    task_id: str, seed: int, env: EnvLike
) -> tuple[float, int, str, list[str]]:
    obs = env.reset(task_id=task_id, seed=seed)
    final_score = 0.0
    warnings: list[str] = []
    provider_enabled = True
    mode_used = "model"
    easy_cache: dict[str, Any] = {}
    rewards_list: list[float] = []

    _print_start(task_id)

    local_steps = 0
    while not obs.done:
        local_steps += 1
        step_error: str | None = None
        if provider_enabled:
            try:
                action_json = call_gpt4o(obs.model_dump_json(), SYSTEM_PROMPT)
                action = Action.model_validate_json(action_json)
            except Exception as exc:
                provider_enabled = False
                mode_used = "fallback"
                warnings.append(f"provider_fallback: {type(exc).__name__}: {exc}")
                step_error = f"{type(exc).__name__}:{exc}"
                action = _fallback_action(task_id, obs, easy_cache)
        else:
            action = _fallback_action(task_id, obs, easy_cache)

        obs, reward, done, _info = env.step(action)

        if (
            task_id == "EASY"
            and getattr(getattr(action, "action_type", None), "value", "") == "EXTRACT"
            and obs.last_action_result is not None
            and obs.last_action_result.success
        ):
            easy_cache.update(obs.last_action_result.output.get("extracted_fields", {}))

        # reward.total is the cumulative normalized episode score at this step.
        final_score = reward.total
        rewards_list.append(reward.total)
        step_number = int(getattr(obs, "step_number", local_steps))
        _print_step(step_number, _action_label(action), reward.total, done, step_error)
        if done:
            break

    end_steps = int(getattr(obs, "step_number", local_steps))
    success = final_score > 0.0
    _print_end(success, end_steps, rewards_list)
    return final_score, end_steps, mode_used, warnings


def run_episode(task_id: str, seed: int, env: EnvLike) -> float:
    score, _steps, _mode, _warnings = _run_episode_with_meta(task_id, seed, env)
    return score


def main() -> None:
    seeds = [42, 137, 999]
    tasks = ["EASY", "MEDIUM", "HARD"]
    env = PharmaTrialsEnv()

    results: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    for task_id in tasks:
        scores: list[float] = []
        step_counts: list[int] = []
        modes: list[str] = []
        for seed in seeds:
            try:
                score, steps, mode_used, episode_warnings = _run_episode_with_meta(
                    task_id, seed, env
                )
                scores.append(score)
                step_counts.append(steps)
                modes.append(mode_used)
                for warning in episode_warnings:
                    warnings.append(
                        {
                            "task_id": task_id,
                            "seed": seed,
                            "warning": warning,
                        }
                    )
            except Exception as exc:
                errors.append(
                    {
                        "task_id": task_id,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        if not scores:
            continue

        results[task_id] = {
            "mean": sum(scores) / len(scores),
            "min": min(scores),
            "max": max(scores),
            "per_seed": dict(zip(seeds, scores)),
            "steps_per_seed": dict(zip(seeds, step_counts)),
            "modes_per_seed": dict(zip(seeds, modes)),
        }

    for required_task in tasks:
        results.setdefault(
            required_task,
            {
                "mean": 0.0,
                "min": 0.0,
                "max": 0.0,
                "per_seed": {seed: 0.0 for seed in seeds},
                "steps_per_seed": {seed: 0 for seed in seeds},
                "modes_per_seed": {seed: "unavailable" for seed in seeds},
            },
        )

    output: dict[str, Any] = {
        "status": (
            "completed" if not errors else ("partial" if any(results.values()) else "failed")
        ),
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "tasks": tasks,
        "seeds": seeds,
        "results": results,
        "errors": errors,
        "warnings": warnings,
    }

    with open("baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
