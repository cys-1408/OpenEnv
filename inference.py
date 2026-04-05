"""
Inference Script — PharmaTrials-Env
=====================================
Mandatory configuration (set as environment variables):

    API_BASE_URL   The API endpoint for the LLM.
                   Default: https://router.huggingface.co/v1
    MODEL_NAME     The model identifier to use for inference.
                   Default: Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN       Your Hugging Face / API key.
    TASK_NAME      Task to run: EASY | MEDIUM | HARD  (default: EASY)
    SEED           Random seed for determinism          (default: 42)

STDOUT FORMAT
-------------
Exactly three line types, in order:

    [START] task=<task_name> env=pharmatrials-env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Rules:
  - One [START] line at episode begin.
  - One [STEP] line per step, immediately after env.step() returns.
  - One [END] line at episode end, always emitted (even on exception).
  - reward and rewards are formatted to 2 decimal places.
  - score is formatted to 3 decimal places.
  - done and success are lowercase booleans: true or false.
  - error is the raw error string, or null if none.
  - All fields on a single line with no newlines within a line.

Example:
    [START] task=EASY env=pharmatrials-env model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=EXTRACT(...) reward=0.06 done=false error=null
    [STEP] step=5 action=SUBMIT(...) reward=0.24 done=true error=null
    [END] success=true steps=5 score=0.235 rewards=0.06,0.09,0.12,0.18,0.24
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from openai import APIError, OpenAI, RateLimitError

from pharmatrials_env import PharmaTrialsEnv
from pharmatrials_env.models import Action, Observation

# ── Environment variables ────────────────────────────────────────────────────


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()

BENCHMARK = "pharmatrials-env"
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
TASK_NAME = os.getenv("TASK_NAME", "EASY")
SEED = int(os.getenv("SEED", "42"))
MAX_RETRIES = 3

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert clinical-trial regulatory specialist AI agent operating "
    "within the PharmaTrials-Env environment. At each step, output exactly one "
    "JSON Action object and no extra text.\n\n"
    "Available action types: EXTRACT, COMPARE, SUMMARISE, ANNOTATE, QUERY, SUBMIT.\n"
    "Use SUBMIT only when you have gathered enough data to answer the task objective.\n"
    'Format: {"action_type": "<TYPE>", "payload": {...}}'
)

# ── Fallback hard-coded strategies (used when LLM unavailable) ───────────────

EASY_BATCHES = [
    ["study_title", "sponsor_name", "protocol_number"],
    ["compound_name", "dose_mg", "route_of_administration"],
    ["dosing_frequency", "num_visits", "total_duration_weeks"],
    ["primary_indication", "irb_name", "compensation_amount_usd"],
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


def fallback_action(task_id: str, obs: Observation, easy_cache: dict[str, Any]) -> Action:
    """Deterministic fallback strategy when the LLM provider is unavailable."""
    if task_id == "EASY":
        if obs.step_number < len(EASY_BATCHES):
            return Action.model_validate(
                {
                    "action_type": "EXTRACT",
                    "payload": {
                        "doc_id": "icf_001",
                        "fields": EASY_BATCHES[obs.step_number],
                        "section_hint": None,
                    },
                }
            )
        extracted = dict(easy_cache)
        if obs.last_action_result is not None:
            extracted.update(obs.last_action_result.output.get("extracted_fields", {}))
        if not extracted:
            extracted = {"protocol_number": "", "dose_mg": 0, "num_visits": 0}
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
        inconsistencies: list[dict[str, Any]] = []
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
    sap_summary: dict[str, Any] = {}
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


# ── Logging helpers (mandatory stdout format) ────────────────────────────────


def log_start(task: str, env_name: str, model: str) -> None:
    print(f"[START] task={task} env={env_name} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Action label helper ──────────────────────────────────────────────────────


def action_label(action: Any) -> str:
    action_type = str(
        getattr(
            getattr(action, "action_type", None),
            "value",
            getattr(action, "action_type", type(action).__name__),
        )
    )
    return f"{action_type}(...)"


# ── LLM call ────────────────────────────────────────────────────────────────


def call_model(observation_json: str) -> str:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN (or OPENAI_API_KEY) is not set.")
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": observation_json},
                ],
                stream=False,
            )
            return completion.choices[0].message.content or "{}"
        except (RateLimitError, APIError) as exc:
            last_error = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise last_error


# ── Episode runner ───────────────────────────────────────────────────────────


def run_episode(
    env: PharmaTrialsEnv,
    task_id: str,
    seed: int,
) -> tuple[float, int, bool, list[float]]:
    obs = env.reset(task_id=task_id, seed=seed)
    rewards: list[float] = []
    easy_cache: dict[str, Any] = {}
    provider_enabled = True
    step_no = 0

    log_start(task=task_id, env_name=BENCHMARK, model=MODEL_NAME)

    try:
        while not obs.done:
            step_no += 1
            step_error: str | None = None

            if provider_enabled:
                try:
                    action_json = call_model(obs.model_dump_json())
                    action = Action.model_validate_json(action_json)
                except Exception as exc:
                    provider_enabled = False
                    step_error = f"{type(exc).__name__}:{exc}"
                    action = fallback_action(task_id, obs, easy_cache)
            else:
                action = fallback_action(task_id, obs, easy_cache)

            obs, reward, done, _info = env.step(action)

            # Cache EXTRACT results for EASY fallback SUBMIT
            if (
                task_id == "EASY"
                and getattr(getattr(action, "action_type", None), "value", "") == "EXTRACT"
                and obs.last_action_result is not None
                and obs.last_action_result.success
            ):
                easy_cache.update(obs.last_action_result.output.get("extracted_fields", {}))

            rewards.append(reward.total)
            log_step(
                step=int(getattr(obs, "step_number", step_no)),
                action=action_label(action),
                reward=reward.total,
                done=done,
                error=step_error,
            )
            if done:
                break

        final_score = rewards[-1] if rewards else 0.0
        success = final_score > 0.0
        return final_score, step_no, success, rewards

    except Exception as exc:  # pragma: no cover
        # Emit a STEP for the current failed step so the log stays complete
        rewards.append(0.0)
        log_step(step=step_no, action="ERROR", reward=0.0, done=True, error=str(exc))
        return 0.0, step_no, False, rewards


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    env = PharmaTrialsEnv()
    final_score, steps, success, rewards = run_episode(env, task_id=TASK_NAME, seed=SEED)
    log_end(success=success, steps=steps, score=final_score, rewards=rewards)

    # Persist results for reproducibility
    result: dict[str, Any] = {
        "task": TASK_NAME,
        "seed": SEED,
        "model": MODEL_NAME,
        "api_base_url": API_BASE_URL,
        "steps": steps,
        "score": final_score,
        "success": success,
        "rewards": rewards,
    }
    with open("inference_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
