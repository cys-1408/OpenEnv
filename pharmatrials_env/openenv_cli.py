from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml  # type: ignore[import-untyped]

EXPECTED_TASKS = {
    "EASY": {"max_steps": 15},
    "MEDIUM": {"max_steps": 25},
    "HARD": {"max_steps": 40},
}

EXPECTED_ENDPOINTS = {
    "reset": "POST /reset",
    "step": "POST /step",
    "state": "GET /state",
}


class ValidationError(Exception):
    pass


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValidationError(f"Manifest not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise ValidationError(f"Failed to parse YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValidationError("Manifest root must be a mapping/object.")
    return data


def _validate_manifest_schema(manifest: dict[str, Any]) -> list[str]:
    checks: list[str] = []
    _check_required_keys(manifest)
    checks.append("required_keys")

    _check_name_version(manifest)
    checks.append("name_version")

    _check_tasks(manifest)
    checks.append("tasks")

    _check_api_endpoints(manifest)
    checks.append("api_endpoints")

    _check_docker(manifest)
    checks.append("docker")

    return checks


def _check_required_keys(manifest: dict[str, Any]) -> None:
    required_keys = [
        "name",
        "version",
        "tasks",
        "api",
        "docker",
        "action_space",
        "reward_space",
        "observation_space",
    ]
    for key in required_keys:
        if key not in manifest:
            raise ValidationError(f"Missing required manifest key: {key}")


def _check_name_version(manifest: dict[str, Any]) -> None:
    if manifest.get("name") != "pharmatrials-env":
        raise ValidationError("Manifest name must be 'pharmatrials-env'.")
    if manifest.get("version") != "1.0.0":
        raise ValidationError("Manifest version must be '1.0.0'.")


def _check_tasks(manifest: dict[str, Any]) -> None:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValidationError("tasks must be a list.")

    seen_ids = {str(t.get("id")) for t in tasks if isinstance(t, dict)}
    if seen_ids != set(EXPECTED_TASKS):
        raise ValidationError(
            f"Task IDs mismatch. Expected {sorted(EXPECTED_TASKS)}, got {sorted(seen_ids)}"
        )

    for task in tasks:
        _check_task_entry(task)


def _check_task_entry(task: Any) -> None:
    if not isinstance(task, dict):
        raise ValidationError("Each task must be an object.")
    task_id = str(task.get("id"))
    expected_steps = EXPECTED_TASKS[task_id]["max_steps"]
    if int(task.get("max_steps", -1)) != expected_steps:
        raise ValidationError(f"Task {task_id} max_steps must be {expected_steps}.")


def _check_api_endpoints(manifest: dict[str, Any]) -> None:
    api = manifest.get("api")
    if not isinstance(api, dict):
        raise ValidationError("api must be an object.")
    endpoints = api.get("endpoints")
    if not isinstance(endpoints, dict):
        raise ValidationError("api.endpoints must be an object.")
    for name, expected in EXPECTED_ENDPOINTS.items():
        if endpoints.get(name) != expected:
            raise ValidationError(f"api.endpoints.{name} must be '{expected}'.")


def _check_docker(manifest: dict[str, Any]) -> None:
    docker = manifest.get("docker")
    if not isinstance(docker, dict):
        raise ValidationError("docker must be an object.")
    if int(docker.get("port", -1)) not in {7860, 8080}:
        raise ValidationError("docker.port must be 7860 or 8080.")


def _validate_runtime(base_url: str) -> list[str]:
    checks: list[str] = []
    timeout = httpx.Timeout(15.0, connect=15.0)

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise ValidationError(f"/health failed with status {health.status_code}")
        checks.append("health")

        tasks = client.get("/tasks")
        if tasks.status_code != 200:
            raise ValidationError(f"/tasks failed with status {tasks.status_code}")
        checks.append("tasks")

        reset = client.post("/reset", json={"task_id": "EASY", "seed": 42})
        if reset.status_code != 200:
            raise ValidationError(f"/reset failed with status {reset.status_code}")
        checks.append("reset")

        step = client.post(
            "/step",
            json={
                "action_type": "QUERY",
                "payload": {
                    "doc_id": "icf_001",
                    "question": "What is the protocol number?",
                },
            },
        )
        if step.status_code != 200:
            raise ValidationError(f"/step failed with status {step.status_code}")
        checks.append("step")

        state = client.get("/state")
        if state.status_code != 200:
            raise ValidationError(f"/state failed with status {state.status_code}")
        checks.append("state")

        manifest = client.get("/openenv.yaml")
        if manifest.status_code != 200:
            raise ValidationError(f"/openenv.yaml failed with status {manifest.status_code}")
        checks.append("openenv_manifest")

    return checks


def run_validate(config_path: str, base_url: str | None = None) -> dict[str, Any]:
    manifest = _load_manifest(Path(config_path))
    schema_checks = _validate_manifest_schema(manifest)

    runtime_checks: list[str] = []
    if base_url:
        runtime_checks = _validate_runtime(base_url)

    return {
        "status": "passed",
        "manifest": config_path,
        "base_url": base_url,
        "schema_checks": schema_checks,
        "runtime_checks": runtime_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="openenv", description="OpenEnv validation utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate an OpenEnv manifest and runtime")
    validate.add_argument(
        "--config",
        required=False,
        default="openenv.yaml",
        help="Path to openenv.yaml (default: openenv.yaml in CWD)",
    )
    validate.add_argument("--base-url", required=False, help="Base URL for runtime endpoint checks")

    args = parser.parse_args()

    try:
        if args.command == "validate":
            result = run_validate(config_path=args.config, base_url=args.base_url)
            print(json.dumps(result, indent=2))
            raise SystemExit(0)
        raise SystemExit(2)
    except ValidationError as exc:
        print(
            json.dumps({"status": "failed", "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
