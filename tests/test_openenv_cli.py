from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

import pharmatrials_env.openenv_cli as cli


def _base_manifest() -> dict[str, object]:
    with open("openenv.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_run_validate_manifest_only() -> None:
    result = cli.run_validate(config_path="openenv.yaml")
    assert result["status"] == "passed"
    assert "required_keys" in result["schema_checks"]


def test_manifest_has_expected_task_ids() -> None:
    with open("openenv.yaml", "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    task_ids = {task["id"] for task in manifest["tasks"]}
    assert task_ids == {"EASY", "MEDIUM", "HARD"}


def test_load_manifest_missing_file_raises() -> None:
    with pytest.raises(cli.ValidationError, match="Manifest not found"):
        cli._load_manifest(Path("does-not-exist.yaml"))


def test_validate_manifest_schema_rejects_bad_name() -> None:
    manifest = _base_manifest()
    manifest["name"] = "wrong-name"
    with pytest.raises(cli.ValidationError, match="Manifest name"):
        cli._validate_manifest_schema(manifest)


def test_validate_manifest_schema_rejects_bad_task_steps() -> None:
    manifest = _base_manifest()
    assert isinstance(manifest["tasks"], list)
    manifest["tasks"][0]["max_steps"] = 999
    with pytest.raises(cli.ValidationError, match="max_steps"):
        cli._validate_manifest_schema(manifest)


def test_load_manifest_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "openenv.yaml"
    path.write_text("- item\n- another\n", encoding="utf-8")
    with pytest.raises(cli.ValidationError, match="mapping/object"):
        cli._load_manifest(path)


def test_check_required_keys_reports_missing_key() -> None:
    manifest = _base_manifest()
    assert isinstance(manifest, dict)
    manifest.pop("api")
    with pytest.raises(cli.ValidationError, match="Missing required manifest key: api"):
        cli._check_required_keys(manifest)


def test_check_name_version_rejects_bad_version() -> None:
    manifest = _base_manifest()
    manifest["version"] = "9.9.9"
    with pytest.raises(cli.ValidationError, match="Manifest version"):
        cli._check_name_version(manifest)


def test_check_tasks_rejects_non_list() -> None:
    with pytest.raises(cli.ValidationError, match="tasks must be a list"):
        cli._check_tasks({"tasks": "not-a-list"})


def test_check_task_entry_rejects_non_object() -> None:
    with pytest.raises(cli.ValidationError, match="Each task must be an object"):
        cli._check_task_entry("EASY")


def test_check_api_endpoints_rejects_wrong_shape() -> None:
    with pytest.raises(cli.ValidationError, match="api must be an object"):
        cli._check_api_endpoints({"api": "nope"})
    with pytest.raises(cli.ValidationError, match="api.endpoints must be an object"):
        cli._check_api_endpoints({"api": {"endpoints": "nope"}})


def test_check_docker_rejects_non_object_and_bad_port() -> None:
    with pytest.raises(cli.ValidationError, match="docker must be an object"):
        cli._check_docker({"docker": "nope"})
    with pytest.raises(cli.ValidationError, match="docker.port"):
        cli._check_docker({"docker": {"port": 1234}})


class _Resp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _Client:
    def __init__(self, statuses: dict[tuple[str, str], int], **_kwargs: object) -> None:
        self._statuses = statuses

    def __enter__(self) -> "_Client":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> bool:
        return False

    def get(self, path: str) -> _Resp:
        return _Resp(self._statuses[("GET", path)])

    def post(self, path: str, json: object) -> _Resp:  # noqa: ARG002
        return _Resp(self._statuses[("POST", path)])


def test_validate_runtime_success(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = {
        ("GET", "/health"): 200,
        ("GET", "/tasks"): 200,
        ("POST", "/reset"): 200,
        ("POST", "/step"): 200,
        ("GET", "/state"): 200,
        ("GET", "/openenv.yaml"): 200,
    }
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: _Client(statuses, **kwargs))

    checks = cli._validate_runtime("http://localhost:7860")
    assert checks == ["health", "tasks", "reset", "step", "state", "openenv_manifest"]


def test_validate_runtime_health_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    statuses = {
        ("GET", "/health"): 500,
        ("GET", "/tasks"): 200,
        ("POST", "/reset"): 200,
        ("POST", "/step"): 200,
        ("GET", "/state"): 200,
        ("GET", "/openenv.yaml"): 200,
    }
    monkeypatch.setattr(cli.httpx, "Client", lambda **kwargs: _Client(statuses, **kwargs))

    with pytest.raises(cli.ValidationError, match="/health failed"):
        cli._validate_runtime("http://localhost:7860")


def test_run_validate_calls_runtime_when_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = {"runtime": False}

    def _fake_runtime(_base_url: str) -> list[str]:
        called["runtime"] = True
        return ["health"]

    monkeypatch.setattr(cli, "_validate_runtime", _fake_runtime)
    result = cli.run_validate(config_path="openenv.yaml", base_url="http://localhost:7860")
    assert called["runtime"] is True
    assert result["runtime_checks"] == ["health"]


def test_main_validate_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "run_validate",
        lambda **_kwargs: {
            "status": "passed",
            "schema_checks": [],
            "runtime_checks": [],
        },
    )
    monkeypatch.setattr(sys, "argv", ["openenv", "validate", "--config", "openenv.yaml"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["status"] == "passed"


def test_main_validate_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom(**_kwargs: object) -> dict[str, object]:
        raise cli.ValidationError("bad manifest")

    monkeypatch.setattr(cli, "run_validate", _boom)
    monkeypatch.setattr(sys, "argv", ["openenv", "validate", "--config", "openenv.yaml"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    err = capsys.readouterr().err
    parsed = json.loads(err)
    assert parsed["status"] == "failed"
    assert "bad manifest" in parsed["error"]
