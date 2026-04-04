from __future__ import annotations

from pathlib import Path

import yaml


def test_openenv_manifest_contract() -> None:
    manifest_path = Path("openenv.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["name"] == "pharmatrials-env"
    assert manifest["version"] == "1.0.0"

    task_ids = {t["id"] for t in manifest["tasks"]}
    assert task_ids == {"EASY", "MEDIUM", "HARD"}

    assert manifest["api"]["endpoints"]["reset"] == "POST /reset"
    assert manifest["api"]["endpoints"]["step"] == "POST /step"
    assert manifest["api"]["endpoints"]["state"] == "GET /state"

    assert manifest["docker"]["port"] == 7860
    assert "8080:7860" in manifest["docker"]["run_command"]
