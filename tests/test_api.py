from __future__ import annotations

from fastapi.testclient import TestClient

from pharmatrials_env.api.server import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        payload = res.json()
        assert payload["status"] == "ok"


def test_tasks_endpoint() -> None:
    with TestClient(app) as client:
        res = client.get("/tasks")
        assert res.status_code == 200
        data = res.json()
        assert any(row["task_id"] == "EASY" for row in data)


def test_reset_step_state_endpoints() -> None:
    with TestClient(app) as client:
        r = client.post("/reset", json={"task_id": "EASY", "seed": 42})
        assert r.status_code == 200
        obs = r.json()
        assert obs["task_id"] == "EASY"

        s = client.post(
            "/step",
            json={
                "action_type": "QUERY",
                "payload": {
                    "doc_id": "icf_001",
                    "question": "What is the protocol number?",
                },
            },
        )
        assert s.status_code == 200
        step_payload = s.json()
        assert "observation" in step_payload
        assert "reward" in step_payload

        st = client.get("/state")
        assert st.status_code == 200
        state_payload = st.json()
        assert state_payload["task_id"] == "EASY"


def test_openenv_manifest_endpoint() -> None:
    with TestClient(app) as client:
        res = client.get("/openenv.yaml")
        assert res.status_code == 200
        assert "name: pharmatrials-env" in res.text
