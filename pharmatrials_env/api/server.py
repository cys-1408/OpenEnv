from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, cast

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pharmatrials_env import PharmaTrialsEnv
from pharmatrials_env.models import Action, Observation, Reward
from pharmatrials_env.state import EnvState


class ResetRequest(BaseModel):
    task_id: str | None = None
    seed: int | None = None


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any]


_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    application.state.env = PharmaTrialsEnv()
    yield
    application.state.env = None


app = FastAPI(title="PharmaTrials-Env", version="1.0.0", lifespan=lifespan)


def _require_env() -> PharmaTrialsEnv:
    env = getattr(app.state, "env", None)
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized.")
    return cast(PharmaTrialsEnv, env)


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "Welcome to PharmaTrials-Env (OpenEnv compliant)",
        "version": "1.0.0",
        "health_check": "/health",
        "tasks": "/tasks",
        "documentation": "/openenv.yaml",
    }


@app.post("/reset", response_model=Observation)
async def reset(body: ResetRequest) -> Observation:
    env = _require_env()
    async with _lock:
        return env.reset(task_id=body.task_id, seed=body.seed)


@app.post("/step", response_model=StepResponse)
async def step(action: Action) -> StepResponse:
    env = _require_env()
    async with _lock:
        obs, reward, done, info = env.step(action)
        return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=EnvState)
async def state() -> EnvState:
    env = _require_env()
    async with _lock:
        return env.state()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
async def tasks() -> list[dict[str, object]]:
    env = _require_env()
    async with _lock:
        return env.task_summaries()


@app.get("/openenv.yaml")
async def openenv_manifest() -> str:
    manifest = Path(__file__).resolve().parents[2] / "openenv.yaml"
    return manifest.read_text(encoding="utf-8")
