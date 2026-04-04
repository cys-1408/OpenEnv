---
title: PharmaTrials-Env
emoji: "🧪"
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - clinical-trials
  - document-intelligence
  - rl-environment
  - ai-agent
pinned: true
license: apache-2.0
---

# PharmaTrials-Env

> **OpenEnv-compliant environment for Clinical-Trial Document Intelligence**  
> An agent-facing environment for structured extraction, cross-document consistency checking,  
> and safety narrative reconciliation over synthetic regulatory documents.

---

## Environment Description & Motivation

Clinical trial operations are among the most documentation-intensive domains in regulated industries. Sponsors, CROs, and regulators must validate that:

- Informed Consent Forms (ICFs) accurately reflect the Study Protocol
- Adverse Event (AE) narratives are correctly coded against the Statistical Analysis Plan (SAP)
- All fields adhere to GCP, ICH E6(R3), MedDRA, and CTCAE standards

**PharmaTrials-Env** simulates this high-stakes document workflow for AI agents. Every episode presents the agent with realistic synthetic documents (ICF, Protocol, SAP, AE narratives), a structured task objective, and a multi-component reward signal that rewards partial progress — not just a final binary success.

**Why this domain?**
- Real-world utility: errors in these workflows lead to trial delays, FDA findings, or patient harm
- Rich language grounding: documents contain structured sections, numeric fields, regulatory codes
- Multi-step reasoning: no single action solves the task; agents must explore, extract, and synthesise
- Deterministic grading: all documents are seeded and ground truth is program-generated — no human annotation needed

All document content is **synthetic and procedurally generated**. No real patient data is used.

---

## Observation Space

Every call to `reset()` and `step()` returns an `Observation` object with these typed fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Task identifier: `EASY`, `MEDIUM`, or `HARD` |
| `task_name` | `enum` | `ICF_EXTRACTION` · `PROTOCOL_ICF_CONSISTENCY` · `SAP_AE_RECONCILIATION` |
| `step_number` | `int` | Current step index (0-indexed before first action) |
| `max_steps` | `int` | Step budget for this task |
| `documents` | `dict[str, DocumentView]` | All documents available, keyed by `doc_id` |
| `last_action_result` | `ActionResult \| None` | Structured result from the previous action |
| `task_instruction` | `str` | Natural-language task brief for the agent |
| `partial_score` | `float ∈ [0, 1]` | Running normalised episode score |
| `done` | `bool` | `True` when the episode has terminated |
| `metadata` | `dict` | Episode ID and seed |

### DocumentView fields

Each entry in `documents` is a `DocumentView`:

| Field | Type | Description |
|---|---|---|
| `doc_id` | `str` | Unique document identifier (e.g. `icf_001`, `sap_001`) |
| `doc_type` | `str` | `ICF` · `PROTOCOL` · `SAP` · `AE_NARRATIVE` |
| `title` | `str` | Document title |
| `content` | `str` | Full document text (line-oriented key: value format) |
| `sections` | `dict[str, str]` | Named subsections for targeted lookup |
| `version` | `str` | Document version string |
| `word_count` | `int` | Document length |

---

## Action Space

Six action types are available. Which ones are **allowed per task** is enforced by the environment.

| Action | Payload fields | Description |
|---|---|---|
| `EXTRACT` | `doc_id`, `fields: list[str]`, `section_hint?` | Extract named fields from a document |
| `COMPARE` | `doc_id_a`, `doc_id_b`, `comparison_fields`, `section_hint_a?`, `section_hint_b?` | Compare field values across two documents |
| `SUMMARISE` | `doc_id`, `focus_areas: list[str]`, `max_words?` | Targeted document summary |
| `ANNOTATE` | `doc_id`, `section`, `label`, `note`, `severity` | Attach a labeled annotation to a section |
| `QUERY` | `doc_id`, `question: str` | Fuzzy factual lookup within document content |
| `SUBMIT` | `answer: dict`, `confidence: float` | Submit final answer for grading (terminates episode) |

### Action JSON example

```json
{
  "action_type": "EXTRACT",
  "payload": {
    "doc_id": "icf_001",
    "fields": ["study_title", "protocol_number", "dose_mg"],
    "section_hint": "Study Procedures"
  }
}
```

### Allowed actions per task

| Task | Allowed actions |
|---|---|
| EASY | `EXTRACT`, `QUERY`, `SUBMIT` |
| MEDIUM | `COMPARE`, `ANNOTATE`, `QUERY`, `SUBMIT` |
| HARD | `EXTRACT`, `COMPARE`, `SUMMARISE`, `ANNOTATE`, `QUERY`, `SUBMIT` |

---

## Task Descriptions

### Task 1 — EASY: ICF Field Extraction

| Property | Value |
|---|---|
| **Difficulty** | Easy |
| **Max steps** | 15 |
| **Documents** | 1 Informed Consent Form (ICF) |
| **Objective** | Extract 12 structured fields from the ICF |
| **Grader** | Per-field accuracy (fuzzy match for strings, tolerance band for numerics) |

**Required fields:** `study_title`, `sponsor_name`, `protocol_number`, `compound_name`, `dose_mg`, `route_of_administration`, `dosing_frequency`, `num_visits`, `total_duration_weeks`, `primary_indication`, `irb_name`, `compensation_amount_usd`

**Scoring:** Score = mean per-field accuracy across all 12 fields. Numeric fields are scored with ±10% tolerance; string fields use fuzzy token-sort ratio.

**Expected difficulty:** An accurate agent should score 0.72–0.85. A random agent scores near 0.

---

### Task 2 — MEDIUM: Protocol-ICF Consistency Check

| Property | Value |
|---|---|
| **Difficulty** | Medium |
| **Max steps** | 25 |
| **Documents** | 1 Study Protocol + 1 ICF |
| **Objective** | Identify and classify 5–8 deliberate inconsistencies between the Protocol and ICF |
| **Grader** | F1 score over (field, severity) pairs; schema completeness bonus |

**Required output schema per inconsistency:**

```json
{
  "field": "dose_mg",
  "doc_a_value": "50",
  "doc_b_value": "100",
  "section_in_protocol": "Study Design",
  "section_in_icf": "Study Procedures",
  "severity": "ERROR",
  "regulatory_basis": "ICH E6(R3) consistency requirement"
}
```

**Severity levels:** `ERROR` (numeric/dosing fields) · `WARNING` (administrative fields)

**Expected difficulty:** A capable agent needs multi-document reasoning. Expected score: 0.45–0.62.

---

### Task 3 — HARD: SAP-AE Narrative Reconciliation

| Property | Value |
|---|---|
| **Difficulty** | Hard |
| **Max steps** | 40 |
| **Documents** | 1 SAP + 1 Protocol + 8–15 AE narratives |
| **Objective** | Extract SAP parameters, annotate each narrative per SAP coding rules, identify reconciliation findings |
| **Grader** | Weighted composite: SAP summary (20%) + narrative extraction (30%) + coding accuracy (20%) + findings F1 (20%) + regulatory impact (10%) |

**Required output structure:**

```json
{
  "sap_ae_analysis_summary": {
    "analysis_population": "...",
    "meddra_coding_level": "PT",
    "ctcae_version": "5.0",
    "sae_definition": "..."
  },
  "narrative_extractions": [
    {
      "narrative_id": "AE-001",
      "ae_term_reported": "Nausea",
      "ae_term_coded_pt": "Nausea",
      "ae_term_coded_soc": "Gastrointestinal disorders",
      "severity_grade_reported": 2,
      "causality": "Possibly related",
      "meets_sae_criteria": false,
      "onset_date": "2024-03-15",
      "resolution_date": "2024-03-18",
      "action_taken": "Dose reduced",
      "outcome": "Recovered"
    }
  ],
  "reconciliation_findings": [
    {
      "narrative_id": "AE-003",
      "finding_type": "MISCODED_SAE",
      "regulatory_impact": "HIGH"
    }
  ]
}
```

**Expected difficulty:** Requires reading 10+ documents and synthesising cross-document logic. Expected score: 0.22–0.38.

---

## Reward Function

Rewards are dense — **every step produces a reward signal**, not just SUBMIT.

```
R(t) = w_acc × Accuracy(t)
     + w_cmp × Completeness(t)
     + w_reg × RegulatoryAlignment(t)
     + w_eff × Efficiency(t)
     − StepPenalty(t)
```

**Component weights (default):**

| Component | Weight | Description |
|---|---|---|
| Accuracy | 0.40 | Field match quality vs. ground truth |
| Completeness | 0.30 | Fraction of required output items addressed |
| Regulatory alignment | 0.20 | Presence of ICH/GCP/MedDRA/CTCAE keywords |
| Efficiency | 0.10 | Steps remaining ratio at SUBMIT (0 otherwise) |

**Penalties:**

| Penalty | Amount |
|---|---|
| Invalid action (disallowed type) | −0.05 |
| Repeated action fingerprint within 3 steps | −0.03 |
| Unused QUERY result at SUBMIT time | −0.02 per result |
| 3 consecutive invalid actions | Early termination |

**Final episode score** = last `reward.total` value (cumulative normalised over `max_steps`).

---

## Setup and Usage

### Python (local)

```bash
# Install
git clone https://your-repo-url
cd pharmatrials-env
pip install -e .
pip install -e .[dev]   # includes pytest, ruff, mypy

# Quick test
python -c "
from pharmatrials_env import PharmaTrialsEnv
from pharmatrials_env.models import Action

env = PharmaTrialsEnv()
obs = env.reset(task_id='EASY', seed=42)
action = Action.model_validate({
    'action_type': 'EXTRACT',
    'payload': {'doc_id': 'icf_001', 'fields': ['study_title', 'protocol_number']}
})
obs, reward, done, info = env.step(action)
print(f'reward={reward.total:.3f}  done={done}')
"
```

### Docker

```bash
# Build
docker build -t pharmatrials-env:latest .

# Run (port 8080 → container 7860)
docker run -p 8080:7860 pharmatrials-env:latest

# Verify
curl http://localhost:8080/health
# → {"status":"ok","version":"1.0.0"}
```

### HTTP API

**PowerShell (Windows):**

```powershell
# Reset environment
Invoke-WebRequest -Uri http://localhost:8080/reset `
  -Method POST -ContentType "application/json" `
  -Body '{"task_id": "EASY", "seed": 42}' -UseBasicParsing

# Take a step
Invoke-WebRequest -Uri http://localhost:8080/step `
  -Method POST -ContentType "application/json" `
  -Body '{"action_type":"EXTRACT","payload":{"doc_id":"icf_001","fields":["study_title","protocol_number","dose_mg"]}}' `
  -UseBasicParsing

# Read state
Invoke-WebRequest -Uri http://localhost:8080/state -UseBasicParsing
```

**Linux / macOS / Git Bash:**

```bash
curl -X POST http://localhost:8080/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "EASY", "seed": 42}'

curl -X POST http://localhost:8080/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"EXTRACT","payload":{"doc_id":"icf_001","fields":["study_title","protocol_number","dose_mg"]}}'

curl http://localhost:8080/state
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (for LLM) | — | HuggingFace / API key for LLM calls |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `TASK_NAME` | No | `EASY` | Task: `EASY`, `MEDIUM`, or `HARD` |
| `SEED` | No | `42` | Random seed for determinism |

---

## Running inference.py

The mandatory inference script runs **one task per invocation**:

```bash
# Linux/macOS
export HF_TOKEN=your-hf-token
export TASK_NAME=EASY
python inference.py

# Windows
set HF_TOKEN=your-hf-token
set TASK_NAME=EASY
python inference.py
```

**Output format:**

```
[START] task=EASY env=pharmatrials-env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=EXTRACT(...) reward=0.06 done=false error=null
[STEP] step=5 action=SUBMIT(...) reward=0.24 done=true error=null
[END] success=true steps=5 score=0.235 rewards=0.06,0.09,0.13,0.18,0.24
```

Results are saved to `inference_results.json`.

---

## Baseline Scores

Baseline results across 3 seeds (42, 137, 999) in **fallback mode** (no LLM, deterministic rule-based agent):

| Task | Fallback score | Expected (Qwen2.5-72B) |
|---|---|---|
| EASY | ~0.235 | 0.72–0.85 |
| MEDIUM | ~0.060–0.072 | 0.45–0.62 |
| HARD | ~0.017 | 0.22–0.38 |

Run the multi-seed aggregate baseline:

```bash
export HF_TOKEN=your-hf-token
python baseline/run_baseline.py
```

Output is saved to `baseline_results.json`.

---

## OpenEnv Validation

```bash
# Offline (manifest schema only)
openenv validate

# Online (with running server)
uvicorn pharmatrials_env.api.server:app --host 0.0.0.0 --port 8080 &
openenv validate --base-url http://localhost:8080
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/reset` | Reset environment, returns initial `Observation` |
| `POST` | `/step` | Execute action, returns `{observation, reward, done, info}` |
| `GET` | `/state` | Full `EnvState` snapshot |
| `GET` | `/health` | Health check — returns `{"status":"ok"}` |
| `GET` | `/tasks` | Task summaries |
| `GET` | `/openenv.yaml` | OpenEnv manifest |

---

## Running Tests

```bash
pytest tests/ -v --tb=short
pytest tests/ --cov=pharmatrials_env --cov-report=term-missing
```

All 41 tests pass against Python 3.11+.

---

## Architecture

```
pharmatrials_env/
├── env.py              ← PharmaTrialsEnv: reset / step / state
├── models.py           ← Pydantic models: Observation, Action, Reward
├── state.py            ← EnvState, StateManager
├── documents/
│   ├── generator.py    ← Seeded synthetic document generation (Jinja2)
│   ├── templates/      ← Jinja2 .j2 templates for ICF, Protocol, SAP, AE
│   └── vocabulary/     ← JSON vocabularies for field values
├── tasks/
│   ├── easy.py         ← EASY task spec
│   ├── medium.py       ← MEDIUM task spec
│   ├── hard.py         ← HARD task spec
│   └── registry.py     ← TaskRegistry
├── graders/
│   ├── icf_grader.py          ← EASY grader
│   ├── consistency_grader.py  ← MEDIUM grader
│   └── reconciliation_grader.py ← HARD grader
├── reward/
│   └── reward_engine.py  ← Multi-component reward computation
├── api/
│   └── server.py         ← FastAPI HTTP server
└── openenv_cli.py        ← openenv validate CLI
```

---

## Repository Layout

```text
pharmatrials-env/
├── pharmatrials_env/       ← Environment package
├── baseline/
│   └── run_baseline.py     ← Multi-seed aggregate baseline
├── tests/                  ← pytest suite (41 tests)
├── inference.py            ← Mandatory inference script
├── openenv.yaml            ← OpenEnv manifest
├── Dockerfile              ← Container definition
├── requirements.txt        ← Runtime dependencies
├── pyproject.toml          ← Package build config
└── README.md               ← This file
```

---

## Citation

```bibtex
@software{pharmatrials_env_2026,
  title   = {PharmaTrials-Env: A Clinical-Trial Document Intelligence Environment for AI Agents},
  author  = {PharmaTrials-Env Contributors},
  year    = {2026},
  version = {1.0.0},
  license = {Apache-2.0}
}
```

---

## License

Apache-2.0 — see `pyproject.toml` and `openenv.yaml`.
