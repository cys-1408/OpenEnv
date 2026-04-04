# Deploying PharmaTrials-Env to Hugging Face Spaces

## Prerequisites

- A Hugging Face account: https://huggingface.co/join
- `git` and `git-lfs` installed
- Docker Desktop installed and running (for local validation)

---

## Step 1 — Local Docker Validation

Run these commands to confirm the image builds and starts cleanly before deploying:

```powershell
# Build the image
docker build -t pharmatrials-env:latest .

# Start the container (in background or a separate terminal)
docker run -p 8080:7860 pharmatrials-env:latest
```

**In a new PowerShell terminal:**

```powershell
# Health check
Invoke-WebRequest -Uri http://localhost:8080/health -UseBasicParsing
# Expected: StatusCode=200  Content={"status":"ok","version":"1.0.0"}

# Reset endpoint (used by the submission validator)
Invoke-WebRequest -Uri http://localhost:8080/reset `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"task_id": "EASY", "seed": 42}' `
  -UseBasicParsing
# Expected: StatusCode=200 with full Observation JSON
```

**Linux / macOS / Git Bash:**

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "EASY", "seed": 42}'
```

---

## Step 2 — Create the HuggingFace Space

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `pharmatrials-env` (or your preferred name)
   - **License**: Apache-2.0
   - **SDK**: Docker
   - **App port**: 7860
3. Click **Create Space**

---

## Step 3 — Push the Repository

```bash
# Clone the empty HF Space
git clone https://huggingface.co/spaces/YOUR_USERNAME/pharmatrials-env
cd pharmatrials-env

# Copy all project files into it
# (or add HF Space as a remote to your existing repo)
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/pharmatrials-env

# Push
git add .
git commit -m "Initial deployment"
git push hf main
```

The README.md frontmatter already contains the required HF Space metadata:

```yaml
---
title: PharmaTrials-Env
emoji: "🧪"
colorFrom: blue
colorTo: teal
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
```

HF Spaces will automatically detect `sdk: docker`, use the `Dockerfile` at the repo root, and expose port 7860.

---

## Step 4 — Set Space Secrets (Optional LLM access)

In your Space settings → **Secrets**, add:

| Secret | Value |
|---|---|
| `HF_TOKEN` | Your HuggingFace token with inference API access |
| `API_BASE_URL` | `https://router.huggingface.co/v1` (already the default) |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` (already the default) |

Without these, the server still starts and all endpoints work — the LLM is only used when running `inference.py`.

---

## Step 5 — Verify Deployment

Once the Space build completes (usually 2–5 minutes):

**PowerShell:**

```powershell
$SPACE_URL = "https://YOUR_USERNAME-pharmatrials-env.hf.space"

# Health check
Invoke-WebRequest -Uri "$SPACE_URL/health" -UseBasicParsing

# Reset (submission validator uses this)
Invoke-WebRequest -Uri "$SPACE_URL/reset" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{}' `
  -UseBasicParsing
```

**Linux / Git Bash:**

```bash
SPACE_URL="https://YOUR_USERNAME-pharmatrials-env.hf.space"
curl $SPACE_URL/health
curl -X POST $SPACE_URL/reset -H "Content-Type: application/json" -d '{}'
# Run the submission pre-validation script
bash validate-submission.sh $SPACE_URL
```

---

## Step 6 — Run inference.py Against the Live Space

**PowerShell:**

```powershell
$env:HF_TOKEN = "your-hf-token"
$env:API_BASE_URL = "https://router.huggingface.co/v1"
$env:MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
$env:TASK_NAME = "EASY"

python inference.py
```

**Linux / macOS:**

```bash
export HF_TOKEN=your-hf-token
export TASK_NAME=EASY
python inference.py
```

---

## Environment Variables Reference

| Variable | Where to set | Description |
|---|---|---|
| `HF_TOKEN` | HF Space Secret | API key for LLM inference |
| `API_BASE_URL` | HF Space Secret or env | LLM endpoint (default: HF router) |
| `MODEL_NAME` | HF Space Secret or env | Model name (default: Qwen2.5-72B) |
| `PORT` | Auto-set by HF Spaces | HTTP port (default: 7860) |
| `TASK_NAME` | Local env only | Task for inference.py (EASY/MEDIUM/HARD) |

---

## Troubleshooting

**Space shows "Build failed"**
- Check the Build logs tab in the Space UI
- Verify `Dockerfile` is at the repo root
- Verify `requirements.txt` and `pyproject.toml` are present

**`/reset` returns 503**
- The application is still starting up (allow 30s after boot)
- Check the Space logs for uvicorn startup messages

**openenv validate fails at Step 1**
- Your Space URL must match exactly: `https://YOUR_USERNAME-pharmatrials-env.hf.space`
- HF Spaces may take a few minutes to become reachable after first deployment
