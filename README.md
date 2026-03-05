# LangGraph Multi-Agent System

A production-grade 6-agent LangGraph multi-agent system for automated software development, designed for Docker / VPS deployment.

## Agents

| Agent | Role | Responsibilities |
|-------|------|-----------------|
| **Orchestrator** | Project Manager | Plans tasks, coordinates agents, manages GitHub issues |
| **Architect** | Software Architect | System design, architecture decisions, technical specs |
| **Developer** | Senior Developer | Code implementation, feature branches, PRs |
| **QA** | QA Engineer | Testing, quality assurance, bug reporting |
| **Security** | Security Engineer | OWASP review, vulnerability assessment, CVE scanning |
| **Documentation** | Technical Writer | README, API docs, changelogs, deployment guides |

## Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                       │
│          (Project Manager / Coordinator)             │
└──────┬──────────────────────────────────────────────┘
       │ routes based on task
       ├──► Architect ──────────► (returns to Orchestrator)
       ├──► Developer ──────────► (returns to Orchestrator)
       ├──► QA         ──────────► (returns to Orchestrator)
       ├──► Security  ──────────► (returns to Orchestrator)
       └──► Documentation ──────► (returns to Orchestrator)
```

Each agent has access to:
- **GitHub Tools**: Create/manage issues, PRs, comments, merges
- **Git Tools**: Clone, branch, commit, push, pull, diff
- **Code Tools**: Read/write files, run commands, search code

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key (or Anthropic)
- GitHub Personal Access Token

### Setup

1. **Clone and configure**:
   ```bash
   git clone https://github.com/clebersantz/langgraph-poc.git
   cd langgraph-poc
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Start with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

3. **Check health**:
   ```bash
   curl http://localhost:8000/health
   ```

4. **Run a workflow**:
   ```bash
   curl -X POST http://localhost:8000/run \
     -H "Content-Type: application/json" \
     -d '{
       "goal": "Create a FastAPI CRUD application for user management",
       "repo_url": "https://github.com/your-org/your-repo",
       "branch": "main",
       "max_iterations": 10
     }'
   ```

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Fill in your API keys
   ```

3. **Run locally**:
   ```bash
   python -m src.main
   # or
   uvicorn src.main:app --reload
   ```

4. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

## API Reference

### `POST /run`

Start a multi-agent workflow.

**Request**:
```json
{
  "goal": "Description of what to build or fix",
  "repo_url": "https://github.com/owner/repo",
  "branch": "main",
  "max_iterations": 10
}
```

**Response**:
```json
{
  "run_id": "uuid",
  "status": "completed",
  "iterations": 5,
  "final_result": {},
  "errors": []
}
```

### `GET /health`

Returns `{"status": "healthy"}`.

### `GET /agents`

Lists all agents with their roles and descriptions.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | LLM provider (`openai` or `anthropic`) |
| `LLM_MODEL` | `gpt-4o` | Model name |
| `LLM_TEMPERATURE` | `0.1` | Model temperature |
| `OPENAI_API_KEY` | - | OpenAI API key |
| `ANTHROPIC_API_KEY` | - | Anthropic API key |
| `GITHUB_TOKEN` | - | GitHub PAT with repo/issues/PR scopes |
| `GITHUB_OWNER` | - | GitHub organization or username |
| `GITHUB_REPO` | - | Repository name |
| `GIT_USER_NAME` | `LangGraph Agent` | Git commit author name |
| `GIT_USER_EMAIL` | `agent@langgraph.local` | Git commit author email |
| `WORKSPACE_DIR` | `/tmp/workspace` | Directory for cloned repositories |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server port |
| `MAX_ITERATIONS` | `10` | Max agent iterations per workflow |
| `RECURSION_LIMIT` | `50` | LangGraph recursion limit |

## VPS Deployment

```bash
# On your VPS
git clone https://github.com/clebersantz/langgraph-poc.git
cd langgraph-poc
cp .env.example .env
nano .env  # fill in your keys

# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Update
git pull && docker-compose up -d --build
```

## CI/CD

Two GitHub Actions workflows ship with this project.

### CI (`ci.yml`) — runs on every push and pull request

| Job | What it does |
|-----|-------------|
| **Lint** | `ruff check` + `ruff format --check` on `src/` and `tests/` |
| **Test** | `pytest` on Python 3.11 and 3.12 with JUnit XML reports |
| **Docker Build Check** | Builds the Docker image (no push) to catch `Dockerfile` errors early |

### CD (`cd.yml`) — runs on push to `main` or a published release

| Job | What it does |
|-----|-------------|
| **Build & Push** | Builds a multi-arch (`amd64`/`arm64`) image and pushes to GHCR as `ghcr.io/<owner>/<repo>:latest` (and `sha-<short>`) |
| **Deploy to VPS** | SSHes into your VPS, updates `.env`, pulls the new image, and runs `docker compose up -d` |

The deploy job only runs when the repository variable `VPS_DEPLOY_ENABLED` is set to `true`.

#### Required secrets & variables

Configure these in **Settings → Secrets and variables → Actions** of your repository.

**Secrets**

| Secret | Description |
|--------|-------------|
| `VPS_HOST` | IP address or hostname of your VPS |
| `VPS_USER` | SSH username (e.g. `ubuntu`) |
| `VPS_SSH_KEY` | Private SSH key for the VPS user |
| `VPS_PORT` | SSH port (optional, default `22`) |
| `OPENAI_API_KEY` | OpenAI API key for the running container |
| `ANTHROPIC_API_KEY` | Anthropic API key (optional) |
| `DEPLOY_GITHUB_TOKEN` | GitHub PAT for the agents (repo + issues + PRs) |

**Variables** (non-secret)

| Variable | Description | Default |
|----------|-------------|---------|
| `VPS_DEPLOY_ENABLED` | Set to `true` to enable VPS deployment | `false` |
| `VPS_PROJECT_DIR` | Directory on VPS to deploy into | `$HOME/langgraph-poc` |
| `LLM_PROVIDER` | `openai` or `anthropic` | `openai` |
| `LLM_MODEL` | Model name | `gpt-4o` |
| `GITHUB_OWNER` | GitHub org/username for agent tools | — |
| `GITHUB_REPO` | Repository name for agent tools | — |

#### Docker image tags

Images are pushed to `ghcr.io/<owner>/<repo>` with the following tags:

- `latest` — always points to the most recent `main` build
- `main` — branch name tag
- `sha-<8chars>` — immutable per-commit tag
- `v1.0.0` (etc.) — semantic version tags created from GitHub releases

## License

MIT
