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

## License

MIT
