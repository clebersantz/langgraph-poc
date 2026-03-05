"""FastAPI application entry point for the multi-agent system."""

from __future__ import annotations

import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

from src.config import get_settings
from src.graph import build_graph
from src.state import AgentState

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting LangGraph Multi-Agent System")
    settings = get_settings()
    os.makedirs(settings.workspace_dir, exist_ok=True)
    yield
    logger.info("Shutting down LangGraph Multi-Agent System")


app = FastAPI(
    title="LangGraph Multi-Agent System",
    description=(
        "Production-grade 6-agent LangGraph system with "
        "Project Manager, Architect, Developer, QA, Security, and Documentation agents."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    """Request body for running the multi-agent workflow."""

    goal: str
    repo_url: str = ""
    branch: str = "main"
    max_iterations: int = 10


class RunResponse(BaseModel):
    """Response from running the multi-agent workflow."""

    run_id: str
    status: str
    iterations: int
    final_result: dict[str, Any]
    errors: list[str]


# ---------------------------------------------------------------------------
# Chat interface models and session storage
# ---------------------------------------------------------------------------

# In-memory session storage: maps session_id -> list of {"role", "content"}
_chat_sessions: dict[str, list[dict[str, str]]] = {}


class ChatRequest(BaseModel):
    """Request body for the conversational chat endpoint."""

    message: str
    session_id: str | None = None
    max_iterations: int = 10


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""

    session_id: str
    reply: str
    run_id: str | None = None
    status: str
    result: dict[str, Any] | None = None


def _extract_repo_url(text: str) -> str:
    """Extract the first GitHub repository URL found in *text*."""
    match = re.search(r"https?://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+", text)
    return match.group(0) if match else ""


def _extract_branch(text: str) -> str:
    """Extract a branch name from *text*, defaulting to 'main'."""
    patterns = [
        r"\bbranch:\s*([a-zA-Z0-9_/.-]+)",
        r"\bbranch\s+([a-zA-Z0-9_/.-]+)",
        r"\bon\s+([a-zA-Z0-9_/.-]+)\s+branch",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return "main"


def _format_chat_reply(state: AgentState) -> str:
    """Convert the completed *state* into a human-readable chat message."""
    lines: list[str] = []
    if state.is_complete:
        lines.append("✅ Workflow completed successfully!")
    else:
        lines.append("⏹ Workflow stopped.")

    if state.final_result:
        lines.append("\n**Results:**")
        for key, value in state.final_result.items():
            lines.append(f"- {key}: {value}")

    if state.errors:
        lines.append("\n**Errors encountered:**")
        for err in state.errors:
            lines.append(f"- {err}")

    lines.append(f"\n*Completed in {state.iteration_count} iteration(s).*")
    return "\n".join(lines)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "langgraph-multi-agent"}


@app.post("/run", response_model=RunResponse)
async def run_workflow(request: RunRequest) -> RunResponse:
    """Run the multi-agent workflow for a given goal."""
    run_id = str(uuid.uuid4())
    logger.info("Starting workflow run %s: %s", run_id, request.goal)

    settings = get_settings()
    workspace_path = os.path.join(settings.workspace_dir, run_id)
    os.makedirs(workspace_path, exist_ok=True)

    initial_state = AgentState(
        project_goal=request.goal,
        project_repo=request.repo_url,
        project_branch=request.branch,
        workspace_path=workspace_path,
        max_iterations=request.max_iterations,
    )

    try:
        graph = build_graph()
        run_config: RunnableConfig = {"recursion_limit": settings.recursion_limit}
        raw_state = await graph.ainvoke(initial_state, config=run_config)
        final_state = AgentState.model_validate(raw_state)

        return RunResponse(
            run_id=run_id,
            status="completed" if final_state.is_complete else "stopped",
            iterations=final_state.iteration_count,
            final_result=final_state.final_result,
            errors=final_state.errors,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Workflow %s failed: %s", run_id, e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/agents")
async def list_agents() -> list[dict[str, str]]:
    """List all available agents and their roles."""
    return [
        {
            "name": "orchestrator",
            "role": "Project Manager / Orchestrator",
            "description": "Plans tasks and coordinates all agents",
        },
        {
            "name": "architect",
            "role": "Software Architect",
            "description": "Designs system architecture and technical specifications",
        },
        {
            "name": "developer",
            "role": "Senior Developer",
            "description": "Implements features and fixes bugs",
        },
        {
            "name": "qa",
            "role": "QA Engineer",
            "description": "Tests, validates, and ensures quality",
        },
        {
            "name": "security",
            "role": "Security Engineer",
            "description": "Reviews security and identifies vulnerabilities",
        },
        {
            "name": "documentation",
            "role": "Technical Writer",
            "description": "Creates and maintains documentation",
        },
    ]


@app.get("/", response_class=HTMLResponse)
async def chat_ui() -> HTMLResponse:
    """Serve the HTML chat interface."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>LangGraph Multi-Agent Chat</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
           background: #0f172a; color: #e2e8f0; height: 100vh;
           display: flex; flex-direction: column; align-items: center; }
    header { width: 100%; max-width: 800px; padding: 1.25rem 1rem;
             border-bottom: 1px solid #1e293b; }
    header h1 { font-size: 1.3rem; font-weight: 700; color: #38bdf8; }
    header p  { font-size: 0.8rem; color: #94a3b8; margin-top: 0.25rem; }
    #chat     { flex: 1; width: 100%; max-width: 800px; overflow-y: auto;
                padding: 1rem; display: flex; flex-direction: column; gap: 0.75rem; }
    .msg      { max-width: 80%; padding: 0.75rem 1rem; border-radius: 12px;
                line-height: 1.5; white-space: pre-wrap; font-size: 0.9rem; }
    .user     { background: #1e40af; align-self: flex-end; border-bottom-right-radius: 4px; }
    .assistant{ background: #1e293b; align-self: flex-start; border-bottom-left-radius: 4px; }
    .system   { background: #134e4a; align-self: center; font-style: italic;
                font-size: 0.8rem; color: #94a3b8; border-radius: 8px; }
    #form     { width: 100%; max-width: 800px; padding: 0.75rem 1rem 1.25rem;
                display: flex; gap: 0.5rem; border-top: 1px solid #1e293b; }
    #input    { flex: 1; padding: 0.65rem 1rem; border-radius: 8px;
                border: 1px solid #334155; background: #1e293b; color: #e2e8f0;
                font-size: 0.9rem; outline: none; resize: none; height: 2.6rem;
                max-height: 8rem; overflow-y: auto; }
    #input:focus { border-color: #38bdf8; }
    button    { padding: 0.65rem 1.25rem; border-radius: 8px; border: none;
                background: #0284c7; color: #fff; font-weight: 600; cursor: pointer;
                font-size: 0.9rem; transition: background 0.2s; }
    button:hover { background: #0369a1; }
    button:disabled { background: #334155; cursor: not-allowed; }
    .spinner  { display: inline-block; width: 14px; height: 14px;
                border: 2px solid #334155; border-top-color: #38bdf8;
                border-radius: 50%; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <header>
    <h1>&#x1F916; LangGraph Multi-Agent Chat</h1>
    <p>Describe your project goal in plain language. Include a GitHub URL if relevant.</p>
  </header>
  <div id="chat"></div>
  <form id="form">
    <textarea id="input" placeholder="e.g. Please clone https://github.com/org/repo and add a new feature..." rows="1"></textarea>
    <button type="submit" id="btn">Send</button>
  </form>
  <script>
    const chat   = document.getElementById('chat');
    const form   = document.getElementById('form');
    const input  = document.getElementById('input');
    const btn    = document.getElementById('btn');
    let sessionId = null;

    function addMsg(role, text) {
      const el = document.createElement('div');
      el.className = 'msg ' + role;
      el.textContent = text;
      chat.appendChild(el);
      chat.scrollTop = chat.scrollHeight;
      return el;
    }

    input.addEventListener('input', () => {
      input.style.height = '2.6rem';
      input.style.height = Math.min(input.scrollHeight, 128) + 'px';
    });

    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); }
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const msg = input.value.trim();
      if (!msg) return;
      input.value = '';
      input.style.height = '2.6rem';
      addMsg('user', msg);

      btn.disabled = true;
      const thinking = addMsg('assistant', '');
      const spinner = document.createElement('span');
      spinner.className = 'spinner';
      thinking.appendChild(spinner);

      try {
        const body = { message: msg, max_iterations: 10 };
        if (sessionId) body.session_id = sessionId;

        const res = await fetch('/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          thinking.textContent = 'Error: ' + (err.detail || res.statusText);
          return;
        }

        const data = await res.json();
        sessionId = data.session_id;
        thinking.textContent = data.reply;
      } catch (err) {
        thinking.textContent = 'Network error: ' + err.message;
      } finally {
        btn.disabled = false;
        input.focus();
      }
    });

    addMsg('system', 'Session started. Type your request below.');
  </script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Conversational endpoint — describe your goal in plain language."""
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = []

    _chat_sessions[session_id].append({"role": "user", "content": request.message})

    repo_url = _extract_repo_url(request.message)
    branch = _extract_branch(request.message)
    goal = request.message

    run_id = str(uuid.uuid4())
    logger.info("Starting chat workflow %s for session %s", run_id, session_id)

    settings = get_settings()
    workspace_path = os.path.join(settings.workspace_dir, run_id)
    os.makedirs(workspace_path, exist_ok=True)

    initial_state = AgentState(
        project_goal=goal,
        project_repo=repo_url,
        project_branch=branch,
        workspace_path=workspace_path,
        max_iterations=request.max_iterations,
    )

    try:
        graph = build_graph()
        run_config: RunnableConfig = {"recursion_limit": settings.recursion_limit}
        raw_state = await graph.ainvoke(initial_state, config=run_config)
        final_state = AgentState.model_validate(raw_state)

        status = "completed" if final_state.is_complete else "stopped"
        reply = _format_chat_reply(final_state)

        _chat_sessions[session_id].append({"role": "assistant", "content": reply})

        return ChatResponse(
            session_id=session_id,
            reply=reply,
            run_id=run_id,
            status=status,
            result=final_state.final_result,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Chat workflow %s failed: %s", run_id, e)
        error_reply = f"An error occurred while processing your request: {e}"
        _chat_sessions[session_id].append({"role": "assistant", "content": error_reply})
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str) -> list[dict[str, str]]:
    """Return the conversation history for a given session."""
    if session_id not in _chat_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _chat_sessions[session_id]


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
