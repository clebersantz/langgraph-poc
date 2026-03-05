"""FastAPI application entry point for the multi-agent system."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
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

# ---------------------------------------------------------------------------
# Agent display helpers
# ---------------------------------------------------------------------------

_AGENT_EMOJIS: dict[str, str] = {
    "orchestrator": "🤖",
    "architect": "🏗️",
    "developer": "💻",
    "qa": "🧪",
    "security": "🔒",
    "documentation": "📝",
}

_AGENT_NAMES: frozenset[str] = frozenset(_AGENT_EMOJIS.keys())


def _extract_last_message_content(messages: list[Any]) -> str:
    """Return the text content of the last message in *messages*."""
    if not messages:
        return ""
    last = messages[-1]
    if hasattr(last, "content"):
        content = last.content
    elif isinstance(last, dict):
        content = last.get("content", "")
    else:
        return ""
    if isinstance(content, list):
        # Mixed-content / tool-call blocks — extract text parts only
        parts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
        return " ".join(p for p in parts if p)
    return str(content)


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


@app.get("/run/{run_id}/files")
async def list_run_files(run_id: str) -> dict[str, Any]:
    """List all files created in a run's workspace directory.

    Useful for verifying that agents produced expected output files.
    Returns a recursive list of relative file paths under the workspace.
    """
    settings = get_settings()
    workspace_path = os.path.join(settings.workspace_dir, run_id)

    if not os.path.exists(workspace_path):
        raise HTTPException(status_code=404, detail=f"Workspace for run '{run_id}' not found")

    files: list[str] = []
    for root, _dirs, filenames in os.walk(workspace_path):
        for filename in filenames:
            abs_path = os.path.join(root, filename)
            rel_path = os.path.relpath(abs_path, workspace_path)
            files.append(rel_path)

    files.sort()
    return {"run_id": run_id, "workspace_path": workspace_path, "files": files}


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
                font-size: 0.8rem; color: #94a3b8; border-radius: 8px; max-width: 90%; }
    .agent    { background: #1a2e1a; align-self: flex-start; border-bottom-left-radius: 4px;
                border-left: 3px solid #22c55e; font-size: 0.85rem; color: #d1fae5;
                max-width: 90%; }
    #form     { width: 100%; max-width: 800px; padding: 0.75rem 1rem 1.25rem;
                display: flex; gap: 0.5rem; flex-wrap: wrap; border-top: 1px solid #1e293b; }
    #input    { flex: 1; min-width: 0; padding: 0.65rem 1rem; border-radius: 8px;
                border: 1px solid #334155; background: #1e293b; color: #e2e8f0;
                font-size: 0.9rem; outline: none; resize: none; height: 2.6rem;
                max-height: 8rem; overflow-y: auto; }
    #input:focus { border-color: #38bdf8; }
    button    { padding: 0.65rem 1.25rem; border-radius: 8px; border: none;
                background: #0284c7; color: #fff; font-weight: 600; cursor: pointer;
                font-size: 0.9rem; transition: background 0.2s; white-space: nowrap; }
    button:hover  { background: #0369a1; }
    button:disabled { background: #334155; cursor: not-allowed; }
    #stop-btn   { background: #dc2626; display: none; }
    #stop-btn:hover { background: #b91c1c; }
    #resend-btn { background: #7c3aed; display: none; }
    #resend-btn:hover { background: #6d28d9; }
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
    <button type="button" id="stop-btn">&#x23F9; Stop</button>
    <button type="button" id="resend-btn">&#x21BA; Resend</button>
  </form>
  <script>
    const chat      = document.getElementById('chat');
    const form      = document.getElementById('form');
    const input     = document.getElementById('input');
    const btn       = document.getElementById('btn');
    const stopBtn   = document.getElementById('stop-btn');
    const resendBtn = document.getElementById('resend-btn');
    let sessionId       = null;
    let lastMessage     = null;
    let abortController = null;

    function addMsg(role, text) {
      const el = document.createElement('div');
      el.className = 'msg ' + role;
      el.textContent = text;
      chat.appendChild(el);
      chat.scrollTop = chat.scrollHeight;
      return el;
    }

    function setProcessing(active) {
      btn.disabled             = active;
      input.disabled           = active;
      stopBtn.style.display    = active ? 'inline-block' : 'none';
      resendBtn.style.display  = 'none';
    }

    function showResend() {
      stopBtn.style.display   = 'none';
      btn.disabled            = false;
      input.disabled          = false;
      if (lastMessage) resendBtn.style.display = 'inline-block';
      input.focus();
    }

    function handleSSEEvent(data, statusEl) {
      switch (data.type) {
        case 'session':
          sessionId = data.session_id;
          break;
        case 'agent_start':
          addMsg('system', data.label || (data.agent + ' is working\u2026'));
          break;
        case 'agent_message':
          if (statusEl) { statusEl.remove(); statusEl = null; }
          addMsg('agent', data.content);
          break;
        case 'done':
          if (statusEl) { statusEl.remove(); statusEl = null; }
          if (data.session_id) sessionId = data.session_id;
          addMsg('assistant', data.reply);
          setProcessing(false);
          input.focus();
          break;
        case 'stopped':
          if (statusEl) { statusEl.remove(); statusEl = null; }
          addMsg('system', '\u23f9 Processing was stopped.');
          showResend();
          break;
        case 'error':
          if (statusEl) { statusEl.remove(); statusEl = null; }
          addMsg('system', '\u274c Error: ' + data.message);
          setProcessing(false);
          input.focus();
          break;
      }
      return statusEl;
    }

    async function sendMessage(msg) {
      if (!msg) return;
      lastMessage = msg;
      resendBtn.style.display = 'none';
      addMsg('user', msg);
      setProcessing(true);

      abortController = new AbortController();
      let statusEl = addMsg('system', '\u23f3 Agents are working on your request\u2026');

      try {
        const body = { message: msg, max_iterations: 10 };
        if (sessionId) body.session_id = sessionId;

        const res = await fetch('/chat/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: abortController.signal,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          if (statusEl) statusEl.remove();
          addMsg('system', 'Error: ' + (err.detail || res.statusText));
          setProcessing(false);
          input.focus();
          return;
        }

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop();
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                statusEl = handleSSEEvent(data, statusEl);
              } catch (_) {}
            }
          }
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          if (statusEl) statusEl.remove();
          addMsg('system', '\u23f9 Processing stopped by user.');
          showResend();
        } else {
          if (statusEl) statusEl.remove();
          addMsg('system', 'Network error: ' + err.message);
          setProcessing(false);
          input.focus();
        }
      }
    }

    stopBtn.addEventListener('click', () => {
      if (abortController) abortController.abort();
    });

    resendBtn.addEventListener('click', async () => {
      if (lastMessage) await sendMessage(lastMessage);
    });

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
      if (!msg || btn.disabled) return;
      input.value = '';
      input.style.height = '2.6rem';
      await sendMessage(msg);
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


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, req: Request) -> StreamingResponse:
    """Streaming chat endpoint using Server-Sent Events (SSE).

    Streams agent progress in real-time.  The client can abort the connection
    at any time to stop processing (STOP button behaviour).

    SSE event payload shapes
    ------------------------
    ``{"type": "session",       "session_id": "..."}``
    ``{"type": "agent_start",   "agent": "developer", "label": "💻 Developer is working…"}``
    ``{"type": "agent_message", "agent": "developer", "content": "💻 Developer: …"}``
    ``{"type": "done",  "reply": "…", "status": "completed", "run_id": "…", "session_id": "…"}``
    ``{"type": "stopped",       "session_id": "…"}``
    ``{"type": "error",         "message": "…",       "session_id": "…"}``
    """

    async def _event_generator() -> AsyncIterator[str]:
        session_id = request.session_id or str(uuid.uuid4())
        if session_id not in _chat_sessions:
            _chat_sessions[session_id] = []
        _chat_sessions[session_id].append({"role": "user", "content": request.message})

        # Emit the session ID first so the client can persist it immediately
        yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

        repo_url = _extract_repo_url(request.message)
        branch = _extract_branch(request.message)
        run_id = str(uuid.uuid4())
        logger.info("Starting streaming chat workflow %s for session %s", run_id, session_id)

        settings = get_settings()
        workspace_path = os.path.join(settings.workspace_dir, run_id)
        os.makedirs(workspace_path, exist_ok=True)

        initial_state = AgentState(
            project_goal=request.message,
            project_repo=repo_url,
            project_branch=branch,
            workspace_path=workspace_path,
            max_iterations=request.max_iterations,
        )

        final_state_dict: dict[str, Any] | None = None

        try:
            graph = build_graph()
            run_config: RunnableConfig = {"recursion_limit": settings.recursion_limit}

            # stream_mode="values" yields the full accumulated state snapshot after
            # every node execution — perfect for real-time agent progress reporting.
            async for snapshot in graph.astream(
                initial_state, config=run_config, stream_mode="values"
            ):
                # Honor client disconnection (STOP button)
                if await req.is_disconnected():
                    logger.info("Client disconnected — stopping stream %s", run_id)
                    yield (f"data: {json.dumps({'type': 'stopped', 'session_id': session_id})}\n\n")
                    return

                final_state_dict = snapshot
                agent_name = snapshot.get("current_agent") or ""
                if agent_name in _AGENT_NAMES:
                    emoji = _AGENT_EMOJIS[agent_name]
                    label = f"{emoji} {agent_name.capitalize()} is working\u2026"
                    yield (
                        f"data: {json.dumps({'type': 'agent_start', 'agent': agent_name, 'label': label})}\n\n"
                    )
                    messages = snapshot.get("messages", [])
                    content = _extract_last_message_content(messages)
                    if content:
                        display = f"{emoji} {agent_name.capitalize()}: {content[:1500]}"
                        yield (
                            f"data: {json.dumps({'type': 'agent_message', 'agent': agent_name, 'content': display})}\n\n"
                        )

            # Build the final reply from the accumulated state
            if final_state_dict:
                final_state = AgentState.model_validate(final_state_dict)
                status = "completed" if final_state.is_complete else "stopped"
                reply = _format_chat_reply(final_state)
                _chat_sessions[session_id].append({"role": "assistant", "content": reply})
                yield (
                    f"data: {json.dumps({'type': 'done', 'reply': reply, 'status': status, 'run_id': run_id, 'session_id': session_id})}\n\n"
                )
            else:
                msg = "Workflow produced no output."
                _chat_sessions[session_id].append({"role": "assistant", "content": msg})
                yield (
                    f"data: {json.dumps({'type': 'done', 'reply': msg, 'status': 'stopped', 'run_id': run_id, 'session_id': session_id})}\n\n"
                )

        except asyncio.CancelledError:
            logger.info("Streaming chat %s cancelled", run_id)
            yield f"data: {json.dumps({'type': 'stopped', 'session_id': session_id})}\n\n"
        except Exception as e:  # noqa: BLE001
            logger.exception("Streaming chat %s failed: %s", run_id, e)
            error_msg = str(e)
            _chat_sessions[session_id].append(
                {"role": "assistant", "content": f"Error: {error_msg}"}
            )
            yield (
                f"data: {json.dumps({'type': 'error', 'message': error_msg, 'session_id': session_id})}\n\n"
            )

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
