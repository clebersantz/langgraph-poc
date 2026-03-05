"""FastAPI application entry point for the multi-agent system."""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        config = {"recursion_limit": settings.recursion_limit}
        final_state = await graph.ainvoke(initial_state, config=config)

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
        {"name": "orchestrator", "role": "Project Manager / Orchestrator", "description": "Plans tasks and coordinates all agents"},
        {"name": "architect", "role": "Software Architect", "description": "Designs system architecture and technical specifications"},
        {"name": "developer", "role": "Senior Developer", "description": "Implements features and fixes bugs"},
        {"name": "qa", "role": "QA Engineer", "description": "Tests, validates, and ensures quality"},
        {"name": "security", "role": "Security Engineer", "description": "Reviews security and identifies vulnerabilities"},
        {"name": "documentation", "role": "Technical Writer", "description": "Creates and maintains documentation"},
    ]


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
