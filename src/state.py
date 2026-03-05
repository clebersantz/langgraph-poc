"""Shared state definitions for the multi-agent system."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Roles of agents in the multi-agent system."""
    ORCHESTRATOR = "orchestrator"
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    QA = "qa"
    SECURITY = "security"
    DOCUMENTATION = "documentation"


class TaskStatus(str, Enum):
    """Status of a task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class Task(BaseModel):
    """Represents a unit of work."""
    id: str
    title: str
    description: str
    assigned_to: AgentRole | None = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=1, ge=1, le=5)
    dependencies: list[str] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    github_issue_number: int | None = None
    github_pr_number: int | None = None
    notes: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """State shared across all agents in the graph."""

    # Conversation history (append-only)
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)

    # Current active task
    current_task: Task | None = None

    # All tasks in the system
    tasks: list[Task] = Field(default_factory=list)

    # Current agent working
    current_agent: AgentRole | None = None

    # Next agent to hand off to
    next_agent: AgentRole | None = None

    # Project context
    project_goal: str = ""
    project_repo: str = ""
    project_branch: str = "main"
    workspace_path: str = ""

    # Iteration tracking
    iteration_count: int = 0
    max_iterations: int = 10

    # Outputs from each agent
    architect_output: dict[str, Any] = Field(default_factory=dict)
    developer_output: dict[str, Any] = Field(default_factory=dict)
    qa_output: dict[str, Any] = Field(default_factory=dict)
    security_output: dict[str, Any] = Field(default_factory=dict)
    documentation_output: dict[str, Any] = Field(default_factory=dict)

    # Error tracking
    errors: list[str] = Field(default_factory=list)

    # Final result
    final_result: dict[str, Any] = Field(default_factory=dict)
    is_complete: bool = False

    model_config = {"arbitrary_types_allowed": True}
