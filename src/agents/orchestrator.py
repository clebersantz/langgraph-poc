"""Project Manager / Orchestrator agent."""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import get_settings
from src.state import AgentRole, AgentState, Task, TaskStatus
from src.tools.github_tools import (
    add_comment_to_issue,
    close_issue,
    create_issue,
    list_issues,
    update_issue,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Project Manager and Orchestrator of a multi-agent software development team.

Your responsibilities:
1. Analyze project goals and decompose them into actionable tasks
2. Assign tasks to the appropriate specialized agents:
   - Architect: System design, architecture decisions, technical specifications
   - Developer: Code implementation, bug fixes, feature development
   - QA: Testing, quality assurance, test case creation
   - Security: Security analysis, vulnerability assessment, threat modeling
   - Documentation: Writing docs, README, API docs, changelogs
3. Manage GitHub issues and track progress
4. Coordinate between agents, resolve blockers
5. Determine when the project goal is complete

When analyzing a request:
- Break it down into clear, prioritized tasks
- Identify dependencies between tasks
- Assign tasks to the right agent
- Track GitHub issues for each task
- Ensure all agents complete their work before declaring success

You have access to GitHub issue management tools to track work.

Always respond with a clear plan in JSON format when decomposing tasks, and clearly state which agent should work next.
"""


def create_orchestrator_agent(llm):
    """Create the orchestrator agent node function."""

    async def orchestrator_node(state: AgentState) -> AgentState:
        """Orchestrator agent that manages task planning and coordination."""
        logger.info("Orchestrator agent processing, iteration %d", state.iteration_count)

        tools = [create_issue, list_issues, update_issue, close_issue, add_comment_to_issue]
        agent_llm = llm.bind_tools(tools)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content=f"""
Current project goal: {state.project_goal}
Current iteration: {state.iteration_count}/{state.max_iterations}
Tasks: {[t.model_dump() for t in state.tasks]}
Architect output: {state.architect_output}
Developer output: {state.developer_output}
QA output: {state.qa_output}
Security output: {state.security_output}
Documentation output: {state.documentation_output}

Based on the current state, what should happen next?
- If no tasks exist, decompose the project goal into tasks and decide which agent to start with.
- If tasks exist and are in progress, check their status and route to the appropriate next agent.
- If all tasks are complete, mark the project as done.

Respond with your analysis and specify the next_agent (one of: architect, developer, qa, security, documentation, or DONE).
"""),
        ])

        messages = prompt.format_messages(messages=state.messages)
        response = await agent_llm.ainvoke(messages)

        # Parse next agent from response
        next_agent = _parse_next_agent(response.content)

        # Update iteration count
        new_iteration = state.iteration_count + 1
        is_complete = next_agent == "done" or new_iteration >= state.max_iterations

        return AgentState(
            **{
                **state.model_dump(exclude={"messages"}),
                "messages": [response],
                "current_agent": AgentRole.ORCHESTRATOR,
                "next_agent": AgentRole(next_agent) if next_agent and next_agent != "done" else None,
                "iteration_count": new_iteration,
                "is_complete": is_complete,
            }
        )

    return orchestrator_node


def _parse_next_agent(content: str) -> str | None:
    """Extract the next agent name from the orchestrator's response."""
    content_lower = content.lower()
    agent_keywords = {
        "architect": AgentRole.ARCHITECT,
        "developer": AgentRole.DEVELOPER,
        "qa": AgentRole.QA,
        "quality assurance": AgentRole.QA,
        "security": AgentRole.SECURITY,
        "documentation": AgentRole.DOCUMENTATION,
        "done": "done",
        "complete": "done",
    }
    for keyword, role in agent_keywords.items():
        if keyword in content_lower:
            return role if isinstance(role, str) else role.value
    return None
