"""Project Manager / Orchestrator agent."""

from __future__ import annotations

import json as _json
import logging
import re as _re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents._tool_executor import sanitize_messages
from src.state import AgentRole, AgentState

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
3. Coordinate between agents, resolve blockers
4. Determine when the project goal is complete

When analyzing a request:
- Break it down into clear, prioritized tasks
- Identify dependencies between tasks
- Assign tasks to the right agent
- Ensure all agents complete their work before declaring success

Always respond with a JSON object specifying the next_agent value:
{
  "analysis": "your analysis here",
  "tasks": [...],
  "next_agent": "developer"   ← one of: architect, developer, qa, security, documentation, or DONE
}
"""


def create_orchestrator_agent(llm):
    """Create the orchestrator agent node function."""

    async def orchestrator_node(state: AgentState) -> AgentState:
        """Orchestrator agent that manages task planning and coordination."""
        logger.info("Orchestrator agent processing, iteration %d", state.iteration_count)

        # The orchestrator only needs to produce routing text — do NOT bind tools
        # so that the LLM always returns a text response that can be parsed for
        # the next_agent. Binding GitHub tools caused Azure OpenAI to return
        # tool_calls (with empty content) instead of routing text, breaking routing.

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                HumanMessage(
                    content=f"""
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
"""
                ),
            ]
        )

        messages = prompt.format_messages(messages=sanitize_messages(state.messages))
        response = await llm.ainvoke(messages)

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
                "next_agent": AgentRole(next_agent)
                if next_agent and next_agent != "done"
                else None,
                "iteration_count": new_iteration,
                "is_complete": is_complete,
            }
        )

    return orchestrator_node


def _parse_next_agent(content: str) -> str | None:
    """Extract the next agent name from the orchestrator's response.

    Parsing order:
    1. JSON ``"next_agent"`` field (most reliable — LLM often returns JSON).
    2. Explicit ``next_agent: <value>`` key-value pattern (text format).
    3. Keyword scan — checks ``done``/``complete`` *before* agent names so that
       a response like "The developer finished; project is DONE" terminates the
       loop rather than routing back to developer.
    """
    content_lower = content.lower()

    # ── 1. JSON "next_agent" field ───────────────────────────────────────────
    # Try the whole content first, then fall back to individual JSON objects.
    for candidate in [content] + _re.findall(r"\{[^{}]+\}", content, _re.DOTALL):
        try:
            data = _json.loads(candidate)
            agent = str(data.get("next_agent", "")).lower().strip()
            if agent:
                if agent in ("done", "complete", "finished"):
                    return "done"
                for role in AgentRole:
                    if agent == role.value:
                        return role.value
        except (ValueError, KeyError, TypeError):
            pass

    # ── 2. Explicit key-value pattern ────────────────────────────────────────
    m = _re.search(r"next[_\s]agent[\"'\s:]+([a-z_]+)", content_lower)
    if m:
        agent = m.group(1).strip()
        if agent in ("done", "complete", "finished"):
            return "done"
        for role in AgentRole:
            if agent == role.value:
                return role.value

    # ── 3. Keyword scan (done/complete checked first) ────────────────────────
    if _re.search(r"\b(done|complete|finished|all\s+tasks?\s+complete)\b", content_lower):
        return "done"

    agent_keywords = {
        "architect": AgentRole.ARCHITECT,
        "developer": AgentRole.DEVELOPER,
        "qa": AgentRole.QA,
        "quality assurance": AgentRole.QA,
        "security": AgentRole.SECURITY,
        "documentation": AgentRole.DOCUMENTATION,
    }
    for keyword, role in agent_keywords.items():
        if keyword in content_lower:
            return role.value

    return None
