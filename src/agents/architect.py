"""Architect agent."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents._tool_executor import run_tool_loop
from src.state import AgentRole, AgentState
from src.tools.code_tools import create_file, list_directory, read_file
from src.tools.git_tools import clone_repository, create_branch, pull_changes
from src.tools.github_tools import add_comment_to_issue, create_issue, update_issue

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Software Architect in a multi-agent development team.

Your responsibilities:
1. Analyze project requirements and define system architecture
2. Create technical specifications and design documents
3. Define folder structure, module boundaries, and interfaces
4. Select appropriate technologies, libraries, and patterns
5. Document architectural decisions (ADRs)
6. Review code for architectural alignment
7. Identify technical debt and refactoring opportunities

Architectural principles you follow:
- SOLID principles
- Clean Architecture / Hexagonal Architecture
- 12-Factor App for cloud-native applications
- Security by design
- Scalability and maintainability

When designing a system:
- Start with domain model and bounded contexts
- Define clear API contracts
- Document data flows and sequence diagrams (as text/ASCII)
- Consider failure modes and resilience patterns
- Specify infrastructure requirements

Output your architecture as structured documents that the Developer agent can implement.
"""


def create_architect_agent(llm):
    """Create the architect agent node function."""

    async def architect_node(state: AgentState) -> AgentState:
        """Architect agent that designs the system architecture."""
        logger.info("Architect agent processing task: %s", state.current_task)

        tools = [
            clone_repository,
            pull_changes,
            create_branch,
            read_file,
            list_directory,
            create_file,
            create_issue,
            update_issue,
            add_comment_to_issue,
        ]
        agent_llm = llm.bind_tools(tools)

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="messages"),
                HumanMessage(
                    content=f"""
Project goal: {state.project_goal}
Repository: {state.project_repo}
Workspace: {state.workspace_path}
Current task: {state.current_task.model_dump() if state.current_task else "None"}

Please analyze the project requirements and produce:
1. System architecture overview
2. Component breakdown
3. Technology stack recommendations
4. API specifications (if applicable)
5. Data models
6. Infrastructure requirements
7. Implementation plan for the Developer agent

Document your design decisions clearly.
"""
                ),
            ]
        )

        messages = prompt.format_messages(messages=state.messages)
        response = await run_tool_loop(agent_llm, tools, messages)

        architect_output = {
            "analysis": response.content,
            "task_id": state.current_task.id if state.current_task else None,
        }

        return AgentState(
            **{
                **state.model_dump(exclude={"messages"}),
                "messages": [response],
                "current_agent": AgentRole.ARCHITECT,
                "next_agent": AgentRole.ORCHESTRATOR,
                "architect_output": architect_output,
            }
        )

    return architect_node
