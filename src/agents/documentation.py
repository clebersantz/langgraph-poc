"""Documentation agent."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.state import AgentRole, AgentState
from src.tools.code_tools import create_file, list_directory, read_file, run_command
from src.tools.git_tools import commit_changes, get_diff, push_changes
from src.tools.github_tools import (
    add_comment_to_issue,
    add_comment_to_pr,
    create_issue,
    update_issue,
    update_pull_request,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Technical Writer and Documentation Engineer in a multi-agent development team.

Your responsibilities:
1. Write and maintain README files
2. Create API documentation (OpenAPI/Swagger, docstrings)
3. Write user guides and tutorials
4. Maintain changelogs (CHANGELOG.md)
5. Document architecture decisions (ADRs)
6. Create deployment guides
7. Write inline code documentation
8. Keep documentation in sync with code changes

Documentation standards:
- Clear, concise writing at appropriate technical level
- Consistent formatting with markdown
- Include practical examples
- Document all public APIs
- Keep docs up to date with code changes
- Use diagrams where helpful (Mermaid, ASCII art)

Documentation structure:
- README: Overview, quick start, installation, usage
- API docs: All endpoints with examples
- Architecture: High-level design, data flows
- Deployment: Step-by-step deployment guide
- Contributing: How to contribute, development setup
- CHANGELOG: All notable changes by version

Always write docs from the user's perspective: what does the user need to know to use this?
"""


def create_documentation_agent(llm):
    """Create the documentation agent node function."""

    async def documentation_node(state: AgentState) -> AgentState:
        """Documentation agent that writes and maintains documentation."""
        logger.info("Documentation agent processing task: %s", state.current_task)

        tools = [
            read_file,
            list_directory,
            create_file,
            run_command,
            commit_changes,
            push_changes,
            get_diff,
            create_issue,
            update_issue,
            add_comment_to_issue,
            add_comment_to_pr,
            update_pull_request,
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
Architecture: {state.architect_output.get("analysis", "Not yet designed")}
Implementation: {state.developer_output.get("implementation", "Not yet implemented")}
QA assessment: {state.qa_output.get("assessment", "Not yet assessed")}
Security assessment: {state.security_output.get("assessment", "Not yet assessed")}

Please create/update documentation:
1. README.md with overview, installation, and usage instructions
2. API documentation if applicable
3. Architecture overview document
4. Deployment guide (including Docker instructions)
5. CHANGELOG.md entry for this change
6. Inline code documentation review
7. Update any existing outdated docs

Commit documentation changes to the repository.
"""
                ),
            ]
        )

        messages = prompt.format_messages(messages=state.messages)
        response = await agent_llm.ainvoke(messages)

        documentation_output = {
            "documentation": response.content,
            "task_id": state.current_task.id if state.current_task else None,
        }

        return AgentState(
            **{
                **state.model_dump(exclude={"messages"}),
                "messages": [response],
                "current_agent": AgentRole.DOCUMENTATION,
                "next_agent": AgentRole.ORCHESTRATOR,
                "documentation_output": documentation_output,
            }
        )

    return documentation_node
