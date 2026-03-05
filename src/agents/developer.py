"""Developer agent."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents._tool_executor import _workspace_is_empty, extract_files_from_content, run_tool_loop
from src.state import AgentRole, AgentState
from src.tools.code_tools import create_file, list_directory, read_file, run_command
from src.tools.git_tools import (
    clone_repository,
    commit_changes,
    create_branch,
    get_diff,
    push_changes,
)
from src.tools.github_tools import (
    add_comment_to_issue,
    create_pull_request,
    update_issue,
    update_pull_request,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Senior Software Developer in a multi-agent development team.

Your responsibilities:
1. Implement features and fix bugs based on architect specifications
2. Write clean, well-tested, production-ready code
3. Follow the project's coding standards and conventions
4. Create feature branches and commit changes with clear messages
5. Open pull requests with detailed descriptions
6. Respond to code review feedback

Development standards you follow:
- Write self-documenting code with clear variable/function names
- Add docstrings to all public functions and classes
- Handle errors gracefully with proper logging
- Write unit tests for new functionality
- Keep functions small and focused (single responsibility)
- Use type hints throughout (Python)
- Follow PEP 8 / project style guide

When implementing:
- Always create a feature branch first
- Commit atomically with meaningful messages
- Test your implementation before committing
- Create a pull request when done
- Reference the relevant GitHub issue in commits and PRs

Git commit message format: <type>(<scope>): <description>
Types: feat, fix, docs, test, refactor, chore
"""


def create_developer_agent(llm):
    """Create the developer agent node function."""

    async def developer_node(state: AgentState) -> AgentState:
        """Developer agent that implements code changes."""
        logger.info("Developer agent processing task: %s", state.current_task)

        tools = [
            clone_repository,
            create_branch,
            commit_changes,
            push_changes,
            get_diff,
            create_file,
            read_file,
            list_directory,
            run_command,
            create_pull_request,
            update_pull_request,
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
Workspace directory (use this for all file operations): {state.workspace_path}
Current branch: {state.project_branch}
Current task: {state.current_task.model_dump() if state.current_task else "None"}
Architecture design: {state.architect_output.get("analysis", "Not yet designed")}

IMPORTANT: You MUST use the available tools to actually create, read, and run files.
Do NOT just describe what you would do — call the tools to perform the actions.

Steps for this task:
1. Use the `create_file` tool to write every source file with its FULL absolute path
   (e.g. `{state.workspace_path}/main.py`). Never use relative paths.
2. Use `run_command` with working_dir="{state.workspace_path}" to install deps, run tests, etc.
3. Use `read_file` to verify file contents after creation.
4. Only use git tools (clone, commit, push, PR) if a repository URL is provided and the
   goal explicitly requires version control. Otherwise skip git steps entirely.

If you cannot call tools, you MUST respond with a ```json code block containing ALL files:
{{
  "files": [
    {{"path": "main.py", "content": "complete file content"}},
    {{"path": "test_main.py", "content": "complete file content"}},
    {{"path": "README.md", "content": "complete file content"}}
  ],
  "summary": "brief description"
}}

Be thorough and write production-quality code.
"""
                ),
            ]
        )

        messages = prompt.format_messages(messages=state.messages)
        response = await run_tool_loop(agent_llm, tools, messages)

        # Fallback: if no files were created via tool_calls, extract them from the
        # LLM response text.  The prompt asks for JSON output when tools can't be
        # called, so we look for a ```json block first, then bold-filename headers.
        if state.workspace_path and _workspace_is_empty(state.workspace_path):
            text = response.content if isinstance(response.content, str) else ""
            created = extract_files_from_content(text, state.workspace_path)
            if created:
                logger.info("Developer: fallback extracted %d file(s): %s", len(created), created)

        developer_output = {
            "implementation": response.content,
            "task_id": state.current_task.id if state.current_task else None,
        }

        return AgentState(
            **{
                **state.model_dump(exclude={"messages"}),
                "messages": [response],
                "current_agent": AgentRole.DEVELOPER,
                "next_agent": AgentRole.ORCHESTRATOR,
                "developer_output": developer_output,
            }
        )

    return developer_node
