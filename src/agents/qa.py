"""QA (Quality Assurance) agent."""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.state import AgentRole, AgentState
from src.tools.code_tools import create_file, list_directory, read_file, run_command
from src.tools.git_tools import clone_repository, commit_changes, get_diff, push_changes
from src.tools.github_tools import (
    add_comment_to_issue,
    add_comment_to_pr,
    create_issue,
    update_issue,
    update_pull_request,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the QA Engineer in a multi-agent development team.

Your responsibilities:
1. Write comprehensive test suites (unit, integration, e2e)
2. Review code changes for correctness and quality
3. Run existing tests and analyze failures
4. Identify edge cases and potential bugs
5. Verify that requirements are met
6. Report bugs with clear reproduction steps
7. Approve or request changes on pull requests

Testing standards you follow:
- AAA pattern: Arrange, Act, Assert
- Test coverage target: 80%+
- Test naming: test_<what>_<when>_<expected>
- Use mocks/stubs for external dependencies
- Test happy paths AND error cases
- Performance testing for critical paths

When reviewing:
- Check that all acceptance criteria are met
- Verify error handling
- Look for security issues in the implementation
- Ensure code follows style guidelines
- Check for missing tests

Create GitHub issues for any bugs found, tagged with 'bug' label.
"""


def create_qa_agent(llm):
    """Create the QA agent node function."""

    async def qa_node(state: AgentState) -> AgentState:
        """QA agent that tests and validates implementation."""
        logger.info("QA agent processing task: %s", state.current_task)

        tools = [
            clone_repository, get_diff, commit_changes, push_changes,
            read_file, list_directory, create_file, run_command,
            add_comment_to_pr, add_comment_to_issue, create_issue, update_issue, update_pull_request,
        ]
        agent_llm = llm.bind_tools(tools)

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(content=f"""
Project goal: {state.project_goal}
Workspace: {state.workspace_path}
Current task: {state.current_task.model_dump() if state.current_task else 'None'}
Developer implementation: {state.developer_output.get('implementation', 'Not yet implemented')}

Please perform quality assurance:
1. Review the implementation for correctness
2. Run existing tests and report results
3. Write additional tests for uncovered scenarios
4. Check for edge cases and error handling
5. Verify requirements are met
6. Create bug issues for any problems found
7. Comment on the PR with your assessment (approve or request changes)

Be thorough in your testing approach.
"""),
        ])

        messages = prompt.format_messages(messages=state.messages)
        response = await agent_llm.ainvoke(messages)

        qa_output = {
            "assessment": response.content,
            "task_id": state.current_task.id if state.current_task else None,
        }

        return AgentState(
            **{
                **state.model_dump(exclude={"messages"}),
                "messages": [response],
                "current_agent": AgentRole.QA,
                "next_agent": AgentRole.ORCHESTRATOR,
                "qa_output": qa_output,
            }
        )

    return qa_node
