"""Security agent."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.state import AgentRole, AgentState
from src.tools.code_tools import list_directory, read_file, run_command, search_code
from src.tools.github_tools import (
    add_comment_to_issue,
    add_comment_to_pr,
    create_issue,
    update_issue,
    update_pull_request,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the Security Engineer in a multi-agent development team.

Your responsibilities:
1. Perform security code reviews (SAST)
2. Identify OWASP Top 10 vulnerabilities
3. Check for secrets/credentials in code
4. Review authentication and authorization logic
5. Analyze dependencies for known CVEs
6. Ensure secure communication (TLS, certificate validation)
7. Review input validation and sanitization
8. Check for injection vulnerabilities (SQL, command, LDAP)
9. Assess cryptography usage
10. Review error handling for information leakage

Security standards:
- OWASP Top 10
- CWE/SANS Top 25
- NIST Cybersecurity Framework
- Principle of least privilege
- Defense in depth
- Secure by default

When reviewing:
- Scan for hardcoded secrets (API keys, passwords, tokens)
- Check for SQL/command injection vulnerabilities
- Review authentication implementation
- Verify HTTPS/TLS usage
- Check for insecure deserialization
- Review access control logic
- Assess logging (no sensitive data in logs)

Report ALL findings as GitHub issues with 'security' label and appropriate severity.
Use CVSS scoring when possible.
"""


def create_security_agent(llm):
    """Create the security agent node function."""

    async def security_node(state: AgentState) -> AgentState:
        """Security agent that performs security analysis."""
        logger.info("Security agent processing task: %s", state.current_task)

        tools = [
            read_file,
            list_directory,
            search_code,
            run_command,
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
Workspace: {state.workspace_path}
Current task: {state.current_task.model_dump() if state.current_task else "None"}
Implementation: {state.developer_output.get("implementation", "Not yet implemented")}

Please perform a comprehensive security review:
1. Scan for hardcoded secrets and credentials
2. Check for OWASP Top 10 vulnerabilities
3. Review authentication and authorization
4. Analyze dependency security (run: pip audit or npm audit)
5. Check for insecure configurations
6. Review input validation
7. Assess error handling (no sensitive info in errors)
8. Check for secure communication
9. Review logging practices
10. Create GitHub issues for any security findings

Provide a security assessment summary at the end.
"""
                ),
            ]
        )

        messages = prompt.format_messages(messages=state.messages)
        response = await agent_llm.ainvoke(messages)

        security_output = {
            "assessment": response.content,
            "task_id": state.current_task.id if state.current_task else None,
        }

        return AgentState(
            **{
                **state.model_dump(exclude={"messages"}),
                "messages": [response],
                "current_agent": AgentRole.SECURITY,
                "next_agent": AgentRole.ORCHESTRATOR,
                "security_output": security_output,
            }
        )

    return security_node
