"""Tests for agent node functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from src.agents.orchestrator import _parse_next_agent
from src.state import AgentRole, AgentState


@pytest.fixture
def sample_state() -> AgentState:
    """Return a minimal AgentState for testing."""
    return AgentState(
        project_goal="Build a REST API",
        project_repo="https://github.com/example/repo",
        project_branch="main",
        workspace_path="/tmp/test-workspace",
        max_iterations=5,
    )


@pytest.fixture
def mock_llm():
    """Return a mock LLM that returns a simple AIMessage."""
    llm = MagicMock()
    bound_llm = AsyncMock()
    bound_llm.ainvoke.return_value = AIMessage(content="Route to architect next.")
    llm.bind_tools.return_value = bound_llm
    return llm


class TestParseNextAgent:
    """Tests for the orchestrator's _parse_next_agent helper."""

    def test_parses_architect(self):
        assert _parse_next_agent("We should route to the architect next.") == "architect"

    def test_parses_developer(self):
        assert _parse_next_agent("The developer should implement this.") == "developer"

    def test_parses_qa(self):
        assert _parse_next_agent("QA should test the implementation.") == "qa"

    def test_parses_quality_assurance(self):
        assert _parse_next_agent("quality assurance review needed") == "qa"

    def test_parses_security(self):
        assert _parse_next_agent("security review is required.") == "security"

    def test_parses_documentation(self):
        assert _parse_next_agent("documentation needs updating.") == "documentation"

    def test_parses_done(self):
        assert _parse_next_agent("The project is done and complete.") == "done"

    def test_parses_complete(self):
        assert _parse_next_agent("Everything is complete.") == "done"

    def test_returns_none_for_unknown(self):
        assert _parse_next_agent("This is an ambiguous response.") is None

    def test_case_insensitive(self):
        assert _parse_next_agent("ARCHITECT should design this.") == "architect"


@pytest.mark.asyncio
class TestOrchestratorAgent:
    """Tests for the orchestrator agent node."""

    async def test_increments_iteration_count(self, sample_state, mock_llm):
        from src.agents.orchestrator import create_orchestrator_agent

        node = create_orchestrator_agent(mock_llm)
        result = await node(sample_state)

        assert result.iteration_count == 1

    async def test_sets_current_agent_to_orchestrator(self, sample_state, mock_llm):
        from src.agents.orchestrator import create_orchestrator_agent

        node = create_orchestrator_agent(mock_llm)
        result = await node(sample_state)

        assert result.current_agent == AgentRole.ORCHESTRATOR

    async def test_marks_complete_when_max_iterations_reached(self, mock_llm):
        from src.agents.orchestrator import create_orchestrator_agent

        state = AgentState(
            project_goal="Test goal",
            max_iterations=1,
            iteration_count=0,
        )
        # Response doesn't mention any agent (returns None)
        bound_llm = AsyncMock()
        bound_llm.ainvoke.return_value = AIMessage(content="No specific agent mentioned.")
        mock_llm.bind_tools.return_value = bound_llm

        node = create_orchestrator_agent(mock_llm)
        result = await node(state)

        assert result.is_complete is True

    async def test_routes_to_architect(self, sample_state, mock_llm):
        from src.agents.orchestrator import create_orchestrator_agent

        bound_llm = AsyncMock()
        bound_llm.ainvoke.return_value = AIMessage(content="Route to architect for design.")
        mock_llm.bind_tools.return_value = bound_llm

        node = create_orchestrator_agent(mock_llm)
        result = await node(sample_state)

        assert result.next_agent == AgentRole.ARCHITECT


@pytest.mark.asyncio
class TestArchitectAgent:
    """Tests for the architect agent node."""

    async def test_sets_architect_output(self, sample_state, mock_llm):
        from src.agents.architect import create_architect_agent

        bound_llm = AsyncMock()
        bound_llm.ainvoke.return_value = AIMessage(content="Architecture design: microservices.")
        mock_llm.bind_tools.return_value = bound_llm

        node = create_architect_agent(mock_llm)
        result = await node(sample_state)

        assert "analysis" in result.architect_output
        assert result.architect_output["analysis"] == "Architecture design: microservices."

    async def test_returns_to_orchestrator(self, sample_state, mock_llm):
        from src.agents.architect import create_architect_agent

        node = create_architect_agent(mock_llm)
        result = await node(sample_state)

        assert result.next_agent == AgentRole.ORCHESTRATOR


@pytest.mark.asyncio
class TestDeveloperAgent:
    """Tests for the developer agent node."""

    async def test_sets_developer_output(self, sample_state, mock_llm):
        from src.agents.developer import create_developer_agent

        bound_llm = AsyncMock()
        bound_llm.ainvoke.return_value = AIMessage(content="Implemented REST API endpoints.")
        mock_llm.bind_tools.return_value = bound_llm

        node = create_developer_agent(mock_llm)
        result = await node(sample_state)

        assert "implementation" in result.developer_output

    async def test_returns_to_orchestrator(self, sample_state, mock_llm):
        from src.agents.developer import create_developer_agent

        node = create_developer_agent(mock_llm)
        result = await node(sample_state)

        assert result.next_agent == AgentRole.ORCHESTRATOR


@pytest.mark.asyncio
class TestQAAgent:
    """Tests for the QA agent node."""

    async def test_sets_qa_output(self, sample_state, mock_llm):
        from src.agents.qa import create_qa_agent

        bound_llm = AsyncMock()
        bound_llm.ainvoke.return_value = AIMessage(content="All tests pass. 95% coverage.")
        mock_llm.bind_tools.return_value = bound_llm

        node = create_qa_agent(mock_llm)
        result = await node(sample_state)

        assert "assessment" in result.qa_output

    async def test_returns_to_orchestrator(self, sample_state, mock_llm):
        from src.agents.qa import create_qa_agent

        node = create_qa_agent(mock_llm)
        result = await node(sample_state)

        assert result.next_agent == AgentRole.ORCHESTRATOR


@pytest.mark.asyncio
class TestSecurityAgent:
    """Tests for the security agent node."""

    async def test_sets_security_output(self, sample_state, mock_llm):
        from src.agents.security import create_security_agent

        bound_llm = AsyncMock()
        bound_llm.ainvoke.return_value = AIMessage(content="No critical vulnerabilities found.")
        mock_llm.bind_tools.return_value = bound_llm

        node = create_security_agent(mock_llm)
        result = await node(sample_state)

        assert "assessment" in result.security_output

    async def test_returns_to_orchestrator(self, sample_state, mock_llm):
        from src.agents.security import create_security_agent

        node = create_security_agent(mock_llm)
        result = await node(sample_state)

        assert result.next_agent == AgentRole.ORCHESTRATOR


@pytest.mark.asyncio
class TestDocumentationAgent:
    """Tests for the documentation agent node."""

    async def test_sets_documentation_output(self, sample_state, mock_llm):
        from src.agents.documentation import create_documentation_agent

        bound_llm = AsyncMock()
        bound_llm.ainvoke.return_value = AIMessage(content="README.md updated.")
        mock_llm.bind_tools.return_value = bound_llm

        node = create_documentation_agent(mock_llm)
        result = await node(sample_state)

        assert "documentation" in result.documentation_output

    async def test_returns_to_orchestrator(self, sample_state, mock_llm):
        from src.agents.documentation import create_documentation_agent

        node = create_documentation_agent(mock_llm)
        result = await node(sample_state)

        assert result.next_agent == AgentRole.ORCHESTRATOR


# ---------------------------------------------------------------------------
# Tool execution loop tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestToolExecutionLoop:
    """Tests that agent nodes execute tool calls returned by the LLM."""

    async def test_developer_executes_tool_calls(self, sample_state):
        """Developer agent must invoke tools when the LLM returns tool calls."""
        from langchain_core.messages import AIMessage

        from src.agents.developer import create_developer_agent

        tool_call = {
            "name": "create_file",
            "args": {"path": "/tmp/test_dev_tool/main.py", "content": "print('hello')"},
            "id": "call_abc123",
            "type": "tool_call",
        }
        first_response = AIMessage(content="", tool_calls=[tool_call])
        final_response = AIMessage(content="File created successfully.")

        llm = MagicMock()
        bound_llm = AsyncMock()
        # First call returns tool calls, second call returns final text
        bound_llm.ainvoke.side_effect = [first_response, final_response]
        llm.bind_tools.return_value = bound_llm

        node = create_developer_agent(llm)
        result = await node(sample_state)

        # LLM must have been called twice: initial + after tool result
        assert bound_llm.ainvoke.call_count == 2
        # Final output must reflect the post-tool response
        assert result.developer_output["implementation"] == "File created successfully."

    async def test_documentation_executes_tool_calls(self, sample_state):
        """Documentation agent must invoke tools when the LLM returns tool calls."""
        from langchain_core.messages import AIMessage

        from src.agents.documentation import create_documentation_agent

        tool_call = {
            "name": "create_file",
            "args": {"path": "/tmp/test_doc_tool/README.md", "content": "# Project"},
            "id": "call_doc123",
            "type": "tool_call",
        }
        first_response = AIMessage(content="", tool_calls=[tool_call])
        final_response = AIMessage(content="README.md created.")

        llm = MagicMock()
        bound_llm = AsyncMock()
        bound_llm.ainvoke.side_effect = [first_response, final_response]
        llm.bind_tools.return_value = bound_llm

        node = create_documentation_agent(llm)
        result = await node(sample_state)

        assert bound_llm.ainvoke.call_count == 2
        assert result.documentation_output["documentation"] == "README.md created."

    async def test_tool_error_is_handled_gracefully(self, sample_state):
        """A tool that raises must not crash the agent — the error is fed back to the LLM."""
        from langchain_core.messages import AIMessage, ToolMessage

        from src.agents.developer import create_developer_agent

        # Patch create_file to raise
        bad_tool = MagicMock()
        bad_tool.name = "create_file"
        bad_tool.ainvoke = AsyncMock(side_effect=OSError("disk full"))

        tool_call = {
            "name": "create_file",
            "args": {"path": "/no/such/path", "content": "data"},
            "id": "call_err1",
            "type": "tool_call",
        }
        first_response = AIMessage(content="", tool_calls=[tool_call])
        final_response = AIMessage(content="Could not create file due to error.")

        llm = MagicMock()
        bound_llm = AsyncMock()
        bound_llm.ainvoke.side_effect = [first_response, final_response]
        # Provide only bad_tool so the executor picks it up by name
        llm.bind_tools.return_value = bound_llm

        # We need to patch the tools inside the node; the simplest way is to
        # intercept the second ainvoke call and confirm a ToolMessage was sent.
        captured_messages: list = []

        async def capture_ainvoke(messages):
            captured_messages.append(messages)
            return bound_llm.ainvoke.side_effect.pop(0)

        bound_llm.ainvoke.side_effect = None
        bound_llm.ainvoke = capture_ainvoke
        bound_llm.ainvoke.side_effect = [first_response, final_response]

        # Re-wire so bound_llm.ainvoke pops from side_effect list sequentially
        responses = [first_response, final_response]
        call_count = [0]

        async def sequential_ainvoke(messages):
            captured_messages.append(messages)
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        bound_llm.ainvoke = sequential_ainvoke
        llm.bind_tools.return_value = bound_llm

        node = create_developer_agent(llm)
        await node(sample_state)

        # The second call must include a ToolMessage with the error
        assert call_count[0] == 2
        second_call_msgs = captured_messages[1]
        tool_messages = [m for m in second_call_msgs if isinstance(m, ToolMessage)]
        assert tool_messages, "Expected a ToolMessage containing the error"
        assert (
            "disk full" in tool_messages[0].content or "error" in tool_messages[0].content.lower()
        )
