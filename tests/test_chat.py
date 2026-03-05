"""Tests for the /chat endpoint and related helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import _extract_branch, _extract_repo_url, _format_chat_reply, app
from src.state import AgentState


class TestExtractRepoUrl:
    """Tests for the _extract_repo_url helper."""

    def test_extracts_github_url(self):
        text = "Please clone https://github.com/org/repo and add a feature."
        assert _extract_repo_url(text) == "https://github.com/org/repo"

    def test_extracts_github_url_with_subpath(self):
        text = "Work on https://github.com/clebersantz/langgraph-poc please"
        assert _extract_repo_url(text) == "https://github.com/clebersantz/langgraph-poc"

    def test_returns_empty_when_no_url(self):
        assert _extract_repo_url("Create a new FastAPI service") == ""

    def test_returns_first_url_when_multiple(self):
        text = "Compare https://github.com/a/repo1 and https://github.com/b/repo2"
        assert _extract_repo_url(text) == "https://github.com/a/repo1"


class TestExtractBranch:
    """Tests for the _extract_branch helper."""

    def test_extracts_branch_colon_syntax(self):
        assert _extract_branch("Work on branch: feature/new-agent") == "feature/new-agent"

    def test_extracts_branch_space_syntax(self):
        assert _extract_branch("checkout branch main and add tests") == "main"

    def test_extracts_on_x_branch(self):
        assert _extract_branch("on develop branch") == "develop"

    def test_defaults_to_main_when_absent(self):
        assert _extract_branch("Create a new endpoint") == "main"

    def test_case_insensitive(self):
        assert _extract_branch("Branch: release/1.0") == "release/1.0"


class TestFormatChatReply:
    """Tests for the _format_chat_reply helper."""

    def test_marks_complete_workflow(self):
        state = AgentState(is_complete=True, iteration_count=3)
        reply = _format_chat_reply(state)
        assert "completed successfully" in reply
        assert "3 iteration" in reply

    def test_marks_stopped_workflow(self):
        state = AgentState(is_complete=False, iteration_count=5)
        reply = _format_chat_reply(state)
        assert "stopped" in reply.lower()

    def test_includes_errors(self):
        state = AgentState(is_complete=False, errors=["Something went wrong"])
        reply = _format_chat_reply(state)
        assert "Something went wrong" in reply

    def test_includes_final_result(self):
        state = AgentState(is_complete=True, final_result={"pr_url": "https://github.com/x"})
        reply = _format_chat_reply(state)
        assert "pr_url" in reply


class TestChatEndpoint:
    """Integration tests for the POST /chat endpoint."""

    @pytest.fixture
    def client(self):
        return TestClient(app, raise_server_exceptions=False)

    @patch("src.main.build_graph")
    def test_chat_returns_session_id(self, mock_build_graph, client):
        mock_graph = AsyncMock()
        completed_state = AgentState(is_complete=True, iteration_count=2)
        mock_graph.ainvoke.return_value = completed_state.model_dump()
        mock_build_graph.return_value = mock_graph

        res = client.post("/chat", json={"message": "Build a REST API"})
        assert res.status_code == 200
        data = res.json()
        assert "session_id" in data
        assert data["status"] == "completed"

    @patch("src.main.build_graph")
    def test_chat_reuses_existing_session(self, mock_build_graph, client):
        mock_graph = AsyncMock()
        state = AgentState(is_complete=True, iteration_count=1)
        mock_graph.ainvoke.return_value = state.model_dump()
        mock_build_graph.return_value = mock_graph

        res1 = client.post("/chat", json={"message": "First message"})
        session_id = res1.json()["session_id"]

        res2 = client.post("/chat", json={"message": "Follow-up", "session_id": session_id})
        assert res2.json()["session_id"] == session_id

    @patch("src.main.build_graph")
    def test_chat_extracts_repo_url_from_message(self, mock_build_graph, client):
        invoked_states: list[AgentState] = []

        async def capture_invoke(state, config=None):
            invoked_states.append(state)
            final = AgentState(is_complete=True, iteration_count=1)
            return final.model_dump()

        mock_graph = MagicMock()
        mock_graph.ainvoke = capture_invoke
        mock_build_graph.return_value = mock_graph

        client.post(
            "/chat",
            json={"message": "Clone https://github.com/org/my-repo and add tests"},
        )
        assert len(invoked_states) == 1
        assert invoked_states[0].project_repo == "https://github.com/org/my-repo"

    @patch("src.main.build_graph")
    def test_chat_returns_500_on_graph_error(self, mock_build_graph, client):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.side_effect = RuntimeError("LLM unavailable")
        mock_build_graph.return_value = mock_graph

        res = client.post("/chat", json={"message": "Do something"})
        assert res.status_code == 500


class TestChatHistoryEndpoint:
    """Tests for GET /chat/history/{session_id}."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_returns_404_for_unknown_session(self, client):
        res = client.get("/chat/history/nonexistent-session-id")
        assert res.status_code == 404

    @patch("src.main.build_graph")
    def test_returns_history_after_chat(self, mock_build_graph, client):
        mock_graph = AsyncMock()
        state = AgentState(is_complete=True, iteration_count=1)
        mock_graph.ainvoke.return_value = state.model_dump()
        mock_build_graph.return_value = mock_graph

        res = client.post("/chat", json={"message": "Hello agents"})
        session_id = res.json()["session_id"]

        history_res = client.get(f"/chat/history/{session_id}")
        assert history_res.status_code == 200
        history = history_res.json()
        assert any(m["role"] == "user" and m["content"] == "Hello agents" for m in history)
        assert any(m["role"] == "assistant" for m in history)


class TestChatUI:
    """Tests for the GET / chat UI endpoint."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_returns_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]
        assert "LangGraph" in res.text

    def test_includes_chat_form(self, client):
        res = client.get("/")
        assert "<form" in res.text
        assert "/chat" in res.text
