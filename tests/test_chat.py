"""Tests for the /chat endpoint and related helpers."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import (
    _extract_branch,
    _extract_last_message_content,
    _extract_repo_url,
    _format_chat_reply,
    app,
)
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
        return TestClient(app)

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

    def test_includes_stop_button(self, client):
        res = client.get("/")
        assert "stop-btn" in res.text

    def test_includes_resend_button(self, client):
        res = client.get("/")
        assert "resend-btn" in res.text

    def test_uses_streaming_endpoint(self, client):
        res = client.get("/")
        assert "/chat/stream" in res.text


class TestExtractLastMessageContent:
    """Tests for the _extract_last_message_content helper."""

    def test_returns_empty_for_no_messages(self):
        assert _extract_last_message_content([]) == ""

    def test_extracts_content_from_object_with_attribute(self):
        class FakeMsg:
            content = "Hello from agent"

        assert _extract_last_message_content([FakeMsg()]) == "Hello from agent"

    def test_extracts_content_from_dict(self):
        assert _extract_last_message_content([{"content": "Dict content"}]) == "Dict content"

    def test_handles_list_content(self):
        msg = {"content": [{"text": "part one"}, {"text": "part two"}]}
        result = _extract_last_message_content([msg])
        assert "part one" in result
        assert "part two" in result

    def test_returns_last_message_only(self):
        msgs = [{"content": "first"}, {"content": "last"}]
        assert _extract_last_message_content(msgs) == "last"


class TestChatStreamEndpoint:
    """Tests for the POST /chat/stream SSE endpoint."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @patch("src.main.build_graph")
    def test_stream_returns_event_stream_content_type(self, mock_build_graph, client):
        """The SSE endpoint must respond with text/event-stream."""
        mock_graph = MagicMock()

        async def fake_astream(state, config=None, stream_mode="updates"):
            completed = AgentState(
                is_complete=True, iteration_count=1, current_agent="orchestrator"
            )
            yield completed.model_dump()

        mock_graph.astream = fake_astream
        mock_build_graph.return_value = mock_graph

        res = client.post(
            "/chat/stream",
            json={"message": "Build something"},
        )
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

    @patch("src.main.build_graph")
    def test_stream_emits_session_event(self, mock_build_graph, client):
        """First SSE event must be a session event containing a session_id."""
        mock_graph = MagicMock()

        async def fake_astream(state, config=None, stream_mode="updates"):
            completed = AgentState(is_complete=True, iteration_count=1)
            yield completed.model_dump()

        mock_graph.astream = fake_astream
        mock_build_graph.return_value = mock_graph

        res = client.post("/chat/stream", json={"message": "hello"})
        lines = [ln for ln in res.text.splitlines() if ln.startswith("data: ")]
        assert lines, "No SSE data lines received"
        first = json.loads(lines[0][6:])
        assert first["type"] == "session"
        assert "session_id" in first

    @patch("src.main.build_graph")
    def test_stream_emits_done_event(self, mock_build_graph, client):
        """A completed workflow must emit a 'done' SSE event."""
        mock_graph = MagicMock()

        async def fake_astream(state, config=None, stream_mode="updates"):
            completed = AgentState(is_complete=True, iteration_count=2)
            yield completed.model_dump()

        mock_graph.astream = fake_astream
        mock_build_graph.return_value = mock_graph

        res = client.post("/chat/stream", json={"message": "hello"})
        events = [json.loads(ln[6:]) for ln in res.text.splitlines() if ln.startswith("data: ")]
        done_events = [e for e in events if e["type"] == "done"]
        assert done_events, "No 'done' event received"
        assert done_events[0]["status"] == "completed"

    @patch("src.main.build_graph")
    def test_stream_emits_agent_events_for_known_agents(self, mock_build_graph, client):
        """A snapshot with current_agent set must produce agent_start + agent_message events."""
        mock_graph = MagicMock()

        async def fake_astream(state, config=None, stream_mode="updates"):
            class FakeMsg:
                content = "I am designing the architecture."

            snapshot = AgentState(
                is_complete=False,
                iteration_count=1,
                current_agent="architect",
            ).model_dump()
            # inject a real-looking message
            snapshot["messages"] = [{"content": "I am designing the architecture."}]
            yield snapshot
            # final completed snapshot
            yield AgentState(is_complete=True, iteration_count=2).model_dump()

        mock_graph.astream = fake_astream
        mock_build_graph.return_value = mock_graph

        res = client.post("/chat/stream", json={"message": "Design a system"})
        events = [json.loads(ln[6:]) for ln in res.text.splitlines() if ln.startswith("data: ")]
        agent_start = [e for e in events if e["type"] == "agent_start"]
        agent_msg = [e for e in events if e["type"] == "agent_message"]
        assert agent_start, "Expected agent_start event"
        assert agent_start[0]["agent"] == "architect"
        assert agent_msg, "Expected agent_message event"
        assert "architect" in agent_msg[0]["content"].lower()

    @patch("src.main.build_graph")
    def test_stream_emits_error_event_on_failure(self, mock_build_graph, client):
        """A graph exception must result in an 'error' SSE event."""
        mock_graph = MagicMock()

        async def failing_astream(state, config=None, stream_mode="updates"):
            raise RuntimeError("LLM is down")
            yield  # pragma: no cover  # keeps function as async generator

        mock_graph.astream = failing_astream
        mock_build_graph.return_value = mock_graph

        res = client.post("/chat/stream", json={"message": "hello"})
        events = [json.loads(ln[6:]) for ln in res.text.splitlines() if ln.startswith("data: ")]
        error_events = [e for e in events if e["type"] == "error"]
        assert error_events, "Expected an error event"
        assert "LLM is down" in error_events[0]["message"]

    @patch("src.main.build_graph")
    def test_stream_reuses_existing_session(self, mock_build_graph, client):
        """Providing an existing session_id must reuse that session."""
        mock_graph = MagicMock()

        async def fake_astream(state, config=None, stream_mode="updates"):
            yield AgentState(is_complete=True, iteration_count=1).model_dump()

        mock_graph.astream = fake_astream
        mock_build_graph.return_value = mock_graph

        # First call to get session id
        res1 = client.post("/chat/stream", json={"message": "first"})
        events1 = [json.loads(ln[6:]) for ln in res1.text.splitlines() if ln.startswith("data: ")]
        session_id = next(e["session_id"] for e in events1 if e["type"] == "session")

        # Second call reusing the same session
        res2 = client.post("/chat/stream", json={"message": "second", "session_id": session_id})
        events2 = [json.loads(ln[6:]) for ln in res2.text.splitlines() if ln.startswith("data: ")]
        session_event = next(e for e in events2 if e["type"] == "session")
        assert session_event["session_id"] == session_id


class TestRunFilesEndpoint:
    """Tests for the GET /run/{run_id}/files endpoint."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_returns_404_for_unknown_run(self, client):
        res = client.get("/run/nonexistent-run-id/files")
        assert res.status_code == 404

    def test_lists_files_in_workspace(self, client):
        """Endpoint must list files present in the run workspace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = "test-run-files-123"
            workspace = os.path.join(tmpdir, run_id)
            os.makedirs(workspace)
            # Create sample files
            open(os.path.join(workspace, "main.py"), "w").close()
            open(os.path.join(workspace, "README.md"), "w").close()
            subdir = os.path.join(workspace, "tests")
            os.makedirs(subdir)
            open(os.path.join(subdir, "test_main.py"), "w").close()

            with patch("src.main.get_settings") as mock_settings:
                settings = MagicMock()
                settings.workspace_dir = tmpdir
                settings.recursion_limit = 50
                mock_settings.return_value = settings

                res = client.get(f"/run/{run_id}/files")

        assert res.status_code == 200
        data = res.json()
        assert data["run_id"] == run_id
        files = data["files"]
        assert any("main.py" in f for f in files)
        assert any("README.md" in f for f in files)
        assert any("test_main.py" in f for f in files)
