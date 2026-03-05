"""Tests for the LangGraph workflow graph."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from src.state import AgentRole, AgentState


class TestRouteAfterOrchestrator:
    """Tests for the routing logic after orchestrator."""

    def test_routes_to_architect(self):
        from src.graph import _route_after_orchestrator

        state = AgentState(next_agent=AgentRole.ARCHITECT)
        assert _route_after_orchestrator(state) == "architect"

    def test_routes_to_developer(self):
        from src.graph import _route_after_orchestrator

        state = AgentState(next_agent=AgentRole.DEVELOPER)
        assert _route_after_orchestrator(state) == "developer"

    def test_routes_to_qa(self):
        from src.graph import _route_after_orchestrator

        state = AgentState(next_agent=AgentRole.QA)
        assert _route_after_orchestrator(state) == "qa"

    def test_routes_to_security(self):
        from src.graph import _route_after_orchestrator

        state = AgentState(next_agent=AgentRole.SECURITY)
        assert _route_after_orchestrator(state) == "security"

    def test_routes_to_documentation(self):
        from src.graph import _route_after_orchestrator

        state = AgentState(next_agent=AgentRole.DOCUMENTATION)
        assert _route_after_orchestrator(state) == "documentation"

    def test_ends_when_complete(self):
        from langgraph.graph import END

        from src.graph import _route_after_orchestrator

        state = AgentState(is_complete=True, next_agent=AgentRole.ARCHITECT)
        assert _route_after_orchestrator(state) == END

    def test_ends_when_no_next_agent(self):
        from langgraph.graph import END

        from src.graph import _route_after_orchestrator

        state = AgentState(next_agent=None)
        assert _route_after_orchestrator(state) == END


class TestBuildGraph:
    """Tests for the build_graph function."""

    @patch("src.graph._create_llm")
    def test_builds_graph_successfully(self, mock_create_llm):
        from src.graph import build_graph

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = MagicMock()
        mock_create_llm.return_value = mock_llm

        graph = build_graph()
        assert graph is not None

    @patch("src.graph._create_llm")
    def test_graph_has_all_nodes(self, mock_create_llm):
        from src.graph import build_graph

        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = MagicMock()
        mock_create_llm.return_value = mock_llm

        graph = build_graph()
        node_names = set(graph.nodes.keys())
        expected = {"orchestrator", "architect", "developer", "qa", "security", "documentation"}
        assert expected.issubset(node_names)
