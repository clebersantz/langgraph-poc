"""LangGraph multi-agent orchestration graph."""

from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.architect import create_architect_agent
from src.agents.developer import create_developer_agent
from src.agents.documentation import create_documentation_agent
from src.agents.orchestrator import create_orchestrator_agent
from src.agents.qa import create_qa_agent
from src.agents.security import create_security_agent
from src.config import get_settings
from src.state import AgentRole, AgentState

logger = logging.getLogger(__name__)


def _create_llm():
    """Instantiate the LLM based on settings."""
    settings = get_settings()
    if settings.llm_provider == "azure":
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_base_url,
            api_key=settings.azure_openai_api_key.get_secret_value(),
            api_version=settings.azure_openai_api_version,
            azure_deployment=settings.llm_model,
            temperature=settings.llm_temperature,
        )
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key.get_secret_value(),
            temperature=settings.llm_temperature,
        )
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=settings.llm_temperature,
    )


def _route_after_orchestrator(state: AgentState) -> str:
    """Route to the next agent based on orchestrator output."""
    if state.is_complete:
        logger.info("Workflow complete after %d iterations", state.iteration_count)
        return END
    if state.next_agent == AgentRole.ARCHITECT:
        return "architect"
    if state.next_agent == AgentRole.DEVELOPER:
        return "developer"
    if state.next_agent == AgentRole.QA:
        return "qa"
    if state.next_agent == AgentRole.SECURITY:
        return "security"
    if state.next_agent == AgentRole.DOCUMENTATION:
        return "documentation"
    # Default: end if no valid next agent
    logger.warning("No valid next agent found, ending workflow")
    return END


def build_graph() -> CompiledStateGraph:
    """Build and compile the multi-agent LangGraph workflow.

    Returns:
        Compiled LangGraph state graph.
    """
    llm = _create_llm()

    # Create agent node functions
    orchestrator = create_orchestrator_agent(llm)
    architect = create_architect_agent(llm)
    developer = create_developer_agent(llm)
    qa = create_qa_agent(llm)
    security = create_security_agent(llm)
    documentation = create_documentation_agent(llm)

    # Build graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("orchestrator", orchestrator)
    graph.add_node("architect", architect)
    graph.add_node("developer", developer)
    graph.add_node("qa", qa)
    graph.add_node("security", security)
    graph.add_node("documentation", documentation)

    # Entry point
    graph.add_edge(START, "orchestrator")

    # Orchestrator routes to agents
    graph.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {
            "architect": "architect",
            "developer": "developer",
            "qa": "qa",
            "security": "security",
            "documentation": "documentation",
            END: END,
        },
    )

    # All agents return to orchestrator
    for agent_name in ["architect", "developer", "qa", "security", "documentation"]:
        graph.add_edge(agent_name, "orchestrator")

    compiled = graph.compile()
    logger.info("Multi-agent graph compiled successfully")
    return compiled


class _GraphHolder:
    """Thread-safe singleton holder for the compiled graph."""

    _instance: CompiledStateGraph | None = None

    @classmethod
    def get(cls) -> CompiledStateGraph:
        """Return (or lazily create) the singleton compiled graph."""
        if cls._instance is None:
            cls._instance = build_graph()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (useful for testing)."""
        cls._instance = None


def get_graph() -> CompiledStateGraph:
    """Return the singleton compiled graph."""
    return _GraphHolder.get()
