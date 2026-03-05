"""LangGraph multi-agent orchestration graph."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage
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

# ---------------------------------------------------------------------------
# Mock LLM — used when LLM_PROVIDER=mock (offline / CI integration testing)
# ---------------------------------------------------------------------------

_MOCK_FILES_JSON = json.dumps(
    {
        "files": [
            {"path": "main.py", "content": 'print("Hello, World!")\n'},
            {
                "path": "test_main.py",
                "content": (
                    "import subprocess\n\n\n"
                    "def test_main():\n"
                    "    result = subprocess.run(\n"
                    '        ["python3", "main.py"],\n'
                    "        capture_output=True,\n"
                    "        text=True,\n"
                    "    )\n"
                    '    assert "Hello" in result.stdout\n'
                ),
            },
            {
                "path": "README.md",
                "content": (
                    "# Hello World\n\n"
                    "A minimal Python hello-world project.\n\n"
                    "## Usage\n\n"
                    "```bash\npython3 main.py\n```\n"
                ),
            },
        ],
        "summary": "Created hello-world project with main.py, tests, and README.",
    },
    indent=2,
)


class _MockChatModel:
    """Offline LLM stub for integration / e2e testing without real credentials.

    Routing rules (detected from the system message content):
    - Orchestrator (first call, no developer_output): route to "developer".
    - Orchestrator (developer output present): return "DONE".
    - Any specialist agent (developer / qa / documentation / …): return a
      ``\\`\\`\\`json`` block with the three hello-world project files so that the
      ``extract_files_from_content`` fallback creates them in the workspace.
    """

    def bind_tools(self, tools: list[Any], **kwargs: Any) -> _MockChatModel:
        """Return self — mock ignores tool binding."""
        return self

    async def ainvoke(self, messages: list[BaseMessage], **kwargs: Any) -> AIMessage:
        """Return a canned response based on which agent is running."""
        # Detect context from the system message.
        sys_content = next(
            (
                str(getattr(m, "content", ""))
                for m in messages
                if getattr(m, "type", "") == "system"
            ),
            "",
        )

        is_orchestrator = "Project Manager" in sys_content

        if is_orchestrator:
            # After the developer has run, route to DONE.
            last_human = next(
                (m for m in reversed(messages) if getattr(m, "type", "") == "human"),
                None,
            )
            human_text = str(getattr(last_human, "content", "")) if last_human else ""
            if "implementation" in human_text:
                return AIMessage(
                    content=json.dumps(
                        {
                            "analysis": "Developer completed all files. Project is done.",
                            "next_agent": "DONE",
                        }
                    )
                )
            return AIMessage(
                content=json.dumps(
                    {
                        "analysis": "Assigning hello-world project to developer.",
                        "tasks": [
                            {"title": "Create hello-world project", "assigned_to": "developer"}
                        ],
                        "next_agent": "developer",
                    }
                )
            )

        # Specialist agent — return JSON file bundle so the fallback extractor
        # writes main.py / test_main.py / README.md into the workspace.
        return AIMessage(content=f"```json\n{_MOCK_FILES_JSON}\n```")


def _create_llm():
    """Instantiate the LLM based on settings."""
    settings = get_settings()

    if settings.llm_provider == "mock":
        logger.info("Using MockChatModel (offline testing mode)")
        return _MockChatModel()

    if settings.llm_provider == "azure":
        api_key = settings.azure_openai_api_key.get_secret_value() or None
        base_url = settings.azure_openai_base_url
        # Deployment name: use explicit override (strip whitespace), or fall back to the model name.
        deployment = settings.azure_openai_deployment.strip() or settings.llm_model

        # New Azure AI Foundry / AI Services endpoints expose an OpenAI-compatible
        # "/v1/" path. Using AzureChatOpenAI with these URLs would double-prefix
        # "/openai/" in the path, causing 404s. Use ChatOpenAI with base_url instead.
        if "/v1/" in base_url:
            return ChatOpenAI(
                model=deployment,
                api_key=api_key,
                base_url=base_url,
                temperature=settings.llm_temperature,
            )

        # Traditional Azure OpenAI endpoint: https://<resource>.openai.azure.com/
        return AzureChatOpenAI(
            azure_endpoint=base_url,
            api_key=api_key,
            api_version=settings.azure_openai_api_version,
            azure_deployment=deployment,
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
