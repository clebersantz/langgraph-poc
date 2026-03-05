"""Configuration management for the multi-agent system."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM settings
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), alias="ANTHROPIC_API_KEY")
    llm_provider: Literal["openai", "anthropic", "azure", "mock"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    # Azure OpenAI settings (used when LLM_PROVIDER=azure)
    azure_openai_api_key: SecretStr = Field(default=SecretStr(""), alias="AZURE_OPENAI_API_KEY")
    azure_openai_base_url: str = Field(default="", alias="AZURE_OPENAI_BASE_URL")
    azure_openai_api_version: str = Field(default="2024-02-01", alias="AZURE_OPENAI_API_VERSION")
    # Optional explicit deployment name; falls back to llm_model when not set.
    # Set AZURE_OPENAI_DEPLOYMENT if your deployment name differs from the model name.
    azure_openai_deployment: str = Field(default="", alias="AZURE_OPENAI_DEPLOYMENT")

    # GitHub settings
    github_token: SecretStr = Field(default=SecretStr(""), alias="GITHUB_TOKEN")
    github_owner: str = Field(default="", alias="GITHUB_OWNER")
    github_repo: str = Field(default="", alias="GITHUB_REPO")

    # Git settings
    git_user_name: str = Field(default="LangGraph Agent", alias="GIT_USER_NAME")
    git_user_email: str = Field(default="agent@langgraph.local", alias="GIT_USER_EMAIL")
    workspace_dir: str = Field(default="/tmp/workspace", alias="WORKSPACE_DIR")

    # Server settings
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    debug: bool = Field(default=False, alias="DEBUG")

    # Agent settings
    max_iterations: int = Field(default=10, alias="MAX_ITERATIONS")
    recursion_limit: int = Field(default=50, alias="RECURSION_LIMIT")

    model_config = {"populate_by_name": True}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
