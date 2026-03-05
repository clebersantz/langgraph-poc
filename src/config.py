"""Configuration management for the multi-agent system."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM settings
    openai_api_key: SecretStr = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    # GitHub settings
    github_token: SecretStr = Field(default="", alias="GITHUB_TOKEN")
    github_owner: str = Field(default="", alias="GITHUB_OWNER")
    github_repo: str = Field(default="", alias="GITHUB_REPO")

    # Git settings
    git_user_name: str = Field(default="LangGraph Agent", alias="GIT_USER_NAME")
    git_user_email: str = Field(
        default="agent@langgraph.local", alias="GIT_USER_EMAIL"
    )
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
