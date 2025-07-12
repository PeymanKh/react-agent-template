"""
Securely loads environment variables both in docker image and local run.
"""
import os
from enum import Enum
from pathlib import Path
from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings


class LogLevel(str, Enum):
    """Allowed log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LangChainConfig(BaseSettings):
    """
    LangChain Backend Server Configuration

    This configuration supports multiple LLM providers, vector stores,
    and deployment environments with GCP Secret Manager integration.
    """

    # APPLICATION SETTINGS
    app_name: str = Field(description="Application name")
    app_version: str = Field(description="Application version")
    environment: str = Field(description="Environment (development/staging/production)")
    debug: bool = Field(description="Enable debug mode")

    # Server configuration - MADE MANDATORY
    host: str = Field(description="Server host")
    port: int = Field(ge=1, le=65535, description="Server port")

    # Logging
    log_level: LogLevel = Field(description="Logging level")
    log_format: str = Field(description="Log format")

    # GCP SETTINGS
    project_id: str = Field(description="GCP Project ID")
    region: str = Field(description="GCP Region")

    # DATABASE SETTINGS
    db_uri: str = Field(description="Database URI")


    # LLM PROVIDER SETTINGS
    langchain_api_key: SecretStr = Field(description="Langchain API key")
    langchain_project: str = Field(description="Langchain project name")
    tavily_api_key: SecretStr = Field(description="Tavily API key")

    # OpenAI
    openai_api_key: SecretStr = Field(description="OpenAI API key")
    openai_model: str = Field(description="OpenAI model name")
    openai_temperature: float = Field(ge=0.0, le=2.0, description="OpenAI temperature")
    openai_max_tokens: int = Field(ge=0, description="OpenAI max tokens")
    openai_top_p: float = Field(ge=0.0, le=1.0, description="OpenAI TOP P")
    openai_top_k: float = Field(ge=1, le=100, description="OpenAI TOP K")


    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"

    class Config:
        env_file = Path(__file__).parent.parent.parent / '.env'
        env_file_encoding = 'utf-8'

        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return (env_settings,)


# Initialize configuration with validation
try:
    config = LangChainConfig()

except Exception as e:
    raise


# Export config instance
__all__ = ['config', 'LangChainConfig']
