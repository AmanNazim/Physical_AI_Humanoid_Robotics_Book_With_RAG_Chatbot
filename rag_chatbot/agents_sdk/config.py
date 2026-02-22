"""
Configuration for the Intelligence Layer (OpenAI Agents SDK) Subsystem.

This module provides configuration settings for the agent system,
including model parameters, safety settings, and integration options.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel
import os


class AgentConfig(BaseModel):
    """
    Configuration class for the Intelligence Layer agents.
    """
    # Model settings
    model: str = "mistral/mistral-large-latest"
    temperature: float = 0.3
    max_tokens: int = 2048

    # Agent behavior settings
    max_iterations: int = 10
    timeout_seconds: int = 30

    # Safety and validation settings
    enable_guardrails: bool = True
    enable_input_validation: bool = True
    enable_output_validation: bool = True
    hallucination_threshold: float = 0.3  # Threshold for detecting hallucinations

    # Context settings
    max_context_length: int = 4096
    context_overlap_threshold: float = 0.8

    # Session and memory settings
    session_ttl_minutes: int = 60
    max_memory_turns: int = 10

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_streaming: bool = True

    # API settings
    api_base_url: str = "https://api.mistral.ai/v1"
    api_key_env_var: str = "MISTRAL_API_KEY"

    class Config:
        arbitrary_types_allowed = True


def get_agent_config() -> AgentConfig:
    """
    Get the agent configuration from environment variables or defaults.

    Returns:
        AgentConfig: Configuration object with settings
    """
    # Override defaults with environment variables if available
    model = os.getenv("AGENT_MODEL", "devstral-latest")
    temperature = float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    max_tokens = int(os.getenv("AGENT_MAX_TOKENS", "2048"))
    timeout_seconds = int(os.getenv("AGENT_TIMEOUT_SECONDS", "30"))

    return AgentConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds
    )


# Default configuration instance
DEFAULT_AGENT_CONFIG = get_agent_config()


def update_config_from_dict(config: AgentConfig, updates: Dict[str, Any]) -> AgentConfig:
    """
    Update configuration from a dictionary of values.

    Args:
        config: The original configuration
        updates: Dictionary of values to update

    Returns:
        Updated configuration object
    """
    # Create a copy of the config with updates
    config_dict = config.dict()
    config_dict.update(updates)
    return AgentConfig(**config_dict)


def get_api_key() -> Optional[str]:
    """
    Get the API key from environment variables.

    Returns:
        API key string or None if not found
    """
    return os.getenv(DEFAULT_AGENT_CONFIG.api_key_env_var)


def validate_config(config: AgentConfig) -> bool:
    """
    Validate the agent configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, False otherwise
    """
    # Check temperature range
    if not 0.0 <= config.temperature <= 1.0:
        return False

    # Check max tokens range
    if config.max_tokens <= 0 or config.max_tokens > 4096:
        return False

    # Check timeout range
    if config.timeout_seconds <= 0 or config.timeout_seconds > 300:
        return False

    return True