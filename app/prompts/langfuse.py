from functools import lru_cache
from langfuse import get_client
from app.core.config import config


@lru_cache(maxsize=1)
def get_langfuse_client():
    """
    Get or create the Langfuse client singleton.

    Returns:
        Langfuse client instance
    """
    return get_client()


def get_system_prompt(prompt_name: str = None):
    """
    Fetch the system prompt from Langfuse Prompt Management.

    Args:
        prompt_name: Name of the prompt in Langfuse. Defaults to config.SYSTEM_PROMPT_NAME

    Returns:
        List of message dicts for LangChain
    """
    if prompt_name is None:
        prompt_name = config.SYSTEM_PROMPT_NAME
    client = get_langfuse_client()
    prompt = client.get_prompt(prompt_name, type="chat")

    # Get LangChain-compatible prompt without variable substitution
    # (variables are hardcoded in Langfuse prompt)
    return prompt.get_langchain_prompt()


def get_model_config(prompt_name: str = None):
    """
    Fetch the model config from Langfuse Prompt Management.

    Args:
        prompt_name: Name of the prompt in Langfuse. Defaults to config.SYSTEM_PROMPT_NAME

    Returns:
        Dict with model configuration (model, temperature)
    """
    if prompt_name is None:
        prompt_name = config.SYSTEM_PROMPT_NAME
    client = get_langfuse_client()
    prompt = client.get_prompt(prompt_name, type="chat")

    # Get config from Langfuse prompt
    cfg = prompt.config

    # Return model config with fallbacks
    return {
        "model": cfg.get("model", config.DEFAULT_MODEL),
        "temperature": cfg.get("temperature", config.DEFAULT_TEMPERATURE)
    }


def get_classification_prompt_obj():
    """Get classification user message template from Langfuse for .compile()

    Returns prompt object from config.CLASSIFICATION_USER_PROMPT_NAME
    with template containing {{reply}} and {{context}} variables.
    """
    client = get_langfuse_client()
    return client.get_prompt(config.CLASSIFICATION_USER_PROMPT_NAME)


def get_classification_config() -> dict:
    """Get classification model config from Langfuse.

    Returns config from config.CLASSIFICATION_CONFIG_PROMPT_NAME.
    """
    client = get_langfuse_client()
    config_prompt = client.get_prompt(config.CLASSIFICATION_CONFIG_PROMPT_NAME)
    return config_prompt.config


def get_classification_system_prompt() -> str:
    """Get classification agent system prompt from Langfuse.

    Returns the system prompt string for the classification agent.
    Fetches from config.CLASSIFICATION_SYSTEM_PROMPT_NAME.
    """
    client = get_langfuse_client()
    prompt = client.get_prompt(config.CLASSIFICATION_SYSTEM_PROMPT_NAME)
    # Get the prompt content as string
    return prompt.prompt  # Returns the prompt template content as string
