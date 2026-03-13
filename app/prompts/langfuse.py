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


def get_agent_prompt():
    """Fetch system prompt and model config for agent from Langfuse in one call.

    Returns:
        {
            "system_prompt": List[dict],  # LangChain chat prompt format
            "model_config": dict          # {"model": str, "temperature": float}
        }
    """
    client = get_langfuse_client()
    prompt = client.get_prompt(config.AGENT_PROMPT_NAME)

    # Get LangChain-compatible prompt without variable substitution
    system_prompt = prompt.get_langchain_prompt()

    return {
        "system_prompt": system_prompt,
        "model_config": prompt.config
    }


def get_classification_prompt() -> dict:
    """Fetch classification prompts and model config in one API call.

    Returns:
        {
            "system_prompt": str,         # Raw prompt template string
            "user_prompt": Prompt object, # For .compile() with variables
            "model_config": dict          # {"model": str, "temperature": float}
        }
    """
    client = get_langfuse_client()
    system_prompt = client.get_prompt(config.CLASSIFICATION_SYSTEM_PROMPT_NAME)
    user_prompt = client.get_prompt(config.CLASSIFICATION_USER_PROMPT_NAME)

    return {
        "system_prompt": system_prompt.prompt,
        "user_prompt": user_prompt,
        "model_config": system_prompt.config
    }


def get_extraction_prompt() -> dict:
    """Fetch extraction prompts and model config in one API call.

    Returns:
        {
            "system_prompt": str,         # Raw prompt template string
            "user_prompt": Prompt object, # For .compile() with variables
            "model_config": dict          # {"model": str, "temperature": float}
        }
    """
    client = get_langfuse_client()
    system_prompt = client.get_prompt(config.EXTRACTION_SYSTEM_PROMPT_NAME)
    user_prompt = client.get_prompt(config.EXTRACTION_USER_PROMPT_NAME)

    return {
        "system_prompt": system_prompt.prompt,
        "user_prompt": user_prompt,
        "model_config": system_prompt.config
    }
