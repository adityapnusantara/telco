from langfuse import get_client
from app.core.config import config

# Initialize Langfuse client once at module level
_langfuse_client = None

def get_langfuse_client():
    """
    Get or create the Langfuse client singleton.

    Returns:
        Langfuse client instance
    """
    global _langfuse_client
    if _langfuse_client is None:
        _langfuse_client = get_client()
    return _langfuse_client

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
    """Fetch classification system prompt and model config in one API call.

    Returns:
        {
            "system_prompt": str,      # Raw prompt template string
            "user_prompt": str,        # Raw user prompt template string
            "model_config": dict       # {"model": str, "temperature": float}
        }
    """
    client = get_langfuse_client()
    system_prompt = client.get_prompt(config.CLASSIFICATION_SYSTEM_PROMPT_NAME)
    user_prompt = client.get_prompt(config.CLASSIFICATION_USER_PROMPT_NAME)

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "model_config": system_prompt.config
    }
