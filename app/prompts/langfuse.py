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

def get_system_prompt(prompt_name: str = "telco-customer-service-agent"):
    """
    Fetch the system prompt from Langfuse Prompt Management.

    Args:
        prompt_name: Name of the prompt in Langfuse

    Returns:
        List of message dicts for LangChain
    """
    client = get_langfuse_client()
    prompt = client.get_prompt(prompt_name, type="chat")

    # Get LangChain-compatible prompt without variable substitution
    # (variables are hardcoded in Langfuse prompt)
    return prompt.get_langchain_prompt()

def get_model_config(prompt_name: str = "telco-customer-service-agent"):
    """
    Fetch the model config from Langfuse Prompt Management.

    Args:
        prompt_name: Name of the prompt in Langfuse

    Returns:
        Dict with model configuration (model, temperature)
    """
    client = get_langfuse_client()
    prompt = client.get_prompt(prompt_name, type="chat")

    # Get config from Langfuse prompt
    cfg = prompt.config

    # Return model config with fallbacks
    return {
        "model": cfg.get("model", config.DEFAULT_MODEL),
        "temperature": cfg.get("temperature", config.DEFAULT_TEMPERATURE)
    }
