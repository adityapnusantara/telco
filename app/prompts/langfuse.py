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

    # Compile with default values
    compiled = prompt.compile(
        company_name="MyTelco",
        escalation_contact="call 123 or use the MyTelco app"
    )

    return compiled
