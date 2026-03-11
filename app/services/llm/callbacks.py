from langfuse.langchain import CallbackHandler

_handler = None

def get_langfuse_handler():
    """Get or create the Langfuse callback handler singleton"""
    global _handler
    if _handler is None:
        _handler = CallbackHandler()
    return _handler
