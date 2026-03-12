from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler


class CallbackHandler:
    """Langfuse callback handler wrapper with explicit initialization"""

    def __init__(self):
        self._handler = LangfuseCallbackHandler()

    @property
    def handler(self):
        """Get the underlying Langfuse callback handler"""
        return self._handler


# Legacy function for backward compatibility (will be removed after migration)
def get_langfuse_handler():
    """Deprecated: Use CallbackHandler class instead"""
    raise DeprecationWarning("Use CallbackHandler class instead")
