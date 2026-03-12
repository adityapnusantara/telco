from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler


class CallbackHandler:
    """Langfuse callback handler wrapper with explicit initialization"""

    def __init__(self):
        self._handler = LangfuseCallbackHandler()

    @property
    def handler(self):
        """Get the underlying Langfuse callback handler"""
        return self._handler
