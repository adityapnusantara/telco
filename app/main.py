import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.chat import router as chat_router
from app.core.config import config
from app.services.rag.vector_store import VectorStore
from app.services.llm.callbacks import CallbackHandler
from app.services.llm.agent import Agent
from app.services.llm.chat import ChatService

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telco Customer Service AI Agent",
    description="AI-powered customer service agent for Telco",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, tags=["chat"])


@app.on_event("startup")
async def startup_event():
    """Initialize all service instances and store in app.state"""
    try:
        logger.info("Initializing VectorStore...")
        app.state.vector_store = VectorStore(
            qdrant_url=config.QDRANT_URL,
            qdrant_api_key=config.QDRANT_API_KEY,
            collection_name=config.QDRANT_COLLECTION_NAME
        )

        logger.info("Initializing CallbackHandler...")
        app.state.callback_handler = CallbackHandler()

        logger.info("Initializing Agent...")
        app.state.agent = Agent(vector_store=app.state.vector_store)

        logger.info("Initializing ChatService...")
        app.state.chat_service = ChatService(
            agent=app.state.agent,
            handler=app.state.callback_handler
        )

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup service connections if needed"""
    logger.info("Shutting down services...")
    logger.info("Shutdown complete")


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "Telco Customer Service AI Agent",
        "version": "0.1.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    return {"status": "healthy"}
