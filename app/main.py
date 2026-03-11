from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes.chat import router as chat_router

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
