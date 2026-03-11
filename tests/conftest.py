import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    """Test client for FastAPI app"""
    from app.main import app
    return TestClient(app)
