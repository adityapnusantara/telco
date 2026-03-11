from fastapi.testclient import TestClient

def test_app_exists():
    """Test that the FastAPI app can be imported"""
    from app.main import app
    assert app is not None

def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Telco Customer Service AI Agent"

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
