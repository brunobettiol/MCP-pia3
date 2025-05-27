from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_health_check():
    """Testa se o endpoint de verificação de saúde está funcionando."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data 