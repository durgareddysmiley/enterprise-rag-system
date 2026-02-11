from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_document_not_found():
    response = client.get("/api/documents/non-existent-id")
    assert response.status_code == 404
    assert response.json() == {"detail": "Document not found"}

def test_query_invalid_body():
    response = client.post("/api/query", json={})
    assert response.status_code == 400
    assert response.json() == {"detail": "Query text is required"}
