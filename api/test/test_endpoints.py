from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


# Define tests for your endpoints
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Working API"}

def test_predict():
    # Example test for the /predict endpoint
    response = client.post("/predict", json={"step": 1, "start_date": "2023-06-01 00:00:00", "end_date": "2023-06-01 03:00:00"})
    assert response.status_code == 200
    assert "predictions" in response.json()
