from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_endpoint():
  response = client.get("/health")
  assert response.status_code == 200
  assert response.json()["status"] == "healthy"


def test_predict_endpoint():
  payload = {
      "day_of_week": 1,
      "month": 5,
      "lag_7": 85.0,
      "lag_14": 82.5,
      "lag_28": 80.0,
      "rolling_mean_7": 84.1,
      "rolling_std_7": 2.1,
  }
  response = client.post("/predict", json=payload)
  assert response.status_code == 200
  json_data = response.json()
  assert "predicted_demand" in json_data
  assert "shap_contributions" in json_data
  assert isinstance(json_data["shap_contributions"], dict)


def test_parse_incident_endpoint():
  payload = {
      "raw_text": (
          "Severe flood and hurricane caused closure, delaying aluminum for 14"
          " days."
      )
  }
  response = client.post("/parse-incident", json=payload)
  assert response.status_code == 200
  json_data = response.json()
  assert json_data["severity"] == "High"
  assert json_data["is_force_majeure"] is True
  assert json_data["delay_days"] == 14
  assert "Aluminum" in json_data["affected_materials"]