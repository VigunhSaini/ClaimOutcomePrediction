"""
test_api.py
Comprehensive test suite for the Flask claim outcome prediction API.
"""

import os
import pytest

# Guard: skip gracefully if model.joblib has not been generated yet
MODEL_FILE = os.path.join(os.path.dirname(__file__), "model.joblib")
if not os.path.exists(MODEL_FILE):
    pytest.skip("model.joblib not found. Run 'python train.py' first.", allow_module_level=True)

from app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_payload(client):
    payload = {
        "payer": "Aetna",
        "provider": "Dr. Rao",
        "cpt_code": "99213",
        "diagnosis_code": "M54.5",
        "billed_amount": 220.00,
        "patient_age": 45,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    assert "predicted_status" in data
    assert data["predicted_status"] in ["Paid", "Partially_Paid", "Denied"]

    probs = data["probabilities"]
    assert list(probs.keys()) == ["Paid", "Partially_Paid", "Denied"]

    # Check sum with a tolerance for 2-decimal presentation rounding
    total_prob = sum(probs.values())
    assert total_prob == pytest.approx(1.0, abs=0.02)


def test_numeric_cpt_code(client):
    # API should accept integer CPT codes (e.g., 99213) and convert to string
    payload = {
        "payer": "UnitedHC",
        "provider": "Dr. Shah",
        "cpt_code": 99214,
        "diagnosis_code": "J06.9",
        "billed_amount": 180.00,
        "patient_age": 52,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["predicted_status"] in ["Paid", "Partially_Paid", "Denied"]


def test_unknown_categories(client):
    # Pipeline should handle unseen categories without throwing 400 or crashing
    payload = {
        "payer": "UnknownInsuranceCo",
        "provider": "Dr. Unknown",
        "cpt_code": "00000",
        "diagnosis_code": "Z99.9",
        "billed_amount": 195.00,
        "patient_age": 30,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()
    assert data["predicted_status"] in ["Paid", "Partially_Paid", "Denied"]


def test_missing_field(client):
    payload = {
        "payer": "Aetna",
        "provider": "Dr. Rao",
        # missing cpt_code
        "diagnosis_code": "M54.5",
        "billed_amount": 220.00,
        "patient_age": 45,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data
    assert "cpt_code" in data["error"]


def test_invalid_billed_amount(client):
    base_payload = {
        "payer": "Aetna",
        "provider": "Dr. Rao",
        "cpt_code": "99213",
        "diagnosis_code": "M54.5",
        "patient_age": 45,
    }

    # Negative amount
    res = client.post("/predict", json={**base_payload, "billed_amount": -50.0})
    assert res.status_code == 400

    # Zero amount
    res = client.post("/predict", json={**base_payload, "billed_amount": 0.0})
    assert res.status_code == 400

    # String amount
    res = client.post("/predict", json={**base_payload, "billed_amount": "two hundred"})
    assert res.status_code == 400

    # Boolean amount (isinstance(True, int) is True in Python, but must be rejected)
    res = client.post("/predict", json={**base_payload, "billed_amount": True})
    assert res.status_code == 400

    # Non-finite amount (NaN / Infinity)
    res = client.post("/predict", json={**base_payload, "billed_amount": float("nan")})
    assert res.status_code == 400


def test_invalid_patient_age(client):
    base_payload = {
        "payer": "Aetna",
        "provider": "Dr. Rao",
        "cpt_code": "99213",
        "diagnosis_code": "M54.5",
        "billed_amount": 220.00,
    }

    # Float age
    res = client.post("/predict", json={**base_payload, "patient_age": 45.5})
    assert res.status_code == 400

    # Boolean age
    res = client.post("/predict", json={**base_payload, "patient_age": False})
    assert res.status_code == 400

    # Negative age
    res = client.post("/predict", json={**base_payload, "patient_age": -1})
    assert res.status_code == 400

    # Unrealistic age (> 120)
    res = client.post("/predict", json={**base_payload, "patient_age": 140})
    assert res.status_code == 400


def test_empty_string_fields(client):
    payload = {
        "payer": "   ",
        "provider": "Dr. Rao",
        "cpt_code": "99213",
        "diagnosis_code": "M54.5",
        "billed_amount": 220.00,
        "patient_age": 45,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 400
    assert "payer" in res.get_json()["error"]

