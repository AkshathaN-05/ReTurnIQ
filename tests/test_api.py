"""
ReTurnIQ -- Milestone 4: API Test Suite
========================================

Executes automated unit and integration tests against the FastAPI REST API
using FastAPI TestClient and the actual saved ML model pipeline artifact.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance managing app lifespan (loads real pipeline artifact)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_order_payload():
    """Realistic valid e-commerce order payload."""
    return {
        "customer_id": 10452,
        "customer_age": 32,
        "customer_tenure_months": 18.0,
        "previous_orders": 12,
        "previous_returns": 1,
        "previous_return_rate": 0.0833,
        "customer_complaint_count": 0.0,
        "average_previous_order_value": 92.50,
        "customer_account_verified": True,
        "product_id": 7821,
        "product_category": "Clothing",
        "product_price": 79.99,
        "product_rating": 4.3,
        "product_return_rate": 0.22,
        "discount_percentage": 10.0,
        "quantity": 1,
        "order_day": "Monday",
        "order_hour": 15,
        "delivery_distance_km": 14.2,
        "payment_method": "Credit Card",
        "order_value": 71.99,
        "is_gift_order": False,
        "is_expedited_shipping": True,
        "size_change_history": 1.0,
        "category_return_history": 0.15,
        "previous_category_orders": 5.0,
        "previous_category_returns": 1.0,
    }


# =====================================================================
# Test 1: Health Endpoint
# =====================================================================

def test_health_endpoint(client):
    """Verify /health returns 200, status=healthy, and model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_name"] == "LogisticRegression"
    assert data["deployment_threshold"] == 0.63
    assert data["feature_count"] == 52
    assert "api_version" in data


# =====================================================================
# Test 2: Root Endpoint
# =====================================================================

def test_root_endpoint(client):
    """Verify / root endpoint returns metadata and documentation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "ReTurnIQ REST API"
    assert data["status"] == "online"
    assert "/docs" in data["documentation"]


# =====================================================================
# Test 3: Valid Prediction with Real Pipeline
# =====================================================================

def test_predict_valid_order(client, valid_order_payload):
    """Verify /predict processes a valid order and returns expected response schema."""
    response = client.post("/predict", json=valid_order_payload)
    assert response.status_code == 200

    data = response.json()
    assert "return_probability" in data
    assert 0.0 <= data["return_probability"] <= 1.0
    assert data["predicted_class"] in [0, 1]
    assert data["deployment_threshold"] == 0.63
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(data["risk_message"], str) and len(data["risk_message"]) > 0
    assert isinstance(data["suggested_intervention"], str) and len(data["suggested_intervention"]) > 0
    assert isinstance(data["top_risk_factors"], list)

    # Consistency check: predicted_class == 1 iff return_probability >= threshold
    if data["return_probability"] >= data["deployment_threshold"]:
        assert data["predicted_class"] == 1
        assert data["risk_level"] == "HIGH"
    else:
        assert data["predicted_class"] == 0


# =====================================================================
# Test 4: Missing Required Field
# =====================================================================

def test_predict_missing_required_field(client, valid_order_payload):
    """Verify /predict returns 422 when a required field is omitted."""
    payload = valid_order_payload.copy()
    del payload["product_price"]  # Required field

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "details" in data or "detail" in data


# =====================================================================
# Test 5: Invalid Customer Age
# =====================================================================

def test_predict_invalid_customer_age(client, valid_order_payload):
    """Verify /predict returns 422 when customer_age < 18 or > 100."""
    payload_too_young = valid_order_payload.copy()
    payload_too_young["customer_age"] = 12  # Below minimum 18

    response = client.post("/predict", json=payload_too_young)
    assert response.status_code == 422

    payload_too_old = valid_order_payload.copy()
    payload_too_old["customer_age"] = 145  # Above maximum 100

    response = client.post("/predict", json=payload_too_old)
    assert response.status_code == 422


# =====================================================================
# Test 6: Invalid Product Rating
# =====================================================================

def test_predict_invalid_product_rating(client, valid_order_payload):
    """Verify /predict returns 422 when product_rating > 5.0 or < 1.0."""
    payload_high = valid_order_payload.copy()
    payload_high["product_rating"] = 5.8  # Exceeds 5.0

    response = client.post("/predict", json=payload_high)
    assert response.status_code == 422

    payload_low = valid_order_payload.copy()
    payload_low["product_rating"] = 0.5  # Below 1.0

    response = client.post("/predict", json=payload_low)
    assert response.status_code == 422


# =====================================================================
# Test 7: Invalid Quantity
# =====================================================================

def test_predict_invalid_quantity(client, valid_order_payload):
    """Verify /predict returns 422 when quantity is 0 or negative."""
    payload_zero = valid_order_payload.copy()
    payload_zero["quantity"] = 0  # Must be >= 1

    response = client.post("/predict", json=payload_zero)
    assert response.status_code == 422

    payload_negative = valid_order_payload.copy()
    payload_negative["quantity"] = -3

    response = client.post("/predict", json=payload_negative)
    assert response.status_code == 422


# =====================================================================
# Test 8: Invalid Discount Percentage
# =====================================================================

def test_predict_invalid_discount_percentage(client, valid_order_payload):
    """Verify /predict returns 422 when discount percentage > 100 or < 0."""
    payload_exceed = valid_order_payload.copy()
    payload_exceed["discount_percentage"] = 120.0  # Above 100%

    response = client.post("/predict", json=payload_exceed)
    assert response.status_code == 422


# =====================================================================
# Test 9: Invalid Categorical Enum Value
# =====================================================================

def test_predict_invalid_category_enum(client, valid_order_payload):
    """Verify /predict rejects uncataloged product categories."""
    payload = valid_order_payload.copy()
    payload["product_category"] = "Automotive Submarines"  # Invalid category

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# =====================================================================
# Test 10: Logical Consistency Validation
# =====================================================================

def test_predict_logical_consistency_returns_exceed_orders(client, valid_order_payload):
    """Verify cross-field validator rejects orders where previous_returns > previous_orders."""
    payload = valid_order_payload.copy()
    payload["previous_orders"] = 3
    payload["previous_returns"] = 7  # Cannot exceed total orders

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# =====================================================================
# Test 11: High-Risk vs Low-Risk Order Profiles
# =====================================================================

def test_predict_high_risk_order(client, valid_order_payload):
    """Verify that a high-risk profile (high return history + size change in clothing) scores high."""
    high_risk = valid_order_payload.copy()
    high_risk.update({
        "previous_orders": 15,
        "previous_returns": 12,
        "previous_return_rate": 0.80,
        "product_category": "Clothing",
        "product_return_rate": 0.35,
        "size_change_history": 6.0,
        "discount_percentage": 50.0,
        "customer_complaint_count": 4.0,
        "payment_method": "Cash on Delivery",
    })

    response = client.post("/predict", json=high_risk)
    assert response.status_code == 200
    data = response.json()
    assert data["return_probability"] >= 0.50
    assert data["risk_level"] in ["MEDIUM", "HIGH"]
    assert len(data["top_risk_factors"]) > 0
