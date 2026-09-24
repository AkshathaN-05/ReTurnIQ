"""
Verification script for ReTurnIQ E2E Storefront & Operations Payloads
"""
import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    """Create a TestClient instance managing app lifespan (loads real pipeline artifact)."""
    with TestClient(app) as test_client:
        yield test_client

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "deployment_threshold" in data

def test_storefront_high_risk_order(client):
    # Simulated Rohan Mehta ordering Denim Jeans with FLASH40 promo
    payload = {
        "customer_id": 5519,
        "product_id": 101,
        "customer_age": 28,
        "customer_tenure_months": 9.0,
        "previous_orders": 14,
        "previous_returns": 7,
        "previous_return_rate": 0.50,
        "customer_complaint_count": 3.0,
        "average_previous_order_value": 115.0,
        "customer_account_verified": True,
        "product_category": "Clothing",
        "product_price": 69.99,
        "product_rating": 4.3,
        "product_return_rate": 0.28,
        "discount_percentage": 40.0,
        "quantity": 1,
        "order_day": "Friday",
        "order_hour": 21,
        "delivery_distance_km": 75.0,
        "payment_method": "Cash on Delivery",
        "order_value": 46.99,
        "is_gift_order": False,
        "is_expedited_shipping": False,
        "size_change_history": 5.0,
        "category_return_history": 0.60,
        "previous_category_orders": 6.0,
        "previous_category_returns": 4.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    data = resp.json()
    assert "return_probability" in data
    assert data["risk_level"] in ["MEDIUM", "HIGH"]
    assert len(data["top_risk_factors"]) >= 1
    assert "suggested_intervention" in data
    assert len(data["suggested_intervention"]) > 0

def test_storefront_low_risk_order(client):
    # Simulated Priya Patel ordering Book with no discount
    payload = {
        "customer_id": 8721,
        "product_id": 401,
        "customer_age": 35,
        "customer_tenure_months": 28.0,
        "previous_orders": 26,
        "previous_returns": 2,
        "previous_return_rate": 0.077,
        "customer_complaint_count": 0.0,
        "average_previous_order_value": 85.0,
        "customer_account_verified": True,
        "product_category": "Books",
        "product_price": 42.99,
        "product_rating": 4.9,
        "product_return_rate": 0.03,
        "discount_percentage": 0.0,
        "quantity": 1,
        "order_day": "Monday",
        "order_hour": 14,
        "delivery_distance_km": 8.0,
        "payment_method": "Credit Card",
        "order_value": 47.99,
        "is_gift_order": False,
        "is_expedited_shipping": True,
        "size_change_history": 0.0,
        "category_return_history": 0.0,
        "previous_category_orders": 5.0,
        "previous_category_returns": 0.0,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    data = resp.json()
    assert data["risk_level"] == "LOW"
    assert data["predicted_class"] == 0

def test_scenario_explorer_comparison(client):
    # Base order vs Modified order
    base_payload = {
        "customer_age": 28,
        "customer_tenure_months": 9.0,
        "previous_orders": 14,
        "previous_returns": 7,
        "previous_return_rate": 0.50,
        "customer_complaint_count": 3.0,
        "average_previous_order_value": 115.0,
        "customer_account_verified": True,
        "product_category": "Clothing",
        "product_price": 69.99,
        "product_rating": 4.3,
        "product_return_rate": 0.28,
        "discount_percentage": 40.0,
        "quantity": 1,
        "order_day": "Friday",
        "order_hour": 21,
        "delivery_distance_km": 75.0,
        "payment_method": "Cash on Delivery",
        "order_value": 46.99,
        "is_gift_order": False,
        "is_expedited_shipping": False,
        "size_change_history": 5.0,
        "category_return_history": 0.60,
        "previous_category_orders": 6.0,
        "previous_category_returns": 4.0,
    }
    # Modified: Lower discount, lower size swaps, lower return rate
    mod_payload = base_payload.copy()
    mod_payload["discount_percentage"] = 0.0
    mod_payload["size_change_history"] = 0.0
    mod_payload["previous_return_rate"] = 0.05
    mod_payload["category_return_history"] = 0.0

    resp_base = client.post("/predict", json=base_payload)
    resp_mod = client.post("/predict", json=mod_payload)

    assert resp_base.status_code == 200
    assert resp_mod.status_code == 200

    prob_base = resp_base.json()["return_probability"]
    prob_mod = resp_mod.json()["return_probability"]

    assert prob_mod < prob_base, f"Modified probability ({prob_mod}) should be strictly less than baseline ({prob_base})"


def test_storefront_typed_customer_order_flow(client):
    """Verify that a typed customer name like 'Akshatha' generates a valid payload and succeeds."""
    from dashboard.app import get_or_create_customer_profile, SIMULATED_CATALOG
    import streamlit as st

    # Initialize mock session state if needed
    if "customer_profiles" not in st.session_state:
        st.session_state.customer_profiles = {}

    profile = get_or_create_customer_profile("Akshatha")
    assert profile["age"] >= 18
    assert profile["previous_returns"] <= profile["previous_orders"]
    assert profile["previous_category_returns"] <= profile["previous_category_orders"]

    # Select product
    prod = SIMULATED_CATALOG[0]  # Slim-Fit Stretch Denim Jeans
    payload = {
        "customer_id": profile["customer_id"],
        "product_id": int(prod["id"].split("-")[1]),
        "customer_age": int(profile["age"]),
        "customer_tenure_months": float(profile["tenure_months"]),
        "previous_orders": int(profile["previous_orders"]),
        "previous_returns": int(profile["previous_returns"]),
        "previous_return_rate": float(profile["previous_return_rate"]),
        "customer_complaint_count": float(profile["customer_complaint_count"]),
        "average_previous_order_value": float(profile["average_previous_order_value"]),
        "customer_account_verified": bool(profile["customer_account_verified"]),
        "product_category": prod["category"],
        "product_price": float(prod["price"]),
        "product_rating": float(prod["rating"]),
        "product_return_rate": float(prod["product_return_rate"]),
        "discount_percentage": 0.0,
        "quantity": 1,
        "order_day": "Friday",
        "order_hour": 14,
        "delivery_distance_km": 12.0,
        "payment_method": "UPI",
        "order_value": float(prod["price"]),
        "is_gift_order": False,
        "is_expedited_shipping": True,
        "size_change_history": float(profile["size_change_history"]),
        "category_return_history": float(profile["category_return_history"]),
        "previous_category_orders": float(profile["previous_category_orders"]),
        "previous_category_returns": float(profile["previous_category_returns"]),
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    data = resp.json()
    assert "return_probability" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "suggested_intervention" in data

