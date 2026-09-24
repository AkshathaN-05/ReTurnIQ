# ReTurnIQ — Milestone 4: FastAPI REST API Documentation & Verification Report

**Date**: 2026-09-18  
**Service Version**: 1.0.0  
**Backend Framework**: FastAPI (Uvicorn ASGI)  
**Schema Validation**: Pydantic v2  
**Production Model Artifact**: `models/best_model_pipeline.joblib` (`LogisticRegression`, 52 preprocessed features)  
**Deployment Decision Threshold**: `0.63` (frozen from validation optimization)

---

## 1. API Architecture

ReTurnIQ exposes an asynchronous RESTful service designed for real-time order ingestion at checkout or pre-fulfillment. 

```
[ Downstream Client / Streamlit Dashboard ]
                     │
                     ▼
          HTTP POST /predict (JSON)
                     │
                     ▼
          [ FastAPI Endpoint ]
                     │
                     ▼
     [ Pydantic Schema Validation ]
       - Type safety & bounds checking
       - Logical consistency cross-checks
                     │
                     ▼
  [ Serialized ReTurnIQ Scikit-Learn Pipeline ]
       - Automatic Feature Engineering (10 domain features)
       - ColumnTransformer (Imputation, One-Hot Encoding, StandardScaler)
       - Logistic Regression Inference
                     │
                     ▼
     [ Attribution & Explainability Engine ]
       - Linear log-odds feature contributions
       - Context-aware operational intervention rules
                     │
                     ▼
        PredictionResponse (JSON)
       - return_probability: 0.0 - 1.0
       - predicted_class: 0 or 1 (threshold = 0.63)
       - risk_level: LOW / MEDIUM / HIGH
       - top_risk_factors: list of grounded factors
       - suggested_intervention: targeted operational action
```

### Model Loading Protocol
* The model pipeline is loaded **once at server startup** using FastAPI's `lifespan` context manager.
* Zero model retraining occurs upon request receipt.
* Memory footprint is kept minimal and constant.

---

## 2. API Endpoints

| Method | Endpoint | Description | Request Model | Response Model |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/` | API service introduction & route directory | None | `JSON` |
| `GET` | `/health` | Liveness & model readiness check | None | `HealthResponse` |
| `POST` | `/predict` | Real-time return risk prediction & intervention prescription | `OrderInput` | `PredictionResponse` |
| `GET` | `/docs` | Interactive Swagger UI documentation | None | `HTML` |
| `GET` | `/redoc` | ReDoc API specification | None | `HTML` |

---

## 3. Input Schema & Validation Rules (`OrderInput`)

| Field | Type | Required | Constraints / Validation Rules | Description |
| :--- | :---: | :---: | :--- | :--- |
| `customer_id` | `int` | No | Optional, ignored by ML pipeline | Identifier |
| `product_id` | `int` | No | Optional, ignored by ML pipeline | Identifier |
| `customer_age` | `int` | **Yes** | $18 \le \text{age} \le 100$ | Customer age in years |
| `customer_tenure_months` | `float` | No | $\ge 0.0, \le 240.0$ | Platform tenure in months |
| `previous_orders` | `int` | **Yes** | $\ge 0$ | Total lifetime orders placed |
| `previous_returns` | `int` | **Yes** | $\ge 0$, $\le \text{previous\_orders}$ | Lifetime returns (cross-validated) |
| `previous_return_rate` | `float` | **Yes** | $0.0 \le \text{rate} \le 1.0$ | Lifetime return rate |
| `customer_complaint_count` | `float` | No | $\ge 0.0$ | Lifetime complaints |
| `average_previous_order_value`| `float` | No | $\ge 0.0$ | Historical order average spend |
| `customer_account_verified` | `bool` | **Yes** | Boolean | Verification flag |
| `product_category` | `str` | **Yes** | Allowed 8 catalog categories | Catalog category enum |
| `product_price` | `float` | **Yes** | $> 0.0$ | Base unit item price |
| `product_rating` | `float` | No | $1.0 \le \text{rating} \le 5.0$ | Catalog review rating |
| `product_return_rate` | `float` | **Yes** | $0.0 \le \text{rate} \le 1.0$ | Product catalog return rate |
| `discount_percentage` | `float` | **Yes** | $0.0 \le \text{pct} \le 100.0$ | Percentage discount |
| `quantity` | `int` | **Yes** | $\ge 1$ | Units ordered |
| `order_day` | `str` | **Yes** | Allowed 7 days of week | Day of order |
| `order_hour` | `int` | **Yes** | $0 \le \text{hour} \le 23$ | Order hour |
| `delivery_distance_km` | `float` | No | $\ge 0.0$ | Estimated transit distance |
| `payment_method` | `str` | **Yes** | Allowed 5 payment methods | Checkout payment method |
| `order_value` | `float` | **Yes** | $> 0.0$ | Total checkout basket value |
| `is_gift_order` | `bool` | **Yes** | Boolean | Gift designation |
| `is_expedited_shipping` | `bool` | **Yes** | Boolean | Shipping speed |
| `size_change_history` | `float` | No | $\ge 0.0$ | Size-swap frequency |
| `category_return_history` | `float` | No | $0.0 \le \text{rate} \le 1.0$ | Category historical return rate |
| `previous_category_orders` | `float` | No | $\ge 0.0$ | Lifetime orders in category |
| `previous_category_returns` | `float` | No | $\ge 0.0$, $\le \text{prev\_cat\_orders}$ | Lifetime returns in category |

---

## 4. Risk Categorization & Deployment Threshold

The API implements three risk tiers grounded in the frozen validation decision threshold (`0.63`):

* **HIGH RISK** (`return_probability >= 0.63`):
  * `predicted_class = 1`
  * Actionable risk requiring proactive pre-fulfillment mitigation.
* **MEDIUM RISK** (`0.30 <= return_probability < 0.63`):
  * `predicted_class = 0`
  * Elevated risk above baseline (~12.9%), handled with digital guidance and tracking.
* **LOW RISK** (`return_probability < 0.30`):
  * `predicted_class = 0`
  * Standard automated fulfillment with no intervention required.

---

## 5. Grounded Explainability & Actionable Interventions

### Explainability Methodology
Because the production estimator is a trained `LogisticRegression` pipeline with standard-scaled features, each feature's contribution to the prediction's log-odds is mathematically:
$$\text{attribution}_i = x_{\text{scaled}, i} \times \beta_i$$
Features with the highest positive attribution values $(\text{attribution}_i > 0)$ directly drive the model towards predicting a return. The API extracts the top contributing factors and translates them into clear, non-technical explanations.

*Disclaimer: Attributions reflect statistical model correlations and do not imply deterministic causality.*

### Prescribed Interventions
* **Sizing variance (Clothing/Footwear)**: Trigger automated pre-dispatch SMS/WhatsApp fit verification and interactive sizing assistance.
* **Heavy discount / impulse buys**: Deliver proactive product specification details and care instructions.
* **Frequent past returners / high complaints**: Route order to proactive customer concierge queue.
* **Standard orders**: Proceed with standard fulfillment.

---

## 6. Automated Test Suite & Results

Unit and integration tests were executed via `pytest tests/test_api.py -v`:

```
============================= test session starts =============================
collected 11 items

tests/test_api.py::test_health_endpoint PASSED                           [  9%]
tests/test_api.py::test_root_endpoint PASSED                             [ 18%]
tests/test_api.py::test_predict_valid_order PASSED                       [ 27%]
tests/test_api.py::test_predict_missing_required_field PASSED            [ 36%]
tests/test_api.py::test_predict_invalid_customer_age PASSED              [ 45%]
tests/test_api.py::test_predict_invalid_product_rating PASSED            [ 54%]
tests/test_api.py::test_predict_invalid_quantity PASSED                  [ 63%]
tests/test_api.py::test_predict_invalid_discount_percentage PASSED       [ 72%]
tests/test_api.py::test_predict_invalid_category_enum PASSED             [ 81%]
tests/test_api.py::test_predict_logical_consistency_returns_exceed_orders PASSED [ 90%]
tests/test_api.py::test_predict_high_risk_order PASSED                   [100%]

============================= 11 passed in 3.34s ==============================
```

All 11 tests passed with zero failures.

---

## 7. Example Request & Response Payloads

### Example 1: High-Risk Order
**Request (`POST /predict`)**:
```json
{
  "customer_age": 25,
  "customer_tenure_months": 6.0,
  "previous_orders": 15,
  "previous_returns": 12,
  "previous_return_rate": 0.80,
  "customer_complaint_count": 3.0,
  "average_previous_order_value": 80.0,
  "customer_account_verified": false,
  "product_category": "Clothing",
  "product_price": 150.0,
  "product_rating": 3.5,
  "product_return_rate": 0.35,
  "discount_percentage": 45.0,
  "quantity": 2,
  "order_day": "Sunday",
  "order_hour": 22,
  "delivery_distance_km": 35.0,
  "payment_method": "Cash on Delivery",
  "order_value": 165.0,
  "is_gift_order": false,
  "is_expedited_shipping": false,
  "size_change_history": 5.0,
  "category_return_history": 0.60,
  "previous_category_orders": 8.0,
  "previous_category_returns": 5.0
}
```

**Response (`200 OK`)**:
```json
{
  "return_probability": 0.9907,
  "predicted_class": 1,
  "deployment_threshold": 0.63,
  "risk_level": "HIGH",
  "risk_message": "This order exhibits elevated return risk (99.1%) exceeding the deployment action threshold (0.63).",
  "top_risk_factors": [
    {
      "factor": "previous_return_rate",
      "impact": "Increases return probability",
      "description": "Customer historical return tendency across all past orders"
    },
    {
      "factor": "product_return_rate",
      "impact": "Increases return probability",
      "description": "Elevated catalog return benchmark for this product category"
    },
    {
      "factor": "size_change_x_high_risk_cat",
      "impact": "Increases return probability",
      "description": "Customer history of size changes in Clothing/Footwear"
    }
  ],
  "suggested_intervention": "Trigger automated pre-dispatch size & fit verification via SMS/WhatsApp. Offer interactive sizing guide and fit-check confirmation before order fulfillment."
}
```

### Example 2: Health Check
**Request (`GET /health`)**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "LogisticRegression",
  "deployment_threshold": 0.63,
  "feature_count": 52,
  "api_version": "1.0.0"
}
```

---

## 8. Known Operational Limitations

1. **Batch Inference**: The `/predict` endpoint is optimized for single-order real-time scoring. For offline bulk scoring of hundreds of thousands of historical orders, a dedicated batch processing job or bulk endpoint would be more suitable.
2. **Cold-Start Customers**: New customers with zero prior order history rely predominantly on catalog-level attributes (`product_return_rate`, `product_category`, `discount_percentage`) and payment method until their customer profile matures.
3. **Linear Attribution Scope**: While log-odds feature contributions provide exact mathematical decomposition for Logistic Regression, complex non-linear interactions are represented through explicitly engineered feature interaction columns.
