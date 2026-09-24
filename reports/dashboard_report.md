# ReTurnIQ — Product-Level Alignment & Interactive Application Report

**Date**: 2026-09-18  
**Component**: Streamlit Frontend Application (`dashboard/app.py`)  
**Backend Target**: FastAPI REST Microservice (`api/main.py` at `http://127.0.0.1:8001`)

---

## 1. Executive Summary

ReTurnIQ delivers a complete, realistic, professional AI-powered e-commerce return risk prediction and prevention system.

The frontend is architected as a decoupled client communicating strictly over HTTP with the FastAPI backend. **The frontend never loads the scikit-learn pipeline directly into memory**, strictly preserving microservice boundaries and preventing model deserialization duplication.

The application serves two distinct user personas through two specialized interfaces:
1. **🛍️ Customer Storefront (NovaCart)**: A realistic e-commerce shopping experience demonstrating how ReTurnIQ is embedded into checkout and fulfillment workflows. Customers type their name and shop normally without seeing technical machine learning questionnaires.
2. **⚙️ ReTurnIQ Operations Center**: An operational command center for fulfillment operators and fraud/risk analysts. Displays real incoming customer orders, detailed order risk attributions, a What-If Risk Explorer, and session analytics.

---

## 2. Main Product Flow & System Architecture

```
┌────────────────────────────────────────────────────────┐
│               Streamlit Dual-Mode Client               │
│                                                        │
│  ┌─────────────────────────┐  ┌─────────────────────┐  │
│  │   Customer Storefront   │  │  Operations Center  │  │
│  │ (Shopping & Checkout UI)│  │ (Live Orders Queue) │  │
│  └───────────┬─────────────┘  └──────────▲──────────┘  │
└──────────────┼───────────────────────────┼─────────────┘
               │                           │
               ▼                           │ Stored Order Record
    Automatic Feature Assembly             │ (orders_history sync)
    (Typed Name -> Profile + Catalog Specs)│
               │                           │
               └─────────────┬─────────────┘
                             │ HTTP POST /predict (JSON)
                             ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Microservice                 │
│  ├ Strict Pydantic Schema Validation (OrderInput)      │
│  ├ Logical Consistency Checks (Returns <= Orders)      │
│  └ Health / Readiness Probes (/health)                 │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│            Pre-Trained ML Pipeline In-Memory           │
│  ├ Custom Pre-Delivery Feature Engineering (11 Feats)  │
│  ├ Imputation & Scalers                                │
│  ├ LogisticRegression Estimator                        │
│  └ Frozen Optimal Threshold Decision (0.63)            │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│             Risk Assessment & Attributions             │
│  ├ Calibrated Return Probability                       │
│  ├ Operational Risk Tier (LOW / MEDIUM / HIGH)         │
│  ├ Model-Associated Driver Attribution                 │
│  └ Automated Actionable Intervention                   │
└────────────────────────────┬───────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────┐
│                   Targeted Response                    │
│  ├ Customer Storefront: Friendly Fit / Care Guidance   │
│  └ Operations Center: Synchronized Order Assessment    │
└────────────────────────────────────────────────────────┘
```

---

## 3. Customer Storefront Flow

### 3.1 Design Rationale & Identity Handling
In real-world e-commerce, customers should never be asked to manually fill in 20+ machine learning features. 
- **Typed Customer Name**: The customer enters their name (e.g. `Akshatha`). The name field is initially empty.
- **Internal Behavioral Profile**: Behind the scenes, the application associates the customer identity with a realistic behavioral profile (e.g., tenure, historical return rates, complaint counts, previous category orders). This profile is cached in session state so returning customers maintain consistent history.
- **Zero Technical ML Exposure**: No API endpoints, model estimator names, feature counts, latency, logits, or raw JSON are displayed to the customer.

### 3.2 End-to-End Shopping & Checkout Workflow
1. **Customer Name Input**: User types their name and receives a personalized welcome greeting.
2. **Product Selection & Sizing**: Choose from a multi-category catalog (Clothing, Footwear, Electronics, Books, Home & Kitchen, Beauty) with size variants and quantities.
3. **Delivery & Checkout**: Configure standard or express delivery, gift packaging, and optional coupons (SAVE15, FLASH40).
4. **Payment Method**: Select between Credit Card, Debit Card, UPI, and Cash on Delivery.
5. **Place Order**: Clicking "Place Order" executes an automated sequence:
   - Order Confirmed (`#ORD-XXXXXX` placed for customer)
   - Payment Confirmed
   - Smart Order Check (Silent background evaluation via `/predict`)
   - Ready for Packing (with customer-friendly guidance)

### 3.3 Customer-Facing Recommendations
- **High Risk**:
  > *"💡 Let's make sure you get the right product  
  > Before we prepare your order, we recommend checking our quick size and fit guide to ensure you receive the exact right fit and avoid the hassle of an exchange."*  
  > Action buttons: `[📏 View Size & Fit Guide]` and `[✅ Confirm Selection & Proceed]`.
- **Medium Risk**:
  > *"ℹ️ One quick suggestion before dispatch  
  > We've noticed that this type of order sometimes benefits from an extra product check. A digital sizing and care guide has been sent to your confirmation email."*
- **Low Risk**:
  > *"🎉 You're all set!  
  > Your order looks good to go. We're packaging your items and will notify you as soon as tracking is available."*

---

## 4. ReTurnIQ Operations Center

The Operations Center provides an operational dashboard for fulfillment teams and analysts. It strictly displays the **exact same prediction** generated during customer checkout without duplicate inference calls.

### 4.1 Operations Overview
- Real-time KPI summary cards based on actual session orders:
  - Orders Today
  - High Risk Orders
  - Medium Risk Orders
  - Low Risk Orders

### 4.2 Recent Orders (Customer -> Operator Sync)
- Prominently displays incoming orders placed from the Customer Storefront.
- Each order card shows:
  - Order ID (e.g. `ORD-123456`)
  - Customer Name (e.g. `Akshatha`)
  - Product & Size (e.g. `Slim-Fit Stretch Denim Jeans (Size: 28)`)
  - Quantity & Order Total (`$74.99`)
  - Payment Method
  - Return Probability (`41.4%`) & Risk Badge (`MEDIUM RISK`)
  - Operational Guidance & Status (`Ready for Packing`)

### 4.3 Order Details Inspector
- Detailed drill-down on any selected order:
  - Full order metadata (date/time, delivery, pricing)
  - Model return risk assessment
  - Grounded risk attribution factors (e.g., category return benchmark, sizing history)
  - Recommended operational intervention

### 4.4 What-If Risk Explorer
- Reframed interactive tool to explore how adjusting key attributes impacts return probability.
- Parameters:
  - Discount percentage (0% to 80%)
  - Customer return history (0.00 to 1.00)
  - Product category
  - Product return benchmark (0.00 to 0.60)
  - Size-change history (0 to 10)
  - Customer complaints (0 to 10)
- Communicates with `/predict` to compute real-time probability shift (`+X.X%` / `-X.X%`).
- Clear statistical disclaimer regarding correlative vs causal interpretation.

### 4.5 Session Analytics
- Grounded strictly in real session orders:
  - Total orders evaluated and average predicted risk
  - Risk tier distribution bar chart
  - Chronological return risk trend line chart
  - Complete orders log table

### 4.6 Advanced Evaluation Tools
- Collapsible secondary section containing the manual 20+ feature form and High/Mid/Low presets for technical demonstration and viva explanation.

---

## 5. Verification & Testing

### 5.1 Automated Endpoint Verification
All 16 unit and integration tests in `tests/` pass with 100% success:
- Health check probe (`GET /health`)
- Valid order scoring (`POST /predict`)
- Missing field handling (422 Unprocessable Content)
- Numerical boundary checks (Age < 18, Rating > 5.0, Discount > 100%)
- Logical consistency validation (`previous_returns > previous_orders`)
- Scenario comparison validation (modified parameters produce lower probability)
- Typed customer storefront order flow verification (`test_storefront_typed_customer_order_flow`)

### 5.2 Decoupled Frontend Verification
- Frontend connects via standard REST JSON payloads to `http://127.0.0.1:8001`.
- Clean minimal sidebar with discreet `● Service Ready` status indicator.
- Fully responsive layout with theme-aware CSS variables ensuring high contrast in both light and dark modes.
