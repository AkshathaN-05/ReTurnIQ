# ReTurnIQ: AI-Powered E-Commerce Return Risk Prediction & Prevention System

## 1. Project Overview
**ReTurnIQ** is an end-to-end, industry-grade machine learning system designed to predict the probability that an e-commerce order will be returned **before the product is delivered**, diagnose the primary driving factors behind the predicted risk, and prescribe targeted, automated preventive interventions.

Rather than treating returns as an unavoidable post-delivery operational loss, ReTurnIQ transforms return management into an active, pre-delivery risk mitigation pipeline.

---

## 2. Problem Statement
Product returns represent one of the most critical margin drains in e-commerce, frequently exceeding 20–30% in categories such as fashion and footwear. Traditional logistics and operations address returns reactively:
- Reverse logistics costs accumulate rapidly (shipping, restocking, damage, depreciation).
- Customer dissatisfaction grows through friction-heavy return and refund cycles.
- Inventory remains locked in transit, reducing turnover efficiency.

Most existing analytical approaches focus on post-delivery reporting rather than proactive interception. By the time an order arrives at the customer's doorstep, the operational costs of fulfillment and return logistics are already locked in.
 
---

## 3. Proposed Solution
ReTurnIQ introduces an intelligent, pre-fulfillment decision engine that operates at the point of order placement:
1. **Pre-Delivery Return Risk Scoring**: Ingests order details, historical customer return tendencies, catalog-level metrics, and pre-fulfillment signals to estimate return probability ($0.0 - 1.0$) and classify orders into risk tiers (`LOW`, `MEDIUM`, `HIGH`).
2. **Explainable Risk Attribution**: Identifies the specific features driving the elevated risk (e.g., high historical size-swap frequency, high category return rate, order discount interaction).
3. **Automated Actionable Interventions**: Generates practical, context-aware operational recommendations to prevent returns before dispatch (e.g., automated size/fit confirmation, proactive customer support outreach, package verification).
4. **Production-Ready Architecture**: Delivered via a modular Scikit-Learn pipeline, a validated FastAPI backend, and an interactive operational dashboard.

---

## 4. Critical Data Leakage & Timing Rules
To ensure real-world validity and prevent artificial model inflation:
> **Strict Pre-Delivery Constraint**: All predictive features must realistically exist and be accessible **before product delivery**.
> 
> The system **strictly forbids** post-delivery or post-outcome variables, including:
> - Customer return requests or reason codes
> - Delivery timestamp discrepancies or courier delay remarks
> - Return tracking numbers or refund statuses
> - Post-delivery customer reviews or inspection reports
>
> The target variable (`returned`) is solely the ground-truth outcome used for training and offline evaluation. All data splits (train/validation/test) and feature transformations are strictly isolated within Scikit-Learn pipelines to prevent data leakage.

---

## 5. Dataset Architecture & Reproducibility
* **Large-Scale Synthetic Data**: Due to the proprietary nature of granular e-commerce return logs, this project utilizes a custom programmatic data generator designed to generate **~1,000,000 order records**.
* **Domain-Consistent Probabilistic Logic**: Unlike simplistic rule-based generators (`if returns > 3 then return=1`), the dataset models complex multivariate interactions, non-linear dependencies, realistic noise, and class overlap.
* **Realistic Properties**:
  - Class imbalance consistent with industry benchmarks (~15–25% baseline return rate).
  - Realistic data anomalies, missing values, and correlated behavioral traits.
  - Strict physical validity constraints (no negative prices, valid age distributions, discount bounds $[0\%, 100\%]$, quantity $\ge 1$, historical returns $\le$ historical orders).
  - Fixed random seed for 100% deterministic reproducibility across environments.

---

## 6. High-Level System Architecture

```
[ Customer Places Order ]
           │
           ▼
[ ReTurnIQ REST API (FastAPI) ]
  ├── Schema Validation (Pydantic)
  ├── Preprocessing & Feature Pipeline (Scikit-learn)
  └── Inference Engine (Trained Estimator)
           │
           ├──► Return Risk Probability & Tier (LOW / MEDIUM / HIGH)
           ├──► Explainability / Risk Attribution Drivers
           └──► Recommended Operational Intervention
           │
           ▼
[ Downstream Clients / Interactive Dashboard (Streamlit) ]
```

---

## 7. Planned Project Milestones

- **Milestone 1: Data Generation Engine**
  - Programmatic generation of ~1,000,000 synthetic e-commerce records.
  - Domain-realistic probabilistic relationships, feature interactions, and realistic noise.
  - Raw storage under `data/raw/` with reproducible random seeding.

- **Milestone 2: Data Preparation & Feature Engineering**
  - Exploratory Data Analysis (EDA) and class imbalance diagnostics.
  - Robust handling of missing values, anomalies, and duplicates.
  - Pre-delivery feature engineering (e.g., customer return rates, category benchmarks, price-discount ratios, complaint rates).
  - Leakage-free Scikit-Learn preprocessing pipelines.

- **Milestone 3: Machine Learning Model Development & Evaluation**
  - Stratified data splitting with strictly held-out test set.
  - Model benchmarking starting with:
    1. *Logistic Regression* (Interpretable linear baseline)
    2. *Random Forest Classifier* (Nonlinear ensemble)
  - Comprehensive imbalanced-class evaluation: Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Confusion Matrix.
  - Pipeline serialization using `joblib`.

- **Milestone 4: Production REST API**
  - FastAPI service with strict Pydantic schema validation.
  - Dynamic loading of saved Scikit-Learn pipeline.
  - Feature attribution explanation module and heuristic intervention recommender.
  - Comprehensive automated test suite (`pytest`) covering valid and invalid payload schemas.

- **Milestone 5: Interactive Operational Dashboard**
  - Real-time simulation interface for live order evaluation.
  - Aggregated risk distribution, risk metrics, and order monitoring.
  - Visual display of contributing risk factors and recommended interventions.

---

## 8. Technology Stack

| Category | Technology / Library |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, Joblib |
| **Visualization** | Matplotlib, Seaborn |
| **API Framework** | FastAPI, Uvicorn, Pydantic |
| **Frontend / Dashboard** | Streamlit, Requests |
| **Testing** | Pytest |

---

## 9. Project Directory Structure

```
ReTurnIQ/
├── data/
│   ├── raw/                 # Generated raw synthetic dataset
│   └── processed/           # Cleaned and processed datasets
├── notebooks/               # Exploratory notebooks and experiments
├── src/                     # Core modular Python packages
│   ├── __init__.py
│   ├── data_generation.py   # (Milestone 1)
│   ├── preprocessing.py     # (Milestone 2)
│   ├── feature_engineering.py # (Milestone 2)
│   ├── eda.py               # (Milestone 2)
│   ├── train.py             # (Milestone 3)
│   └── evaluate.py          # (Milestone 3)
├── models/                  # Serialized pipelines and model artifacts
├── api/                     # FastAPI application
│   ├── __init__.py
│   ├── main.py              # (Milestone 4)
│   └── schemas.py           # (Milestone 4)
├── tests/                   # Automated pytest suite
│   ├── __init__.py
│   └── ...                  # (Milestone 4)
├── dashboard/               # Interactive UI application
│   └── ...                  # (Milestone 5)
├── reports/                 # Evaluation figures and documentation
│   └── figures/
├── requirements.txt         # Project dependencies
├── .gitignore               # Git exclusion rules
└── README.md                # Project documentation and architectural overview
```
