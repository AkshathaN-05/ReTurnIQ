"""
ReTurnIQ -- Milestone 4: FastAPI REST API
==========================================

Production REST API providing real-time pre-delivery e-commerce return risk
scoring, transparent explainability attribution, and actionable operational
interventions.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    OrderInput,
    PredictionResponse,
    RiskFactor,
    RiskLevelEnum,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("returniq_api")

MODEL_PATH = "models/best_model_pipeline.joblib"
METADATA_PATH = "models/model_metadata.json"


# =====================================================================
# Lifespan Context Manager (Load model once on startup)
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load trained ML pipeline and metadata once at application startup."""
    logger.info("Initializing ReTurnIQ API service...")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Trained pipeline artifact not found at '{MODEL_PATH}'")

    if not os.path.exists(METADATA_PATH):
        raise RuntimeError(f"Model metadata artifact not found at '{METADATA_PATH}'")

    # Load artifacts into app.state
    logger.info("Loading pipeline artifact from %s...", MODEL_PATH)
    app.state.pipeline = joblib.load(MODEL_PATH)

    logger.info("Loading metadata from %s...", METADATA_PATH)
    with open(METADATA_PATH, "r") as f:
        app.state.metadata = json.load(f)

    # Extract threshold and model details
    threshold = app.state.metadata.get("recommended_threshold", 0.63)
    app.state.threshold = float(threshold)
    app.state.model_name = app.state.metadata.get("best_model", "LogisticRegression")
    app.state.feature_names = app.state.metadata.get("feature_names", [])

    logger.info(
        "Successfully loaded %s pipeline. Deployment threshold: %.2f (%d features).",
        app.state.model_name,
        app.state.threshold,
        len(app.state.feature_names),
    )

    yield

    logger.info("Shutting down ReTurnIQ API service...")


# =====================================================================
# FastAPI Application Initialization
# =====================================================================

app = FastAPI(
    title="ReTurnIQ REST API",
    description=(
        "AI-Powered Pre-Delivery Return Risk Prediction & Prevention Engine. "
        "Predicts the probability of order return before delivery, highlights contributing "
        "risk factors, and prescribes targeted pre-fulfillment interventions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# =====================================================================
# Custom Exception Handlers
# =====================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format Pydantic schema validation errors cleanly."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append({"field": field, "message": msg, "type": err.get("type")})

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "The submitted order payload contains invalid fields.",
            "details": errors,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled runtime errors with meaningful logging."""
    logger.error("Unhandled exception during request processing: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred during prediction processing.",
            "detail": str(exc),
        },
    )


# =====================================================================
# Explainability & Attribution Helpers
# =====================================================================

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "complaint_x_return_rate": "Customer complaints combined with previous return history contributed to the model's predicted risk for this order.",
    "discount_percentage": "A higher promotional discount percentage contributed to the model's predicted risk for this order.",
    "previous_orders": "Previous order history contributed to the model's prediction for this order.",
    "previous_returns": "Previous return history contributed to the model's predicted risk for this order.",
    "previous_return_rate": "Customer historical return rate contributed to the model's predicted risk for this order.",
    "product_return_rate": "Higher category benchmark return rate contributed to the model's predicted risk for this order.",
    "size_change_history": "Customer history of size exchanges contributed to the model's predicted risk for this order.",
    "size_change_x_high_risk_cat": "Customer history of size changes in high-risk categories contributed to the model's predicted risk for this order.",
    "return_rate_x_product_rate": "Interaction of customer return tendency and product category risk contributed to the model's predicted risk for this order.",
    "discount_amount": "A higher discount amount relative to unit price contributed to the model's predicted risk for this order.",
    "price_discount_interaction": "Synergy of item price and discount percentage contributed to the model's predicted risk for this order.",
    "complaint_rate": "Customer historical complaint rate contributed to the model's predicted risk for this order.",
    "customer_complaint_count": "Customer historical complaint count contributed to the model's predicted risk for this order.",
    "product_category_Clothing": "Clothing category benchmark contributed to the model's predicted risk for this order.",
    "product_category_Footwear": "Footwear category benchmark contributed to the model's predicted risk for this order.",
    "payment_method_Cash on Delivery": "Cash on Delivery payment method contributed to the model's predicted risk for this order.",
    "delivery_distance_km": "Longer delivery transit distance contributed to the model's predicted risk for this order.",
    "order_value": "Higher total basket value contributed to the model's predicted risk for this order.",
    "quantity": "Higher item quantity contributed to the model's predicted risk for this order.",
}


def extract_risk_factors(
    pipeline: Any,
    feature_names: List[str],
    df_raw: pd.DataFrame,
    top_k: int = 3,
) -> List[RiskFactor]:
    """
    Extract model-grounded feature attributions for an individual order.
    For Logistic Regression, computes log-odds attribution = (scaled_x_i * coef_i).
    """
    factors: List[RiskFactor] = []

    try:
        classifier = pipeline.named_steps.get("classifier")
        if classifier is None or not hasattr(classifier, "coef_"):
            return factors

        # Transform raw record through feature engineer and preprocessor
        feat_eng = pipeline.named_steps["feature_engineer"]
        preproc = pipeline.named_steps["preprocessor"]

        df_eng = feat_eng.transform(df_raw)
        X_trans = preproc.transform(df_eng)

        coefs = classifier.coef_[0]
        # Log-odds contribution
        contributions = X_trans[0] * coefs

        # Identify features that increase return probability (contribution > 0)
        pos_indices = np.where(contributions > 0.0)[0]
        if len(pos_indices) == 0:
            return factors

        sorted_indices = pos_indices[np.argsort(-contributions[pos_indices])][:top_k]

        for idx in sorted_indices:
            feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            desc = FEATURE_DESCRIPTIONS.get(
                feat_name,
                f"A higher value of feature '{feat_name}' contributed to the model's predicted risk for this order.",
            )
            factors.append(
                RiskFactor(
                    factor=feat_name,
                    impact="Increases return probability",
                    description=desc,
                )
            )
    except Exception as e:
        logger.warning("Feature attribution extraction failed gracefully: %s", str(e))

    return factors


def prescribe_intervention(
    risk_level: RiskLevelEnum,
    top_factors: List[RiskFactor],
    category: str,
) -> str:
    """Recommend targeted, context-aware operational intervention."""
    factor_names = [f.factor for f in top_factors]

    if risk_level == RiskLevelEnum.HIGH:
        if any("size" in fn for fn in factor_names) or category in ["Clothing", "Footwear"]:
            return (
                "Trigger automated pre-dispatch size & fit verification via SMS/WhatsApp. "
                "Offer interactive sizing guide and fit-check confirmation before order fulfillment."
            )
        elif any("discount" in fn for fn in factor_names):
            return (
                "Send proactive digital order confirmation highlighting detailed product "
                "specifications and usage guide to mitigate impulse purchase returns."
            )
        elif any("complaint" in fn or "return_rate" in fn for fn in factor_names):
            return (
                "Flag for customer care pre-fulfillment outreach to confirm product expectations "
                "and ensure delivery readiness."
            )
        else:
            return (
                "Prioritize quality inspection and package verification before dispatch. "
                "Include clear product satisfaction instructions inside the packaging."
            )

    elif risk_level == RiskLevelEnum.MEDIUM:
        return (
            "Dispatch proactive order progress notification with electronic product guide and "
            "standard automated delivery tracking."
        )

    else:
        return "Proceed with standard automated fulfillment (no pre-delivery intervention required)."


# =====================================================================
# Endpoints
# =====================================================================

@app.get("/", tags=["General"])
async def root():
    """Root entry point providing service metadata."""
    return {
        "service": "ReTurnIQ REST API",
        "description": "AI-Powered Pre-Delivery Return Risk Prediction System",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Health check endpoint confirming model status and deployment threshold."""
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded",
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_name=app.state.model_name,
        deployment_threshold=app.state.threshold,
        threshold_used=app.state.threshold,
        feature_count=len(app.state.feature_names),
        api_version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_return_risk(order: OrderInput):
    """
    Generate a pre-delivery return risk prediction for an e-commerce order.

    Accepts raw pre-delivery order fields, executes automatic feature engineering,
    imputation, and scaling, and applies the deployment threshold stored in model metadata.
    """
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML pipeline artifact is not loaded.",
        )

    threshold = app.state.threshold
    feature_names = app.state.feature_names

    # Convert incoming validated Pydantic model to DataFrame
    order_dict = order.model_dump()
    df_raw = pd.DataFrame([order_dict])

    try:
        # Predict class probabilities using full pipeline
        prob_arr = pipeline.predict_proba(df_raw)
        return_prob = float(prob_arr[0, 1])
    except Exception as exc:
        logger.error("Inference failure: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline inference failed: {str(exc)}",
        )

    # Determine binary class using the frozen validation threshold
    predicted_class = 1 if return_prob >= threshold else 0

    # Categorize risk level transparently
    if return_prob >= threshold:
        risk_level = RiskLevelEnum.HIGH
        risk_message = (
            f"This order exhibits elevated return risk ({return_prob:.1%}) exceeding the "
            f"deployment action threshold ({threshold:.2f})."
        )
    elif return_prob >= 0.30:
        risk_level = RiskLevelEnum.MEDIUM
        risk_message = (
            f"This order exhibits moderate return risk ({return_prob:.1%}) above baseline "
            f"levels, but below the high-risk action threshold ({threshold:.2f})."
        )
    else:
        risk_level = RiskLevelEnum.LOW
        risk_message = (
            f"This order exhibits low return risk ({return_prob:.1%}) comfortably below "
            "baseline risk levels."
        )

    # Grounded explainability: extract top model-contributing features
    top_factors = extract_risk_factors(
        pipeline=pipeline,
        feature_names=feature_names,
        df_raw=df_raw,
        top_k=3,
    )

    # Contextual operational intervention recommendation
    suggested_intervention = prescribe_intervention(
        risk_level=risk_level,
        top_factors=top_factors,
        category=order.product_category.value,
    )

    return PredictionResponse(
        return_probability=round(return_prob, 4),
        predicted_class=predicted_class,
        deployment_threshold=round(threshold, 4),
        threshold_used=round(threshold, 4),
        risk_level=risk_level,
        risk_message=risk_message,
        top_risk_factors=top_factors,
        suggested_intervention=suggested_intervention,
    )
