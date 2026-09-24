"""
ReTurnIQ -- Milestone 4: Pydantic Schemas
==========================================

Defines strict input validation schemas, enumeration types, and response
structures for the ReTurnIQ REST API.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


# =====================================================================
# Allowed Categorical Enums
# =====================================================================

class ProductCategoryEnum(str, Enum):
    BEAUTY = "Beauty & Personal Care"
    BOOKS = "Books"
    CLOTHING = "Clothing"
    ELECTRONICS = "Electronics"
    FOOTWEAR = "Footwear"
    HOME_KITCHEN = "Home & Kitchen"
    SPORTS = "Sports & Outdoors"
    TOYS = "Toys & Games"


class OrderDayEnum(str, Enum):
    MONDAY = "Monday"
    TUESDAY = "Tuesday"
    WEDNESDAY = "Wednesday"
    THURSDAY = "Thursday"
    FRIDAY = "Friday"
    SATURDAY = "Saturday"
    SUNDAY = "Sunday"


class PaymentMethodEnum(str, Enum):
    COD = "Cash on Delivery"
    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    EMI = "EMI"
    UPI = "UPI"


class RiskLevelEnum(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


# =====================================================================
# Request Schema
# =====================================================================

class OrderInput(BaseModel):
    """
    Schema for a new e-commerce order evaluated prior to delivery.
    Includes only pre-delivery features; post-delivery attributes are strictly forbidden.
    """

    # Optional identifiers (excluded from model input features)
    customer_id: Optional[int] = Field(
        default=None,
        description="Optional customer ID (excluded from prediction features)",
    )
    product_id: Optional[int] = Field(
        default=None,
        description="Optional product ID (excluded from prediction features)",
    )

    # Customer Attributes
    customer_age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Customer age in years (18-100)",
        examples=[34],
    )
    customer_tenure_months: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=240.0,
        description="Tenure with platform in months",
        examples=[14.5],
    )
    previous_orders: int = Field(
        ...,
        ge=0,
        description="Total lifetime orders placed by customer",
        examples=[10],
    )
    previous_returns: int = Field(
        ...,
        ge=0,
        description="Total lifetime returns by customer",
        examples=[1],
    )
    previous_return_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Historical return rate (previous_returns / max(previous_orders, 1))",
        examples=[0.10],
    )
    customer_complaint_count: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Lifetime customer complaints logged",
        examples=[0.0],
    )
    average_previous_order_value: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Historical average order spend in currency units",
        examples=[75.50],
    )
    customer_account_verified: bool = Field(
        ...,
        description="Whether customer phone/email is verified",
        examples=[True],
    )

    # Product Catalog Attributes
    product_category: ProductCategoryEnum = Field(
        ...,
        description="Product catalog category",
        examples=[ProductCategoryEnum.CLOTHING],
    )
    product_price: float = Field(
        ...,
        gt=0.0,
        description="Base unit price of product",
        examples=[89.99],
    )
    product_rating: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=5.0,
        description="Average catalog rating (1.0 - 5.0)",
        examples=[4.2],
    )
    product_return_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Catalog benchmark return rate for this product",
        examples=[0.18],
    )
    discount_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Discount percentage applied to product (0 - 100)",
        examples=[15.0],
    )
    quantity: int = Field(
        ...,
        ge=1,
        description="Item quantity ordered (must be >= 1)",
        examples=[1],
    )

    # Order Logistics & Context
    order_day: OrderDayEnum = Field(
        ...,
        description="Day of week when order was placed",
        examples=[OrderDayEnum.MONDAY],
    )
    order_hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day (0-23) when order was placed",
        examples=[14],
    )
    delivery_distance_km: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Estimated delivery distance in kilometers",
        examples=[12.5],
    )
    payment_method: PaymentMethodEnum = Field(
        ...,
        description="Payment method used for checkout",
        examples=[PaymentMethodEnum.CREDIT_CARD],
    )
    order_value: float = Field(
        ...,
        gt=0.0,
        description="Total order transaction value",
        examples=[76.49],
    )
    is_gift_order: bool = Field(
        ...,
        description="Whether marked as gift order",
        examples=[False],
    )
    is_expedited_shipping: bool = Field(
        ...,
        description="Whether customer selected express shipping",
        examples=[True],
    )

    # Granular Category Behavioral History
    size_change_history: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Customer lifetime size-swap frequency",
        examples=[1.0],
    )
    category_return_history: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Customer historical return rate in this specific category",
        examples=[0.20],
    )
    previous_category_orders: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Customer lifetime orders in this product category",
        examples=[4.0],
    )
    previous_category_returns: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Customer lifetime returns in this product category",
        examples=[1.0],
    )

    @model_validator(mode="after")
    def validate_logical_consistency(self) -> "OrderInput":
        """Validate physical business logic across correlated fields."""
        if self.previous_returns > self.previous_orders:
            raise ValueError(
                f"previous_returns ({self.previous_returns}) cannot exceed previous_orders ({self.previous_orders})"
            )

        if (
            self.previous_category_orders is not None
            and self.previous_category_returns is not None
            and self.previous_category_returns > self.previous_category_orders
        ):
            raise ValueError(
                f"previous_category_returns ({self.previous_category_returns}) cannot exceed "
                f"previous_category_orders ({self.previous_category_orders})"
            )

        return self


# =====================================================================
# Response Schemas
# =====================================================================

class RiskFactor(BaseModel):
    """Grounded contributing factor identified by model attribution."""
    factor: str = Field(..., description="Feature name driving return risk")
    impact: str = Field(..., description="Directional impact on return probability")
    description: str = Field(..., description="Plain language explanation of risk contributor")


class PredictionResponse(BaseModel):
    """Comprehensive prediction response payload."""
    return_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that this order will be returned (0.0 - 1.0)",
        examples=[0.724],
    )
    predicted_class: int = Field(
        ...,
        description="Binary prediction: 1 (Return) if probability >= deployment threshold, else 0 (Non-Return)",
        examples=[1],
    )
    deployment_threshold: float = Field(
        ...,
        description="Frozen deployment decision threshold (derived from validation set)",
        examples=[0.63],
    )
    threshold_used: Optional[float] = Field(
        default=None,
        description="Alias for deployment_threshold for backward compatibility",
        examples=[0.63],
    )
    risk_level: RiskLevelEnum = Field(
        ...,
        description="Categorical return risk tier (LOW, MEDIUM, HIGH)",
        examples=[RiskLevelEnum.HIGH],
    )
    risk_message: str = Field(
        ...,
        description="Contextual description of predicted risk level",
        examples=["This order has elevated predicted return risk exceeding the deployment threshold."],
    )
    top_risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Key model-derived features contributing to the risk score (correlative, not causal)",
    )
    suggested_intervention: str = Field(
        ...,
        description="Actionable pre-dispatch operational intervention recommended to prevent return",
        examples=["Trigger automated size/fit confirmation and provide detailed sizing assistance prior to shipping."],
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., description="Service health status", examples=["healthy"])
    model_loaded: bool = Field(..., description="Whether the ML pipeline artifact is loaded", examples=[True])
    model_name: str = Field(..., description="Active production estimator name", examples=["LogisticRegression"])
    deployment_threshold: float = Field(..., description="Active deployment decision threshold", examples=[0.63])
    threshold_used: Optional[float] = Field(default=None, description="Alias for deployment_threshold", examples=[0.63])
    feature_count: int = Field(..., description="Number of preprocessed feature columns", examples=[52])
    api_version: str = Field(..., description="ReTurnIQ API version", examples=["1.0.0"])
