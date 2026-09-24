"""
ReTurnIQ -- Milestone 2: Feature Engineering
=============================================

Creates new domain-based features from existing columns.  Every engineered
feature uses ONLY pre-delivery information.  The target column ``returned``
is never read or used during feature construction.

Leakage prevention
------------------
* No feature is derived from the ``returned`` target.
* No target-encoding is performed (would require the target).
* No rolling/temporal features are created because the dataset lacks a
  meaningful chronological ordering.

Usage
-----
    from src.feature_engineering import engineer_features
    df = engineer_features(df)
"""

import numpy as np
import pandas as pd


# ===================================================================
# Public helpers
# ===================================================================

def get_engineered_feature_names() -> list[str]:
    """Return the names of all features created by ``engineer_features``."""
    return [
        "discount_amount",
        "discounted_price",
        "price_per_quantity",
        "complaint_rate",
        "category_return_ratio",
        "price_discount_interaction",
        "return_rate_x_product_rate",
        "is_high_risk_category",
        "size_change_x_high_risk_cat",
        "complaint_x_return_rate",
    ]


# ===================================================================
# Main transformer
# ===================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 engineered features to *df* and return it **in-place** (no copy
    of the full DataFrame is made).

    Parameters
    ----------
    df : pd.DataFrame
        The raw (or partially processed) dataset.  Must contain the columns
        referenced below.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with new columns appended.
    """

    # ------------------------------------------------------------------
    # 1. discount_amount  =  product_price * discount_percentage / 100
    #    Absolute dollar value of the discount.  An expensive item with a
    #    small % discount can still have a large absolute discount.
    # ------------------------------------------------------------------
    df["discount_amount"] = np.round(
        df["product_price"] * df["discount_percentage"] / 100, 2
    )

    # ------------------------------------------------------------------
    # 2. discounted_price  =  product_price - discount_amount
    #    What the customer actually pays per unit -- captures value
    #    perception better than raw price alone.
    # ------------------------------------------------------------------
    df["discounted_price"] = np.round(
        df["product_price"] - df["discount_amount"], 2
    )

    # ------------------------------------------------------------------
    # 3. price_per_quantity  =  order_value / quantity
    #    Effective per-unit spend.  Safe because quantity >= 1.
    # ------------------------------------------------------------------
    df["price_per_quantity"] = np.round(
        df["order_value"] / df["quantity"].clip(lower=1), 2
    )

    # ------------------------------------------------------------------
    # 4. complaint_rate  =  complaints / max(previous_orders, 1)
    #    Normalised complaint tendency -- customers with many orders and
    #    few complaints are different from those with few orders and many.
    # ------------------------------------------------------------------
    df["complaint_rate"] = np.round(
        df["customer_complaint_count"]
        / df["previous_orders"].clip(lower=1),
        4,
    )

    # ------------------------------------------------------------------
    # 5. category_return_ratio  =  cat_returns / max(cat_orders, 1)
    #    Category-specific return tendency for this customer.
    # ------------------------------------------------------------------
    df["category_return_ratio"] = np.round(
        df["previous_category_returns"]
        / df["previous_category_orders"].clip(lower=1),
        4,
    )

    # ------------------------------------------------------------------
    # 6. price_discount_interaction  =  price * discount_pct / 100
    #    Same value as discount_amount but semantically framed as an
    #    interaction term -- high price + high discount may drive impulse
    #    purchases that are more likely to be returned.
    #    (Identical numerically to discount_amount; kept as an explicit
    #     interaction name for clarity.  We can drop one later if needed.)
    # ------------------------------------------------------------------
    df["price_discount_interaction"] = df["discount_amount"]

    # ------------------------------------------------------------------
    # 7. return_rate_x_product_rate
    #    Customer return propensity x product return propensity.
    #    Captures the compounding risk when a high-return customer buys a
    #    high-return product.
    # ------------------------------------------------------------------
    df["return_rate_x_product_rate"] = np.round(
        df["previous_return_rate"] * df["product_return_rate"], 6
    )

    # ------------------------------------------------------------------
    # 8. is_high_risk_category  (binary flag)
    #    Clothing and Footwear have the highest return rates (~25 % and
    #    22 % respectively).  A simple binary flag helps tree models find
    #    this split quickly.
    # ------------------------------------------------------------------
    df["is_high_risk_category"] = df["product_category"].isin(
        ["Clothing", "Footwear"]
    ).astype(int)

    # ------------------------------------------------------------------
    # 9. size_change_x_high_risk_cat
    #    Size-change history is only meaningful for Clothing / Footwear.
    #    This interaction zeroes out size_change_history for other
    #    categories so the signal is concentrated where it matters.
    # ------------------------------------------------------------------
    df["size_change_x_high_risk_cat"] = (
        df["size_change_history"] * df["is_high_risk_category"]
    )

    # ------------------------------------------------------------------
    # 10. complaint_x_return_rate
    #     Dissatisfied repeat returners -- customers who both complain
    #     and return at high rates are the riskiest segment.
    # ------------------------------------------------------------------
    df["complaint_x_return_rate"] = np.round(
        df["customer_complaint_count"] * df["previous_return_rate"], 4
    )

    return df
