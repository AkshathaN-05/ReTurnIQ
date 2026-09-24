"""
ReTurnIQ -- Milestone 3: Data Preprocessing Pipeline
=====================================================

Implements a reusable, leakage-safe preprocessing architecture using
Scikit-Learn ``Pipeline``, ``ColumnTransformer``, and custom transformers.

Key guarantees:
* All transformations are fit strictly on training data.
* Raw identifiers ('customer_id', 'product_id') are dropped.
* Target column ('returned') is strictly excluded.
* Categorical features are imputed and one-hot encoded (handle_unknown='ignore').
* Numerical and boolean features are imputed (median) and scaled (StandardScaler).
* Preprocessor can be saved and loaded alongside estimators via Joblib.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import engineer_features, get_engineered_feature_names


# =====================================================================
# Column Definitions
# =====================================================================

ID_COLUMNS: List[str] = ["customer_id", "product_id"]
TARGET_COLUMN: str = "returned"

CATEGORICAL_COLUMNS: List[str] = [
    "product_category",
    "order_day",
    "payment_method",
]

BOOLEAN_COLUMNS: List[str] = [
    "customer_account_verified",
    "is_gift_order",
    "is_expedited_shipping",
]

RAW_NUMERICAL_COLUMNS: List[str] = [
    "customer_age",
    "customer_tenure_months",
    "previous_orders",
    "previous_returns",
    "previous_return_rate",
    "customer_complaint_count",
    "average_previous_order_value",
    "product_price",
    "product_rating",
    "product_return_rate",
    "discount_percentage",
    "quantity",
    "order_hour",
    "delivery_distance_km",
    "order_value",
    "size_change_history",
    "category_return_history",
    "previous_category_orders",
    "previous_category_returns",
]

ENGINEERED_COLUMNS: List[str] = get_engineered_feature_names()

ALL_NUMERICAL_COLUMNS: List[str] = (
    RAW_NUMERICAL_COLUMNS + BOOLEAN_COLUMNS + ENGINEERED_COLUMNS
)


# =====================================================================
# Custom Transformers
# =====================================================================

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer that executes ``engineer_features``
    from ``src.feature_engineering`` within the pipeline.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X_copy = X.copy()
        # Ensure booleans are cast to float/int before downstream handling if needed
        for col in BOOLEAN_COLUMNS:
            if col in X_copy.columns:
                X_copy[col] = X_copy[col].astype(float)
        return engineer_features(X_copy)


# =====================================================================
# Preprocessor Construction
# =====================================================================

def build_column_transformer(include_scaling: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer for numerical and categorical preprocessing.

    Parameters
    ----------
    include_scaling : bool, default=True
        Whether to include StandardScaler in the numerical pipeline.

    Returns
    -------
    ColumnTransformer
    """
    # Numerical sub-pipeline: Median Imputation + optional Scaling
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if include_scaling:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=num_steps)

    # Categorical sub-pipeline: Most-frequent Imputation + One-Hot Encoding
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ALL_NUMERICAL_COLUMNS),
            ("cat", cat_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",  # Drops ID columns and any untracked columns
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_pipeline(
    estimator=None,
    include_scaling: bool = True,
    use_sampling: bool = False,
) -> Pipeline:
    """Construct a complete end‑to‑end Pipeline.

    1. Feature Engineering (adds domain features)
    2. ColumnTransformer (impute, one‑hot, scale, drop IDs)
+   3. Optional RandomOverSampler for training‑only class balance (inserted before classifier)
    4. Classifier (e.g., LogisticRegression or RandomForest)

    Parameters
    ----------
    estimator: estimator instance or None
        Classifier to append.
    include_scaling: bool, default=True
        Apply StandardScaler to numerical features.
    use_sampling: bool, default=False
        If True, a ``RandomOverSampler`` step is added using imblearn's Pipeline.
        The sampler is only fitted on the training set; during validation/test the
        same pipeline is used without resampling because ``fit_resample`` is only
        invoked on the training data.

    Returns
    -------
    Pipeline
    """
    steps = [
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", build_column_transformer(include_scaling=include_scaling)),
    ]
    if use_sampling:
        from imblearn.over_sampling import RandomOverSampler
        from imblearn.pipeline import Pipeline as ImbPipeline
        steps.append(("sampler", RandomOverSampler(random_state=42)))
        pipeline_cls = ImbPipeline
    else:
        pipeline_cls = Pipeline
    if estimator is not None:
        steps.append(("classifier", estimator))
    return pipeline_cls(steps=steps)


def get_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """
    Extract output feature names from a fitted Pipeline containing a ColumnTransformer.
    """
    col_trans: ColumnTransformer = fitted_pipeline.named_steps["preprocessor"]
    return list(col_trans.get_feature_names_out())
