"""
ReTurnIQ -- Milestone 3: Model Training Pipeline
=================================================

Trains, validates, and serializes machine learning models for pre-delivery
return risk prediction.

Models trained:
1. Baseline: Logistic Regression (interpretable linear baseline with balanced weights)
2. Ensemble: Random Forest Classifier (non-linear ensemble with balanced weights)

Leakage Prevention:
* Data is split into Train (70%), Validation (15%), and Test (15%) using stratified sampling.
* Pipelines encapsulate Feature Engineering + Preprocessing + Estimator.
* Preprocessor is fitted strictly on the Training set.
* Serialized models saved to ``models/`` using Joblib.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Tuple, Any

import joblib
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_pipeline, get_feature_names, TARGET_COLUMN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("returniq_train")


# =====================================================================
# Data Loading and Splitting
# =====================================================================

def load_and_split_data(
    data_path: str = "data/raw/return_data.csv",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    save_splits: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load raw dataset and perform stratified split into Train, Val, and Test partitions.
    """
    logger.info("Loading raw dataset from %s...", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows and %d columns.", df.shape[0], df.shape[1])

    y = df[TARGET_COLUMN].astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    # First split: train+val vs test
    test_size = test_ratio
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Second split: train vs val
    val_relative_ratio = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_ratio,
        random_state=random_state,
        stratify=y_train_val,
    )

    logger.info(
        "Split sizes: Train=%d (%.1f%%, pos=%.2f%%), Val=%d (%.1f%%, pos=%.2f%%), Test=%d (%.1f%%, pos=%.2f%%)",
        len(X_train),
        len(X_train) / len(df) * 100,
        y_train.mean() * 100,
        len(X_val),
        len(X_val) / len(df) * 100,
        y_val.mean() * 100,
        len(X_test),
        len(X_test) / len(df) * 100,
        y_test.mean() * 100,
    )

    if save_splits:
        os.makedirs("data/processed", exist_ok=True)
        # Save test set with target for evaluation and validation reproducibility
        test_df = X_test.copy()
        test_df[TARGET_COLUMN] = y_test.values
        test_path = "data/processed/test_data.csv"
        test_df.to_csv(test_path, index=False)
        logger.info("Saved held-out test split to %s (%d rows)", test_path, len(test_df))

        val_df = X_val.copy()
        val_df[TARGET_COLUMN] = y_val.values
        val_path = "data/processed/val_data.csv"
        val_df.to_csv(val_path, index=False)
        logger.info("Saved validation split to %s (%d rows)", val_path, len(val_df))

    return X_train, X_val, X_test, y_train, y_val, y_test


# =====================================================================
# Model Factory
# =====================================================================

def build_logistic_regression_pipeline(random_state: int = 42) -> Pipeline:
    """Build Logistic Regression pipeline with balanced weights."""
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=random_state,
        n_jobs=-1,
    )
    return build_pipeline(estimator=clf, include_scaling=True)


def build_random_forest_pipeline(
    n_estimators: int = 100,
    max_depth: int = 14,
    min_samples_leaf: int = 20,
    max_samples: float = 0.5,
    random_state: int = 42,
    n_jobs: int = 4,
) -> Pipeline:
    """Build Random Forest pipeline with balanced weights and memory-safe limits."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        max_samples=max_samples,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    # Scaling is optional for trees, but keeping uniform scaling enables direct comparison
    return build_pipeline(estimator=clf, include_scaling=True)


# =====================================================================
# Training & Validation Helper
# =====================================================================

def train_and_validate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Dict[str, Any]:
    """Train pipeline on training set and evaluate on validation set."""
    logger.info("--- Training %s ---", name)
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    duration = time.time() - start_time
    logger.info("%s training completed in %.2f seconds.", name, duration)

    # Validation predictions
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    y_val_pred = pipeline.predict(X_val)

    roc_auc = roc_auc_score(y_val, y_val_prob)
    pr_auc = average_precision_score(y_val, y_val_prob)
    f1 = f1_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred)

    logger.info(
        "Validation [%s] -> ROC-AUC: %.4f | PR-AUC: %.4f | F1: %.4f | Precision: %.4f | Recall: %.4f",
        name,
        roc_auc,
        pr_auc,
        f1,
        precision,
        recall,
    )

    return {
        "name": name,
        "train_time_sec": round(duration, 2),
        "val_roc_auc": round(float(roc_auc), 4),
        "val_pr_auc": round(float(pr_auc), 4),
        "val_f1": round(float(f1), 4),
        "val_precision": round(float(precision), 4),
        "val_recall": round(float(recall), 4),
    }


# =====================================================================
# Main Training Routine
# =====================================================================

def train_all_models(
    data_path: str = "data/raw/return_data.csv",
    output_dir: str = "models",
) -> Dict[str, Any]:
    """
    Orchestrates end-to-end model training, validation, and artifact serialization.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and split data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(
        data_path=data_path,
        random_state=42,
        save_splits=True,
    )

    # Apply RandomOverSampler to training set only (no leakage)
    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    results = {}

    # 2. Train Logistic Regression
    pipe_lr = build_logistic_regression_pipeline(random_state=42)
    lr_metrics = train_and_validate_model(
        "LogisticRegression", pipe_lr, X_train_resampled, y_train_resampled, X_val, y_val
    )
    results["LogisticRegression"] = lr_metrics

    # Save Logistic Regression pipeline
    lr_path = os.path.join(output_dir, "logistic_regression_pipeline.joblib")
    joblib.dump(pipe_lr, lr_path)
    logger.info("Saved Logistic Regression pipeline to %s", lr_path)

    # 3. Train Random Forest Classifier
    pipe_rf = build_random_forest_pipeline(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=20,
        max_samples=0.5,
        random_state=42,
        n_jobs=4,
    )
    rf_metrics = train_and_validate_model(
        "RandomForestClassifier", pipe_rf, X_train_resampled, y_train_resampled, X_val, y_val
    )
    results["RandomForestClassifier"] = rf_metrics

    # Save Random Forest pipeline
    rf_path = os.path.join(output_dir, "random_forest_pipeline.joblib")
    joblib.dump(pipe_rf, rf_path)
    logger.info("Saved Random Forest pipeline to %s", rf_path)

    # 4. Select Best Model by Validation PR-AUC
    best_name = (
        "RandomForestClassifier"
        if rf_metrics["val_pr_auc"] >= lr_metrics["val_pr_auc"]
        else "LogisticRegression"
    )
    best_pipeline = pipe_rf if best_name == "RandomForestClassifier" else pipe_lr
    best_path = os.path.join(output_dir, "best_model_pipeline.joblib")
    joblib.dump(best_pipeline, best_path)
    logger.info("Best model selected: %s (PR-AUC: %.4f). Saved to %s", best_name, results[best_name]["val_pr_auc"], best_path)

    # 5. Extract Feature Names and Save Model Metadata
    feature_names = get_feature_names(best_pipeline)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "target_prevalence_train": float(round(y_train.mean(), 4)),
        "models": results,
        "best_model": best_name,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", metadata_path)

    return metadata


if __name__ == "__main__":
    train_all_models()
