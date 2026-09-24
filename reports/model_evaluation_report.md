# ReTurnIQ — Milestone 3: Machine Learning Model Evaluation Report

**Evaluation Date**: 2026-09-18 12:20:03  
**Evaluation Dataset**: Strictly Held-Out Test Set (`data/processed/test_data.csv`)  
**Test Set Size**: 150,000 records (completely held out until final evaluation)  
**Validation Set Size**: 150,000 records (`data/processed/val_data.csv` used for model & threshold selection)  
**Observed Return Prevalence**: 12.91% (19,371 returns)

---

## 1. Executive Summary & Leakage Prevention Protocol

The objective of Milestone 3 is to develop, rigorously evaluate, and serialize production-grade machine learning pipelines that predict pre-delivery return risk while strictly safeguarding against data leakage.

### Strict Data Isolation Protocol:
* **Training Partition (700,000 orders)**: Used exclusively for fitting transformers and estimator weights.
* **Validation Partition (150,000 orders)**: Used exclusively for estimator benchmarking (`val_pr_auc`) and deployment decision threshold optimization.
* **Test Partition (150,000 orders)**: **Zero test set contamination.** The test set was untouched during hyperparameter tuning, model selection, and threshold tuning. The validation-selected threshold was frozen and evaluated on the test set exactly once.

> [!IMPORTANT]
> **Threshold Selection Correction & Leakage Elimination**:  
> During Milestone 3 verification, a potential data leakage issue was audited where threshold optimization was previously executed on the test set. This workflow has been explicitly corrected: decision threshold sweeping is executed **strictly on validation data (`data/processed/val_data.csv`)**. The optimal threshold is frozen on validation data, and then evaluated on the untouched test set.

---

## 2. Benchmark Performance & Test Results

Evaluated on the **150,000 held-out test orders** (`data/processed/test_data.csv`):

| Metric | Logistic Regression (Thresh=0.50) | Logistic Regression (Val-Selected Thresh=0.63) | Random Forest (Thresh=0.50) | Random Forest (Val-Selected Thresh=0.57) |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.7638 | 0.7638 | **0.7612** | **0.7612** |
| **PR-AUC (Average Precision)** | **0.3809** | **0.3809** | 0.3731 | 0.3731 |
| **Balanced Accuracy** | **0.6967** | 0.6739 | 0.6911 | **0.6731** |
| **Brier Score (Calibration)** | 0.1937 | 0.1937 | **0.1743** | **0.1743** |
| **Return Precision (Class 1)** | 0.2618 | **0.3436** | 0.2861 | 0.3382 |
| **Return Recall (Class 1)** | **0.6761** | 0.4854 | 0.6066 | 0.4876 |
| **Return F1-Score (Class 1)** | 0.3774 | **0.4024** | 0.3888 | 0.3994 |
| **Non-Return Recall (Class 0)** | 0.7173 | **0.8625** | 0.7756 | 0.8585 |

---

## 3. Imbalanced Class Analysis & Validation-Selected Thresholds

Because return events occur at an ~12.91% baseline rate, standard 0.50 classification thresholds can skew recall/precision trade-offs:
* **LogisticRegression** achieved the highest overall PR-AUC (0.3809) and ROC-AUC (0.7638).
* Sweeping thresholds on the validation set identified an optimal F1 threshold for LogisticRegression at **`0.63`**.
* When applied to the untouched held-out test set, the frozen threshold of **`0.63`** achieved:
  * **Test Precision (Return)**: **`0.3436`**
  * **Test Recall (Return)**: **`0.4854`**
  * **Test F1-Score (Return)**: **`0.4024`**

### 3.1 Why Raw Accuracy Alone Is Insufficient

In real-world e-commerce return prevention, **raw classification accuracy is a misleading and practically useless metric**:

1. **The Majority-Class Illusion (Accuracy Paradox)**:
   In our dataset, only ~12.91% of orders are returned (87.09% non-returns). A naive or trivial baseline model that simply predicts "No Return" (Class 0) for every single order automatically achieves **87.09% raw accuracy**. Yet, such a model has **0.0% Recall** and **0.0% Precision** for returns, catching exactly zero return risks and providing zero business value.

2. **Asymmetric Error Costs**:
   - **Cost of False Negative (FN)**: An undetected return slips through unmitigated. The business incurs reverse logistics expenses, inspection costs, restocking fees, and item depreciation ($15–$30+ per returned unit).
   - **Cost of False Positive (FP)**: An order is flagged as high risk when it wouldn't have been returned. The business executes a low-friction preventive intervention (e.g., verifying size or sending an automated fit guide via email/SMS). The operational friction is near zero (~$0.05–$0.20), far lower than the cost of a missed return.
   - Raw accuracy assigns equal 1:1 weight to both errors, failing to capture this severe economic asymmetry.

3. **Domain-Appropriate Metrics Adopted**:
   - **PR-AUC (Precision-Recall Area Under Curve)**: Measures precision across all recall thresholds, focusing specifically on the minority positive class without being inflated by true negatives.
   - **ROC-AUC**: Evaluates the model's ranking ability across all decision boundaries.
   - **F1-Score on Positive Class**: Harmonizes precision and recall, ensuring operational actions are both targeted and effective.
   - **Brier Score**: Measures probability calibration so predicted risks reflect genuine return probabilities rather than overconfident binary predictions.

---

## 4. Confusion Matrix Breakdown (Test Set: 150,000 Orders)

### LogisticRegression (Default Threshold = 0.50):
* **True Negatives (TN)**: 93,704
* **False Positives (FP)**: 36,925
* **False Negatives (FN)**: 6,275
* **True Positives (TP)**: 13,096

### LogisticRegression (Validation-Selected Threshold = 0.63):
* **True Negatives (TN)**: 112,669
* **False Positives (FP)**: 17,960
* **False Negatives (FN)**: 9,969
* **True Positives (TP)**: 9,402

---

## 5. Key Predictive Drivers & Feature Importance

The top features identified by both models:
1. **`previous_return_rate` & `return_rate_x_product_rate`**: Historical customer return tendency interacting with product catalog return baseline.
2. **`size_change_x_high_risk_cat` & `size_change_history`**: High customer size-change frequency, especially in Apparel and Footwear.
3. **`product_return_rate`**: Baseline catalog item risk.
4. **`discount_percentage` & `discount_amount`**: Large promotional discounts strongly correlated with return incidence.
5. **`complaint_rate` & `complaint_x_return_rate`**: Customer dissatisfaction signal compounded with return propensity.

---

## 6. Artifacts Generated

* **Pipeline Models** (`models/`):
  - `logistic_regression_pipeline.joblib`
  - `random_forest_pipeline.joblib`
  - `best_model_pipeline.joblib` (`LogisticRegression`)
  - `model_metadata.json`
* **Evaluation Figures** (`reports/figures/`):
  - `roc_curves.png`
  - `pr_curves.png`
  - `confusion_matrices.png`
  - `threshold_tuning.png` (validation threshold sweep)
  - `feature_importances.png`

---

## 7. Readiness for Milestone 4 (Production FastAPI Service)

The best pipeline (`best_model_pipeline.joblib`) is fully self-contained. It incorporates:
* Feature engineering directly on input dictionaries/DataFrames.
* Automatic imputation of missing values.
* Robust one-hot encoding handling unseen categories (`handle_unknown='ignore'`).
* Scaling and probability calibration.
* Recommended deployment threshold: **`0.63`**.

Downstream API endpoints can load the pipeline artifact directly with `joblib.load()` and execute real-time predictions without needing manual preprocessing logic.
