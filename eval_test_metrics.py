import joblib, pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# Load processed test data
X_test = pd.read_csv('data/processed/test_data.csv')
# Separate target
y_test = X_test.pop('returned')

# Load best model pipeline
model = joblib.load('models/best_model_pipeline.joblib')

# Predict probabilities
proba = model.predict_proba(X_test)[:, 1]
# Threshold selected during validation (stored in model metadata)
threshold = 0.63
pred = (proba >= threshold).astype(int)

print('Test ROC-AUC:', roc_auc_score(y_test, proba))
print('Test PR-AUC:', average_precision_score(y_test, proba))
print('Test Precision:', precision_score(y_test, pred))
print('Test Recall:', recall_score(y_test, pred))
print('Test F1:', f1_score(y_test, pred))
