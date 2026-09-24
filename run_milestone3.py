"""
ReTurnIQ -- Milestone 3: End-to-End Execution Runner
=====================================================

Executes:
1. Training and validation of Logistic Regression and Random Forest models on 1,000,000-order dataset.
2. Pipeline artifact serialization to ``models/``.
3. Full evaluation on the strictly held-out test set.
4. Threshold tuning and report & figure generation under ``reports/``.
"""

import sys
import time
from src.train import train_all_models
from src.evaluate import run_evaluation


def main():
    print("=" * 70)
    print("   ReTurnIQ -- STARTING MILESTONE 3 (MODEL TRAINING & EVALUATION)")
    print("=" * 70)
    start_total = time.time()

    # Step 1: Model Training
    print("\n>>> STEP 1: TRAINING & SERIALIZATION <<<")
    train_metadata = train_all_models(
        data_path="data/raw/return_data.csv",
        output_dir="models",
    )
    print(f"[SUCCESS] Training complete. Best model: {train_metadata.get('best_model')}")

    # Step 2: Evaluation
    print("\n>>> STEP 2: TEST SET EVALUATION & VISUALIZATION <<<")
    eval_results = run_evaluation(
        val_data_path="data/processed/val_data.csv",
        test_data_path="data/processed/test_data.csv",
        models_dir="models",
        reports_dir="reports",
        figures_dir="reports/figures",
    )
    print("[SUCCESS] Evaluation complete. Reports and figures saved.")

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"   ReTurnIQ -- MILESTONE 3 COMPLETED IN {total_time:.1f} SECONDS")
    print("=" * 70)


if __name__ == "__main__":
    main()
