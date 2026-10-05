"""
Master Execution Script.
Starts the Vehicle Claim Fraud Detection Portal:
1. Verifies MySQL Database connection and records (with graceful fallback)
2. Verifies trained XGBoost pipeline
3. Launches the clean web application on http://127.0.0.1:8000
"""

import os
import sys
import uvicorn
from config import MODELS_DIR
from init_mysql_database import init_database
from src.train_and_evaluate import run_training_pipeline

def main():
    print("=" * 75)
    print("      VEHICLE CLAIM FRAUD DETECTION SYSTEM (POWERED BY XGBOOST)      ")
    print("=" * 75)

    # Step 1: Ensure MySQL Database is initialized and populated
    print("\n[1/3] Verifying MySQL Database connection & records...")
    try:
        init_database()
    except Exception as e:
        print(f"[!] Note: MySQL connection skipped ({e}). Continuing with trained model pipeline.")

    # Step 2: Ensure Models are trained and packaged
    model_path = os.path.join(MODELS_DIR, "fraud_xgboost_pipeline.joblib")
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        print("\n[2/3] Training XGBoost pipeline and generating model artifacts...")
        run_training_pipeline()
    else:
        print("\n[2/3] Trained XGBoost model pipeline verified in models/.")

    # Step 3: Launch Web Application
    print("\n[3/3] Starting Web Application...")
    print("=" * 75)
    print("   >> Open in Browser:  http://127.0.0.1:8000")
    print("   >> MySQL Workbench: Open 'workbench_setup.sql' in MySQL Workbench")
    print("=" * 75)
    print("Press CTRL+C in this terminal to stop the server.\n")

    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
