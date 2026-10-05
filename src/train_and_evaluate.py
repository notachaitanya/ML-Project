"""
Model Training, Evaluation, and Lifecycle Pipeline.
Implements:
- CO1: ML Lifecycle Pipeline Integration & Tracing Precursors
- CO2: Logistic Regression with Regularization and Feature Scaling (Linear Baseline)
- CO3: XGBoost Classifier with Feature Importances & Splitting Criteria Analysis
- CO4: PCA (Principal Component Analysis) for Unsupervised Structure Discovery
- CO5: Hyperparameter Tuning (GridSearchCV with Stratified CV) & Classification Metrics
- CO6: Model Packaging & Baseline Statistics Generation for Drift Monitoring
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

from config import MODELS_DIR, OUTPUTS_DIR
from src.db_connector import fetch_claims_from_db
from src.data_cleaning import clean_raw_data, get_preprocessor

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

def find_best_threshold(y_true, y_probs):
    """Finds probability threshold that maximizes F1-score on imbalanced data."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    # thresholds array has length len(precisions) - 1
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return float(best_thresh)

def run_training_pipeline():
    print("=" * 75)
    print("INSURANCE CLAIM FRAUD DETECTION - COMPLETE MACHINE LEARNING PIPELINE")
    print("=" * 75)

    # -------------------------------------------------------------
    # STAGE 1: Pull Data Exclusively from Live MySQL Database (CO1)
    # -------------------------------------------------------------
    print("\n[STAGE 1 / CO1] Querying raw data from live MySQL database...")
    raw_df = fetch_claims_from_db()
    total_records = len(raw_df)
    
    # -------------------------------------------------------------
    # STAGE 2: Data Cleaning & Preprocessing (CO1)
    # -------------------------------------------------------------
    print("\n[STAGE 2 / CO1] Cleaning raw database records...")
    X, y, cat_cols, num_cols, median_age, mode_day, mode_month = clean_raw_data(raw_df)
    print(f"[*] Extracted {X.shape[0]} samples with {X.shape[1]} features.")
    print(f"[*] Target Distribution -> Fraud: {y.sum()} ({y.mean()*100:.2f}%), Genuine: {(y == 0).sum()} ({(1-y.mean())*100:.2f}%)")

    # Stratified Train/Test Split (80/20) preserving fraud proportion (CO5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[*] Stratified Split -> Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Preprocessor (StandardScaler for numeric, OneHotEncoder for categoricals)
    preprocessor = get_preprocessor(cat_cols, num_cols)
    preprocessor.fit(X_train)
    
    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    ohe_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + ohe_cols
    print(f"[*] Total transformed feature dimensions: {len(all_feature_names)}")

    # -------------------------------------------------------------
    # STAGE 3: CO4 - Unsupervised Learning (PCA Structure Discovery)
    # -------------------------------------------------------------
    print("\n[STAGE 3 / CO4] Applying Unsupervised Learning: Principal Component Analysis (PCA)...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_train_proc)
    evr = pca.explained_variance_ratio_
    print(f"[*] PCA Explained Variance Ratio -> PC1: {evr[0]*100:.2f}%, PC2: {evr[1]*100:.2f}% (Total: {sum(evr)*100:.2f}%)")
    
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=y_train, cmap='coolwarm', alpha=0.55, s=15, edgecolors='none'
    )
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Genuine (0)', 'Fraud (1)'])
    plt.title(f"CO4: PCA 2D Feature Space Projection (Explained Variance: {sum(evr)*100:.1f}%)", fontsize=13, fontweight='bold')
    plt.xlabel(f"Principal Component 1 ({evr[0]*100:.1f}% Variance)")
    plt.ylabel(f"Principal Component 2 ({evr[1]*100:.1f}% Variance)")
    plt.tight_layout()
    pca_plot_path = os.path.join(OUTPUTS_DIR, "pca_fraud_structure.png")
    plt.savefig(pca_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved PCA visualization: {pca_plot_path}")

    # -------------------------------------------------------------
    # STAGE 4: CO2 - Linear Supervised Model (Logistic Regression Baseline)
    # -------------------------------------------------------------
    print("\n[STAGE 4 / CO2] Training Linear Supervised Model: Logistic Regression (L2 Regularization)...")
    # L2 regularized logistic regression with class balancing
    lr_model = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_proc, y_train)
    
    y_test_pred_lr = lr_model.predict(X_test_proc)
    y_test_prob_lr = lr_model.predict_proba(X_test_proc)[:, 1]
    
    lr_metrics = {
        "model": "Logistic Regression (Linear L2)",
        "accuracy": float(accuracy_score(y_test, y_test_pred_lr)),
        "precision": float(precision_score(y_test, y_test_pred_lr, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred_lr)),
        "f1_score": float(f1_score(y_test, y_test_pred_lr)),
        "roc_auc": float(roc_auc_score(y_test, y_test_prob_lr)),
        "pr_auc": float(average_precision_score(y_test, y_test_prob_lr))
    }

    # -------------------------------------------------------------
    # STAGE 5: Basic XGBoost Model (Standard Settings from Abstract)
    # -------------------------------------------------------------
    print("\n[STAGE 5A / CO3] Training Basic XGBoost (Standard Default Settings)...")
    xgb_basic = XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42,
        tree_method='hist'
    )
    xgb_basic.fit(X_train_proc, y_train)
    y_test_pred_basic = xgb_basic.predict(X_test_proc)
    y_test_prob_basic = xgb_basic.predict_proba(X_test_proc)[:, 1]

    basic_metrics = {
        "model": "Basic XGBoost (Standard Defaults)",
        "accuracy": float(accuracy_score(y_test, y_test_pred_basic)),
        "precision": float(precision_score(y_test, y_test_pred_basic, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred_basic)),
        "f1_score": float(f1_score(y_test, y_test_pred_basic)),
        "roc_auc": float(roc_auc_score(y_test, y_test_prob_basic)),
        "pr_auc": float(average_precision_score(y_test, y_test_prob_basic))
    }

    # -------------------------------------------------------------
    # STAGE 6: CO3 & CO5 - Tuned XGBoost with GridSearchCV & Imbalance Handling
    # -------------------------------------------------------------
    print("\n[STAGE 5B / CO3 & CO5] Hyperparameter Tuning XGBoost using GridSearchCV (5-Fold Stratified CV)...")
    imbalance_ratio = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"[*] Class Imbalance Ratio (Genuine/Fraud): {imbalance_ratio:.2f}")

    xgb_tune_base = XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42,
        tree_method='hist'
    )

    param_grid = {
        'max_depth': [3, 4, 6],
        'learning_rate': [0.03, 0.08],
        'n_estimators': [100, 150],
        'scale_pos_weight': [round(imbalance_ratio * 0.5), round(imbalance_ratio)]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=xgb_tune_base,
        param_grid=param_grid,
        scoring='average_precision',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid_search.fit(X_train_proc, y_train)
    best_xgb = grid_search.best_estimator_
    print(f"[+] Optimal Hyperparameters Found: {grid_search.best_params_}")
    print(f"[+] Best 5-Fold Stratified CV PR-AUC: {grid_search.best_score_:.4f}")

    # Evaluate Tuned XGBoost on Unseen Test Set
    y_test_pred_xgb = best_xgb.predict(X_test_proc)
    y_test_prob_xgb = best_xgb.predict_proba(X_test_proc)[:, 1]

    # Calculate optimal F1 threshold
    optimal_thresh = find_best_threshold(y_train, best_xgb.predict_proba(X_train_proc)[:, 1])
    y_test_pred_opt = (y_test_prob_xgb >= optimal_thresh).astype(int)

    tuned_metrics = {
        "model": "Improved Tuned XGBoost",
        "best_hyperparameters": grid_search.best_params_,
        "cv_pr_auc": float(grid_search.best_score_),
        "optimal_threshold": float(optimal_thresh),
        "accuracy": float(accuracy_score(y_test, y_test_pred_xgb)),
        "precision": float(precision_score(y_test, y_test_pred_xgb, zero_division=0)),
        "recall": float(recall_score(y_test, y_test_pred_xgb)),
        "f1_score": float(f1_score(y_test, y_test_pred_xgb)),
        "roc_auc": float(roc_auc_score(y_test, y_test_prob_xgb)),
        "pr_auc": float(average_precision_score(y_test, y_test_prob_xgb)),
        "optimal_threshold_metrics": {
            "accuracy": float(accuracy_score(y_test, y_test_pred_opt)),
            "precision": float(precision_score(y_test, y_test_pred_opt, zero_division=0)),
            "recall": float(recall_score(y_test, y_test_pred_opt)),
            "f1_score": float(f1_score(y_test, y_test_pred_opt))
        }
    }

    # Print Comparison Table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL COMPARISON (CO2 vs CO3 vs CO5)")
    print("=" * 80)
    print(f"{'Metric':<18} | {'CO2: Logistic Reg':<20} | {'CO3: Basic XGB':<18} | {'CO3+5: Tuned XGB':<18}")
    print("-" * 80)
    for m in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']:
        print(f"{m.upper():<18} | {lr_metrics[m]:<20.4f} | {basic_metrics[m]:<18.4f} | {tuned_metrics[m]:<18.4f}")
    print("=" * 80)

    # -------------------------------------------------------------
    # STAGE 7: Feature Importance Analysis (CO3)
    # -------------------------------------------------------------
    print("\n[STAGE 7 / CO3] Computing Feature Importances (Tree Splitting Gain)...")
    importances = best_xgb.feature_importances_
    feat_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    feat_df = feat_df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    top_feats = feat_df.head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_feats, x='importance', y='feature', hue='feature', palette='Blues_r', legend=False)
    plt.title("CO3: Top 15 Tree-Based Feature Importances (XGBoost Splitting Gain)", fontsize=13, fontweight='bold')
    plt.xlabel("Relative Importance Score (Gain)")
    plt.ylabel("Feature")
    plt.tight_layout()
    feat_plot_path = os.path.join(OUTPUTS_DIR, "feature_importances.png")
    plt.savefig(feat_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved Feature Importance plot: {feat_plot_path}")

    # -------------------------------------------------------------
    # STAGE 8: Performance Visualizations (CO5)
    # -------------------------------------------------------------
    print("\n[STAGE 8 / CO5] Generating Performance Evaluation Curves...")
    
    # ROC Curves Comparison
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_prob_lr)
    fpr_basic, tpr_basic, _ = roc_curve(y_test, y_test_prob_basic)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_test_prob_xgb)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_xgb, tpr_xgb, label=f"Tuned XGBoost (AUC = {tuned_metrics['roc_auc']:.3f})", color='#1f77b4', lw=2.5)
    plt.plot(fpr_basic, tpr_basic, label=f"Basic XGBoost (AUC = {basic_metrics['roc_auc']:.3f})", color='#2ca02c', linestyle=':', lw=2)
    plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {lr_metrics['roc_auc']:.3f})", color='#ff7f0e', linestyle='--', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Guess")
    plt.title("CO5: Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight='bold')
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_plot_path = os.path.join(OUTPUTS_DIR, "roc_curve.png")
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()

    # Precision-Recall Curves Comparison
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_test_prob_lr)
    prec_basic, rec_basic, _ = precision_recall_curve(y_test, y_test_prob_basic)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_test_prob_xgb)
    
    plt.figure(figsize=(8, 6))
    plt.plot(rec_xgb, prec_xgb, label=f"Tuned XGBoost (PR-AUC = {tuned_metrics['pr_auc']:.3f})", color='#1f77b4', lw=2.5)
    plt.plot(rec_basic, prec_basic, label=f"Basic XGBoost (PR-AUC = {basic_metrics['pr_auc']:.3f})", color='#2ca02c', linestyle=':', lw=2)
    plt.plot(rec_lr, prec_lr, label=f"Logistic Regression (PR-AUC = {lr_metrics['pr_auc']:.3f})", color='#ff7f0e', linestyle='--', lw=2)
    plt.axhline(y=y_test.mean(), color='r', linestyle=':', label=f"No-Skill Baseline ({y_test.mean():.3f})")
    plt.title("CO5: Precision-Recall (PR) Curve Comparison (Imbalanced Fraud Detection)", fontsize=13, fontweight='bold')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="upper right")
    plt.tight_layout()
    pr_plot_path = os.path.join(OUTPUTS_DIR, "precision_recall_curve.png")
    plt.savefig(pr_plot_path, dpi=300)
    plt.close()

    # Confusion Matrix (Tuned XGBoost)
    cm_xgb = confusion_matrix(y_test, y_test_pred_xgb)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Genuine (0)', 'Fraud (1)'],
                yticklabels=['Genuine (0)', 'Fraud (1)'])
    plt.title("CO5: Confusion Matrix - Tuned XGBoost", fontsize=13, fontweight='bold')
    plt.ylabel("Actual Claim Status")
    plt.xlabel("Predicted Claim Status")
    plt.tight_layout()
    cm_plot_path = os.path.join(OUTPUTS_DIR, "confusion_matrix_xgb.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # STAGE 9: ML Engineering - Model Packaging & Drift Baselines (CO6)
    # -------------------------------------------------------------
    print("\n[STAGE 9 / CO6] Packaging Model Pipeline & Computing Drift Baseline Statistics...")
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', best_xgb)
    ])
    
    model_artifact_path = os.path.join(MODELS_DIR, "fraud_xgboost_pipeline.joblib")
    joblib.dump(full_pipeline, model_artifact_path)
    print(f"[+] Packaged production model pipeline to: {model_artifact_path}")

    # Compute baseline statistics on numerical features for drift monitoring (CO6)
    num_stats = {}
    for col in num_cols:
        num_stats[col] = {
            "mean": float(X_train[col].mean()),
            "std": float(X_train[col].std()),
            "min": float(X_train[col].min()),
            "max": float(X_train[col].max())
        }

    metadata = {
        "project": "Insurance Claim Fraud Detection Using XGBoost with Hyperparameter Tuning",
        "dataset_source": "Live MySQL Database (insurance_fraud_db.vehicle_claims)",
        "total_records": total_records,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_balance": {
            "genuine_count": int((y == 0).sum()),
            "fraud_count": int(y.sum()),
            "fraud_percentage": float(y.mean() * 100)
        },
        "imputation_constants": {
            "median_age": float(median_age),
            "mode_day_claimed": str(mode_day),
            "mode_month_claimed": str(mode_month)
        },
        "feature_schema": {
            "categorical": cat_cols,
            "numerical": num_cols,
            "all_features": X.columns.tolist()
        },
        "numerical_baseline_stats": num_stats,
        "metrics_comparison": {
            "logistic_regression": lr_metrics,
            "basic_xgboost": basic_metrics,
            "xgboost_tuned": tuned_metrics
        }
    }

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[+] Saved production metadata and drift baselines to: {metadata_path}")

    print("\n[SUCCESS] ML Pipeline Execution Completed Successfully!")
    return full_pipeline, metadata

if __name__ == "__main__":
    run_training_pipeline()
