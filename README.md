# Insurance Claim Fraud Detection Using XGBoost with Hyperparameter Tuning

**Course:** Machine Learning (A.Y. 2026-2027)  
**Institution:** KL Deemed to be University, Bachupally, Hyderabad  
**Team Number:** 8 | **Section:** 9  

### Team Members:
1. **Vadlapudi Nitish Sri Sai** (Roll No: 2520030179)
2. **Kondapalli Chaitanya** (Roll No: 2520030210)
3. **Kudumula Rama Shashank Reddy** (Roll No: 2520030215)

**Under the Guidance of:**  
**Dr. Ravindra Gaddam**, Associate Professor, Department of Computer Science and Engineering

---

## 📌 Abstract
Insurance fraud is a major problem for insurance companies, causing huge financial losses every year and leading to higher premiums for honest customers. Fraudulent claims can appear in many forms, such as exaggerated damage reports, fake accidents, or false medical claims. Traditionally, insurance companies have relied on manual checking and fixed rules to catch fraud, but these methods are slow, costly, and often fail to catch new or clever types of fraud. As the number of claims keeps increasing, there is a growing need for smarter and automated systems that can detect fraud quickly and accurately. This project aims to solve this problem by building a fraud detection system that uses machine learning to automatically identify whether an insurance claim is genuine or fraudulent. The system is trained using past claim records that include details such as the policyholder's information, claim amount, type of incident, and past claim history. Before training the model, the data is cleaned and prepared by fixing missing information, converting text details into a usable format, and balancing the data so the model does not become biased toward genuine claims, since fraud cases are usually much fewer in number. The model is first trained using standard settings to check its basic performance. Then, its settings are carefully adjusted and improved through a tuning process that tests different combinations of values to find the best possible configuration for accurate predictions. This helps the model make better decisions and avoid mistakes when working with new, unseen data. The results show that the improved model performs significantly better than the basic version, successfully identifying more fraudulent claims while reducing the number of genuine claims wrongly marked as fraud. This project demonstrates how modern data-driven techniques can help insurance companies detect fraud more effectively, save money, speed up the claims process, and build greater trust between insurers and customers.

---

## 🎯 Course Outcomes (CO) Implementation (1 Topic Per CO)

| Course Outcome | Syllabus Description | Implemented Topic in Project | Implementation Details |
| :--- | :--- | :--- | :--- |
| **CO1** | Analyze the end-to-end ML system lifecycle — data → features → training → evaluation → deployment → monitoring — and trace one prediction request from a deployed model back through every preceding lifecycle stage. | **End-to-End Prediction Request Tracing** | Implemented `/trace-prediction` endpoint that traces a prediction backwards through Stage 6 (Drift) → Stage 5 (XGBoost Inference) → Stage 4 (Decision Threshold) → Stage 3 (StandardScaler + OneHotEncoder) → Stage 2 (Cleaning & Imputation) → Stage 1 (Raw MySQL DB row in `insurance_fraud_db.vehicle_claims`). |
| **CO2** | Apply linear supervised-learning models — linear regression, ridge, lasso, elastic net, logistic regression, multinomial logistic regression — and reason about regularisation, the bias-variance trade-off, and feature scaling. | **Logistic Regression with L2 Regularization & Feature Scaling** | Fitted `LogisticRegression(penalty='l2', C=1.0)` on `StandardScaler`-normalized features. Evaluated regularisation effect preventing weight explosion and analyzed why linear decision boundaries exhibit high bias on non-linear fraud interactions. |
| **CO3** | Apply tree-based supervised-learning models — decision trees, random forests, gradient-boosted trees (XGBoost, LightGBM) — and reason about feature importance, splitting criteria, and ensemble variance reduction. | **XGBoost Classifier with Tree Splitting Gain Feature Importance** | Built gradient boosted tree ensemble optimizing 2nd-order Taylor expansion loss; ranked top 15 features by node splitting gain (Fault, Vehicle Category, Deductible, Past Claims); evaluated ensemble variance reduction. |
| **CO4** | Apply unsupervised-learning techniques — k-means and hierarchical clustering, density-based clustering (DBSCAN), dimensionality reduction (PCA, t-SNE, UMAP), and anomaly detection — to discover structure in unlabelled data. | **Principal Component Analysis (PCA)** | Applied PCA to reduce 145 unlabelled preprocessed feature dimensions into 2 principal components; visualized geometric structure and separation between genuine and fraudulent claims (`outputs/pca_fraud_structure.png`). |
| **CO5** | Analyze model performance using cross-validation, train/validation/test splits, classification metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC), regression metrics, calibration, and hyperparameter search. | **Hyperparameter Search (GridSearchCV) with 5-Fold Stratified CV** | Tuned `max_depth`, `learning_rate`, `n_estimators`, and `scale_pos_weight` using 5-Fold Stratified Cross-Validation; optimized decision threshold; plotted ROC curve, Precision-Recall curve, and Confusion Matrix. |
| **CO6** | Apply ML-engineering practices — feature stores, training-serving skew avoidance, model packaging, REST-based serving, basic monitoring for drift and performance degradation. | **REST-based Serving & Feature Drift Monitoring** | Built FastAPI service with interactive Web Dashboard (`/dashboard`), packaged end-to-end `Pipeline` via Joblib, eliminated training-serving skew with `ColumnTransformer`, and implemented feature drift monitor tracking incoming request z-scores vs training baselines. |

---

## 🗄️ MySQL Database Setup & Workbench Integration

This project connects to a live MySQL Database. All data used for cleaning, training, and testing is extracted **exclusively from this database**.

- **Database Name:** `insurance_fraud_db`
- **Table Name:** `vehicle_claims` (15,420 records)
- **Primary Key:** `PolicyNumber`

### Opening in MySQL Workbench:
1. Open **MySQL Workbench**.
2. Connect to your local instance (`Mysql@127.0.0.1:3306`).
3. Click **File -> Open SQL Script...** and select `workbench_setup.sql`.
4. Run the queries to inspect the live dataset:
   - Check total counts and genuine vs fraud ratio.
   - Check fraud rate by vehicle category, fault, deductible, driver rating, etc.

---

## 🚀 Quick Start Guide

### 1. Initialize Database & Populate MySQL
```powershell
py init_mysql_database.py
```
*Creates `insurance_fraud_db.vehicle_claims` and inserts all 15,420 records.*

### 2. Train Models & Generate Visualizations
```powershell
py src/train_and_evaluate.py
```
*Pulls from MySQL, runs PCA (CO4), trains Logistic Regression (CO2), performs 5-Fold Stratified CV GridSearchCV on XGBoost (CO3, CO5), and packages the model to `models/` (CO6).*

### 3. Launch REST API & Interactive Web Dashboard
```powershell
py run_project.py
```
Then open in your browser:
- **Interactive Web Dashboard:** [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
- **Interactive Swagger REST API:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📊 Model Evaluation Results

| Evaluation Metric | CO2: Logistic Regression (L2) | CO3: Basic XGBoost (Defaults) | CO3+5: Tuned XGBoost (Optimal) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.8064 | 0.8480 | **0.8380** |
| **PR-AUC (Precision-Recall)** | 0.1561 | 0.2496 | **0.2418** |
| **Recall (Fraud Catch Rate)** | 0.8865 | 0.0811 | **0.5081** |
| **Precision** | 0.1319 | 0.4412 | **0.2030** |
| **F1-Score** | 0.2297 | 0.1370 | **0.2901** |
| **Overall Accuracy** | 0.6433 | 0.9387 | **0.8508** |

> **Key Finding:** Standard default XGBoost only catches 8.1% of fraudulent claims due to class imbalance (94% genuine vs 6% fraud). After systematic hyperparameter tuning (`scale_pos_weight = 8`, `max_depth = 6`, `learning_rate = 0.08`), fraud catch rate jumps over 6x to **50.81%** with high accuracy (85.08%), outperforming Logistic Regression's excessive false alarm rate.
