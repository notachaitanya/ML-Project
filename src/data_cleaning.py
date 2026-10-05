"""
Data Cleaning and Preprocessing Module.
Cleans raw data pulled directly from the MySQL database, performs
imputation for data anomalies, and builds a robust ColumnTransformer.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_raw_data(raw_df: pd.DataFrame):
    """
    Cleans raw DataFrame extracted from MySQL.
    
    Cleaning Operations:
    1. Impute invalid 0 values in 'DayOfWeekClaimed' and 'MonthClaimed' with column mode.
    2. Impute placeholder 0 values in 'Age' with the median age.
    3. Drop unique identifiers ('PolicyNumber') to prevent memorization / data leakage.
    4. Separate target variable 'FraudFound_P' from feature matrix X.
    
    Returns:
        X (pd.DataFrame): Cleaned feature matrix.
        y (pd.Series): Binary target vector (0 = Genuine, 1 = Fraud).
        cat_cols (list): List of categorical column names.
        num_cols (list): List of numerical column names.
    """
    df = raw_df.copy()
    
    # 1. Clean anomaly string values ('0' in day/month claimed)
    mode_day = df[df['DayOfWeekClaimed'] != '0']['DayOfWeekClaimed'].mode()[0]
    df['DayOfWeekClaimed'] = df['DayOfWeekClaimed'].replace('0', mode_day)
    
    mode_month = df[df['MonthClaimed'] != '0']['MonthClaimed'].mode()[0]
    df['MonthClaimed'] = df['MonthClaimed'].replace('0', mode_month)
    
    # 2. Impute invalid Age == 0 with median of non-zero ages
    median_age = df[df['Age'] > 0]['Age'].median()
    df.loc[df['Age'] == 0, 'Age'] = median_age
    
    # 3. Separate target y and drop primary key / non-predictive columns
    if 'FraudFound_P' in df.columns:
        y = df['FraudFound_P'].astype(int)
        X = df.drop(columns=['FraudFound_P'])
    else:
        y = None
        X = df.copy()
        
    if 'PolicyNumber' in X.columns:
        X = X.drop(columns=['PolicyNumber'])
        
    # 4. Identify categorical and numerical columns
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    
    return X, y, cat_cols, num_cols, median_age, mode_day, mode_month

def get_preprocessor(cat_cols, num_cols):
    """
    Constructs a ColumnTransformer preprocessor:
    - Numerical features: Standardized (Zero mean, unit variance for Logistic Regression & PCA)
    - Categorical features: One-Hot Encoded with handle_unknown='ignore' (avoids training-serving skew)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor

if __name__ == "__main__":
    from src.db_connector import fetch_claims_from_db
    raw_df = fetch_claims_from_db(limit=20)
    X, y, cat_cols, num_cols, med_age, m_day, m_month = clean_raw_data(raw_df)
    print("Cleaned Features Shape:", X.shape)
    print("Categorical Columns:", len(cat_cols))
    print("Numerical Columns:", len(num_cols))
    print("Median Age used for imputation:", med_age)
