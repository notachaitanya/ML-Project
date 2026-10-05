"""
Database Connector Module.
Connects directly to the live MySQL database 'insurance_fraud_db'
and extracts raw claim records into a pandas DataFrame.
Includes resilient fallback to dataset file if MySQL server is offline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymysql
import pandas as pd
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, TABLE_NAME, CSV_PATH

def get_db_connection(as_dict=False):
    """Establish and return a connection to the live MySQL database."""
    cursorclass = pymysql.cursors.DictCursor if as_dict else pymysql.cursors.Cursor
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=cursorclass,
        connect_timeout=2
    )
    return conn

def fetch_claims_from_db(limit=None):
    """
    Query and extract vehicle claim records directly from MySQL,
    with fallback to dataset if database is offline.
    """
    try:
        print(f"[*] Connecting to live MySQL database '{DB_NAME}' at {DB_HOST}:{DB_PORT}...")
        conn = get_db_connection(as_dict=False)
        try:
            sql = f"SELECT * FROM `{TABLE_NAME}`"
            if limit is not None:
                sql += f" LIMIT {int(limit)}"
            
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchall()
                cols = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(rows, columns=cols)
                
            print(f"[+] Successfully pulled {len(df)} records from MySQL table `{TABLE_NAME}`.")
            return df
        finally:
            conn.close()
    except Exception as e:
        print(f"[!] MySQL connection failed ({e}). Falling back to local dataset at {CSV_PATH}...")
        df = pd.read_csv(CSV_PATH)
        if limit is not None:
            df = df.head(int(limit))
        return df

def fetch_single_claim_by_id(policy_number: int):
    """
    Query a single claim by its primary key PolicyNumber from MySQL,
    or dataset fallback if database is offline.
    """
    try:
        conn = get_db_connection(as_dict=True)
        try:
            sql = f"SELECT * FROM `{TABLE_NAME}` WHERE `PolicyNumber` = %s"
            with conn.cursor() as cursor:
                cursor.execute(sql, (policy_number,))
                record = cursor.fetchone()
            if record:
                return record
        finally:
            conn.close()
    except Exception:
        pass

    # Fallback to local dataset
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        row = df[df['PolicyNumber'] == policy_number]
        if not row.empty:
            return row.iloc[0].to_dict()
            
    return None

if __name__ == "__main__":
    df_check = fetch_claims_from_db(limit=5)
    print("\nDatabase pull preview:")
    print(df_check[['PolicyNumber', 'Make', 'VehicleCategory', 'FraudFound_P', 'Deductible']])
    
    single = fetch_single_claim_by_id(1)
    print("\nSingle claim fetched by PolicyNumber=1:")
    print(f"Policy: {single['PolicyNumber']}, Make: {single['Make']}, Fraud: {single['FraudFound_P']}")
