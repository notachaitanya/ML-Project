"""
Database Initializer and Data Ingestion Script.
Creates database 'insurance_fraud_db' and table 'vehicle_claims',
then populates it with all records from the Kaggle dataset if not already populated.
"""

import sys
import pandas as pd
import pymysql
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, TABLE_NAME, CSV_PATH

def init_database(force=False):
    print(f"[*] Connecting to MySQL Server at {DB_HOST}:{DB_PORT} as '{DB_USER}'...")
    try:
        # Step 1: Connect to server
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            autocommit=True
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}`;")
        cursor.execute(f"USE `{DB_NAME}`;")

        # Check if table already exists and has records
        cursor.execute(f"SHOW TABLES LIKE '{TABLE_NAME}';")
        table_exists = cursor.fetchone() is not None

        if table_exists and not force:
            cursor.execute(f"SELECT COUNT(*) FROM `{TABLE_NAME}`;")
            count = cursor.fetchone()[0]
            if count > 0:
                print(f"[+] MySQL table '{TABLE_NAME}' already populated with {count} records. Ready to use!")
                cursor.close()
                conn.close()
                return True

        # Step 2: Create Table
        print(f"[*] Creating table '{TABLE_NAME}'...")
        cursor.execute(f"DROP TABLE IF EXISTS `{TABLE_NAME}`;")
        create_table_sql = f"""
        CREATE TABLE `{TABLE_NAME}` (
            PolicyNumber INT PRIMARY KEY,
            Month VARCHAR(10) NOT NULL,
            WeekOfMonth INT NOT NULL,
            DayOfWeek VARCHAR(15) NOT NULL,
            Make VARCHAR(20) NOT NULL,
            AccidentArea VARCHAR(10) NOT NULL,
            DayOfWeekClaimed VARCHAR(15),
            MonthClaimed VARCHAR(10),
            WeekOfMonthClaimed INT NOT NULL,
            Sex VARCHAR(10) NOT NULL,
            MaritalStatus VARCHAR(15) NOT NULL,
            Age INT NOT NULL,
            Fault VARCHAR(25) NOT NULL,
            PolicyType VARCHAR(35) NOT NULL,
            VehicleCategory VARCHAR(20) NOT NULL,
            VehiclePrice VARCHAR(25) NOT NULL,
            FraudFound_P INT NOT NULL,
            RepNumber INT NOT NULL,
            Deductible INT NOT NULL,
            DriverRating INT NOT NULL,
            Days_Policy_Accident VARCHAR(20) NOT NULL,
            Days_Policy_Claim VARCHAR(20) NOT NULL,
            PastNumberOfClaims VARCHAR(20) NOT NULL,
            AgeOfVehicle VARCHAR(20) NOT NULL,
            AgeOfPolicyHolder VARCHAR(20) NOT NULL,
            PoliceReportFiled VARCHAR(10) NOT NULL,
            WitnessPresent VARCHAR(10) NOT NULL,
            AgentType VARCHAR(15) NOT NULL,
            NumberOfSuppliments VARCHAR(20) NOT NULL,
            AddressChange_Claim VARCHAR(25) NOT NULL,
            NumberOfCars VARCHAR(20) NOT NULL,
            Year INT NOT NULL,
            BasePolicy VARCHAR(20) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        cursor.execute(create_table_sql)

        # Step 3: Ingest Data from CSV
        print(f"[*] Reading dataset from {CSV_PATH}...")
        df = pd.read_csv(CSV_PATH)
        print(f"[*] Dataset loaded into memory: {df.shape[0]} rows, {df.shape[1]} columns.")

        columns = [
            'PolicyNumber', 'Month', 'WeekOfMonth', 'DayOfWeek', 'Make', 'AccidentArea',
            'DayOfWeekClaimed', 'MonthClaimed', 'WeekOfMonthClaimed', 'Sex', 'MaritalStatus',
            'Age', 'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice', 'FraudFound_P',
            'RepNumber', 'Deductible', 'DriverRating', 'Days_Policy_Accident', 'Days_Policy_Claim',
            'PastNumberOfClaims', 'AgeOfVehicle', 'AgeOfPolicyHolder', 'PoliceReportFiled',
            'WitnessPresent', 'AgentType', 'NumberOfSuppliments', 'AddressChange_Claim',
            'NumberOfCars', 'Year', 'BasePolicy'
        ]
        df_sql = df[columns].copy()
        df_sql = df_sql.fillna("")

        placeholders = ", ".join(["%s"] * len(columns))
        col_names = ", ".join([f"`{c}`" for c in columns])
        insert_sql = f"INSERT INTO `{TABLE_NAME}` ({col_names}) VALUES ({placeholders})"

        print(f"[*] Ingesting {len(df_sql)} records into MySQL `{DB_NAME}`.`{TABLE_NAME}`...")
        data_tuples = [tuple(row) for row in df_sql.values]
        batch_size = 2000
        for i in range(0, len(data_tuples), batch_size):
            batch = data_tuples[i:i + batch_size]
            cursor.executemany(insert_sql, batch)

        cursor.execute(f"SELECT COUNT(*) FROM `{TABLE_NAME}`;")
        count = cursor.fetchone()[0]
        print(f"\n[SUCCESS] Populated {count} records into MySQL table '{TABLE_NAME}'.")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    force_run = "--force" in sys.argv
    init_database(force=force_run)
