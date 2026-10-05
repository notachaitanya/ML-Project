import os

# MySQL Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Iamchaitanya@13542")
DB_NAME = os.getenv("DB_NAME", "insurance_fraud_db")
TABLE_NAME = "vehicle_claims"

# File Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.path.join(DATA_DIR, "fraud_oracle.csv")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
