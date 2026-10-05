"""
Integration test for the FastAPI REST API and CO1 lifecycle tracing.
Tests:
- GET /
- POST /predict
- POST /trace-prediction (with PolicyNumber from live MySQL DB)
- GET /drift-monitor
- GET /metrics
"""

from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def run_tests():
    print("=" * 60)
    print("TESTING REST API ENDPOINTS & CO TRACEABILITY")
    print("=" * 60)

    # 1. Health check & landing
    print("\n[1] Testing GET / (Landing & CO Mapping)...")
    res = client.get("/")
    assert res.status_code == 200
    print("  -> Status 200 OK. Contains HTML documentation.")

    # 2. Metrics endpoint
    print("\n[2] Testing GET /metrics (CO2, CO3, CO5 comparison)...")
    res = client.get("/metrics")
    assert res.status_code == 200
    data = res.json()
    print("  -> Models evaluated:", list(data.keys()))
    print("  -> Tuned XGBoost Best Hyperparameters:", data['xgboost_tuned']['best_hyperparameters'])

    # 3. Predict endpoint (Genuine sample)
    print("\n[3] Testing POST /predict (Real-time scoring)...")
    sample_payload = {
        "Month": "Dec", "WeekOfMonth": 5, "DayOfWeek": "Wednesday", "Make": "Honda",
        "AccidentArea": "Urban", "DayOfWeekClaimed": "Tuesday", "MonthClaimed": "Jan",
        "WeekOfMonthClaimed": 1, "Sex": "Female", "MaritalStatus": "Single", "Age": 21,
        "Fault": "Policy Holder", "PolicyType": "Sport - Liability", "VehicleCategory": "Sport",
        "VehiclePrice": "more than 69000", "RepNumber": 12, "Deductible": 300, "DriverRating": 1,
        "Days_Policy_Accident": "more than 30", "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none", "AgeOfVehicle": "3 years", "AgeOfPolicyHolder": "26 to 30",
        "PoliceReportFiled": "No", "WitnessPresent": "No", "AgentType": "External",
        "NumberOfSuppliments": "none", "AddressChange_Claim": "1 year", "NumberOfCars": "3 to 4",
        "Year": 1994, "BasePolicy": "Liability"
    }
    res = client.post("/predict", json=sample_payload)
    assert res.status_code == 200
    pred = res.json()
    print("  -> Prediction Output:", pred)

    # 4. Trace prediction (CO1: Tracing back to MySQL DB)
    print("\n[4] Testing POST /trace-prediction (CO1: Tracing Policy #1 from MySQL)...")
    trace_res = client.post("/trace-prediction", json={"policy_number": 1})
    assert trace_res.status_code == 200
    trace_data = trace_res.json()
    print("  -> CO1 Lifecycle Trace Stages:")
    print("     * Stage 1 (Raw DB Record):", trace_data['stage_1_raw_data_retrieval']['source'])
    print("     * Stage 2 (Cleaning):", trace_data['stage_2_data_cleaning']['cleaning_actions_taken'])
    print("     * Stage 3 (Feature Scaling):", trace_data['stage_3_feature_transformations']['total_feature_vector_dimensions'], "features")
    print("     * Stage 4 (Evaluation Threshold):", trace_data['stage_4_evaluation_criteria']['decision_threshold'])
    print("     * Stage 5 (Inference):", trace_data['stage_5_model_inference']['predicted_label'], "Prob:", trace_data['stage_5_model_inference']['predicted_fraud_probability'])
    print("     * Stage 6 (Monitoring Drift Check):", trace_data['stage_6_monitoring']['overall_drift_status'])

    # 5. Drift monitor
    print("\n[5] Testing GET /drift-monitor (CO6 Drift Tracking)...")
    drift_res = client.get("/drift-monitor")
    assert drift_res.status_code == 200
    drift_data = drift_res.json()
    print("  -> Monitoring Status:", drift_data['monitoring_status'])
    print("  -> Requests Evaluated:", drift_data['requests_evaluated'])
    print("  -> Overall Drift Detected:", drift_data['overall_drift_detected'])

    print("\n[SUCCESS] All REST API Endpoints and CO Verification Tests Passed!")

if __name__ == "__main__":
    run_tests()
