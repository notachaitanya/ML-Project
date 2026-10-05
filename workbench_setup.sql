-- ====================================================================
-- PROJECT: Insurance Claim Fraud Detection Using XGBoost
-- INSTITUTION: KL University Hyderabad (KLH Bachupally)
-- AUTHORS: Vadlapudi Nitish Sri Sai, Kondapalli Chaitanya, Kudumula Rama Shashank Reddy
-- GUIDE: Dr. Ravindra Gaddam
-- ====================================================================
-- This SQL script sets up the MySQL database, tables, and analytical queries
-- for presentation and verification in MySQL Workbench.
-- ====================================================================

-- 1. Create and Select Database
CREATE DATABASE IF NOT EXISTS insurance_fraud_db;
USE insurance_fraud_db;

-- 2. Create the Claims Table
DROP TABLE IF EXISTS vehicle_claims;

CREATE TABLE vehicle_claims (
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
    FraudFound_P INT NOT NULL,  -- Target Variable: 0 = Genuine, 1 = Fraud
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
);

-- ====================================================================
-- VERIFICATION & DEMO QUERIES FOR MYSQL WORKBENCH
-- (Run each query below in MySQL Workbench to show data to evaluation panel)
-- ====================================================================

-- Query 1: View first 10 claims
SELECT * FROM vehicle_claims LIMIT 10;

-- Query 2: Total number of claims and class balance (Fraud vs Genuine)
SELECT 
    CASE WHEN FraudFound_P = 1 THEN 'Fraudulent (Class 1)' ELSE 'Genuine (Class 0)' END AS Claim_Status,
    COUNT(*) AS Total_Claims,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM vehicle_claims), 2) AS Percentage
FROM vehicle_claims
GROUP BY FraudFound_P;

-- Query 3: Fraud rate by Vehicle Category (Sedan, Sport, Utility)
SELECT 
    VehicleCategory,
    COUNT(*) AS Total_Claims,
    SUM(FraudFound_P) AS Fraud_Count,
    ROUND(SUM(FraudFound_P) * 100.0 / COUNT(*), 2) AS Fraud_Rate_Pct
FROM vehicle_claims
GROUP BY VehicleCategory
ORDER BY Fraud_Rate_Pct DESC;

-- Query 4: Fraud rate by Fault (Policy Holder vs Third Party)
SELECT 
    Fault,
    COUNT(*) AS Total_Claims,
    SUM(FraudFound_P) AS Fraud_Count,
    ROUND(SUM(FraudFound_P) * 100.0 / COUNT(*), 2) AS Fraud_Rate_Pct
FROM vehicle_claims
GROUP BY Fault;

-- Query 5: Fraud rate by Policy Type
SELECT 
    PolicyType,
    COUNT(*) AS Total_Claims,
    SUM(FraudFound_P) AS Fraud_Claims,
    ROUND(SUM(FraudFound_P) * 100.0 / COUNT(*), 2) AS Fraud_Rate_Pct
FROM vehicle_claims
GROUP BY PolicyType
ORDER BY Fraud_Rate_Pct DESC;

-- Query 6: Deductible distribution vs Fraud
SELECT 
    Deductible,
    COUNT(*) AS Claims_Count,
    SUM(FraudFound_P) AS Fraud_Count,
    ROUND(SUM(FraudFound_P) * 100.0 / COUNT(*), 2) AS Fraud_Rate_Pct
FROM vehicle_claims
GROUP BY Deductible
ORDER BY Deductible;

-- Query 7: Fraud rate based on Past Number of Claims
SELECT 
    PastNumberOfClaims,
    COUNT(*) AS Total_Claims,
    SUM(FraudFound_P) AS Fraud_Count,
    ROUND(SUM(FraudFound_P) * 100.0 / COUNT(*), 2) AS Fraud_Rate_Pct
FROM vehicle_claims
GROUP BY PastNumberOfClaims
ORDER BY Fraud_Count DESC;

-- Query 8: High risk cases where Address Change occurred right before claim
SELECT 
    AddressChange_Claim,
    COUNT(*) AS Total_Claims,
    SUM(FraudFound_P) AS Fraud_Count,
    ROUND(SUM(FraudFound_P) * 100.0 / COUNT(*), 2) AS Fraud_Rate_Pct
FROM vehicle_claims
GROUP BY AddressChange_Claim
ORDER BY Fraud_Rate_Pct DESC;

-- Query 9: Fraud occurrence by Driver Rating
SELECT 
    DriverRating,
    COUNT(*) AS Total_Claims,
    SUM(FraudFound_P) AS Fraud_Count,
    ROUND(SUM(FraudFound_P) * 100.0 / COUNT(*), 2) AS Fraud_Rate_Pct
FROM vehicle_claims
GROUP BY DriverRating
ORDER BY DriverRating;

-- Query 10: Summary of claims by Year and Fraud detected
SELECT 
    Year,
    COUNT(*) AS Total_Claims,
    SUM(FraudFound_P) AS Fraud_Claims,
    ROUND(SUM(FraudFound_P) * 100.0 / COUNT(*), 2) AS Fraud_Rate_Pct
FROM vehicle_claims
GROUP BY Year
ORDER BY Year;
