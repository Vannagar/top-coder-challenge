#!/usr/bin/env python3

import sys
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Global variables for model and scaler
model = None
scaler = None

def train_model():
    global model, scaler
    
    # Load training data
    with open('public_cases.json', 'r') as f:
        test_cases = json.load(f)
    
    # Prepare features and target
    X = np.array([
        [case['input']['trip_duration_days'],
         case['input']['miles_traveled'],
         case['input']['total_receipts_amount']]
        for case in test_cases
    ])
    y = np.array([case['expected_output'] for case in test_cases])
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

def calculate_reimbursement(trip_duration_days, miles_traveled, total_receipts_amount):
    global model, scaler
    
    # Initialize model if not already done
    if model is None or scaler is None:
        train_model()
    
    # Prepare input
    X_new = np.array([[trip_duration_days, miles_traveled, total_receipts_amount]])
    X_new_scaled = scaler.transform(X_new)
    
    # Make prediction
    prediction = model.predict(X_new_scaled)[0]
    
    return prediction

def main():
    if len(sys.argv) != 4:
        print("Usage: python3 calculate_reimbursement.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        trip_duration = float(sys.argv[1])
        miles = float(sys.argv[2])
        receipts = float(sys.argv[3])
        
        result = calculate_reimbursement(trip_duration, miles, receipts)
        print(result)
    except ValueError:
        print("Error: All arguments must be numbers")
        sys.exit(1)

if __name__ == "__main__":
    main()
