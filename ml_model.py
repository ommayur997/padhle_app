import joblib
import numpy as np


def predict_risk(input_data):
    """Predict risk score using the trained model without fallback heuristic"""
    # Load trained model and scaler
    model = joblib.load('trained_model.pkl')
    scaler = joblib.load('trained_scaler.pkl')

    # Scale input data
    scaled_data = scaler.transform(input_data)

    # Predict risk
    risk_score = model.predict(scaled_data)[0]
    is_at_risk = risk_score > 60
    return risk_score, is_at_risk