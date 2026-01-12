from fastapi import FastAPI, HTTPException
import numpy as np
import joblib

from api.schema import TransactionInput
from api.logger import logger

app = FastAPI(title="Fraud Detection API")

# Load trained model and scaler
try:
    model = joblib.load("models/isolation_forest.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    raise RuntimeError("Model or scaler could not be loaded") from e


@app.post("/predict")
def predict_fraud(txn: TransactionInput):
    try:
        
        
      
        log_amount = np.log1p(txn.amount)
        is_night = 1 if txn.hour < 6 or txn.hour > 22 else 0
        is_weekend = 1 if txn.day_of_week >= 5 else 0
        distance = txn.distance_from_home
        amount_distance_ratio = txn.amount / (txn.distance_from_home + 1)

        features = np.array([[
            log_amount,
            is_night,
            is_weekend,
            distance,
            amount_distance_ratio
        ]])

        # Scale features
        features_scaled = scaler.transform(features)

                # Model prediction
       
        
        raw_score = -model.decision_function(features_scaled)[0]

        # Normalize to probability-like score
        fraud_probability = 1 / (1 + np.exp(-raw_score))

        THRESHOLD = 0.6
        decision = "fraud" if fraud_probability >= THRESHOLD else "legitimate"


        
        # Reasoning (Explainability)
       
        reasons = []
        if txn.amount > 100000:
            reasons.append("high transaction amount")
        if txn.distance_from_home > 100:
            reasons.append("large distance from home")
        if is_night:
            reasons.append("unusual transaction time")
        if is_weekend:
            reasons.append("weekend transaction")

        if decision == "legitimate":
            reasoning = "normal transaction behavior"
        else:
            reasoning = ", ".join(reasons) if reasons else "anomalous pattern detected"


        
        # Logging
        
        logger.info(
            f"amount={txn.amount}, distance={txn.distance_from_home}, "
            f"raw_score={raw_score:.4f}, decision={decision}"
        )

        
        # API Response
        
        return {
            "fraud_probability": round(fraud_probability, 4),
            "decision": decision,
            "reasoning": reasoning
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
