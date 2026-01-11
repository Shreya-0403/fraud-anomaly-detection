from fastapi import FastAPI, HTTPException
import numpy as np
import joblib

from api.schema import TransactionInput
from api.logger import logger

app = FastAPI(title="Fraud Detection API")

# Load trained model
try:
    model = joblib.load("models/isolation_forest.pkl")
except Exception as e:
    raise RuntimeError("Model could not be loaded") from e


@app.post("/predict")
def predict_fraud(txn: TransactionInput):
    try:
        # Convert input to model format
        features = np.array([[
            txn.amount,
            txn.hour,
            txn.day_of_week,
            txn.month,
            txn.distance_from_home
        ]])

        # Model prediction
        prediction = model.predict(features)[0]

        # Anomaly score
        raw_score = -model.decision_function(features)[0]

        # Normalize to 0–1 range (sigmoid)
        fraud_probability = 1 / (1 + np.exp(-raw_score))

        # Final decision
        decision = "fraud" if prediction == -1 else "legitimate"

        # Reasoning logic
        reasons = []
        if txn.amount > 100000:
            reasons.append("high transaction amount")
        if txn.distance_from_home > 100:
            reasons.append("large distance from home")
        if txn.hour < 6 or txn.hour > 22:
            reasons.append("unusual transaction time")

        reasoning = ", ".join(reasons) if reasons else "normal transaction behavior"

        # Logging
        logger.info(
            f"amount={txn.amount}, distance={txn.distance_from_home}, "
            f"raw_score={raw_score:.4f}, decision={decision}"
        )

        # API response
        return {
            "fraud_probability": round(fraud_probability, 4),
            "decision": decision,
            "reasoning": reasoning
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
