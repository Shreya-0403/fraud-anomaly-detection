Fraud Detection Using Anomaly Detection

Project Overview
This project focuses on detecting fraudulent financial transactions using anomaly detection techniques.
Given the highly imbalanced nature of fraud data, traditional supervised learning models are less effective. Therefore, unsupervised and semi-supervised models were explored to identify abnormal transaction behavior.
The project evaluates multiple anomaly detection models and deploys the final selected model using FastAPI to simulate a real-world fraud detection system.
Objectives
•	Analyze and preprocess a large, imbalanced transaction dataset
•	Apply anomaly detection models for fraud detection
•	Compare model performance using appropriate metrics
•	Select the most suitable model for production use
•	Deploy the final model as an API using FastAPI
1. fraudTest (Raw Dataset)
•	Rows: ~555,000+
•	Columns: ~20–22
•	Description: Original transaction dataset containing identifiers, timestamps, merchant details, and fraud labels
•	Usage: Exploratory analysis and feature engineering
•	Note: Not included in the repository due to GitHub file size limits
        dataset link - https://drive.google.com/file/d/1T4rUZTGMl-YcvXOdEbrGSo_FrnxP91DV/view?usp=sharing
2. transactions_cleaned (Processed Dataset)
•	Rows: 150,000
•	Columns: 15
•	Selected Features:
o	amount
o	hour
o	day_of_week
o	month
o	distance_from_home
o	is_fraud
•	Description: Cleaned, reduced, and model-ready dataset used for training and evaluation


Feature Engineering (Final Feature Set)
Fraud detection works best on behavioral deviation features rather than raw identifiers.
Final features used for modeling:
•	log_amount – log-scaled transaction amount
•	is_night – late-night transaction indicator
•	is_weekend – weekend transaction indicator
•	distance_from_home – distance between customer and merchant
•	amount_distance_ratio – spending relative to distance
These features capture unusual spending behavior, temporal risk, and location deviation.

Models Implemented
One-Class SVM
•	Trained only on legitimate transactions
•	High recall but very low precision
•	Poor scalability on large datasets
•	Used as a baseline model
Autoencoder
•	Neural network trained to reconstruct normal behavior
•	Fraud detected via high reconstruction error
•	Captures non-linear patterns but shows low recall
•	Higher latency than tree-based models
Isolation Forest (Final Model)
•	Tree-based anomaly detection algorithm
•	Best balance between precision and recall
•	Highest F1-score and AUC-ROC
•	Fast, scalable, and production-ready






Model Comparison Summary
Model	Precision (Fraud)	Recall (Fraud)	F1-Score	AUC-ROC
One-Class SVM	Low	High	Low	Good
Autoencoder	Low	Low	Low	Moderate
Isolation Forest	Balanced	Balanced	Best	Best
Final Selection: Isolation Forest

API Deployment (FastAPI)
The selected Isolation Forest model is deployed using FastAPI.
Endpoint
POST /predict
Input
{
  "amount": 100000,
  "hour": 23,
  "day_of_week": 6,
  "month": 1,
  "distance_from_home": 50
}
Output
{
  "fraud_probability": 0.82,
  "decision": "fraud",
  "reasoning": "high transaction amount, unusual transaction time, weekend transaction"
}
API Features
•	Input validation using Pydantic
•	Error handling for robust predictions
•	Logging for monitoring and audit
•	Human-readable reasoning for explainability


How to Run
Install Dependencies
pip install -r requirements.txt
Run Jupyter Notebook
jupyter notebook
Start FastAPI Server
uvicorn api.main:app --reload
Open API docs at:
http://127.0.0.1:8000/docs

Version Control
•	Incremental Git commits maintained throughout development
•	Large datasets excluded using .gitignore
•	Commit history reflects experimentation and model selection

Conclusion
This project demonstrates that anomaly detection techniques are well-suited for fraud detection in highly imbalanced datasets.
After systematic experimentation, Isolation Forest proved to be the most effective and practical model for real-world deployment.


