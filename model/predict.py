# model/predict.py
import pandas as pd
from joblib import load

# Load the trained XGBoost model
model_path = r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\artifacts\best_xgb_model.joblib"
best_xgb_model = load(model_path)

# Load a sample dataset (replace with actual data for prediction)
data_path = r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\data\churn_data.csv"
df = pd.read_csv(data_path)

# Drop the target column (assuming 'churn' is the target variable)
X_new = df.drop(columns=['churn'])

# Convert categorical features using One-Hot Encoding (must match training)
categorical_features = ['payment_method', 'has_contract']
X_new = pd.get_dummies(X_new, columns=categorical_features)

# Make predictions
predictions = best_xgb_model.predict(X_new)
print("\n🚀 Predictions:", predictions)
