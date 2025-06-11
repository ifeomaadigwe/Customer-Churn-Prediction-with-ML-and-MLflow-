# models/hyperparameter.py

import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from joblib import dump

# Configuration
DATA_PATH = Path(r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\data\churn_data.csv")
ARTIFACTS_DIR = Path(r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
RANDOM_STATE = 42

# 1️⃣ Data Preparation
def prepare_data():
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop('churn', axis=1)
    y = df['churn']

    # Convert categorical features BEFORE applying SMOTE
    categorical_features = ['payment_method', 'has_contract']
    X = pd.get_dummies(X, columns=categorical_features)  # One-Hot Encoding

    # Apply SMOTE to balance dataset
    smote = SMOTE(sampling_strategy=0.3, random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Define preprocessing (only numerical scaling needed now)
    numeric_features = ['tenure', 'monthly_charges', 'support_calls']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=RANDOM_STATE, stratify=y_resampled)

    return X_train, X_test, y_train, y_test, preprocessor

# 2️⃣ Define Hyperparameter Space (Prefixed with `classifier__`)
param_distributions = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.3],
    'classifier__max_depth': [3, 5, 7, 9],
    'classifier__min_child_weight': [1, 3, 5],
    'classifier__gamma': [0, 0.1, 0.2, 0.3],
    'classifier__colsample_bytree': [0.3, 0.5, 0.7, 1.0],
    'classifier__scale_pos_weight': [1, 5, 10]  # Adjust for class imbalance
}

# 3️⃣ Hyperparameter Tuning with RandomizedSearchCV
def tune_xgboost(X_train, y_train, preprocessor):
    xgb = XGBClassifier(random_state=RANDOM_STATE)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb)
    ])

    random_search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=20, 
        scoring='f1', cv=3, verbose=2, random_state=RANDOM_STATE, n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    print("\n🔍 Best Parameters Found:")
    print(random_search.best_params_)

    return random_search.best_estimator_

# 4️⃣ Execute Tuning
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    best_xgb_model = tune_xgboost(X_train, y_train, preprocessor)

    # Save best model for deployment
    dump(best_xgb_model, ARTIFACTS_DIR / "best_xgb_model.joblib")

    print("\n🚀 Best XGBoost model saved!")
