# model/model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)

# Handling Imbalanced Data
from imblearn.over_sampling import SMOTE

# Configuration
DATA_PATH = Path(r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\data\churn_data.csv")
ARTIFACTS_DIR = Path(r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)  # Ensure artifacts folder exists
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

    # Apply SMOTE only after categorical encoding
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

# 2️⃣ Model Training and Evaluation
def train_and_evaluate(model, model_name, X_train, X_test, y_train, y_test, preprocessor):
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Handle probability predictions
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = pipeline.decision_function(X_test)  # SVM uses decision function

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=1),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    # Visualization
    plot_confusion_matrix(y_test, y_pred, model_name)
    plot_roc_curve(y_test, y_proba, model_name)

    return pipeline, metrics

# 3️⃣ Visualization Functions
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(ARTIFACTS_DIR / f'confusion_matrix_{model_name.lower()}.png', bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(ARTIFACTS_DIR / f'roc_curve_{model_name.lower()}.png', bbox_inches='tight')
    plt.close()

# 4️⃣ Main Execution
if __name__ == "__main__":
    # Prepare data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # Define models with class weighting
    models = {
        'LogisticRegression': LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE),
        'XGBoost': XGBClassifier(scale_pos_weight=1/(9/191), random_state=RANDOM_STATE),
        'SVM': SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    }

    # Train and evaluate
    results = {}
    for name, model in models.items():
        print(f"\n🔥 Training {name}...")
        pipeline, metrics = train_and_evaluate(
            model, name, X_train, X_test, y_train, y_test, preprocessor)
        results[name] = metrics

    # Compare models
    results_df = pd.DataFrame(results).T
    print("\n📊 Model Comparison:")
    print(results_df)
    results_df.to_csv(ARTIFACTS_DIR / 'model_comparison.csv')
