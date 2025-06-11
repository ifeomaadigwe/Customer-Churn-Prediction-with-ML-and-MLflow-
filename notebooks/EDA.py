import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set up paths and style
DATA_PATH = Path(r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\data\churn_data.csv")
ARTIFACTS_DIR = Path(r"C:\Users\IfeomaAugustaAdigwe\Desktop\Customer_Churn_Prediction_and_Model\artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)  # Ensure the directory exists

# Use modern styles
plt.style.use('seaborn-v0_8')  # Updated style name
sns.set_theme(style="whitegrid")  
sns.set_palette("pastel")

# Load data
df = pd.read_csv(DATA_PATH)

### 📊 1. Distribution of Customer Tenure
plt.figure(figsize=(10, 6))
sns.histplot(df['tenure'], bins=20, kde=True)
plt.title('Distribution of Customer Tenure (Months)')
plt.xlabel('Tenure (months)')
plt.ylabel('Count')
plt.savefig(ARTIFACTS_DIR / 'tenure_distribution.png', bbox_inches='tight')
plt.close()

### 📊 2. Churn Rate by Payment Method
plt.figure(figsize=(10, 6))
churn_payment = df.groupby('payment_method')['churn'].mean().sort_values().mul(100)
churn_payment.plot(kind='bar', color='skyblue')
plt.title('Churn Rate by Payment Method')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=45)
plt.savefig(ARTIFACTS_DIR / 'churn_by_payment_method.png', bbox_inches='tight')
plt.close()

### 📊 3. Customer Churn Distribution
plt.figure(figsize=(6, 6))
churn_counts = df['churn'].value_counts()
churn_labels = ['Retained' if x == 0 else 'Churned' for x in churn_counts.index]
df['churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=churn_labels, colors=['lightblue', 'lightcoral'])
plt.title('Customer Churn Distribution')
plt.ylabel('')
plt.savefig(ARTIFACTS_DIR / 'churn_distribution.png', bbox_inches='tight')
plt.close()

### 📊 4. Monthly Charges vs Tenure
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='tenure', y='monthly_charges', hue='churn', alpha=0.7)
plt.title('Monthly Charges vs Tenure')
plt.xlabel('Tenure (months)')
plt.ylabel('Monthly Charges ($)')
plt.savefig(ARTIFACTS_DIR / 'monthly_charges_vs_tenure.png', bbox_inches='tight')
plt.close()

### 📊 5. Churn Rate by Contract Status
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='has_contract', hue='churn')
plt.title('Churn Rate by Contract Status')
plt.xlabel('Has Contract')
plt.ylabel('Customer Count')
plt.xticks([0, 1], ['No Contract', 'Has Contract'])
plt.savefig(ARTIFACTS_DIR / 'churn_by_contract_status.png', bbox_inches='tight')
plt.close()

### 📊 6. Support Calls vs Churn
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='churn', y='support_calls')
plt.title('Support Calls by Churn Status')
plt.xlabel('Churn Status')
plt.ylabel('Number of Support Calls')
plt.xticks([0, 1], ['Retained', 'Churned'])
plt.savefig(ARTIFACTS_DIR / 'support_calls_vs_churn.png', bbox_inches='tight')
plt.close()

### 📊 7. Box Plot of Monthly Charges by Churn
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='churn', y='monthly_charges')
plt.title('Monthly Charges by Churn Status')
plt.xlabel('Churn Status')
plt.ylabel('Monthly Charges ($)')
plt.xticks([0, 1], ['Retained', 'Churned'])
plt.savefig(ARTIFACTS_DIR / 'monthly_charges_by_churn.png', bbox_inches='tight')
plt.close()

### 📊 8. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.savefig(ARTIFACTS_DIR / 'feature_correlation_heatmap.png', bbox_inches='tight')
plt.close()

print(f"✅ All visualizations saved to {ARTIFACTS_DIR}")
