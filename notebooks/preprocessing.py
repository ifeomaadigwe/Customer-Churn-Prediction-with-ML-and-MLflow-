import pandas as pd
from pathlib import Path

# 1. Load the Dataset
def load_data():
    data_path = Path("../data/churn_data.csv")
    df = pd.read_csv(data_path)
    print("✅ Dataset loaded successfully!")
    return df

# 2. Data Inspection
def inspect_data(df):
    print("\n=== First 5 Rows ===")
    print(df.head())
    
    print("\n=== Dataset Info ===")
    print(df.info())
    
    print("\n=== Descriptive Statistics ===")
    print(df.describe(include='all'))
    
    print("\n=== Missing Values ===")
    print(df.isnull().sum())

# 3. Handle Missing Values
def handle_missing_values(df):
    # Check initial missing values
    missing_before = df.isnull().sum().sum()
    
    # Fill numerical columns with median
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    
    # Fill categorical columns with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    missing_after = df.isnull().sum().sum()
    print(f"\n🔥 Missing values handled: {missing_before} → {missing_after} remaining")
    return df

# 4. Save Processed Data
def save_processed_data(df):
    output_path = Path("../data/processed_churn_data.csv")
    df.to_csv(output_path, index=False)
    print(f"\n💾 Processed data saved to: {output_path}")

# Main Pipeline
if __name__ == "__main__":
    print("🚀 Starting Data Preprocessing Pipeline...")
    
    # Step 1: Load
    df = load_data()
    
    # Step 2: Inspect
    inspect_data(df)
    
    # Step 3: Clean
    clean_df = handle_missing_values(df)
    
    # Step 4: Save
    save_processed_data(clean_df)
    
    print("\n✅ Preprocessing completed successfully!")