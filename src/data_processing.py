import pandas as pd
import os

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # ---- 1. Parse datetime ----
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])

    # ---- 2. Create time features ----
    df["TransactionHour"] = df["TransactionStartTime"].dt.hour
    df["TransactionDay"] = df["TransactionStartTime"].dt.day
    df["TransactionMonth"] = df["TransactionStartTime"].dt.month
    df["TransactionYear"] = df["TransactionStartTime"].dt.year

    # ---- 3. Aggregate per customer ----
    agg_df = df.groupby("CustomerId").agg(
        TotalTransactionAmount=("Amount", "sum"),
        AvgTransactionAmount=("Amount", "mean"),
        TransactionCount=("Amount", "count"),
        StdTransactionAmount=("Amount", "std")
    ).reset_index()

    # Merge aggregates back
    df = df.merge(agg_df, on="CustomerId", how="left")

    return df  # return full DataFrame

# =========================
# SCRIPT ENTRY POINT
# =========================

if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)

    # Load raw data
    df_raw = pd.read_csv("data/raw/data.csv")

    # Process
    processed_df = process_data(df_raw)

    # Save output
    output_path = "data/processed/processed_features.csv"
    processed_df.to_csv(output_path, index=False)

    print("Feature engineering completed successfully.")
    print(f"Saved to: {output_path}")

import joblib

imputer = joblib.load("models/imputer.pkl")
scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/model.pkl")
