import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# =========================
# TASK 4: PROXY TARGET
# =========================

def create_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    # ---- 0. Make sure datetime ----
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    
    # ---- 1. Compute RFM metrics ----
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby("CustomerId").agg(
        Recency=("TransactionStartTime", lambda x: (snapshot_date - x.max()).days),
        Frequency=("TransactionId", "count"),
        Monetary=("Amount", "sum")
    ).reset_index()

    # ---- 2. Scale RFM features ----
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    # ---- 3. K-Means clustering (3 clusters) ----
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    rfm["Cluster"] = clusters

    # ---- 4. Define high-risk cluster ----
    # High-risk: cluster with highest average Recency, lowest Frequency/Monetary
    cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = cluster_summary.sort_values(
        by=["Recency", "Frequency", "Monetary"], 
        ascending=[False, True, True]
    ).index[0]

    rfm["is_high_risk"] = (rfm["Cluster"] == high_risk_cluster).astype(int)

    # ---- 5. Merge back to main DataFrame ----
    df = df.merge(rfm[["CustomerId", "is_high_risk"]], on="CustomerId", how="left")

    return df

# =========================
# SCRIPT ENTRY POINT
# =========================

if __name__ == "__main__":
    import os
    import pandas as pd

    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv("data/processed/processed_features.csv")
    df_with_target = create_proxy_target(df)  # <- this is the variable

    output_path = "data/processed/processed_features_with_target.csv"
    df_with_target.to_csv(output_path, index=False)  # <- use the same variable name

    print(" Proxy target added successfully.")
    print(f" Saved to: {output_path}")

