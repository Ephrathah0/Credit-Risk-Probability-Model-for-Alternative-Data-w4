import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import os

# =========================
# Load data
# =========================
df = pd.read_csv("data/processed/processed_features_with_target.csv")

# Features and target
X = df.drop(columns=["is_high_risk", "TransactionStartTime", "TransactionId", "BatchId", 
                     "AccountId", "SubscriptionId", "CustomerId", "CurrencyCode", 
                     "CountryCode", "ProviderId", "ProductId", "ProductCategory", "ChannelId", 
                     "PricingStrategy", "FraudResult", "Value"])  # keep numeric/time features
y = df["is_high_risk"]

# =========================
# Train/test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.impute import SimpleImputer

# =========================
# Scale & Impute features
# =========================
imputer = SimpleImputer(strategy="median")  # or mean if you prefer
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# =========================
# MLflow setup
# =========================
mlflow.set_experiment("Credit_Risk_Model")

# =========================
# Models to train
# =========================
models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "RandomForest": RandomForestClassifier(random_state=42)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        
        # Log metrics & model
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)
        mlflow.sklearn.log_model(model, "model")
        
        print(f" {name} trained and logged. ROC-AUC: {roc:.4f}")

# =========================
# Save scaler for future use
# =========================
os.makedirs("models", exist_ok=True)
import joblib
joblib.dump(scaler, "models/scaler.pkl")
print("Scaler saved at models/scaler.pkl")
