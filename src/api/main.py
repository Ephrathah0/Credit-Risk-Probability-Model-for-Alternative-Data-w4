from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow.sklearn

# Load scaler, imputer, and best model (adjust path if needed)
scaler = joblib.load("models/scaler.pkl")
imputer = joblib.load("models/imputer.pkl")
model = mlflow.sklearn.load_model("mlruns/0/<RUN_ID>/artifacts/model")  # replace <RUN_ID> with your best run id

app = FastAPI(title="Credit Risk API")

# Pydantic model for input
class CustomerData(BaseModel):
    # list all feature names except target and ID
    Feature1: float
    Feature2: float
    Feature3: float
    # ... continue with all features

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    df_imputed = imputer.transform(df)
    df_scaled = scaler.transform(df_imputed)
    proba = model.predict_proba(df_scaled)[:, 1][0]
    return {"risk_probability": proba}
