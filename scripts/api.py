import joblib
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("models/model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

app = FastAPI()

class PredictionResponse(BaseModel):
    churn: int
    churn_probability: float
    
class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict", response_model=PredictionResponse)

def predict(data: Customer):
    df = pd.DataFrame([data.dict()])
    X = preprocessor.transform(df)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]
    print("🚀 Input data:", data.dict())
    df = pd.DataFrame([data.dict()])
    print("📊 DataFrame:\n", df)
    return {"churn": int(pred), "churn_probability": round(proba, 3)}
    
    
if __name__ == "__main__":
    uvicorn.run("scripts.api:app", host="0.0.0.0", port=8000, reload=True)

