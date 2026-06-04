from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.decision_service import record_decision
from api.schemas import DecisionInput, RiskInput
from api.ml_service import predict_risk

app = FastAPI(title="Risk Profiling ML API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://risk-profiler-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: RiskInput):

    result = predict_risk(data)

    return {
        "status": "success",
        "data": {
            "merchant_id": data.merchant_id,
            "prediction": result
        }
    }

@app.post("/decisions")
def save_decision(data: DecisionInput):
    decision = record_decision(data)

    return {
        "status": "success",
        "data": decision,
    }
