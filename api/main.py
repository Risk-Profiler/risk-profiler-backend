from fastapi import FastAPI
from api.schemas import RiskInput
from api.ml_service import predict_risk

app = FastAPI(title="Risk Profiling ML API")

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