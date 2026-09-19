from pathlib import Path

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


MODEL_PATH = Path(__file__).resolve().parent / "kmeans_model.pkl"

PERSONA_MAP = {
    0: "KANJA PISNARI",
    1: "SELEVALI",
    2: "UDHARI",
}


class CustomerRequest(BaseModel):
    annual_income_k: float
    spending_score: float


try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as exc:
    model = None
    print(f"Warning: Model not found ({exc})")


app = FastAPI(
    title="Customer Segmentation API",
    description="API for predicting customer segments based on annual income and spending score.",
    version="1.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "status": "Backend running",
        "model_ready": model is not None,
    }


@app.post("/predict")
def predict(payload: CustomerRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model file not found.",
        )

    try:
        input_data = np.array([
            [payload.annual_income_k, payload.spending_score]
        ])

        cluster_id = int(model.predict(input_data)[0])
        persona = PERSONA_MAP.get(cluster_id, "Unknown")

        return {
            "cluster": cluster_id,
            "persona": persona,
        }

    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(err)}",
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
    )
