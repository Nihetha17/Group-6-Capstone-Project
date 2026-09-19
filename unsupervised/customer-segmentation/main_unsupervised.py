from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans

MODEL_PATH = Path(__file__).resolve().parent / "kmeans_model.pkl"
PERSONA_MAP = {
    0: "KANJA PISNARI",
    1: "SELEVALI",
    2: "UDHARI",
}


def resolve_persona(income: float, score: float) -> str:
    """Match the personas to the actual KMeans centroid pattern.
    High income + high spend is SELEVALI, low income + high spend is UDHARI,
    and high income + low spend is KANJA PISNARI.
    """
    if income >= 80 and score >= 75:
        return "SELEVALI"
    if income <= 40 and score >= 70:
        return "UDHARI"
    return "KANJA PISNARI"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as exc:
    model = None
    print(f"Warning: Model not found ({exc})")

app = FastAPI(
    title="Customer Segmentation API",
    description="API for predicting customer segments based on annual income and spending score.",
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
    return {"status": "Backend running", "model_ready": model is not None}


@app.post("/predict")
def predict(payload: dict):
    if model is None:
        raise HTTPException(status_code=503, detail="Model file not found.")

    if "annual_income_k" not in payload or "spending_score" not in payload:
        raise HTTPException(
            status_code=422,
            detail="Payload must include 'annual_income_k' and 'spending_score'.",
        )

    try:
        income = float(payload["annual_income_k"])
        score = float(payload["spending_score"])
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=422,
            detail="'annual_income_k' and 'spending_score' must be numeric values.",
        )

    try:
        input_data = np.array([[income, score]])
        cluster_id = int(model.predict(input_data)[0])
        persona = resolve_persona(income, score)
        return {"cluster": cluster_id, "persona": persona}
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(err)}")


def run_training_demo():
    customer_data = pd.read_csv("customers.csv")
    X = customer_data[["annual_income_k", "spending_score"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    customer_data["Cluster"] = kmeans.labels_
    print(customer_data)
    print("Cluster Centers:")
    print(kmeans.cluster_centers_)

    plt.scatter(
        customer_data["annual_income_k"],
        customer_data["spending_score"],
        c=customer_data["Cluster"],
        cmap="viridis",
    )
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        s=200,
        marker="X",
        color="red",
    )

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segmentation using K-Means")
    plt.show()

    while True:
        a = int(input("Enter Annual Income (k$): "))
        b = int(input("Enter Spending Score (1-100): "))

        new_customer = pd.DataFrame(
            [[a, b]],
            columns=["annual_income_k", "spending_score"],
        )

        cluster = kmeans.predict(new_customer)

        if cluster[0] == 2:
            print("UDHARI")
        elif cluster[0] == 1:
            print("SELEVALI")
        else:
            print("KANJA PISNARI")

        print("Cluster:", cluster[0])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

