from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib


# Create FastAPI application
app = FastAPI(
    title="Email Spam Detection API",
    description="API for detecting whether an email is spam or not spam.",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "spam_model.pkl"

model = joblib.load(MODEL_PATH)

# Request model
class EmailRequest(BaseModel):
    email: str


# Root endpoint
@app.get("/")
def home():
    return {
        "message": "Email Spam Detection API is running"
    }


# Prediction endpoint
@app.post("/predict")
def predict(request: EmailRequest):

    # Make prediction
    prediction = model.predict([request.email])[0]

    # Get probability
    probabilities = model.predict_proba([request.email])[0]
    confidence = max(probabilities)

    # Convert numerical prediction to label
    if prediction == 1:
        result = "Spam"
    else:
        result = "Not Spam"

    return {
        "email": request.email,
        "prediction": result,
        "label": int(prediction),
        "confidence": round(float(confidence), 4)
    }
