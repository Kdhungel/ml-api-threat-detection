
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from pathlib import Path

# Create API
app = FastAPI()

# Load trained ML model
model_path = Path(__file__).resolve().parent / "breast_cancer_model.pkl"
model = joblib.load(model_path)

# Input format
class CancerInput(BaseModel):
    features: list[float] = Field(
        min_length=30,
        max_length=30
    )

# Home endpoint
@app.get("/")
def home():
    return {"message": "Breast Cancer API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: CancerInput):

    prediction = model.predict([data.features])[0]

    if prediction == "B":
        result = "Benign"
    else:
        result = "Malignant"

    return {
        "prediction": result
    }