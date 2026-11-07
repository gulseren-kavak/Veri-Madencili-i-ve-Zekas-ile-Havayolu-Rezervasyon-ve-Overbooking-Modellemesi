"""API router for overbooking risk calculations."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

from .config import MODEL_PATH, RISK_THRESHOLD, FLIGHT_DATA_FILE

router = APIRouter()

# Load model once (FastAPI will import this module at startup)
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None

class FlightInput(BaseModel):
    """Minimal flight information required for a prediction.
    In a real system you would pass the flight_id and look‑up details in the DB.
    """
    airline: str
    flight_number: int
    origin_airport: str
    destination_airport: str
    scheduled_departure: int  # HHMM (e.g., 1330)
    distance: float
    # Add more fields as needed (weather index, special event flag, etc.)

def _prepare_features(inp: FlightInput) -> pd.DataFrame:
    """Convert the pydantic model into a one‑row DataFrame matching the training schema.
    This function is very simple – in production you would encode categorical columns
    with the same encoders used during training.
    """
    # Basic numeric features
    data = {
        "AIRLINE": [inp.airline],
        "FLIGHT_NUMBER": [inp.flight_number],
        "ORIGIN_AIRPORT": [inp.origin_airport],
        "DESTINATION_AIRPORT": [inp.destination_airport],
        "SCHEDULED_DEPARTURE": [inp.scheduled_departure],
        "DISTANCE": [inp.distance],
    }
    df = pd.DataFrame(data)
    # Simple preprocessing – encode categories with pandas factorize (fit on training data later)
    for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        df[col] = pd.factorize(df[col])[0]
    return df

@router.post("/predict")
def predict_risk(flight: FlightInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    X = _prepare_features(flight)
    prob = model.predict_proba(X)[:, 1][0]
    risk = prob >= RISK_THRESHOLD
    return {"flight": flight.dict(), "risk_probability": prob, "high_risk": risk}

@router.get("/risky_flights")
def list_risky_flights(limit: int = 20):
    """Return the next *limit* flights with predicted probability >= threshold.
    This is a very light‑weight demo that loads the CSV, scores the whole set and
    returns the top risky rows.
    """
    if not FLIGHT_DATA_FILE.exists():
        raise HTTPException(status_code=404, detail="Flight data not found")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    df = pd.read_csv(FLIGHT_DATA_FILE, low_memory=False)
    # Keep only columns needed for the model – adjust as you extend the schema
    needed = ["AIRLINE", "FLIGHT_NUMBER", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "DISTANCE"]
    for col in needed:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Missing column {col} in data")
    X = df[needed].copy()
    # Encode categorical columns the same way as in _prepare_features (factorize)
    for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        X[col] = pd.factorize(X[col])[0]
    probs = model.predict_proba(X)[:, 1]
    df["risk_probability"] = probs
    risky = df[df["risk_probability"] >= RISK_THRESHOLD]
    risky = risky.sort_values("risk_probability", ascending=False).head(limit)
    # Convert to a list of dicts for JSON response
    return risky.to_dict(orient="records")

