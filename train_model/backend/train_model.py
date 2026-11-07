"""Simple training script for the overbooking risk model.
It reads the flight CSV, creates a binary target ``overbook`` that is
``True`` when the historical ``DEPARTURE_DELAY`` exceeds a configurable
threshold (default 15 minutes).  The script trains a GradientBoostingClassifier,
persists it with joblib, and stores the artifact under ``backend/model``.

You can adapt the feature engineering and target definition to match the
project requirements (weather, special‑event flags, etc.).
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# Paths – adjust if you move the project folder
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FLIGHT_CSV = DATA_DIR / "flights.csv"

# Parameters (you may expose them via env vars or CLI args)
DELAY_THRESHOLD_MIN = int(os.getenv("DELAY_THRESHOLD_MIN", "15"))  # minutes
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

def load_data() -> pd.DataFrame:
    if not FLIGHT_CSV.exists():
        raise FileNotFoundError(f"{FLIGHT_CSV} not found. Place the flights CSV there.")
    df = pd.read_csv(FLIGHT_CSV, low_memory=False)
    return df

def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # Basic binary target: overbooking risk = large departure delay
    df["overbook"] = (df["DEPARTURE_DELAY"] > DELAY_THRESHOLD_MIN).astype(int)
    # Select a few illustrative features – extend as needed
    features = [
        "AIRLINE",
        "FLIGHT_NUMBER",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "SCHEDULED_DEPARTURE",
        "DISTANCE",
    ]
    X = df[features].copy()
    # Simple encoding – factorize categorical columns
    for col in ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        X[col] = pd.factorize(X[col])[0]
    y = df["overbook"]
    return X, y

def train():
    df = load_data()
    X, y = prepare(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    # Evaluation
    prob_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob_test)
    print(f"Test ROC‑AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))
    # Persist model
    model_path = MODEL_DIR / "overbooking_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train()

