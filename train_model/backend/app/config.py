import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data files (you can place them under a 'data' folder)
DATA_DIR = BASE_DIR / "data"
FLIGHT_DATA_FILE = DATA_DIR / "flights.csv"
AIRPORT_DATA_FILE = DATA_DIR / "airports.csv"

# Model artifact
MODEL_PATH = BASE_DIR / "model" / "overbooking_model.joblib"

# Default risk threshold (can be overridden via env variable)
RISK_THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.7"))

