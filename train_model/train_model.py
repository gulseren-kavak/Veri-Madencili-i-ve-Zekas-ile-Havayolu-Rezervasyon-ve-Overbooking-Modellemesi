"""
Gelişmiş Overbooking Risk Modeli — Eğitim Scripti
===================================================
Features:
  - Uçuş bilgileri (havayolu, rota, saat, mesafe)
  - Kapasite / rezervasyon oranı
  - Hava durumu verileri
  - Özel etkinlik etkisi
  - Mevsimsel ve gün faktörleri
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("🤖 OVERBOOKING RİSK MODELİ EĞİTİMİ")
print("=" * 60)

# ── 1. VERİ YÜKLEME ──
print("\n📂 Veriler yükleniyor...")
flights_df = pd.read_csv(DATA_DIR / "flights.csv")
print(f"   Uçuş verisi: {len(flights_df)} kayıt")
print(f"   Overbooked: {flights_df['IS_OVERBOOKED'].sum()} (%{flights_df['IS_OVERBOOKED'].mean()*100:.1f})")

# ── 2. ÖZELLİK MÜHENDİSLİĞİ ──
print("\n⚙️  Özellik mühendisliği...")

# Encode categorical
le_airline = LabelEncoder()
le_origin = LabelEncoder()
le_dest = LabelEncoder()
le_aircraft = LabelEncoder()

flights_df["AIRLINE_ENC"] = le_airline.fit_transform(flights_df["AIRLINE"])
flights_df["ORIGIN_ENC"] = le_origin.fit_transform(flights_df["ORIGIN_AIRPORT"])
flights_df["DEST_ENC"] = le_dest.fit_transform(flights_df["DESTINATION_AIRPORT"])
flights_df["AIRCRAFT_ENC"] = le_aircraft.fit_transform(flights_df["AIRCRAFT_TYPE"])

# Departure hour
flights_df["DEP_HOUR"] = flights_df["SCHEDULED_DEPARTURE"] // 100

# Is weekend
flights_df["IS_WEEKEND"] = (flights_df["DAY_OF_WEEK"] >= 5).astype(int)

# Features
FEATURE_COLS = [
    "AIRLINE_ENC", "ORIGIN_ENC", "DEST_ENC", "AIRCRAFT_ENC",
    "MONTH", "DAY_OF_WEEK", "IS_WEEKEND", "DEP_HOUR",
    "DISTANCE", "CAPACITY", "TICKETS_SOLD", "BOOKING_RATIO",
    "NO_SHOW_RATE", "HAS_EVENT", "EVENT_IMPACT",
    "WEATHER_DELAY_RISK", "SEASONAL_FACTOR", "DAY_OF_WEEK_FACTOR"
]
TARGET = "IS_OVERBOOKED"

X = flights_df[FEATURE_COLS]
y = flights_df[TARGET]

print(f"   Özellik sayısı: {len(FEATURE_COLS)}")
print(f"   Özellikler: {', '.join(FEATURE_COLS)}")

# ── 3. TRAIN / TEST BÖLME ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ── 4. MODEL EĞİTİMİ ──
print("\n🔧 Model eğitiliyor (GradientBoosting)...")
model_gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
    min_samples_split=10,
    min_samples_leaf=5
)
model_gb.fit(X_train, y_train)

print("🔧 Model eğitiliyor (RandomForest)...")
model_rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model_rf.fit(X_train, y_train)

# ── 5. DEĞERLENDİRME ──
def evaluate_model(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'─'*40}")
    print(f"📈 {name} Sonuçları:")
    print(f"{'─'*40}")
    print(f"   ROC-AUC:   {auc:.4f}")
    print(f"   Accuracy:  {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall:    {rec:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Confusion Matrix:")
    print(f"     TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    print(f"     FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    
    return {"name": name, "auc": auc, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "model": model}

results_gb = evaluate_model(model_gb, "GradientBoosting", X_test, y_test)
results_rf = evaluate_model(model_rf, "RandomForest", X_test, y_test)

# ── 6. EN İYİ MODELİ KAYDET ──
best = results_gb if results_gb["auc"] >= results_rf["auc"] else results_rf
best_model = best["model"]
print(f"\n🏆 En iyi model: {best['name']} (AUC={best['auc']:.4f})")

# Save model
joblib.dump(best_model, MODEL_DIR / "overbooking_model.joblib")
print(f"💾 Model kaydedildi: model/overbooking_model.joblib")

# Save label encoders
encoders = {
    "airline": le_airline,
    "origin": le_origin,
    "destination": le_dest,
    "aircraft": le_aircraft,
}
joblib.dump(encoders, MODEL_DIR / "label_encoders.joblib")
print(f"💾 Encoder'lar kaydedildi: model/label_encoders.joblib")

# Save feature importance
importances = best_model.feature_importances_
feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
print(f"\n📊 Özellik Önem Sıralaması:")
for feat, imp in feat_imp:
    bar = "█" * int(imp * 100)
    print(f"   {feat:25s} {imp:.4f} {bar}")

# Save metrics
metrics = {
    "best_model": best["name"],
    "auc": best["auc"],
    "accuracy": best["accuracy"],
    "precision": best["precision"],
    "recall": best["recall"],
    "f1": best["f1"],
    "features": FEATURE_COLS,
    "feature_importance": {f: round(float(i), 4) for f, i in feat_imp},
    "train_size": len(X_train),
    "test_size": len(X_test),
}
with open(MODEL_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print(f"💾 Metrikler kaydedildi: model/model_metrics.json")

# Save feature columns for the server
with open(MODEL_DIR / "feature_columns.json", "w") as f:
    json.dump(FEATURE_COLS, f)

print(f"\n{'='*60}")
print(f"✅ EĞİTİM TAMAMLANDI!")
print(f"{'='*60}")

