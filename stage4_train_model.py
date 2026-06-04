
from dotenv import load_dotenv
load_dotenv()

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone

from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import joblib



logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MONGODB_URI   = os.getenv("MONGODB_URI", "")
MONGODB_DB    = "aqi_pipeline"
FEATURES_COL  = "weather_aqi_features"
REGISTRY_COL  = "model_registry"          # ← Model Registry collection
OUTPUT_DIR    = Path("aqi_model_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET = "aqi"


# ── Load Data ────────────────────────────────────────────────────────────────
def load_data():
    log.info("Loading features from MongoDB...")
    client = MongoClient(MONGODB_URI)
    db     = client[MONGODB_DB]
    docs   = list(db[FEATURES_COL].find({}, {"_id": 0}))
    client.close()
    df = pd.DataFrame(docs)
    log.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


# ── Preprocess ───────────────────────────────────────────────────────────────
def preprocess(df):
    df = df.copy()

    if "timestamp" in df.columns:
        df["timestamp"]   = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["hour"]        = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"]       = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        df["is_night"]    = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
        df = df.sort_values("timestamp").reset_index(drop=True)

    if "city" in df.columns:
        le = LabelEncoder()
        df["city_encoded"] = le.fit_transform(df["city"].astype(str))

    drop_cols = ["timestamp", "city", "dominant_poll", "aqi_cat_label", "aqi_cat_ordinal"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df = df.dropna(subset=[TARGET])

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    X = df[feature_cols]
    y = df[TARGET]

    return X, y, feature_cols


# ── Train ────────────────────────────────────────────────────────────────────
def train(X, y):
    split_idx   = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp  = imputer.transform(X_test)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train_imp, y_train)

    preds = model.predict(X_test_imp)
    metrics = {
        "mae":  round(float(mean_absolute_error(y_test, preds)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "r2":   round(float(r2_score(y_test, preds)), 4),
        "train_size": int(len(X_train)),
        "test_size":  int(len(X_test)),
    }

    log.info(f"MAE={metrics['mae']} | RMSE={metrics['rmse']} | R²={metrics['r2']}")
    return model, imputer, metrics


# ── Save Locally ─────────────────────────────────────────────────────────────
def save_artifacts(model, imputer, feature_cols, metrics):
    joblib.dump(model,   OUTPUT_DIR / "aqi_xgb_model.pkl")
    joblib.dump(imputer, OUTPUT_DIR / "imputer.pkl")
    with open(OUTPUT_DIR / "feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "latest_metrics.csv", index=False)
    log.info(f"Model artifacts saved to {OUTPUT_DIR}/")


# ── Register in MongoDB ───────────────────────────────────────────────────────
def register_model(metrics, feature_cols):
    if not MONGODB_URI:
        log.warning("MONGODB_URI not set — skipping model registry.")
        return

    client = MongoClient(MONGODB_URI)
    db     = client[MONGODB_DB]
    registry = db[REGISTRY_COL]

    # Check if this model is better than the current best
    best = registry.find_one({"status": "champion"})
    is_better = True
    if best:
        is_better = metrics["r2"] > best.get("metrics", {}).get("r2", -999)
        if not is_better:
            log.info(f"New model R²={metrics['r2']} not better than champion R²={best['metrics']['r2']} — logged as challenger.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    registry_doc = {
        "run_id":       run_id,
        "model_name":   "XGBoost_AQI",
        "version":      run_id,
        "status":       "champion" if is_better else "challenger",
        "metrics":      metrics,
        "feature_cols": feature_cols,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "model_file":   str(OUTPUT_DIR / "aqi_xgb_model.pkl"),
    }

    # Demote old champion
    if is_better and best:
        registry.update_one({"status": "champion"}, {"$set": {"status": "retired"}})
        log.info("Previous champion demoted to retired.")

    registry.insert_one(registry_doc)
    client.close()

    status = "👑 CHAMPION" if is_better else "🔁 CHALLENGER"
    log.info(f"Model registered in MongoDB registry as {status} | run_id={run_id}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("=" * 50)
    log.info("Stage 4: Daily Model Training Started")
    log.info("=" * 50)

    df              = load_data()
    X, y, feat_cols = preprocess(df)
    model, imputer, metrics = train(X, y)
    save_artifacts(model, imputer, feat_cols, metrics)
    register_model(metrics, feat_cols)

    log.info("=" * 50)
    log.info("Stage 4 Complete ")
    log.info(f"  MAE  : {metrics['mae']}")
    log.info(f"  RMSE : {metrics['rmse']}")
    log.info(f"  R²   : {metrics['r2']}")
    log.info("=" * 50)