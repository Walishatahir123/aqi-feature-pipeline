"""
STAGE 3 — Feature Store Upload (Hopsworks)
Hopsworks free tier: https://app.hopsworks.ai  (sign up free)

What this does:
  - Creates a Feature Group called "weather_aqi_features"
  - Upserts each new row by timestamp (idempotent)
  - Creates a Feature View for easy model training retrieval
  - Saves a local CSV backup in  data/features_store_backup/

Setup:
  1. Sign up at https://app.hopsworks.ai
  2. Create a project (e.g. "AQI_Forecaster")
  3. Go to Account → API Keys → Create Key
  4. Set env var:  export HOPSWORKS_API_KEY="your_key_here"
  5. Set env var:  export HOPSWORKS_PROJECT="AQI_Forecaster"
  Or add them as GitHub Secrets for CI/CD.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "AQI_Forecaster")
FEATURE_GROUP_NAME    = "weather_aqi_features"
FEATURE_GROUP_VERSION = 1
BACKUP_DIR = Path("data/features_store_backup")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature Group Schema (metadata) ────────────────────────────────────────

FEATURE_DESCRIPTIONS = {
    # Time
    "timestamp":                  "UTC timestamp of the reading",
    "city":                       "City name",
    "hour":                       "Hour of day (0–23)",
    "day_of_week":                "Day of week (0=Mon, 6=Sun)",
    "month":                      "Month (1–12)",
    "day_of_year":                "Day of year (1–365)",
    "hour_sin":                   "Cyclic sine encoding of hour",
    "hour_cos":                   "Cyclic cosine encoding of hour",
    "dow_sin":                    "Cyclic sine encoding of day of week",
    "dow_cos":                    "Cyclic cosine encoding of day of week",
    "month_sin":                  "Cyclic sine encoding of month",
    "month_cos":                  "Cyclic cosine encoding of month",
    "is_weekend":                 "1 if Saturday or Sunday",
    "is_peak_hour":               "1 if rush hour (7–9am or 5–7pm)",
    "is_night":                   "1 if between 10pm and 6am",
    "season":                     "0=Winter 1=Spring 2=Summer 3=Autumn",
    # Weather
    "temp_c":                     "Air temperature in Celsius",
    "humidity_pct":               "Relative humidity %",
    "windspeed_kmh":              "Wind speed km/h",
    "winddir_deg":                "Wind direction degrees",
    "windgusts_kmh":              "Wind gusts km/h",
    "pressure_hpa":               "Atmospheric pressure hPa",
    "uv_index":                   "UV Index",
    "dew_point_c":                "Dew point Celsius",
    "cloudcover_pct":             "Cloud cover %",
    "visibility_m":               "Visibility metres",
    "precipitation_mm":           "Total precipitation mm",
    "rain_mm":                    "Rain mm",
    "heat_index_c":               "Heat index Celsius (>27C only)",
    "wind_chill_c":               "Wind chill Celsius (<10C only)",
    "apparent_temp_delta":        "Apparent temp minus actual temp",
    "weathercode":                "WMO weather interpretation code",
    # Pollutants
    "aqi":                        "Overall AQI from AQICN",
    "dominant_poll":              "Dominant pollutant identifier",
    "pm25_iaqi":                  "PM2.5 sub-index",
    "pm10_iaqi":                  "PM10 sub-index",
    "no2_iaqi":                   "NO2 sub-index",
    "o3_iaqi":                    "O3 sub-index",
    "co_iaqi":                    "CO sub-index",
    "so2_iaqi":                   "SO2 sub-index",
    "aqi_cat_ordinal":            "AQI category 0–5",
    "aqi_cat_label":              "AQI category text label",
    # Derived
    "pollution_accumulation_idx": "PM2.5 × humidity / wind speed proxy",
    "ventilation_idx":            "Wind / AQI × 100 (dispersion proxy)",
    "humidity_pm25_interaction":  "Humidity × PM2.5 / 100",
    "uv_o3_interaction":          "UV Index × O3 (photochemical proxy)",
    "temp_range_c":               "Daily temp range from hourly data",
    "temp_variance":              "Hourly temperature variance",
    "wind_variance":              "Hourly wind speed variance",
    "aqi_lag_1h":                 "AQI 1 hour ago",
    "aqi_lag_3h":                 "AQI 3 hours ago",
    "aqi_lag_6h":                 "AQI 6 hours ago",
    "aqi_lag_24h":                "AQI 24 hours ago (same hour yesterday)",
    "aqi_rolling_3h_mean":        "Rolling 3h mean AQI",
    "aqi_rolling_3h_std":         "Rolling 3h std AQI",
    "aqi_rolling_6h_mean":        "Rolling 6h mean AQI",
    "aqi_rolling_24h_mean":       "Rolling 24h mean AQI",
    "aqi_rolling_24h_max":        "Rolling 24h max AQI",
    "aqi_change_rate_1h":         "AQI delta vs 1h ago",
    "aqi_change_rate_6h":         "AQI delta per hour over last 6h",
    "aqi_change_rate_24h":        "AQI delta per hour over last 24h",
    # Targets
    "target_aqi_current":         "TARGET: current AQI (regression)",
    "target_aqi_cat_current":     "TARGET: current AQI category (classification)",
    "target_pm25_day1_avg":       "TARGET: forecast PM2.5 avg tomorrow",
    "target_pm25_day2_avg":       "TARGET: forecast PM2.5 avg day+2",
    "target_aqi_cat_day1":        "TARGET: AQI category tomorrow",
    "target_aqi_cat_day2":        "TARGET: AQI category day+2",
    "target_trend_direction":     "TARGET: -1 improving / 0 stable / +1 worsening",
}

# ── Hopsworks Upload ───────────────────────────────────────────────────────

def upload_to_hopsworks(row: dict) -> bool:
    """
    Upload one feature row to Hopsworks Feature Store.
    Returns True on success, False on failure (falls back to CSV backup).
    """
    if not HOPSWORKS_API_KEY:
        log.warning("HOPSWORKS_API_KEY not set — skipping Hopsworks upload.")
        return False

    try:
        import hopsworks                          # pip install hopsworks
        import hsfs.feature as hsf

        log.info(f"Connecting to Hopsworks project: {HOPSWORKS_PROJECT}")
        project = hopsworks.login(
            host="app.hopsworks.ai",
            api_key_value=HOPSWORKS_API_KEY,
            project=HOPSWORKS_PROJECT,
        )
        fs = project.get_feature_store()

        # Build DataFrame
        df = build_dataframe(row)

        # Get or create Feature Group
        try:
            fg = fs.get_feature_group(
                name=FEATURE_GROUP_NAME,
                version=FEATURE_GROUP_VERSION,
            )
            log.info(f"Found existing feature group: {FEATURE_GROUP_NAME} v{FEATURE_GROUP_VERSION}")
        except Exception:
            log.info(f"Creating new feature group: {FEATURE_GROUP_NAME}")
            fg = fs.create_feature_group(
                name=FEATURE_GROUP_NAME,
                version=FEATURE_GROUP_VERSION,
                description="Weather, AQI, and pollutant features for air quality forecasting",
                primary_key=["timestamp", "city"],
                event_time="timestamp",
                online_enabled=True,          # real-time serving
                features=[
                    hsf.Feature(
                        name=col,
                        type=dtype_to_hopsworks(df[col].dtype),
                        description=FEATURE_DESCRIPTIONS.get(col, ""),
                    )
                    for col in df.columns
                ],
            )
            fg.save(df)
            log.info("Feature group created & data saved.")
            return True

        # Upsert (insert_or_overwrite for existing group)
        fg.insert(df, write_options={"wait_for_job": True})
        log.info(f"Inserted {len(df)} rows into Hopsworks Feature Store ✅")

        # Create / update Feature View (for model training)
        try:
            fv = fs.get_feature_view(name="aqi_training_view", version=1)
        except Exception:
            query = fg.select_all()
            fv = fs.create_feature_view(
                name="aqi_training_view",
                version=1,
                description="Training view: all weather + AQI features",
                query=query,
                labels=["target_aqi_current", "target_aqi_cat_current",
                        "target_trend_direction"],
            )
            log.info("Feature View 'aqi_training_view' created.")

        return True

    except ImportError:
        log.error("hopsworks package not installed. Run: pip install hopsworks")
        return False
    except Exception as e:
        log.error(f"Hopsworks upload failed: {e}")
        return False


def dtype_to_hopsworks(dtype) -> str:
    """Map pandas dtype to Hopsworks type string."""
    s = str(dtype)
    if "int"   in s: return "bigint"
    if "float" in s: return "double"
    if "bool"  in s: return "boolean"
    return "string"


def build_dataframe(row: dict) -> pd.DataFrame:
    """Clean row and return as typed DataFrame for Hopsworks."""
    df = pd.DataFrame([row])

    # Ensure timestamp column is proper datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Convert numeric columns
    numeric_cols = [c for c in df.columns if c not in ("timestamp", "city",
                    "dominant_poll", "aqi_cat_label", "weathercode")]
    for col in numeric_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    return df

# ── CSV Backup (always written, regardless of Hopsworks) ──────────────────

def save_backup_csv(row: dict) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    path     = BACKUP_DIR / f"feature_store_backup_{date_str}.csv"
    df       = pd.DataFrame([row])

    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)

    log.info(f"Backup CSV → {path}")
    return path


# ── Feature Store Status Checker ──────────────────────────────────────────

def check_feature_store_status() -> dict:
    """Return metadata about current feature store state."""
    backup_files = sorted(BACKUP_DIR.glob("*.csv"))
    total_rows   = 0
    for f in backup_files:
        try:
            total_rows += len(pd.read_csv(f))
        except Exception:
            pass

    return {
        "backend":       "Hopsworks" if HOPSWORKS_API_KEY else "CSV backup only",
        "backup_files":  len(backup_files),
        "total_rows":    total_rows,
        "feature_count": len(FEATURE_DESCRIPTIONS),
        "last_backup":   str(backup_files[-1]) if backup_files else "none",
    }


# ── Entry Point ────────────────────────────────────────────────────────────

def store_features(row: dict) -> dict:
    """
    Main entry: try Hopsworks, always write CSV backup.
    Returns status dict.
    """
    log.info("Stage 3: Storing features...")

    # Always save CSV backup
    backup_path = save_backup_csv(row)

    # Try Hopsworks
    hw_success = upload_to_hopsworks(row)

    status = {
        "hopsworks_upload": hw_success,
        "csv_backup":       str(backup_path),
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "feature_count":    len(row),
    }

    if hw_success:
        log.info("✅ Stage 3 complete: stored in Hopsworks + CSV backup")
    else:
        log.info("✅ Stage 3 complete: stored in CSV backup (Hopsworks skipped)")

    return status


if __name__ == "__main__":
    # Standalone test with dummy data
    feat_files = sorted(Path("data/features").glob("features_*.csv"))
    if not feat_files:
        print("No feature CSV found. Run stage2_compute_features.py first.")
        exit(1)

    df  = pd.read_csv(feat_files[-1])
    row = df.iloc[-1].to_dict()

    status = store_features(row)
    print(f"\n✅ Stage 3 complete.")
    print(f"   Hopsworks: {'✅' if status['hopsworks_upload'] else '⚠️  skipped (set HOPSWORKS_API_KEY)'}")
    print(f"   CSV backup: {status['csv_backup']}")

    store_info = check_feature_store_status()
    print(f"\n📦 Feature Store Status:")
    for k, v in store_info.items():
        print(f"   {k:20s} = {v}")
