"""
STAGE 2 — Feature Engineering
Transforms raw API snapshots into ML-ready features + targets.

Feature Groups:
  A. Time-based      : hour, day_of_week, month, season, is_weekend, is_peak_hour
  B. Weather         : temp, humidity, wind, UV, pressure, dew point, heat index
  C. Pollutant       : PM2.5, PM10, NO2, O3, CO, SO2 absolute values
  D. Derived         : AQI change rate, rolling averages, lag features,
                       heat index, wind chill, apparent_temp_delta
  E. Target (labels) : next-hour AQI, next-day AQI category, AQI trend direction
"""

import json
import math
import logging
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW_DIR     = Path("data/raw")
FEAT_DIR    = Path("data/features")
DB_PATH     = Path("data/features.db")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

# ── AQI Category Lookup ────────────────────────────────────────────────────

AQI_CATEGORIES = {
    (0,   50):  (0, "Good"),
    (51,  100): (1, "Moderate"),
    (101, 150): (2, "Unhealthy for Sensitive Groups"),
    (151, 200): (3, "Unhealthy"),
    (201, 300): (4, "Very Unhealthy"),
    (301, 500): (5, "Hazardous"),
}

def aqi_to_category(aqi) -> tuple[int, str]:
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return (-1, "Unknown")
    aqi = int(aqi)
    for (lo, hi), (idx, label) in AQI_CATEGORIES.items():
        if lo <= aqi <= hi:
            return (idx, label)
    return (5, "Hazardous")

# ── A. Time-based Features ─────────────────────────────────────────────────

def time_features(ts: datetime) -> dict:
    hour          = ts.hour
    day_of_week   = ts.weekday()          # 0=Monday, 6=Sunday
    month         = ts.month
    day_of_year   = ts.timetuple().tm_yday

    # Cyclic encoding (preserves circular distance, e.g. 23:00 close to 00:00)
    hour_sin      = math.sin(2 * math.pi * hour        / 24)
    hour_cos      = math.cos(2 * math.pi * hour        / 24)
    dow_sin       = math.sin(2 * math.pi * day_of_week / 7)
    dow_cos       = math.cos(2 * math.pi * day_of_week / 7)
    month_sin     = math.sin(2 * math.pi * month       / 12)
    month_cos     = math.cos(2 * math.pi * month       / 12)
    doy_sin       = math.sin(2 * math.pi * day_of_year / 365)
    doy_cos       = math.cos(2 * math.pi * day_of_year / 365)

    # Derived time flags
    is_weekend     = int(day_of_week >= 5)
    is_peak_hour   = int(hour in range(7, 10) or hour in range(17, 20))   # rush hour
    is_night       = int(hour < 6 or hour >= 22)

    # Season (Northern Hemisphere)
    season_map = {12: 0, 1: 0, 2: 0,   # Winter
                  3:  1, 4: 1, 5: 1,   # Spring
                  6:  2, 7: 2, 8: 2,   # Summer
                  9:  3, 10:3, 11:3}   # Autumn
    season = season_map[month]

    return {
        "hour":          hour,
        "day_of_week":   day_of_week,
        "month":         month,
        "day_of_year":   day_of_year,
        "hour_sin":      round(hour_sin,  6),
        "hour_cos":      round(hour_cos,  6),
        "dow_sin":       round(dow_sin,   6),
        "dow_cos":       round(dow_cos,   6),
        "month_sin":     round(month_sin, 6),
        "month_cos":     round(month_cos, 6),
        "doy_sin":       round(doy_sin,   6),
        "doy_cos":       round(doy_cos,   6),
        "is_weekend":    is_weekend,
        "is_peak_hour":  is_peak_hour,
        "is_night":      is_night,
        "season":        season,
    }

# ── B. Weather Features ────────────────────────────────────────────────────

def weather_features(meteo: dict) -> dict:
    temp     = meteo.get("current_temp_c")   or meteo.get("hourly_temp_c", 0)
    humidity = meteo.get("hourly_humidity", 0)
    wind     = meteo.get("current_wind_kmh") or meteo.get("windspeed_kmh", 0)
    pressure = meteo.get("pressure_hpa", 1013)
    dew      = meteo.get("dewpoint_c", 0)
    uv       = meteo.get("uv_index", 0)

    # Heat Index (Rothfusz, valid when temp > 27°C)
    heat_index = None
    if temp is not None and temp > 27 and humidity is not None:
        T, R = temp * 9/5 + 32, humidity   # convert to °F for formula
        hi = (-42.379 + 2.04901523*T + 10.14333127*R
              - 0.22475541*T*R - 0.00683783*T**2
              - 0.05481717*R**2 + 0.00122874*T**2*R
              + 0.00085282*T*R**2 - 0.00000199*T**2*R**2)
        heat_index = round((hi - 32) * 5/9, 2)   # back to °C

    # Wind Chill (valid when temp < 10°C, wind > 4.8 km/h)
    wind_chill = None
    if temp is not None and temp < 10 and wind is not None and wind > 4.8:
        wc = (13.12 + 0.6215*temp - 11.37*(wind**0.16) + 0.3965*temp*(wind**0.16))
        wind_chill = round(wc, 2)

    apparent  = meteo.get("apparent_temp_c")
    delta_app = round(apparent - temp, 2) if (apparent is not None and temp is not None) else None

    return {
        "temp_c":              temp,
        "humidity_pct":        humidity,
        "windspeed_kmh":       wind,
        "winddir_deg":         meteo.get("winddir_deg") or meteo.get("current_wind_dir"),
        "windgusts_kmh":       meteo.get("windgusts_kmh"),
        "pressure_hpa":        pressure,
        "uv_index":            uv,
        "dew_point_c":         dew,
        "cloudcover_pct":      meteo.get("cloudcover_pct"),
        "visibility_m":        meteo.get("visibility_m"),
        "precipitation_mm":    meteo.get("precipitation_mm", 0),
        "rain_mm":             meteo.get("rain_mm", 0),
        "heat_index_c":        heat_index,
        "wind_chill_c":        wind_chill,
        "apparent_temp_delta": delta_app,
        "weathercode":         meteo.get("weathercode"),
    }

# ── C. Pollutant Features ──────────────────────────────────────────────────

def pollutant_features(aqicn: dict) -> dict:
    return {
        "aqi":             aqicn.get("aqi"),
        "dominant_poll":   aqicn.get("dominant_pollutant"),
        "pm25_iaqi":       aqicn.get("pm25_iaqi"),
        "pm10_iaqi":       aqicn.get("pm10_iaqi"),
        "no2_iaqi":        aqicn.get("no2_iaqi"),
        "o3_iaqi":         aqicn.get("o3_iaqi"),
        "co_iaqi":         aqicn.get("co_iaqi"),
        "so2_iaqi":        aqicn.get("so2_iaqi"),
        # AQI category label + ordinal
        "aqi_cat_ordinal": aqi_to_category(aqicn.get("aqi"))[0],
        "aqi_cat_label":   aqi_to_category(aqicn.get("aqi"))[1],
    }

# ── D. Derived / Lag Features ──────────────────────────────────────────────

def derived_features(aqicn: dict, meteo: dict, history_df: Optional[pd.DataFrame]) -> dict:
    feats = {}

    # Pollution-weather interaction features
    temp     = meteo.get("current_temp_c",    20)
    humidity = meteo.get("hourly_humidity",   50)
    wind     = meteo.get("current_wind_kmh",  10)
    pm25     = aqicn.get("pm25_iaqi")         or 0
    aqi      = aqicn.get("aqi")              or 0

    # Higher temperature + low wind → pollution accumulation proxy
    feats["pollution_accumulation_idx"] = round(
        (pm25 * (1 + humidity/100)) / max(wind, 0.1), 4
    )

    # Ventilation index (higher wind = better dispersion)
    feats["ventilation_idx"] = round(wind / max(aqi, 1) * 100, 4)

    # Humidity × PM2.5 interaction (humidity worsens PM2.5 effects)
    feats["humidity_pm25_interaction"] = round(humidity * pm25 / 100, 4)

    # UV drives O3 formation
    o3  = aqicn.get("o3_iaqi") or 0
    uv  = meteo.get("uv_index", 0) or 0
    feats["uv_o3_interaction"] = round(uv * o3, 4)

    # Hourly temperature range (diurnal swing)
    hourly_temps = meteo.get("hourly_temps", [])
    if len(hourly_temps) >= 2:
        feats["temp_range_c"]    = round(max(hourly_temps) - min(hourly_temps), 2)
        feats["temp_variance"]   = round(float(np.var(hourly_temps)), 4)
    else:
        feats["temp_range_c"]    = None
        feats["temp_variance"]   = None

    # Wind variability
    hourly_winds = meteo.get("hourly_winds", [])
    if len(hourly_winds) >= 2:
        feats["wind_variance"]   = round(float(np.var(hourly_winds)), 4)
    else:
        feats["wind_variance"]   = None

    # ── Lag + Rolling features from local SQLite history ──────────────────
    if history_df is not None and not history_df.empty and "aqi" in history_df.columns:
        hdf = history_df.sort_values("timestamp").tail(48)  # last 48 rows max

        # Lag features
        aqi_vals = hdf["aqi"].dropna().values
        if len(aqi_vals) >= 1:
            feats["aqi_lag_1h"]  = float(aqi_vals[-1])
        if len(aqi_vals) >= 3:
            feats["aqi_lag_3h"]  = float(aqi_vals[-3])
        if len(aqi_vals) >= 6:
            feats["aqi_lag_6h"]  = float(aqi_vals[-6])
        if len(aqi_vals) >= 24:
            feats["aqi_lag_24h"] = float(aqi_vals[-24])

        # Rolling averages
        if len(aqi_vals) >= 3:
            feats["aqi_rolling_3h_mean"]  = round(float(np.mean(aqi_vals[-3:])),  2)
            feats["aqi_rolling_3h_std"]   = round(float(np.std(aqi_vals[-3:])),   2)
        if len(aqi_vals) >= 6:
            feats["aqi_rolling_6h_mean"]  = round(float(np.mean(aqi_vals[-6:])),  2)
        if len(aqi_vals) >= 24:
            feats["aqi_rolling_24h_mean"] = round(float(np.mean(aqi_vals[-24:])), 2)
            feats["aqi_rolling_24h_max"]  = round(float(np.max(aqi_vals[-24:])),  2)

        # AQI Change Rate (delta per hour)
        if len(aqi_vals) >= 2:
            feats["aqi_change_rate_1h"]  = round(float(aqi_vals[-1] - aqi_vals[-2]), 4)
        if len(aqi_vals) >= 6:
            feats["aqi_change_rate_6h"]  = round(float(aqi_vals[-1] - aqi_vals[-6]) / 6, 4)
        if len(aqi_vals) >= 24:
            feats["aqi_change_rate_24h"] = round(float(aqi_vals[-1] - aqi_vals[-24]) / 24, 4)

    return feats

# ── E. Target Engineering ──────────────────────────────────────────────────

def target_features(aqicn: dict) -> dict:
    """
    Targets are derived from AQICN's built-in 3-day forecast.
    In production you'd store today's value, then use tomorrow's actual
    reading as the target (label) in a separate labelling job.
    Here we use the forecast as a proxy target.
    """
    forecast_pm25 = aqicn.get("forecast_pm25", [])
    current_aqi   = aqicn.get("aqi") or 0

    targets = {
        "target_aqi_current":          current_aqi,
        "target_aqi_cat_current":      aqi_to_category(current_aqi)[0],
    }

    # Forecast targets (day+1, day+2)
    for i, fc in enumerate(forecast_pm25[:2], start=1):
        avg = fc.get("avg")
        targets[f"target_pm25_day{i}_avg"]   = avg
        targets[f"target_aqi_cat_day{i}"]    = aqi_to_category(avg)[0] if avg else None

    # Trend direction: -1 (improving), 0 (stable), +1 (worsening)
    if len(forecast_pm25) >= 1 and forecast_pm25[0].get("avg") is not None:
        delta = forecast_pm25[0]["avg"] - (aqicn.get("pm25_iaqi") or 0)
        targets["target_trend_direction"] = 1 if delta > 5 else (-1 if delta < -5 else 0)
    else:
        targets["target_trend_direction"] = None

    return targets

# ── SQLite History (for lag computation) ──────────────────────────────────

def load_history(n: int = 48) -> Optional[pd.DataFrame]:
    if not DB_PATH.exists():
        return None
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            f"SELECT * FROM features ORDER BY timestamp DESC LIMIT {n}", con
        )
        return df
    except Exception:
        return None
    finally:
        con.close()

# ── Master Feature Builder ─────────────────────────────────────────────────

def build_features(aqicn: dict, meteo: dict, ts: datetime) -> dict:
    history = load_history()

    row = {
        "timestamp": ts.isoformat(),
        "city":      "Lahore",
    }
    row.update(time_features(ts))
    row.update(weather_features(meteo))
    row.update(pollutant_features(aqicn))
    row.update(derived_features(aqicn, meteo, history))
    row.update(target_features(aqicn))

    log.info(f"Built {len(row)} features")
    return row

# ── Save Feature Row ───────────────────────────────────────────────────────

def save_features(row: dict) -> Path:
    # CSV
    date_str = datetime.now().strftime("%Y-%m-%d")
    csv_path = FEAT_DIR / f"features_{date_str}.csv"
    df       = pd.DataFrame([row])

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    log.info(f"Features saved → {csv_path}")

#     # SQLite (for lag lookups in future runs)
#     con = sqlite3.connect(DB_PATH)
#     # df.to_sql("features", con, if_exists="append", index=False)
# try:
#     df.to_sql("features", con, if_exists="append", index=False)
# except Exception:
#     con.execute("DROP TABLE IF EXISTS features")
#     df.to_sql("features", con, if_exists="replace", index=False)
#     log.warning("SQLite schema updated — table recreated.")
#     con.close()
#     log.info(f"Features stored in SQLite → {DB_PATH}")
#     return csv_path
# SQLite (for lag lookups in future runs)
    con = sqlite3.connect(DB_PATH)
    try:
        df.to_sql("features", con, if_exists="append", index=False)
    except Exception:
        con.execute("DROP TABLE IF EXISTS features")
        df.to_sql("features", con, if_exists="replace", index=False)
        log.warning("SQLite schema updated — table recreated.")
    finally:
        con.close()
    log.info(f"Features stored in SQLite → {DB_PATH}")
    return csv_path


# ── Entry Point ────────────────────────────────────────────────────────────

def compute_features(aqicn: dict, meteo: dict) -> tuple[dict, Path]:
    ts  = datetime.now(timezone.utc)
    row = build_features(aqicn, meteo, ts)
    path = save_features(row)
    return row, path


if __name__ == "__main__":
    # Standalone test using latest raw file
    raw_files = sorted(RAW_DIR.glob("raw_*.json"))
    if not raw_files:
        print("No raw data found. Run stage1_fetch_raw.py first.")
        exit(1)

    with open(raw_files[-1]) as f:
        snap = json.load(f)

    row, path = compute_features(snap["aqicn"], snap["openmeteo"])
    print(f"\n Stage 2 complete. Features saved to: {path}")
    print(f"   Feature count : {len(row)}")
    print(f"\n   Sample features:")
    for k in ["aqi", "pm25_iaqi", "temp_c", "hour", "hour_sin",
              "is_peak_hour", "season", "aqi_change_rate_1h",
              "pollution_accumulation_idx", "target_trend_direction"]:
        print(f"   {k:35s} = {row.get(k)}")
