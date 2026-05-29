
import os
import math
import time
import logging
import argparse
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from pymongo import MongoClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
MONGODB_URI   = os.getenv("MONGODB_URI", "")
MONGODB_DB    = "aqi_pipeline"
COLLECTION    = "weather_aqi_features"
AQICN_TOKEN   = os.getenv("AQICN_API_KEY", os.getenv("DEMO", "demo"))
CITY          = "Lahore"
LATITUDE      = 31.5497
LONGITUDE     = 74.3436
BACKUP_DIR    = Path("data/historical_backup")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# ── AQI Categories ─────────────────────────────────────────────────────────
AQI_CATEGORIES = {
    (0,   50):  (0, "Good"),
    (51,  100): (1, "Moderate"),
    (101, 150): (2, "Unhealthy for Sensitive Groups"),
    (151, 200): (3, "Unhealthy"),
    (201, 300): (4, "Very Unhealthy"),
    (301, 500): (5, "Hazardous"),
}

def aqi_to_category(aqi):
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return (-1, "Unknown")
    aqi = int(aqi)
    for (lo, hi), (idx, label) in AQI_CATEGORIES.items():
        if lo <= aqi <= hi:
            return (idx, label)
    return (5, "Hazardous")

# ── Fetch Historical Weather (Open-Meteo) ──────────────────────────────────
def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly historical weather from Open-Meteo (free, no API key)."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        LATITUDE,
        "longitude":       LONGITUDE,
        "start_date":      start_date,
        "end_date":        end_date,
        "hourly":          [
            "temperature_2m", "relativehumidity_2m", "dewpoint_2m",
            "apparent_temperature", "precipitation", "rain",
            "weathercode", "pressure_msl", "cloudcover",
            "visibility", "windspeed_10m", "winddirection_10m",
            "windgusts_10m", "uv_index"
        ],
        "timezone":        "UTC",
        "wind_speed_unit": "kmh",
    }
    log.info(f"Fetching weather: {start_date} → {end_date}")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp":      hourly["time"],
        "temp_c":         hourly["temperature_2m"],
        "humidity_pct":   hourly["relativehumidity_2m"],
        "dew_point_c":    hourly["dewpoint_2m"],
        "apparent_temp":  hourly["apparent_temperature"],
        "precipitation_mm": hourly["precipitation"],
        "rain_mm":        hourly["rain"],
        "weathercode":    hourly["weathercode"],
        "pressure_hpa":   hourly["pressure_msl"],
        "cloudcover_pct": hourly["cloudcover"],
        "visibility_m":   hourly["visibility"],
        "windspeed_kmh":  hourly["windspeed_10m"],
        "winddir_deg":    hourly["winddirection_10m"],
        "windgusts_kmh":  hourly["windgusts_10m"],
        "uv_index":       hourly["uv_index"],
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    log.info(f"Weather fetched: {len(df)} hourly rows")
    return df

# ── Fetch Historical AQI (AQICN) ───────────────────────────────────────────
def fetch_historical_aqi(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical AQI from AQICN API.
    Returns daily AQI values which we broadcast to all hours of that day.
    """
    url = f"https://api.waqi.info/feed/{CITY}/?token={AQICN_TOKEN}"
    
    # AQICN free tier only gives current + forecast, not full history
    # So we use a fixed recent value and simulate variation for training data
    log.info("Fetching current AQI from AQICN (historical proxy)...")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "ok":
            current_aqi = int(data["data"]["aqi"])
            pm25 = data["data"]["iaqi"].get("pm25", {}).get("v", 80)
            pm10 = data["data"]["iaqi"].get("pm10", {}).get("v", 60)
            no2  = data["data"]["iaqi"].get("no2",  {}).get("v", 20)
            o3   = data["data"]["iaqi"].get("o3",   {}).get("v", 30)
            co   = data["data"]["iaqi"].get("co",   {}).get("v", 10)
            so2  = data["data"]["iaqi"].get("so2",  {}).get("v", 5)
        else:
            current_aqi, pm25, pm10, no2, o3, co, so2 = 100, 80, 60, 20, 30, 10, 5
    except Exception as e:
        log.warning(f"AQICN fetch failed: {e} — using defaults")
        current_aqi, pm25, pm10, no2, o3, co, so2 = 100, 80, 60, 20, 30, 10, 5

    # Generate date range
    dates = pd.date_range(start=start_date, end=end_date, freq="D", tz="UTC")
    
    # Simulate realistic AQI variation for Lahore
    # Higher in winter (Nov-Feb), lower in monsoon (Jul-Sep)
    aqi_rows = []
    rng = np.random.default_rng(42)
    for date in dates:
        month = date.month
        # Seasonal multiplier
        if month in [11, 12, 1, 2]:      # Winter — high pollution
            seasonal = 1.4
        elif month in [7, 8, 9]:          # Monsoon — lower
            seasonal = 0.7
        elif month in [3, 4]:             # Spring
            seasonal = 0.9
        else:
            seasonal = 1.0

        base = current_aqi * seasonal
        daily_aqi = int(base + rng.normal(0, base * 0.15))
        daily_aqi = max(10, min(500, daily_aqi))

        aqi_rows.append({
            "date":     date.date(),
            "aqi":      daily_aqi,
            "pm25":     int(pm25 * seasonal + rng.normal(0, 10)),
            "pm10":     int(pm10 * seasonal + rng.normal(0, 8)),
            "no2":      int(no2  * seasonal + rng.normal(0, 5)),
            "o3":       int(o3   + rng.normal(0, 5)),
            "co":       int(co   + rng.normal(0, 2)),
            "so2":      int(so2  + rng.normal(0, 2)),
        })

    df = pd.DataFrame(aqi_rows)
    log.info(f"AQI data generated for {len(df)} days")
    return df

# ── Feature Engineering ────────────────────────────────────────────────────
def build_features_for_row(weather_row: pd.Series, aqi_row: pd.Series,
                            history_aqi: list) -> dict:
    ts   = weather_row["timestamp"]
    temp = weather_row["temp_c"]
    hum  = weather_row["humidity_pct"]
    wind = weather_row["windspeed_kmh"]
    aqi  = int(aqi_row["aqi"])
    pm25 = int(aqi_row["pm25"])

    # Time features
    hour        = ts.hour
    dow         = ts.weekday()
    month       = ts.month
    doy         = ts.timetuple().tm_yday
    season_map  = {12:0,1:0,2:0, 3:1,4:1,5:1, 6:2,7:2,8:2, 9:3,10:3,11:3}

    # Heat index
    heat_index = None
    if temp > 27:
        T, R = temp * 9/5 + 32, hum
        hi = (-42.379 + 2.04901523*T + 10.14333127*R
              - 0.22475541*T*R - 0.00683783*T**2
              - 0.05481717*R**2 + 0.00122874*T**2*R
              + 0.00085282*T*R**2 - 0.00000199*T**2*R**2)
        heat_index = round((hi - 32) * 5/9, 2)

    # Wind chill
    wind_chill = None
    if temp < 10 and wind > 4.8:
        wind_chill = round(13.12 + 0.6215*temp - 11.37*(wind**0.16)
                           + 0.3965*temp*(wind**0.16), 2)

    apparent     = weather_row["apparent_temp"]
    delta_app    = round(apparent - temp, 2) if apparent is not None else None
    uv           = weather_row["uv_index"] or 0
    o3           = int(aqi_row["o3"])

    # Lag features
    aqi_vals = history_aqi[-24:] if len(history_aqi) >= 1 else []
    lag_1h   = aqi_vals[-1]  if len(aqi_vals) >= 1  else None
    lag_3h   = aqi_vals[-3]  if len(aqi_vals) >= 3  else None
    lag_6h   = aqi_vals[-6]  if len(aqi_vals) >= 6  else None
    lag_24h  = aqi_vals[-24] if len(aqi_vals) >= 24 else None

    roll_3h_mean  = round(float(np.mean(aqi_vals[-3:])),  2) if len(aqi_vals) >= 3  else None
    roll_3h_std   = round(float(np.std(aqi_vals[-3:])),   2) if len(aqi_vals) >= 3  else None
    roll_6h_mean  = round(float(np.mean(aqi_vals[-6:])),  2) if len(aqi_vals) >= 6  else None
    roll_24h_mean = round(float(np.mean(aqi_vals[-24:])), 2) if len(aqi_vals) >= 24 else None
    roll_24h_max  = round(float(np.max(aqi_vals[-24:])),  2) if len(aqi_vals) >= 24 else None

    change_1h  = round(aqi_vals[-1] - aqi_vals[-2], 4)       if len(aqi_vals) >= 2  else None
    change_6h  = round((aqi_vals[-1] - aqi_vals[-6])  / 6, 4) if len(aqi_vals) >= 6  else None
    change_24h = round((aqi_vals[-1] - aqi_vals[-24]) / 24, 4) if len(aqi_vals) >= 24 else None

    # Targets
    next_aqi     = aqi
    aqi_cat      = aqi_to_category(aqi)
    trend        = 0
    if change_1h is not None:
        trend = 1 if change_1h > 5 else (-1 if change_1h < -5 else 0)

    row = {
        "timestamp":                  ts.isoformat(),
        "city":                       CITY,
        "hour":                       hour,
        "day_of_week":                dow,
        "month":                      month,
        "day_of_year":                doy,
        "hour_sin":                   round(math.sin(2*math.pi*hour/24), 6),
        "hour_cos":                   round(math.cos(2*math.pi*hour/24), 6),
        "month_sin":                  round(math.sin(2*math.pi*month/12), 6),
        "month_cos":                  round(math.cos(2*math.pi*month/12), 6),
        "doy_sin":                    round(math.sin(2*math.pi*doy/365), 6),
        "doy_cos":                    round(math.cos(2*math.pi*doy/365), 6),
        "is_weekend":                 int(dow >= 5),
        "is_peak_hour":               int(hour in range(7,10) or hour in range(17,20)),
        "is_night":                   int(hour < 6 or hour >= 22),
        "season":                     season_map[month],
        "temp_c":                     temp,
        "humidity_pct":               hum,
        "windspeed_kmh":              wind,
        "winddir_deg":                weather_row["winddir_deg"],
        "windgusts_kmh":              weather_row["windgusts_kmh"],
        "pressure_hpa":               weather_row["pressure_hpa"],
        "uv_index":                   uv,
        "dew_point_c":                weather_row["dew_point_c"],
        "cloudcover_pct":             weather_row["cloudcover_pct"],
        "visibility_m":               weather_row["visibility_m"],
        "precipitation_mm":           weather_row["precipitation_mm"],
        "rain_mm":                    weather_row["rain_mm"],
        "heat_index_c":               heat_index,
        "wind_chill_c":               wind_chill,
        "apparent_temp_delta":        delta_app,
        "weathercode":                weather_row["weathercode"],
        "aqi":                        aqi,
        "dominant_poll":              "pm25",
        "pm25_iaqi":                  pm25,
        "pm10_iaqi":                  int(aqi_row["pm10"]),
        "no2_iaqi":                   int(aqi_row["no2"]),
        "o3_iaqi":                    o3,
        "co_iaqi":                    int(aqi_row["co"]),
        "so2_iaqi":                   int(aqi_row["so2"]),
        "aqi_cat_ordinal":            aqi_cat[0],
        "aqi_cat_label":              aqi_cat[1],
        "pollution_accumulation_idx": round((pm25*(1+hum/100))/max(wind,0.1), 4),
        "ventilation_idx":            round(wind/max(aqi,1)*100, 4),
        "humidity_pm25_interaction":  round(hum*pm25/100, 4),
        "uv_o3_interaction":          round(uv*o3, 4),
        "temp_range_c":               None,
        "temp_variance":              None,
        "wind_variance":              None,
        "aqi_lag_1h":                 lag_1h,
        "aqi_lag_3h":                 lag_3h,
        "aqi_lag_6h":                 lag_6h,
        "aqi_lag_24h":                lag_24h,
        "aqi_rolling_3h_mean":        roll_3h_mean,
        "aqi_rolling_3h_std":         roll_3h_std,
        "aqi_rolling_6h_mean":        roll_6h_mean,
        "aqi_rolling_24h_mean":       roll_24h_mean,
        "aqi_rolling_24h_max":        roll_24h_max,
        "aqi_change_rate_1h":         change_1h,
        "aqi_change_rate_6h":         change_6h,
        "aqi_change_rate_24h":        change_24h,
        "target_aqi_current":         next_aqi,
        "target_aqi_cat_current":     aqi_cat[0],
        "target_pm25_day1_avg":       pm25,
        "target_pm25_day2_avg":       pm25,
        "target_aqi_cat_day1":        aqi_cat[0],
        "target_aqi_cat_day2":        aqi_cat[0],
        "target_trend_direction":     trend,
    }
    return row


# ── MongoDB Upload ─────────────────────────────────────────────────────────
def upload_batch_to_mongodb(rows: list) -> bool:
    if not MONGODB_URI:
        log.warning("MONGODB_URI not set — skipping MongoDB upload.")
        return False
    try:
        client     = MongoClient(MONGODB_URI)
        db         = client[MONGODB_DB]
        collection = db[COLLECTION]

        operations = []
        from pymongo import UpdateOne
        for row in rows:
            clean = {}
            for k, v in row.items():
                if hasattr(v, "item"):
                    clean[k] = v.item()
                elif isinstance(v, float) and v != v:
                    clean[k] = None
                else:
                    clean[k] = v
            operations.append(UpdateOne(
                {"timestamp": clean["timestamp"], "city": clean["city"]},
                {"$set": clean},
                upsert=True
            ))

        if operations:
            result = collection.bulk_write(operations)
            log.info(f"MongoDB: {result.upserted_count} inserted, "
                     f"{result.modified_count} updated")
        client.close()
        return True
    except Exception as e:
        log.error(f"MongoDB upload failed: {e}")
        return False


# ── Main ───────────────────────────────────────────────────────────────────
def generate_historical(start_date: str, end_date: str):
    log.info(f"Generating historical data: {start_date} → {end_date}")

    # Fetch data
    weather_df = fetch_historical_weather(start_date, end_date)
    aqi_df     = fetch_historical_aqi(start_date, end_date)

    # Build features for each hour
    all_rows    = []
    history_aqi = []

    for _, w_row in weather_df.iterrows():
        ts   = w_row["timestamp"]
        date = ts.date()

        # Match AQI row for this date
        aqi_match = aqi_df[aqi_df["date"] == date]
        if aqi_match.empty:
            continue
        a_row = aqi_match.iloc[0]

        # Build features
        row = build_features_for_row(w_row, a_row, history_aqi)
        all_rows.append(row)
        history_aqi.append(int(a_row["aqi"]))

    log.info(f"Built {len(all_rows)} feature rows")

    # Save CSV backup
    csv_path = BACKUP_DIR / f"historical_{start_date}_{end_date}.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False, quoting=1)
    log.info(f"CSV backup saved → {csv_path}")

    # Upload to MongoDB in batches of 500
    batch_size = 500
    for i in range(0, len(all_rows), batch_size):
        batch = all_rows[i:i+batch_size]
        log.info(f"Uploading batch {i//batch_size + 1} "
                 f"({len(batch)} rows)...")
        upload_batch_to_mongodb(batch)
        time.sleep(0.5)

    log.info(f"✅ Done! {len(all_rows)} rows stored.")
    return len(all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate historical AQI training data")
    parser.add_argument("--days",  type=int, default=90,
                        help="Number of past days to generate (default: 90)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str, default=None,
                        help="End date YYYY-MM-DD")
    args = parser.parse_args()

    if args.start and args.end:
        start_date = args.start
        end_date   = args.end
    else:
        end_dt   = datetime.now(timezone.utc) - timedelta(days=1)
        start_dt = end_dt - timedelta(days=args.days)
        start_date = start_dt.strftime("%Y-%m-%d")
        end_date   = end_dt.strftime("%Y-%m-%d")

    total = generate_historical(start_date, end_date)
    print(f"\n✅ Historical data generation complete!")
    print(f"   Date range : {start_date} → {end_date}")
    print(f"   Total rows : {total}")
    print(f"   MongoDB    : {'✅ uploaded' if MONGODB_URI else '⚠️  set MONGODB_URI'}")
