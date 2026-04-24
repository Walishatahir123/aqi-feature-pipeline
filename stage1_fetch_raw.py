"""
STAGE 1 — Raw Data Fetcher
APIs:
  - AQICN  (https://aqicn.org/api/)       → AQI + Pollutants  [free, token needed]
  - Open-Meteo (https://open-meteo.com/)  → Weather           [free, no key]
Outputs raw JSON snapshots to  data/raw/  folder.
"""

import os
import json
import time
import requests
import logging
from datetime import datetime, timezone
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
CITY         = os.getenv("CITY", "Lahore")
LATITUDE     = float(os.getenv("LATITUDE",  "31.5497"))
LONGITUDE    = float(os.getenv("LONGITUDE", "74.3436"))

# Free token from https://aqicn.org/data-platform/token/
AQICN_TOKEN  = os.getenv("AQICN_TOKEN", "demo")   # replace with yours

RAW_DIR      = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── AQICN Fetcher ──────────────────────────────────────────────────────────

def fetch_aqicn() -> dict:
    """
    AQICN Real-time World Air Quality API.
    Docs: https://aqicn.org/json-api/doc/
    Free token: https://aqicn.org/data-platform/token/
    Returns: AQI + individual pollutant IAQIs + met data
    """
    url = f"https://api.waqi.info/feed/geo:{LATITUDE};{LONGITUDE}/"
    params = {"token": AQICN_TOKEN}

    log.info("Fetching AQICN data...")
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    body = r.json()

    if body.get("status") != "ok":
        raise RuntimeError(f"AQICN error: {body.get('data', body)}")

    d = body["data"]
    iaqi = d.get("iaqi", {})

    result = {
        "source":        "aqicn",
        "station":       d.get("city", {}).get("name", CITY),
        "aqi":           d.get("aqi"),                          # overall AQI
        "dominant_pollutant": d.get("dominentpol"),
        # Pollutant IAQIs (sub-indices)
        "pm25_iaqi":     iaqi.get("pm25",  {}).get("v"),
        "pm10_iaqi":     iaqi.get("pm10",  {}).get("v"),
        "no2_iaqi":      iaqi.get("no2",   {}).get("v"),
        "o3_iaqi":       iaqi.get("o3",    {}).get("v"),
        "co_iaqi":       iaqi.get("co",    {}).get("v"),
        "so2_iaqi":      iaqi.get("so2",   {}).get("v"),
        # Met from AQICN station
        "temperature_c": iaqi.get("t",     {}).get("v"),
        "humidity_pct":  iaqi.get("h",     {}).get("v"),
        "pressure_hpa":  iaqi.get("p",     {}).get("v"),
        "wind_speed":    iaqi.get("w",     {}).get("v"),
        # Forecast (next 3 days) — useful for target engineering
        "forecast_pm25": d.get("forecast", {}).get("daily", {}).get("pm25", []),
        "forecast_o3":   d.get("forecast", {}).get("daily", {}).get("o3",   []),
        "forecast_uvi":  d.get("forecast", {}).get("daily", {}).get("uvi",  []),
    }
    log.info(f"  AQICN → AQI={result['aqi']}, PM2.5={result['pm25_iaqi']}")
    return result


# ── Open-Meteo Fetcher ─────────────────────────────────────────────────────

def fetch_openmeteo() -> dict:
    """
    Open-Meteo API — no API key required.
    Docs: https://open-meteo.com/en/docs
    Returns: Current + hourly weather variables
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":        LATITUDE,
        "longitude":       LONGITUDE,
        "current_weather": True,
        "hourly": ",".join([
            "temperature_2m", "relativehumidity_2m", "apparent_temperature",
            "precipitation", "rain", "windspeed_10m", "winddirection_10m",
            "windgusts_10m", "uv_index", "visibility", "surface_pressure",
            "cloudcover", "dewpoint_2m", "et0_fao_evapotranspiration",
        ]),
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "windspeed_10m_max", "uv_index_max",
            "sunrise", "sunset",
        ]),
        "timezone":      "Asia/Karachi",
        "forecast_days": 3,
    }

    log.info("Fetching Open-Meteo data...")
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    # Current hour index
    now_hour = datetime.now().hour
    hourly   = data["hourly"]
    idx      = min(now_hour, len(hourly["temperature_2m"]) - 1)

    result = {
        "source":             "open-meteo",
        # Current conditions
        "current_temp_c":     data["current_weather"]["temperature"],
        "current_wind_kmh":   data["current_weather"]["windspeed"],
        "current_wind_dir":   data["current_weather"]["winddirection"],
        "weathercode":        data["current_weather"]["weathercode"],
        # Hourly snapshot at current hour
        "hourly_temp_c":      hourly["temperature_2m"][idx],
        "hourly_humidity":    hourly["relativehumidity_2m"][idx],
        "apparent_temp_c":    hourly["apparent_temperature"][idx],
        "precipitation_mm":   hourly["precipitation"][idx],
        "rain_mm":            hourly["rain"][idx],
        "windspeed_kmh":      hourly["windspeed_10m"][idx],
        "winddir_deg":        hourly["winddirection_10m"][idx],
        "windgusts_kmh":      hourly["windgusts_10m"][idx],
        "uv_index":           hourly["uv_index"][idx],
        "visibility_m":       hourly["visibility"][idx],
        "pressure_hpa":       hourly["surface_pressure"][idx],
        "cloudcover_pct":     hourly["cloudcover"][idx],
        "dewpoint_c":         hourly["dewpoint_2m"][idx],
        # Full hourly arrays (for lag feature computation later)
        "hourly_temps":       hourly["temperature_2m"][:24],
        "hourly_winds":       hourly["windspeed_10m"][:24],
        "hourly_humidity_arr":hourly["relativehumidity_2m"][:24],
        # Daily summary
        "daily":              data.get("daily", {}),
    }
    log.info(f"  Open-Meteo → Temp={result['current_temp_c']}°C, Wind={result['current_wind_kmh']} km/h")
    return result


# ── Save Raw Snapshot ──────────────────────────────────────────────────────

def save_raw(aqicn_data: dict, meteo_data: dict) -> Path:
    ts  = datetime.now(timezone.utc)
    tag = ts.strftime("%Y%m%d_%H%M%S")

    snapshot = {
        "fetched_at_utc": ts.isoformat(),
        "city":           CITY,
        "latitude":       LATITUDE,
        "longitude":      LONGITUDE,
        "aqicn":          aqicn_data,
        "openmeteo":      meteo_data,
    }

    out = RAW_DIR / f"raw_{tag}.json"
    with open(out, "w") as f:
        json.dump(snapshot, f, indent=2)

    log.info(f"Raw snapshot saved → {out}")
    return out


# ── Entry Point ────────────────────────────────────────────────────────────

def fetch_all() -> tuple[dict, dict, Path]:
    aqicn_data = fetch_aqicn()
    time.sleep(1)                    # polite pause between API calls
    meteo_data = fetch_openmeteo()
    path       = save_raw(aqicn_data, meteo_data)
    return aqicn_data, meteo_data, path


if __name__ == "__main__":
    a, m, p = fetch_all()
    print(f"\nStage 1 complete. Raw data saved to: {p}")
