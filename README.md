# 🧠 AQI Feature Pipeline

A 3-stage ML feature pipeline: **Fetch → Engineer → Store**, running hourly via GitHub Actions CI/CD.

---

## Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌──────────────────────────┐
│  STAGE 1            │    │  STAGE 2             │    │  STAGE 3                 │
│  Raw Data Fetch     │───▶│  Feature Engineering │───▶│  Feature Store Upload    │
│                     │    │                      │    │                          │
│  • AQICN API        │    │  • Time features     │    │  • Hopsworks (free tier) │
│  • Open-Meteo API   │    │  • Weather features  │    │  • CSV backup (always)   │
│  • Saves raw JSON   │    │  • Pollutant features│    │  • SQLite for lag lookups│
│                     │    │  • Derived features  │    │                          │
│  data/raw/*.json    │    │  • Lag features      │    │  data/features_store_    │
│                     │    │  • Target labels     │    │  backup/*.csv            │
└─────────────────────┘    │                      │    └──────────────────────────┘
                           │  data/features/*.csv │
                           └─────────────────────┘
```

---

## Files

```
├── stage1_fetch_raw.py        # Fetch from AQICN + Open-Meteo
├── stage2_compute_features.py # Engineer all features + targets
├── stage3_store_features.py   # Upload to Hopsworks Feature Store
├── run_pipeline.py            # Orchestrator (runs all 3 stages)
├── requirements.txt
└── .github/workflows/
    └── pipeline.yml           # CI/CD: runs every hour
```

---

## Features Engineered (50+)

### A. Time-Based
| Feature | Description |
|---------|-------------|
| `hour`, `day_of_week`, `month`, `day_of_year` | Raw time values |
| `hour_sin`, `hour_cos` | Cyclic encoding (preserves 23:00 ≈ 00:00) |
| `dow_sin`, `dow_cos`, `month_sin`, `month_cos` | Cyclic encodings |
| `is_weekend`, `is_peak_hour`, `is_night` | Binary flags |
| `season` | 0=Winter, 1=Spring, 2=Summer, 3=Autumn |

### B. Weather
`temp_c`, `humidity_pct`, `windspeed_kmh`, `pressure_hpa`, `uv_index`, `dew_point_c`, `cloudcover_pct`, `visibility_m`, `precipitation_mm`, `heat_index_c`, `wind_chill_c`, `apparent_temp_delta`

### C. Pollutants
`aqi`, `pm25_iaqi`, `pm10_iaqi`, `no2_iaqi`, `o3_iaqi`, `co_iaqi`, `so2_iaqi`, `aqi_cat_ordinal`, `aqi_cat_label`

### D. Derived / Lag
| Feature | Formula |
|---------|---------|
| `aqi_change_rate_1h` | AQI[now] − AQI[t-1h] |
| `aqi_change_rate_6h` | (AQI[now] − AQI[t-6h]) / 6 |
| `aqi_rolling_3h_mean` | Rolling 3-hour mean AQI |
| `aqi_rolling_24h_max` | Rolling 24-hour peak AQI |
| `pollution_accumulation_idx` | PM2.5 × humidity / wind |
| `ventilation_idx` | Wind / AQI × 100 |
| `humidity_pm25_interaction` | humidity × PM2.5 / 100 |
| `uv_o3_interaction` | UV × O3 (photochemical smog proxy) |

### E. Targets (Labels)
| Target | Type | Use Case |
|--------|------|----------|
| `target_aqi_current` | float | Regression |
| `target_aqi_cat_current` | 0–5 | Classification |
| `target_trend_direction` | -1/0/+1 | Trend prediction |
| `target_aqi_cat_day1` | 0–5 | Next-day forecast |
| `target_pm25_day1_avg` | float | Next-day PM2.5 |

---

## Setup

### 1. Get Free API Keys

**AQICN** (AQI + Pollutants):
- Sign up: https://aqicn.org/data-platform/token/
- Free token, no credit card

**Hopsworks** (Feature Store):
- Sign up: https://app.hopsworks.ai (free tier: 1 project, unlimited rows)
- Create project → Account → API Keys

### 2. Add GitHub Secrets & Variables

Go to: `Settings → Secrets and variables → Actions`

**Secrets** (sensitive):
```
AQICN_TOKEN        = your_aqicn_token
HOPSWORKS_API_KEY  = your_hopsworks_key
```

**Variables** (non-sensitive):
```
CITY               = Lahore
LATITUDE           = 31.5497
LONGITUDE          = 74.3436
HOPSWORKS_PROJECT  = AQI_Forecaster
```

### 3. Run Locally

```bash
pip install -r requirements.txt

export AQICN_TOKEN="your_token"
export HOPSWORKS_API_KEY="your_key"    # optional

python run_pipeline.py
```

### 4. Push to GitHub → CI/CD starts automatically

```bash
git add .
git commit -m "init feature pipeline"
git push origin main
```

The pipeline runs **every hour** and commits updated CSVs back to the repo.

---

## Hopsworks Free Tier Limits

| Resource | Free Limit |
|----------|-----------|
| Projects | 1 |
| Feature Groups | Unlimited |
| Rows | Unlimited |
| Online Feature Store | ✅ Included |
| Feature Views | ✅ Included |
| Training Datasets | ✅ Included |

Sign up at: https://app.hopsworks.ai

---

## Next Steps

- Add a **Training Pipeline** that reads from Hopsworks Feature View
- Train an XGBoost or LSTM model on the stored features
- Add a **Inference Pipeline** for hourly AQI forecasts
- Build a Streamlit dashboard to visualize trends
