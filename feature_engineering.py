"""
feature_engineering.py
=======================
Standalone feature engineering module for AQI prediction.
Import this in any notebook or script:

    from feature_engineering import load_and_engineer, FEATURE_SETS

Exports:
    load_and_engineer(path)  → returns (df_engineered, feature_sets_dict)
    LEAKY_FOR_CURRENT        → list of features to drop for current-AQI targets
    get_X_y(df, target)      → returns (X, y, feature_names)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

# Always drop — raw duplicates, zero-variance, or redundant
DROP_ALWAYS = [
    "timestamp", "city",
    "hour", "month", "day_of_year", "day_of_week",   # encoded cyclically
    "uv_index", "uv_o3_interaction",                  # all-zero in dataset
    "rain_mm", "heat_index_c", "wind_chill_c",        # derived from other cols
    "apparent_temp_delta",
    "weathercode", "dominant_poll", "so2_iaqi",       # low signal / categorical
    "temp_range_c", "temp_variance", "wind_variance", # noise
    "aqi_cat_label",                                   # string label
    "is_peak_hour", "is_night",                        # flat in this dataset
]

# Drop these ONLY when predicting current-moment AQI
# (they are mathematically derived from the target → leakage)
LEAKY_FOR_CURRENT = [
    "aqi",                         # IS the current AQI
    "aqi_cat_ordinal",             # derived from current AQI
    "pm25_iaqi", "pm10_iaqi",      # AQI formula inputs
    "no2_iaqi", "o3_iaqi", "co_iaqi",
    "pollution_accumulation_idx",  # built from current pollutants
    "humidity_pm25_interaction",   # contains current pm25
    "log_pm25_iaqi", "log_pm10_iaqi",
]

ALL_TARGETS = [
    "target_aqi_current", "target_aqi_cat_current",
    "target_pm25_day1_avg", "target_pm25_day2_avg",
    "target_aqi_cat_day1", "target_aqi_cat_day2",
    "target_trend_direction",
]

CURRENT_TARGETS = ["target_aqi_current", "target_aqi_cat_current"]

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD & BASIC CLEAN
# ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"[load] {len(df):,} rows × {df.shape[1]} cols from '{path}'")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    valid_targets = [t for t in ALL_TARGETS if t in df.columns]
    df = df.dropna(subset=valid_targets)
    print(f"[clean] Dropped {before - len(df)} rows with null targets → {len(df):,} remain")

    # Forward-fill lag features (short gaps only)
    lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
    df[lag_cols] = df[lag_cols].ffill(limit=3)

    # Clip extreme outliers
    for t in ["target_aqi_current", "target_pm25_day1_avg", "target_pm25_day2_avg"]:
        if t in df.columns:
            df[t] = df[t].clip(0, 500)

    return df

# ─────────────────────────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[feat_eng] Creating new features...")

    # ── Pressure interactions (pressure r=0.43, strongest weather predictor)
    if {"pressure_hpa", "temp_c"}.issubset(df.columns):
        df["pressure_x_temp"]     = df["pressure_hpa"] * df["temp_c"]
        df["pressure_x_humidity"] = df["pressure_hpa"] * df["humidity_pct"]
        print("  + pressure_x_temp, pressure_x_humidity")

    # ── Wind dispersion index (ventilation capacity)
    if {"windspeed_kmh", "pressure_hpa"}.issubset(df.columns):
        denom = (df["pressure_hpa"] - 990).clip(lower=0.5)
        df["wind_dispersion"] = (df["windspeed_kmh"] / denom).clip(-100, 100)
        print("  + wind_dispersion")

    # ── Season encoding + interaction with pressure
    if "season" in df.columns:
        season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3, "Fall": 3}
        df["season_num"] = df["season"].map(season_map).fillna(0).astype(int)
        if "pressure_hpa" in df.columns:
            df["season_x_pressure"] = df["season_num"] * df["pressure_hpa"]
        print("  + season_num, season_x_pressure")

    # ── Humidity × cloud cover (stagnation indicator)
    if {"humidity_pct", "cloudcover_pct"}.issubset(df.columns):
        df["humidity_x_cloud"] = df["humidity_pct"] * df["cloudcover_pct"] / 100
        print("  + humidity_x_cloud")

    # ── Dew point depression (lower = more humid/stagnant)
    if {"temp_c", "dew_point_c"}.issubset(df.columns):
        df["dew_depression"] = df["temp_c"] - df["dew_point_c"]
        print("  + dew_depression")

    # ── AQI momentum (rate × lag)
    if {"aqi_change_rate_1h", "aqi_lag_1h"}.issubset(df.columns):
        df["aqi_momentum"] = df["aqi_change_rate_1h"] * df["aqi_lag_1h"].replace(0, 1)
        print("  + aqi_momentum")

    # ── Thermal inversion proxy (high pressure + low wind = trapped pollution)
    if {"pressure_hpa", "windspeed_kmh"}.issubset(df.columns):
        df["inversion_risk"] = df["pressure_hpa"] / (df["windspeed_kmh"].clip(lower=0.1))
        df["inversion_risk"] = df["inversion_risk"].clip(upper=df["inversion_risk"].quantile(0.99))
        print("  + inversion_risk")

    # ── Pollution persistence (how long AQI has stayed elevated)
    if {"aqi_lag_1h", "aqi_lag_6h"}.issubset(df.columns):
        df["pollution_persistence"] = (df["aqi_lag_1h"] + df["aqi_lag_6h"]) / 2
        print("  + pollution_persistence")

    # ── Log transforms for skewed lag/pollutant features
    log_targets = ["aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_24h",
                   "pm25_iaqi", "pm10_iaqi"]
    for col in log_targets:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    print(f"  + log_* transforms for {[c for c in log_targets if c in df.columns]}")

    # ── Rolling AQI trend slope (3h window)
    if "aqi_rolling_3h_mean" in df.columns and "aqi_rolling_6h_mean" in df.columns:
        df["aqi_trend_slope"] = df["aqi_rolling_3h_mean"] - df["aqi_rolling_6h_mean"]
        print("  + aqi_trend_slope")

    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — BUILD X, y  FOR A GIVEN TARGET
# ─────────────────────────────────────────────────────────────

def get_X_y(df: pd.DataFrame, target: str):
    """
    Returns (X, y, feature_names) with correct leakage handling per target.

    Parameters
    ----------
    df     : engineered dataframe from load_and_engineer()
    target : one of ALL_TARGETS

    Returns
    -------
    X              : pd.DataFrame of features
    y              : np.ndarray of target values
    feature_names  : list of column names in X
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in dataframe.")

    extra_drop = LEAKY_FOR_CURRENT if target in CURRENT_TARGETS else []
    drop_cols  = set(DROP_ALWAYS + ALL_TARGETS + extra_drop + ["season"])

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    # Encode any remaining categoricals
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.fillna(X.median(numeric_only=True))

    # Replace inf values produced by any division
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    

    y = df[target].values

    if target in CURRENT_TARGETS or "cat" in target or "trend" in target:
        le = LabelEncoder()
        y  = le.fit_transform(y.astype(str))
        print(f"[get_X_y] '{target}' → {X.shape[1]} features | "
              f"classes: {dict(enumerate(le.classes_))}")
        return X, y, list(X.columns), le
    else:
        print(f"[get_X_y] '{target}' → {X.shape[1]} features | "
              f"y range: [{y.min():.1f}, {y.max():.1f}]")
        return X, y, list(X.columns), None


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

# def load_and_engineer(path: str) -> pd.DataFrame:
#     """
#     Full pipeline: load → clean → engineer features.
#     Returns the engineered dataframe ready for get_X_y().
#     """
#     df = df.dropna()
#     df = load_data(path)
#     df = clean_data(df)
#     df = engineer_features(df)
#     print(f"\n[done] DataFrame ready: {df.shape[0]} rows × {df.shape[1]} cols\n")
#     return df
def load_and_engineer(path: str) -> pd.DataFrame:
    df = load_data(path)        # MUST exist first
    df = clean_data(df)
    df = engineer_features(df)
    print(df.head())
    df = df.dropna()

    print(f"\n[done] DataFrame ready: {df.shape[0]} rows × {df.shape[1]} cols\n")
    return df


# ─────────────────────────────────────────────────────────────
# QUICK TEST (run: python feature_engineering.py --data <path>)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    args = p.parse_args()

    df = load_and_engineer(args.data)
    print("\nSample engineered columns:")
    new_cols = ["pressure_x_temp", "wind_dispersion", "dew_depression",
                "inversion_risk", "pollution_persistence", "aqi_trend_slope"]
    print(df[[c for c in new_cols if c in df.columns]].describe().round(2))

    print("\nAll available targets:")
    for t in ALL_TARGETS:
        if t in df.columns:
            print(f"  ✔ {t}")

