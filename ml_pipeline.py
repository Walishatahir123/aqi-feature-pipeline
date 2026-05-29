"""
Rolling-Window Multi-Model AQI Forecasting Pipeline
=====================================================
Models   : Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
Target   : target_aqi_current
Window   : Train=87 days | Test=3 days | Step=7 days
"""

import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Sklearn
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib

# ── Directories ─────────────────────────────────────────────────────────────
MODEL_DIR  = Path("models")
REPORT_DIR = Path("data/ml_reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────────────
DROP_COLS = [
    "timestamp", "city", "dominant_poll", "aqi_cat_label",
    "temp_range_c", "temp_variance", "wind_variance",
    "wind_chill_c", "visibility_m", "heat_index_c",
    "target_aqi_cat_current", "target_pm25_day1_avg",
    "target_pm25_day2_avg", "target_aqi_cat_day1",
    "target_aqi_cat_day2", "target_trend_direction",
    "aqi_cat_ordinal",
]
TARGET      = "target_aqi_current"
TRAIN_DAYS  = 87    # 2088 hours
TEST_DAYS   = 3     #   72 hours
STEP_DAYS   = 7     # slide window by 7 days each iteration
HOURS       = 24    # hourly data

# ── Model registry ───────────────────────────────────────────────────────────
def build_models() -> dict:
    """
    Each model is wrapped in a StandardScaler → Estimator Pipeline.
    Ridge & Lasso benefit most from scaling; tree models are robust to it.
    """
    return {
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=1.0)),
        ]),
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Lasso(alpha=0.1, max_iter=10_000)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=200, max_depth=None,
                min_samples_leaf=2, n_jobs=-1, random_state=42,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, subsample=0.8, random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  XGBRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=4, subsample=0.8, colsample_bytree=0.8,
                n_jobs=-1, random_state=42, verbosity=0,
            )),
        ]),
    }

# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load CSV, parse timestamps, sort, drop unused columns."""
    df = pd.read_csv(csv_path)
    log.info(f"Loaded  {csv_path}  →  {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Parse timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Drop configured columns that exist in the dataframe
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Drop rows where TARGET is missing
    df = df.dropna(subset=[TARGET])

    # Fill remaining NaNs with column median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    log.info(f"After prep  →  {df.shape[0]:,} rows × {df.shape[1]} features  (target={TARGET})")
    return df


# ── Rolling-window training loop ─────────────────────────────────────────────
def rolling_window_evaluation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Slide a (TRAIN_DAYS → TEST_DAYS) window over the dataset, stepping STEP_DAYS
    each iteration. Returns per-window metrics and aggregated predictions.
    """
    TRAIN_H = TRAIN_DAYS * HOURS   # 2088
    TEST_H  = TEST_DAYS  * HOURS   #   72
    STEP_H  = STEP_DAYS  * HOURS   #  168

    feature_cols = [c for c in df.columns if c != TARGET]
    X_all = df[feature_cols].values
    y_all = df[TARGET].values
    n     = len(df)

    model_names = list(build_models().keys())
    all_metrics  = {m: [] for m in model_names}   # list of per-window metric dicts
    all_preds    = {m: [] for m in model_names}   # (y_true, y_pred) pairs
    window_ids   = []

    start = 0
    window_num = 0

    while start + TRAIN_H + TEST_H <= n:
        train_end = start + TRAIN_H
        test_end  = train_end + TEST_H

        X_train, y_train = X_all[start:train_end],  y_all[start:train_end]
        X_test,  y_test  = X_all[train_end:test_end], y_all[train_end:test_end]

        window_tag = f"W{window_num:02d}_rows{start}-{test_end}"
        window_ids.append(window_tag)
        log.info(f"  Window {window_num:02d}: train [{start:5d}:{train_end:5d}]  "
                 f"test [{train_end:5d}:{test_end:5d}]")

        models = build_models()
        for name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = compute_metrics(y_test, y_pred)
            metrics["window"] = window_tag
            all_metrics[name].append(metrics)
            all_preds[name].append((y_test.copy(), y_pred.copy()))

            # Save best pipeline per model on final window
            joblib.dump(pipeline, MODEL_DIR / f"{name}_latest.pkl")

        start     += STEP_H
        window_num += 1

    if window_num == 0:
        raise ValueError(
            f"Dataset too small for even one window. "
            f"Need at least {TRAIN_H + TEST_H:,} rows, got {n:,}."
        )

    log.info(f"Completed {window_num} rolling windows.")

    # ── Aggregate metrics across windows ──────────────────────────────────
    summary_rows = []
    for name in model_names:
        df_m = pd.DataFrame(all_metrics[name])
        row  = {"Model": name}
        for col in ["RMSE", "MAE", "R2", "MAPE"]:
            row[f"{col}_mean"] = df_m[col].mean()
            row[f"{col}_std"]  = df_m[col].std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).set_index("Model")
    return summary_df, all_metrics, all_preds, window_ids, feature_cols


# ── Reporting ─────────────────────────────────────────────────────────────────
PALETTE = ["#4361EE", "#F72585", "#7209B7", "#3A0CA3", "#4CC9F0"]
BG      = "#0F1117"
PANEL   = "#1A1D27"
WHITE   = "#FFFFFF"


def _ax_style(ax, title=""):
    ax.set_facecolor(PANEL)
    ax.spines[:].set_visible(False)
    ax.tick_params(colors=WHITE, labelsize=8)
    plt.setp(ax.get_xticklabels(), color=WHITE)
    plt.setp(ax.get_yticklabels(), color=WHITE)
    if title:
        ax.set_title(title, color=WHITE, fontsize=10, pad=8)


def plot_results(summary_df: pd.DataFrame,
                 all_metrics: dict,
                 all_preds: dict,
                 window_ids: list,
                 feature_cols: list,
                 model_names: list,
                 save_path: Path):

    fig = plt.figure(figsize=(24, 18))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

    colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(model_names)}

    # ── 1. Mean RMSE bar chart ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    _ax_style(ax1, "Mean RMSE across Rolling Windows (lower = better)")
    vals = summary_df["RMSE_mean"]
    errs = summary_df["RMSE_std"]
    bars = ax1.bar(model_names, vals, color=[colors[m] for m in model_names],
                   edgecolor="none", width=0.55,
                   yerr=errs, capsize=5, error_kw={"ecolor": WHITE, "alpha": 0.6})
    for bar, v, e in zip(bars, vals, errs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 v + e + vals.max() * 0.02,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9, color=WHITE)
    ax1.set_ylabel("RMSE", color=WHITE, fontsize=9)

    # ── 2. Mean R² bar chart ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    _ax_style(ax2, "Mean R²")
    r2_vals = summary_df["R2_mean"]
    ax2.barh(model_names, r2_vals, color=[colors[m] for m in model_names], edgecolor="none")
    for i, v in enumerate(r2_vals):
        ax2.text(max(v, 0) + 0.005, i, f"{v:.3f}", va="center", fontsize=8, color=WHITE)
    ax2.set_xlim(min(0, r2_vals.min() - 0.05), 1.05)
    ax2.axvline(1.0, color=WHITE, lw=0.5, alpha=0.3)

    # ── 3. Metrics heatmap ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    _ax_style(ax3, "Aggregated Metrics Heatmap (mean ± std)")
    metric_labels = ["RMSE", "MAE", "R2", "MAPE"]
    heat_mean = summary_df[[f"{m}_mean" for m in metric_labels]].copy()
    heat_mean.columns = metric_labels
    # Normalise: for RMSE/MAE/MAPE lower=better → invert after norm
    normed = heat_mean.apply(lambda c: (c - c.min()) / (c.max() - c.min() + 1e-9), axis=0)
    normed[["RMSE", "MAE", "MAPE"]] = 1 - normed[["RMSE", "MAE", "MAPE"]]
    import seaborn as sns
    sns.heatmap(normed.T, annot=heat_mean.T.round(3), fmt=".3f",
                cmap="YlOrRd", ax=ax3, cbar=False,
                linewidths=0.5, linecolor=BG,
                annot_kws={"size": 9, "color": WHITE})
    ax3.tick_params(colors=WHITE, labelsize=9)
    plt.setp(ax3.get_xticklabels(), rotation=20, color=WHITE)
    plt.setp(ax3.get_yticklabels(), rotation=0, color=WHITE)

    # ── 4. RMSE per window (line chart) ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    _ax_style(ax4, "RMSE per Window")
    x_ticks = range(len(window_ids))
    for name in model_names:
        rmse_per_win = [d["RMSE"] for d in all_metrics[name]]
        ax4.plot(x_ticks, rmse_per_win, marker="o", ms=4,
                 color=colors[name], label=name, lw=1.5)
    ax4.set_xticks(x_ticks)
    ax4.set_xticklabels([f"W{i}" for i in x_ticks], fontsize=7, color=WHITE)
    ax4.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.7)
    ax4.set_ylabel("RMSE", color=WHITE, fontsize=8)

    # ── 5. Actual vs Predicted — last window, all models ─────────────────
    for idx, name in enumerate(model_names):
        ax = fig.add_subplot(gs[2, idx % 3])
        _ax_style(ax, f"{name}\nActual vs Predicted (last window)")
        y_true, y_pred = all_preds[name][-1]
        ax.plot(y_true,  color=WHITE,        lw=1.0, alpha=0.8, label="Actual")
        ax.plot(y_pred, color=colors[name], lw=1.2, alpha=0.9, label="Predicted", ls="--")
        ax.legend(fontsize=7, facecolor=PANEL, labelcolor=WHITE, framealpha=0.7)
        ax.set_xlabel("Hour", color=WHITE, fontsize=8)
        # Annotate RMSE
        rmse = all_metrics[name][-1]["RMSE"]
        r2   = all_metrics[name][-1]["R2"]
        ax.text(0.97, 0.05, f"RMSE={rmse:.2f}\nR²={r2:.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=WHITE,
                bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=colors[name], alpha=0.8))

    fig.suptitle(
        f"Rolling-Window AQI Forecast — {TRAIN_DAYS}d train / {TEST_DAYS}d test / {STEP_DAYS}d step",
        color=WHITE, fontsize=16, fontweight="bold", y=0.99,
    )
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    log.info(f"Plot saved → {save_path}")


def save_reports(summary_df: pd.DataFrame,
                 all_metrics: dict,
                 model_names: list):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Summary CSV
    summary_path = REPORT_DIR / f"model_summary_{ts}.csv"
    summary_df.round(4).to_csv(summary_path)
    log.info(f"Summary CSV  → {summary_path}")

    # Per-window detail CSV
    rows = []
    for name in model_names:
        for d in all_metrics[name]:
            rows.append({"model": name, **d})
    detail_path = REPORT_DIR / f"per_window_metrics_{ts}.csv"
    pd.DataFrame(rows).to_csv(detail_path, index=False)
    log.info(f"Detail CSV   → {detail_path}")

    return summary_path, detail_path


# ── Demo data generator (replace with your CSV path) ─────────────────────────
def generate_demo_data(n_hours: int = 2400) -> str:
    """
    Generates a synthetic hourly AQI dataset so the pipeline runs without
    a real file. Replace csv_path in main() with your actual CSV.
    """
    rng  = np.random.default_rng(42)
    path = Path("data/demo_aqi.csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    ts = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    df = pd.DataFrame({"timestamp": ts})

    # Environmental features
    df["temp_c"]          = 20 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_hours)) + rng.normal(0, 2, n_hours)
    df["humidity_pct"]    = 60 + 20 * np.cos(np.linspace(0, 4 * np.pi, n_hours)) + rng.normal(0, 5, n_hours)
    df["wind_speed_ms"]   = np.abs(5 + rng.normal(0, 2, n_hours))
    df["wind_dir_deg"]    = rng.uniform(0, 360, n_hours)
    df["pressure_hpa"]    = 1013 + rng.normal(0, 3, n_hours)
    df["dew_point_c"]     = df["temp_c"] - 10 + rng.normal(0, 1, n_hours)
    df["uv_index"]        = np.clip(5 + 3 * np.sin(np.linspace(0, 4 * np.pi, n_hours)) + rng.normal(0, 1, n_hours), 0, 12)
    df["precip_mm"]       = np.abs(rng.normal(0, 0.5, n_hours))
    df["cloud_cover_pct"] = np.clip(rng.uniform(0, 100, n_hours), 0, 100)
    df["pm25"]            = np.abs(35 + 15 * np.sin(np.linspace(0, 8 * np.pi, n_hours)) + rng.normal(0, 8, n_hours))
    df["pm10"]            = df["pm25"] * 1.5 + rng.normal(0, 5, n_hours)
    df["no2_ppb"]         = np.abs(20 + rng.normal(0, 5, n_hours))
    df["o3_ppb"]          = np.abs(30 + rng.normal(0, 8, n_hours))
    df["co_ppm"]          = np.abs(0.5 + rng.normal(0, 0.1, n_hours))
    df["so2_ppb"]         = np.abs(5 + rng.normal(0, 2, n_hours))
    df["aqi"]             = df["pm25"] * 1.2 + df["no2_ppb"] * 0.5 + rng.normal(0, 5, n_hours)
    df["hour"]            = ts.hour
    df["day_of_week"]     = ts.dayofweek
    df["month"]           = ts.month
    df["is_weekend"]      = (ts.dayofweek >= 5).astype(int)
    df["city"]            = "DemoCity"

    # Target
    df[TARGET] = (df["pm25"] * 1.1 + df["no2_ppb"] * 0.4 +
                  df["o3_ppb"] * 0.3 + rng.normal(0, 4, n_hours)).clip(0)

    df.to_csv(path, index=False)
    log.info(f"Demo data written → {path}  ({n_hours} rows)")
    return str(path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(csv_path: str = None):
    log.info("=" * 65)
    log.info("  ROLLING-WINDOW MULTI-MODEL AQI PIPELINE")
    log.info("=" * 65)

    if csv_path is None:
        log.info("No CSV path provided → generating demo data …")
        csv_path = generate_demo_data(n_hours=2400)

    df = load_and_prepare(csv_path)

    log.info(f"\n  Config: TRAIN={TRAIN_DAYS}d | TEST={TEST_DAYS}d | STEP={STEP_DAYS}d")
    log.info(f"  Models: Ridge | Lasso | RandomForest | GradientBoosting | XGBoost\n")

    summary_df, all_metrics, all_preds, window_ids, feature_cols = rolling_window_evaluation(df)

    model_names = list(build_models().keys())

    # ── Print summary table ───────────────────────────────────────────────
    log.info("\n" + "─" * 65)
    log.info("  AGGREGATED RESULTS  (mean ± std across windows)")
    log.info("─" * 65)
    display_cols = ["RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean", "MAPE_mean"]
    print(summary_df[display_cols].round(4).to_string())
    log.info("─" * 65)
    best = summary_df["RMSE_mean"].idxmin()
    log.info(f"  🏆  Best model (lowest RMSE): {best}  "
             f"(RMSE={summary_df.loc[best,'RMSE_mean']:.4f}  "
             f"R²={summary_df.loc[best,'R2_mean']:.4f})")
    log.info("─" * 65)

    # ── Save artefacts ────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = REPORT_DIR / f"model_comparison_{ts}.png"
    plot_results(summary_df, all_metrics, all_preds,
                 window_ids, feature_cols, model_names, plot_path)

    summary_path, detail_path = save_reports(summary_df, all_metrics, model_names)

    log.info("\n  Saved artefacts:")
    log.info(f"    Plot    → {plot_path}")
    log.info(f"    Summary → {summary_path}")
    log.info(f"    Detail  → {detail_path}")
    log.info("  Pipeline complete ✅\n")

    return summary_df, plot_path, summary_path, detail_path


if __name__ == "__main__":
    import sys
    csv_input = sys.argv[1] if len(sys.argv) > 1 else None
    main(csv_input)
