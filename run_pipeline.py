"""
Feature Pipeline Orchestrator
Runs Stage 1 → Stage 2 → Stage 3 in sequence.
Usage:  python run_pipeline.py
"""

import logging
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from stage1_fetch_raw       import fetch_all
from stage2_compute_features import compute_features
from stage3_store_features   import store_features, check_feature_store_status

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/pipeline.log"),
    ],
)
log = logging.getLogger(__name__)

RUN_LOG = Path("data/run_history.json")

def log_run(stage: str, status: str, detail: str):
    runs = []
    if RUN_LOG.exists():
        with open(RUN_LOG) as f:
            runs = json.load(f)
    runs.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage":     stage,
        "status":    status,
        "detail":    detail,
    })
    runs = runs[-200:]
    with open(RUN_LOG, "w") as f:
        json.dump(runs, f, indent=2)


def run():
    start = datetime.now()
    log.info("=" * 60)
    log.info("🚀 Feature Pipeline Starting")
    log.info("=" * 60)

    # ── Stage 1: Fetch Raw ──────────────────────────────────────────────
    try:
        log.info("\n📡 STAGE 1: Fetching raw data...")
        aqicn_data, meteo_data, raw_path = fetch_all()
        log_run("stage1", "success", str(raw_path))
        log.info(f"   Raw data → {raw_path}")
    except Exception as e:
        log.error(f"Stage 1 failed: {e}")
        log_run("stage1", "error", str(e))
        sys.exit(1)

    # ── Stage 2: Compute Features ───────────────────────────────────────
    try:
        log.info("\n⚙️  STAGE 2: Computing features...")
        features, feat_path = compute_features(aqicn_data, meteo_data)
        log_run("stage2", "success", str(feat_path))
        log.info(f"   Features ({len(features)}) → {feat_path}")
    except Exception as e:
        log.error(f"Stage 2 failed: {e}")
        log_run("stage2", "error", str(e))
        sys.exit(1)

    # ── Stage 3: Store ──────────────────────────────────────────────────
    try:
        log.info("\n🗄️  STAGE 3: Storing in Feature Store...")
        status = store_features(features)
        log_run("stage3", "success", json.dumps(status))
    except Exception as e:
        log.error(f"Stage 3 failed: {e}")
        log_run("stage3", "error", str(e))
        sys.exit(1)

    elapsed = (datetime.now() - start).total_seconds()

    # ── Summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("✅ Pipeline completed successfully")
    log.info(f"   Duration     : {elapsed:.1f}s")
    log.info(f"   AQI          : {features.get('aqi')} ({features.get('aqi_cat_label')})")
    log.info(f"   PM2.5        : {features.get('pm25_iaqi')}")
    log.info(f"   Temp         : {features.get('temp_c')}°C")
    log.info(f"   AQI change/h : {features.get('aqi_change_rate_1h', 'N/A (first run)')}")
    log.info(f"   Trend target : {features.get('target_trend_direction')}")
    log.info(f"   Feature store: {'Hopsworks ✅' if status['hopsworks_upload'] else 'CSV only ⚠️'}")
    log.info("=" * 60)

    fs_info = check_feature_store_status()
    log.info(f"\n📦 Feature Store: {fs_info['total_rows']} total rows, "
             f"{fs_info['feature_count']} features, "
             f"backend={fs_info['backend']}")

if __name__ == "__main__":
    run()
