
import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MONGODB_URI        = os.getenv("MONGODB_URI", "")
MONGODB_DB         = "aqi_pipeline"
MONGODB_COLLECTION = "weather_aqi_features"
BACKUP_DIR         = Path("data/features_store_backup")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)


# ── MongoDB Upload ─────────────────────────────────────────────────────────

def upload_to_mongodb(row: dict) -> bool:
    """
    Upload one feature row to MongoDB Atlas.
    Returns True on success, False on failure.
    """
    if not MONGODB_URI:
        log.warning("MONGODB_URI not set — skipping MongoDB upload.")
        return False

    try:
        from pymongo import MongoClient

        client     = MongoClient(MONGODB_URI)
        db         = client[MONGODB_DB]
        collection = db[MONGODB_COLLECTION]

        # Make row serializable (convert numpy types etc.)
        clean_row = {}
        for k, v in row.items():
            try:
                if hasattr(v, "item"):          # numpy scalar
                    clean_row[k] = v.item()
                elif isinstance(v, float) and (v != v):  # NaN
                    clean_row[k] = None
                else:
                    clean_row[k] = v
            except Exception:
                clean_row[k] = str(v)

        # Upsert by timestamp + city (idempotent)
        collection.update_one(
            {"timestamp": clean_row.get("timestamp"), "city": clean_row.get("city")},
            {"$set": clean_row},
            upsert=True,
        )

        log.info(f"Inserted/updated row in MongoDB: {clean_row.get('timestamp')}")
        client.close()
        return True

    except ImportError:
        log.error("pymongo not installed. Run: pip install 'pymongo[srv]'")
        return False
    except Exception as e:
        log.error(f"MongoDB upload failed: {e}")
        return False


# ── CSV Backup (always written) ────────────────────────────────────────────

def save_backup_csv(row: dict) -> Path:
    date_str = datetime.now().strftime("%Y-%m-%d")
    path     = BACKUP_DIR / f"feature_store_backup_{date_str}.csv"
    df       = pd.DataFrame([row])

    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False, quoting=1)
    else:
        df.to_csv(path, index=False, quoting=1)

    log.info(f"Backup CSV → {path}")
    return path


# ── Feature Store Status ───────────────────────────────────────────────────

def check_feature_store_status() -> dict:
    backup_files = sorted(BACKUP_DIR.glob("*.csv"))
    total_rows   = 0
    for f in backup_files:
        try:
            total_rows += len(pd.read_csv(f, on_bad_lines='skip', engine='python'))
        except Exception:
            pass

    return {
        "backend":      "MongoDB Atlas" if MONGODB_URI else "CSV backup only",
        "backup_files": len(backup_files),
        "total_rows":   total_rows,
        "last_backup":  str(backup_files[-1]) if backup_files else "none",
    }


# ── Entry Point ────────────────────────────────────────────────────────────

def store_features(row: dict) -> dict:
    """
    Main entry: try MongoDB, always write CSV backup.
    Returns status dict.
    """
    log.info("Stage 3: Storing features...")

    # Always save CSV backup
    backup_path = save_backup_csv(row)

    # Try MongoDB
    mongo_success = upload_to_mongodb(row)

    status = {
        "mongodb_upload": mongo_success,
        "csv_backup":     str(backup_path),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
        "feature_count":  len(row),
    }

    if mongo_success:
        log.info("Stage 3 complete: stored in MongoDB Atlas + CSV backup")
    else:
        log.info("Stage 3 complete: stored in CSV backup (MongoDB skipped)")

    return status


if __name__ == "__main__":
    feat_files = sorted(Path("data/features").glob("features_*.csv"))
    if not feat_files:
        print("No feature CSV found. Run stage2_compute_features.py first.")
        exit(1)

    df  = pd.read_csv(feat_files[-1], on_bad_lines='skip', engine='python')
    row = df.iloc[-1].to_dict()

    status = store_features(row)
    print(f"\n Stage 3 complete.")
    print(f"   MongoDB: {'✅' if status['mongodb_upload'] else '⚠️  skipped (set MONGODB_URI)'}")
    print(f"   CSV backup: {status['csv_backup']}")

    store_info = check_feature_store_status()
    print(f"\nFeature Store Status:")
    for k, v in store_info.items():
        print(f"   {k:20s} = {v}")
