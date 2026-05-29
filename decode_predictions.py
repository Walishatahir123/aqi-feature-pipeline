import joblib
import pandas as pd

# Load model bundle
bundle = joblib.load("aqi_model_outputs/model_clf_target_trend_direction_RandomForest.pkl")
model = bundle["model"]
le    = bundle["label_encoder"]

print("Class mapping:")
for i, label in enumerate(le.classes_):
    print(f"  {i} → {label}")

# Load your data and prepare same features
df = pd.read_csv("data/historical_backup/historical_2026-02-09_2026-05-10.csv")

# Quick check — just decode the training labels
raw_preds = model.predict(model.feature_importances_.reshape(1,-1))  # dummy
# Better: show the class map only