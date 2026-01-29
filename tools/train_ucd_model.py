import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# ============================================================
# PATHS
# ============================================================

DATA_PARQUET = "data/pixels_baseline_2019.parquet"
OUT_MODEL = "models/ucd_model.joblib"

# ============================================================
# FEATURES & TARGET
# ============================================================

FEATURES = ["T2", "RH", "bvol", "bldh", "pop", "infil", "ahem"]
TARGET = "y_ucd_wm2"

# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_PARQUET):
    raise FileNotFoundError(f"Missing baseline file: {DATA_PARQUET}")

df = pd.read_parquet(DATA_PARQUET)

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in parquet: {missing}")

df = df.dropna(subset=FEATURES + [TARGET]).copy()

X = df[FEATURES].values
y = df[TARGET].values

print("===================================")
print("Baseline pixels:", len(df))
print("Target summary:")
print(pd.Series(y).describe())
print("===================================")

# ============================================================
# TRAIN / TEST SPLIT (SANITY CHECK ONLY)
# ============================================================

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# RANDOM FOREST MODEL (CODESPACES SAFE)
# ============================================================

model = RandomForestRegressor(
    n_estimators=600,      # high quality response surface
    max_depth=20,          # prevents runaway trees
    min_samples_leaf=2,
    n_jobs=1,              # Codespaces stability
    random_state=42
)

print("Training RandomForest (this may take 2–5 minutes)...")
model.fit(Xtr, ytr)
print("Training finished.")

# ============================================================
# EVALUATION
# ============================================================

pred = model.predict(Xte)

r2 = r2_score(yte, pred)
rmse = mean_squared_error(yte, pred) ** 0.5
mae = mean_absolute_error(yte, pred)

print("===================================")
print(f"TEST R2   = {r2:.3f}")
print(f"TEST RMSE = {rmse:.3f} W m-2")
print(f"TEST MAE  = {mae:.3f} W m-2")
print("===================================")

# ============================================================
# SAVE MODEL BUNDLE
# ============================================================

os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)

bundle = {
    "model": model,
    "features": FEATURES,
    "target": TARGET,
    "baseline": "pixels_baseline_2019.parquet",
}

joblib.dump(bundle, OUT_MODEL)

print(f"✅ Saved model: {OUT_MODEL}")
