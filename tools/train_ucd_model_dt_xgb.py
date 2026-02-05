import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from xgboost import XGBRegressor

# ============================================================
# PATHS (NEW: do NOT overwrite old files)
# ============================================================

DATA_PARQUET = "data/pixels_baseline_2019_dt.parquet"
OUT_MODEL = "models/ucd_model_dt_xgb.joblib"

# ============================================================
# FEATURES & TARGET
# ============================================================

FEATURES = ["dT_cool", "RH", "bvol", "bldh", "pop", "infil", "ahem"]
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

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET]).copy()

# safety clamps (optional)
df["RH"] = df["RH"].clip(0, 100)
df["infil"] = df["infil"].clip(0, 1.0)
df["ahem"] = df["ahem"].clip(lower=0)
df["dT_cool"] = df["dT_cool"].clip(lower=0)

X = df[FEATURES].astype(float).values
y = df[TARGET].astype(float).values

print("===================================")
print("Baseline pixels:", len(df))
print("Using FEATURES:", FEATURES)
print("dT_cool summary:")
print(pd.Series(df["dT_cool"]).describe())
print("Target summary:")
print(pd.Series(y).describe())
print("===================================")

# ============================================================
# TRAIN / TEST SPLIT (SANITY CHECK)
# ============================================================

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================
# XGBOOST MODEL (fast + accurate)
# ============================================================

model = XGBRegressor(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    reg_alpha=0.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,   # adjust if needed
)

print("Training XGBoost (this may take a few minutes)...")
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
    "baseline": os.path.basename(DATA_PARQUET),
    "model_type": "XGBRegressor",
}

joblib.dump(bundle, OUT_MODEL)

print(f"✅ Saved model: {OUT_MODEL}")
print("Saved bundle features:", FEATURES)
