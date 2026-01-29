import os
import pandas as pd

# ------------------------------------------------------------
# INPUT: your Block A CSV inside data/
# ------------------------------------------------------------
BLOCKA_FILE = "data/KOLKATA_PIXEL_ML_v5_5_FULL_FINAL_4x2_COLORGROUPS_BLOCKA_TRAINING_TABLE_UCD_Wm2_AND_Wpp.csv"

# OUTPUT: app-ready baseline file
OUT_PARQUET = "data/pixels_baseline_2019.parquet"

BASELINE_YEAR = 2019

# ------------------------------------------------------------
# LOAD (CSV)
# ------------------------------------------------------------
if not os.path.exists(BLOCKA_FILE):
    raise FileNotFoundError(f"Cannot find input file: {BLOCKA_FILE}")

df = pd.read_csv(BLOCKA_FILE)
print("Loaded rows:", len(df))
print("Columns:", list(df.columns))

# ------------------------------------------------------------
# FILTER YEAR (2019 baseline)
# ------------------------------------------------------------
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df["year"] == BASELINE_YEAR].copy()
print("Rows after year filter:", len(df))

# ------------------------------------------------------------
# KEEP COLUMNS needed for the app
# ------------------------------------------------------------
keep_cols = [
    "pixel_group", "iy", "ix", "lon", "lat",
    "T2", "RH",
    "pop", "infil", "bvol", "bldh", "ahem",
    "y_ucd_wm2", "y_wpp"
]

missing = [c for c in keep_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in Block A file: {missing}")

df = df[keep_cols].replace([float("inf"), float("-inf")], pd.NA).dropna()

# One row per pixel (safety)
df = df.groupby("pixel_group", as_index=False).mean(numeric_only=True)

# ------------------------------------------------------------
# SAVE PARQUET
# ------------------------------------------------------------
os.makedirs("data", exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False)

print("✅ Saved:", OUT_PARQUET)
print("Final pixel count:", len(df))
print(df.head())
