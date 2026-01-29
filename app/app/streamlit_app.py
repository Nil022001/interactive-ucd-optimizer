import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="UCD Optimizer", layout="wide")

st.title("Interactive City-wide Cooling Demand Optimizer")
st.write("Sweep one attribute at a time and find the minimum predicted city-wide demand.")

# -------------------------------------------------------------------
# REQUIRED FILES IN YOUR REPO
# -------------------------------------------------------------------
DATA_PATH = "data/baseline_2019_pixels.csv"          # you will upload later
MODEL_PATH = "models/model_BLOCKA_UCD_BH_noBV.joblib"  # you will upload later

# -------------------------------------------------------------------
# Load baseline + model
# -------------------------------------------------------------------
@st.cache_data
def load_baseline(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(DATA_PATH):
    st.error(f"Missing baseline file: {DATA_PATH}")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Missing model file: {MODEL_PATH}")
    st.stop()

df = load_baseline(DATA_PATH)
model = load_model(MODEL_PATH)

# -------------------------------------------------------------------
# CONFIG: Edit these to match your trained model feature list
# -------------------------------------------------------------------
FEATURE_COLS = ["T2", "RH", "infil", "bldh", "ahem"]   # BH_noBV example
TARGET_KIND = st.sidebar.selectbox("Target kind", ["UCD (W/m²)", "W/person"])

CELL_AREA_M2 = st.sidebar.number_input("Cell area (m²)", value=1_000_000.0, step=10_000.0)

if TARGET_KIND == "W/person" and "pop" not in df.columns:
    st.error("For W/person target you must have 'pop' column in baseline table.")
    st.stop()

# -------------------------------------------------------------------
# Choose sweep feature + sweep range
# -------------------------------------------------------------------
st.sidebar.header("Policy sweep")
sweep_feature = st.sidebar.selectbox("Feature to sweep", FEATURE_COLS)

q05 = float(df[sweep_feature].quantile(0.05))
q95 = float(df[sweep_feature].quantile(0.95))

x_min = st.sidebar.number_input("Sweep min", value=q05)
x_max = st.sidebar.number_input("Sweep max", value=q95)

n_points = st.sidebar.slider("Number of sweep points", 5, 60, 25)

policy_mode = st.sidebar.radio("Policy mode", ["set_uniform", "clip"], index=0)

# -------------------------------------------------------------------
# Helper: aggregate citywide
# -------------------------------------------------------------------
def aggregate_citywide(pred, df_base):
    pred = np.asarray(pred, dtype=float)

    if TARGET_KIND == "UCD (W/m²)":
        total_W = np.nansum(pred * CELL_AREA_M2)
        total_MW = total_W / 1e6
        return total_MW, None

    # W/person
    pop = df_base["pop"].astype(float).values
    total_W = np.nansum(pred * pop)
    total_MW = total_W / 1e6
    mean_Wpp = total_W / (np.nansum(pop) + 1e-12)
    return total_MW, mean_Wpp

# -------------------------------------------------------------------
# Sweep + predict
# -------------------------------------------------------------------
xs = np.linspace(x_min, x_max, n_points)
mw = []
wpp = []

X_base = df[FEATURE_COLS].copy()

for xv in xs:
    Xmod = X_base.copy()

    if policy_mode == "set_uniform":
        Xmod[sweep_feature] = xv
    else:
        # clip each pixel to within [x_min, x_max]
        Xmod[sweep_feature] = np.clip(Xmod[sweep_feature].astype(float).values, x_min, x_max)

    pred = model.predict(Xmod.values)
    total_MW, mean_Wpp = aggregate_citywide(pred, df)
    mw.append(total_MW)
    wpp.append(mean_Wpp if mean_Wpp is not None else np.nan)

mw = np.array(mw, dtype=float)

best_idx = int(np.nanargmin(mw))
best_x = float(xs[best_idx])
best_mw = float(mw[best_idx])

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(xs, mw, marker="o", linewidth=1.2)
ax.axvline(best_x, linestyle="--")
ax.scatter([best_x], [best_mw], s=60)
ax.set_xlabel(sweep_feature)
ax.set_ylabel("Predicted city-wide demand (MW)")
ax.set_title(f"Optimization sweep ({policy_mode}) — optimum at {best_x:.3g} → {best_mw:.2f} MW")
st.pyplot(fig)

st.success(f"✅ Optimum {sweep_feature} ≈ {best_x:.4g} → {best_mw:.2f} MW")

if TARGET_KIND == "W/person" and np.isfinite(wpp).any():
    st.info(f"City-mean W/person at optimum ≈ {float(wpp[best_idx]):.3f}")
