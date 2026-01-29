import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="UCD Optimizer", layout="wide")

st.title("Interactive City-wide Cooling Demand Optimizer")
st.write("Sweep one attribute at a time and find the minimum predicted city-wide demand.")

# ---------------------------------------------------------
# Files you will upload later into these repo folders:
#   data/baseline_2019_pixels.csv
#   models/model_BLOCKA_UCD_BH_noBV.joblib
# ---------------------------------------------------------
DATA_PATH = "data/baseline_2019_pixels.csv"
MODEL_PATH = "models/model_BLOCKA_UCD_BH_noBV.joblib"

@st.cache_data
def load_baseline(path):
    return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(DATA_PATH):
    st.error(f"Missing baseline file: {DATA_PATH} (upload it to the data/ folder)")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error(f"Missing model file: {MODEL_PATH} (upload it to the models/ folder)")
    st.stop()

df = load_baseline(DATA_PATH)
model = load_model(MODEL_PATH)

# ---------------------------------------------------------
# IMPORTANT: Edit these feature names to match your model
# ---------------------------------------------------------
FEATURE_COLS = ["T2", "RH", "infil", "bldh", "ahem"]  # BH_noBV example

TARGET_KIND = st.sidebar.selectbox("Target kind", ["UCD (W/m²)", "W/person"])

CELL_AREA_M2 = st.sidebar.number_input("Cell area (m²)", value=1_000_000.0, step=10_000.0)

if TARGET_KIND == "W/person" and "pop" not in df.columns:
    st.error("For W/person target, baseline must contain a 'pop' column.")
    st.stop()

st.sidebar.header("Policy sweep")
sweep_feature = st.sidebar.selectbox("Feature to sweep", FEATURE_COLS)

q05 = float(df[sweep_feature].quantile(0.05))
q95 = float(df[sweep_feature].quantile(0.95))

x_min = st.sidebar.number_input("Sweep min", value=q05)
x_max = st.sidebar.number_input("Sweep max", value=q95)

n_points = st.sidebar.slider("Number of sweep points", 5, 60, 25)

policy_mode = st.sidebar.radio("Policy mode", ["set_uniform", "clip"], index=0)

def aggregate_citywide(pred, df_base):
    pred = np.asarray(pred, dtype=float)

    if TARGET_KIND == "UCD (W/m²)":
        total_W = np.nansum(pred * CELL_AREA_M2)
        total_MW = total_W / 1e6
        return total_MW, None

    pop = df_base["pop"].astype(float).values
    total_W = np.nansum(pred * pop)
    total_MW = total_W / 1e6
    mean_Wpp = total_W / (np.nansum(pop) + 1e-12)
    return total_MW, mean_Wpp

xs = np.linspace(x_min, x_max, n_points)
mw = []
wpp = []

X_base = df[FEATURE_COLS].copy()

for xv in xs:
    Xmod = X_base.copy()

    if policy_mode == "set_uniform":
        Xmod[sweep_feature] = xv
    else:
        Xmod[sweep_feature] = np.clip(Xmod[sweep_feature].astype(float).values, x_min, x_max)

    pred = model.predict(Xmod.values)
    total_MW, mean_Wpp = aggregate_citywide(pred, df)
    mw.append(total_MW)
    wpp.app
