import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import branca.colormap as cm

st.set_page_config(page_title="Kolkata UCD Estimator (1km) — v2", layout="wide")

DATA_PARQUET = "data/pixels_baseline_2019.parquet"
MODEL_FILE = "models/ucd_model.joblib"

@st.cache_data
def load_pixels():
    return pd.read_parquet(DATA_PARQUET)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

st.title("Interactive Urban Cooling Demand (UCD) Estimator — Kolkata (1 km × 1 km)")
st.caption("Yellow = lower demand, Red = higher demand. Use sliders to test ‘what-if’ changes.")

df0 = load_pixels()
bundle = load_model()
model = bundle["model"]
features = bundle["features"]

# -----------------------------
# Sidebar: optimizer sliders
# -----------------------------
st.sidebar.header("Optimizer sliders (global perturbations)")

st.sidebar.subheader("Urban morphology")
bvol_pct = st.sidebar.slider("Building volume change (Δ bvol, %)", -50, 200, 0, 5)
bldh_pct = st.sidebar.slider("Building height change (Δ bldh, %)", -50, 200, 0, 5)

st.sidebar.subheader("Meteorology")
dT2 = st.sidebar.slider("Air temperature change (Δ T2, K)", -5.0, 8.0, 0.0, 0.1)
dRH = st.sidebar.slider("Relative humidity change (Δ RH, % points)", -30.0, 30.0, 0.0, 0.5)

st.sidebar.subheader("Population")
pop_pct = st.sidebar.slider("Population change (Δ pop, %)", -50, 200, 0, 5)

st.sidebar.subheader("Surface / land properties")
infil_pct = st.sidebar.slider("Infiltration change (Δ infil, %)", -100, 300, 0, 10)

# OPTIONAL (remove if you don't want AHEM slider)
ahem_pct = st.sidebar.slider("Anthropogenic heat change (Δ ahem, %)", -50, 300, 0, 10)

# -----------------------------
# Prepare working dataframe
# -----------------------------
df = df0.copy()

# Baseline prediction
df["ucd_base_pred"] = model.predict(df[features].values)

# Apply perturbations to features (global scenario)
df["bvol_adj"] = df["bvol"] * (1.0 + bvol_pct / 100.0)
df["bldh_adj"] = df["bldh"] * (1.0 + bldh_pct / 100.0)

df["T2_adj"] = df["T2"] + dT2
df["RH_adj"] = np.clip(df["RH"] + dRH, 0.0, 100.0)

df["pop_adj"] = df["pop"] * (1.0 + pop_pct / 100.0)

# infil and ahem (new)
df["infil_adj"] = df["infil"] * (1.0 + infil_pct / 100.0)
df["ahem_adj"]  = df["ahem"]  * (1.0 + ahem_pct  / 100.0)

# Build X for prediction using model feature ordering
feature_map = {
    "T2": "T2_adj",
    "RH": "RH_adj",
    "bvol": "bvol_adj",
    "bldh": "bldh_adj",
    "pop": "pop_adj",
    "infil": "infil_adj",
    "ahem": "ahem_adj",
}

Xmat = np.column_stack([df[feature_map.get(f, f)].values for f in features])

df["ucd_new_pred"] = model.predict(Xmat)
df["d_ucd"] = df["ucd_new_pred"] - df["ucd_base_pred"]

# -----------------------------
# Map mode
# -----------------------------
st.sidebar.header("Map mode")
mode = st.sidebar.selectbox(
    "Show",
    ["New predicted UCD (W m⁻²)", "ΔUCD (new - baseline) (W m⁻²)", "Baseline predicted UCD (W m⁻²)"],
    index=0
)
if mode.startswith("New"):
    val_col = "ucd_new_pred"
elif mode.startswith("ΔUCD"):
    val_col = "d_ucd"
else:
    val_col = "ucd_base_pred"

# -----------------------------
# Colormap
# -----------------------------
vals = df[val_col].astype(float).values
vmin = float(np.nanpercentile(vals, 2))
vmax = float(np.nanpercentile(vals, 98))
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals)) if float(np.nanmax(vals)) > vmin else (vmin + 1.0)

if val_col == "d_ucd":
    colormap = cm.LinearColormap(
        colors=["#2c7bb6", "#ffffbf", "#d7191c"],  # blue -> yellow -> red
        vmin=vmin,
        vmax=vmax
    )
    colormap.caption = "ΔUCD (W m⁻²)"
else:
    colormap = cm.LinearColormap(
        colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],  # yellow -> red
        vmin=vmin,
        vmax=vmax
    )
    colormap.caption = f"{mode}"

# -----------------------------
# Map
# -----------------------------
center_lat = float(df["lat"].mean())
center_lon = float(df["lon"].mean())
m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

for _, r in df.iterrows():
    val = float(r[val_col])
    color = colormap(val)

    tooltip = (
        f"pixel: {r['pixel_group']}<br>"
        f"UCD_base_pred: {r['ucd_base_pred']:.3f} W/m²<br>"
        f"UCD_new_pred: {r['ucd_new_pred']:.3f} W/m²<br>"
        f"ΔUCD: {r['d_ucd']:.3f} W/m²<br><br>"
        f"T2: {r['T2']:.2f}→{r['T2_adj']:.2f} K | "
        f"RH: {r['RH']:.2f}→{r['RH_adj']:.2f}%<br>"
        f"bvol: {r['bvol']:.1f}→{r['bvol_adj']:.1f} | "
        f"bldh: {r['bldh']:.2f}→{r['bldh_adj']:.2f}<br>"
        f"pop: {r['pop']:.0f}→{r['pop_adj']:.0f}<br>"
        f"infil: {r['infil']:.3f}→{r['infil_adj']:.3f} | "
        f"ahem: {r['ahem']:.1f}→{r['ahem_adj']:.1f}"
    )

    folium.CircleMarker(
        location=[float(r["lat"]), float(r["lon"])],
        radius=4,
        color=color,
        weight=1,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        tooltip=tooltip,
    ).add_to(m)

colormap.add_to(m)

# -----------------------------
# Layout
# -----------------------------
c1, c2 = st.columns([2.2, 1.0], gap="large")

with c1:
    st.subheader("Kolkata 1 km pixels (interactive)")
    out = st_folium(m, width=1100, height=650)

with c2:
    st.subheader("Scenario summary")
    st.write(f"Δ bvol: **{bvol_pct:+d}%**")
    st.write(f"Δ bldh: **{bldh_pct:+d}%**")
    st.write(f"Δ T2: **{dT2:+.1f} K**")
    st.write(f"Δ RH: **{dRH:+.1f} %**")
    st.write(f"Δ pop: **{pop_pct:+d}%**")
    st.write(f"Δ infil: **{infil_pct:+d}%**")
    st.write(f"Δ ahem: **{ahem_pct:+d}%**")

    st.markdown("---")
    st.write("Citywide stats (predicted)")
    st.metric("Baseline mean UCD (W m⁻²)", f"{df['ucd_base_pred'].mean():.3f}")
    st.metric("New mean UCD (W m⁻²)", f"{df['ucd_new_pred'].mean():.3f}")
    st.metric("Mean ΔUCD (W m⁻²)", f"{df['d_ucd'].mean():+.3f}")

st.subheader("Clicked pixel (nearest)")
if out and out.get("last_clicked"):
    click_lat = out["last_clicked"]["lat"]
    click_lon = out["last_clicked"]["lng"]
    d = (df["lat"] - click_lat) ** 2 + (df["lon"] - click_lon) ** 2
    idx = d.idxmin()
    row = df.loc[idx].copy()
    st.write("Nearest pixel_group:", row["pixel_group"])
    st.dataframe(row.to_frame("value"))
else:
    st.info("Click on a pixel marker to see nearest pixel details.")
