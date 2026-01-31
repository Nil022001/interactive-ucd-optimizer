import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import branca.colormap as cm

st.set_page_config(page_title="Kolkata UCD Estimator (1km) — v3", layout="wide")

DATA_PARQUET = "data/pixels_baseline_2019.parquet"
MODEL_FILE = "models/ucd_model.joblib"

@st.cache_data
def load_pixels():
    return pd.read_parquet(DATA_PARQUET)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

# -----------------------------
# Title
# -----------------------------
st.title("Urban Cooling Demand (UCD) Estimator — Kolkata (1 km × 1 km)")
st.caption("Planner workflow: City-scale scenario → Pixel-scale intervention → Citywide impact recomputed")

df0 = load_pixels()
bundle = load_model()
model = bundle["model"]
features = bundle["features"]

# ==========================================================
# SIDEBAR — CITY-LEVEL (GLOBAL) SCENARIO SLIDERS (Maroon)
# ==========================================================
st.sidebar.markdown(
    "<h3 style='color:#7a0019;'>City-level scenario (applies to all pixels)</h3>",
    unsafe_allow_html=True
)

st.sidebar.subheader("Urban morphology")
bvol_city = st.sidebar.slider("Δ Building volume (%)", -50, 200, 0, 5, key="bvol_city")
bldh_city = st.sidebar.slider("Δ Building height (%)", -50, 200, 0, 5, key="bldh_city")

st.sidebar.subheader("Meteorology")
dT2_city = st.sidebar.slider("Δ Air temperature (K)", -5.0, 8.0, 0.0, 0.1, key="dT2_city")
dRH_city = st.sidebar.slider("Δ Relative humidity (% points)", -30.0, 30.0, 0.0, 0.5, key="dRH_city")

st.sidebar.subheader("Population")
pop_city = st.sidebar.slider("Δ Population (%)", -50, 200, 0, 5, key="pop_city")

st.sidebar.subheader("Surface / land properties")
infil_city = st.sidebar.slider("Δ Infiltration (%)", -100, 300, 0, 10, key="infil_city")
ahem_city  = st.sidebar.slider("Δ Anthropogenic heat (%)", -50, 300, 0, 10, key="ahem_city")

# ==========================================================
# BASELINE + CITY-LEVEL PERTURBATION
# ==========================================================
df = df0.copy()

# Baseline prediction (no perturbation)
df["ucd_base_pred"] = model.predict(df[features].values)

# Apply city-level perturbations
df["bvol_adj"]  = df["bvol"]  * (1.0 + bvol_city / 100.0)
df["bldh_adj"]  = df["bldh"]  * (1.0 + bldh_city / 100.0)
df["pop_adj"]   = df["pop"]   * (1.0 + pop_city  / 100.0)
df["T2_adj"]    = df["T2"]    + dT2_city
df["RH_adj"]    = np.clip(df["RH"] + dRH_city, 0.0, 100.0)
df["infil_adj"] = np.clip(df["infil"] * (1.0 + infil_city / 100.0), 0.0, None)
df["ahem_adj"]  = np.clip(df["ahem"]  * (1.0 + ahem_city  / 100.0), 0.0, None)

feature_map = {
    "T2": "T2_adj",
    "RH": "RH_adj",
    "bvol": "bvol_adj",
    "bldh": "bldh_adj",
    "pop": "pop_adj",
    "infil": "infil_adj",
    "ahem": "ahem_adj",
}

X_city = np.column_stack([df[feature_map[f]].values for f in features])
df["ucd_city_pred"] = model.predict(X_city)

# City metrics (baseline vs city-scenario)
city_base_mean = float(df["ucd_base_pred"].mean())
city_scn_mean  = float(df["ucd_city_pred"].mean())
city_scn_delta = city_scn_mean - city_base_mean

# ==========================================================
# MAP SETUP — MULTIPLE BASEMAP LAYERS (ESRI + OSM)
# ==========================================================
center_lat = float(df["lat"].mean())
center_lon = float(df["lon"].mean())

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=None)

folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="ESRI World Imagery (Satellite)",
).add_to(m)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="ESRI Light Gray (Base)",
).add_to(m)

# ==========================================================
# COLORBAR — FULL RANGE (TRUE MIN → TRUE MAX)
# ==========================================================
mode = st.sidebar.selectbox(
    "Map value",
    ["City-scenario UCD (W m⁻²)", "Baseline UCD (W m⁻²)", "ΔUCD (city-scenario - baseline)"],
    index=0
)

if mode.startswith("City-scenario"):
    val_col = "ucd_city_pred"
elif mode.startswith("Baseline"):
    val_col = "ucd_base_pred"
else:
    df["d_ucd_city"] = df["ucd_city_pred"] - df["ucd_base_pred"]
    val_col = "d_ucd_city"

vals = df[val_col].astype(float).values
vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
    vmin, vmax = 0.0, 1.0

if val_col == "d_ucd_city":
    colormap = cm.LinearColormap(
        colors=["#2c7bb6", "#ffffbf", "#d7191c"],
        vmin=vmin,
        vmax=vmax
    )
    colormap.caption = "ΔUCD (W m⁻²)"
else:
    colormap = cm.LinearColormap(
        colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#800026"],
        vmin=vmin,
        vmax=vmax
    )
    colormap.caption = f"{mode}"

# ==========================================================
# DRAW PIXELS (circles for speed; 1 km squares is next step)
# ==========================================================
for _, r in df.iterrows():
    color = colormap(float(r[val_col]))

    folium.CircleMarker(
        location=[float(r["lat"]), float(r["lon"])],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        weight=0.6,
        tooltip=(
            f"Pixel: {r['pixel_group']}<br>"
            f"Baseline UCD: {r['ucd_base_pred']:.2f}<br>"
            f"City-scenario UCD: {r['ucd_city_pred']:.2f}<br>"
            f"ΔUCD: {(r['ucd_city_pred']-r['ucd_base_pred']):+.2f}"
        ),
    ).add_to(m)

colormap.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)

# ==========================================================
# LAYOUT
# ==========================================================
c1, c2 = st.columns([2.3, 1.0], gap="large")

with c1:
    st.subheader("UCD map (layer toggle: OSM / ESRI satellite / ESRI gray)")
    out = st_folium(m, width=1100, height=650)

with c2:
    st.subheader("City-wide impact (global scenario)")
    st.metric("City mean UCD (baseline)", f"{city_base_mean:.2f} W m⁻²")
    st.metric("City mean UCD (city-scenario)", f"{city_scn_mean:.2f} W m⁻²", delta=f"{city_scn_delta:+.2f}")
    st.caption("Now click a pixel to apply a local intervention and recompute citywide impact.")

# ==========================================================
# PIXEL-LEVEL LOCAL INTERVENTION (Prussian Blue)
# + CITY IMPACT RECOMPUTATION AFTER LOCAL CHANGE
# ==========================================================
st.markdown(
    "<h3 style='color:#003153;'>Pixel-level intervention (applies ONLY to selected pixel)</h3>",
    unsafe_allow_html=True
)

if out and out.get("last_clicked"):
    click_lat = out["last_clicked"]["lat"]
    click_lon = out["last_clicked"]["lng"]

    idx = ((df["lat"] - click_lat)**2 + (df["lon"] - click_lon)**2).idxmin()
    row = df.loc[idx].copy()

    st.write(f"Selected pixel: **{row['pixel_group']}**")

    # Pixel-level sliders (local intervention)
    st.markdown("<b style='color:#003153;'>Local controls</b>", unsafe_allow_html=True)
    bvol_px  = st.slider("Pixel Δ bvol (%)", -50, 200, 0, 5, key="bvol_px")
    bldh_px  = st.slider("Pixel Δ bldh (%)", -50, 200, 0, 5, key="bldh_px")
    infil_px = st.slider("Pixel Δ infil (%)", -100, 300, 0, 10, key="infil_px")
    ahem_px  = st.slider("Pixel Δ ahem (%)", -50, 300, 0, 10, key="ahem_px")

    # Apply pixel-level changes on top of city scenario (hierarchical)
    row["bvol_adj"]  *= (1.0 + bvol_px  / 100.0)
    row["bldh_adj"]  *= (1.0 + bldh_px  / 100.0)
    row["infil_adj"] *= (1.0 + infil_px / 100.0)
    row["ahem_adj"]  *= (1.0 + ahem_px  / 100.0)

    X_px = np.array([[row[feature_map[f]] for f in features]])
    ucd_px_new = float(model.predict(X_px)[0])

    # Citywide recomputation after local intervention:
    # overwrite just that pixel's city-scenario prediction with the new local value
    df_after_local = df.copy()
    df_after_local.loc[idx, "ucd_city_pred"] = ucd_px_new

    city_after_local_mean = float(df_after_local["ucd_city_pred"].mean())
    city_local_delta = city_after_local_mean - city_scn_mean

    # Show pixel metrics
    st.subheader("Selected pixel impact")
    st.metric("Pixel UCD (city-scenario)", f"{float(df.loc[idx,'ucd_city_pred']):.2f} W m⁻²")
    st.metric("Pixel UCD (after local action)", f"{ucd_px_new:.2f} W m⁻²", delta=f"{ucd_px_new - float(df.loc[idx,'ucd_city_pred']):+.2f}")

    # Show city metrics after local intervention
    st.subheader("Citywide impact AFTER local action")
    st.metric("City mean UCD (city-scenario)", f"{city_scn_mean:.2f} W m⁻²")
    st.metric("City mean UCD (after local action)", f"{city_after_local_mean:.2f} W m⁻²", delta=f"{city_local_delta:+.4f}")

    st.caption("This is the correct planning interpretation: a local intervention changes the selected pixel, and citywide stats are recomputed by aggregation.")

else:
    st.info("Click a pixel on the map to activate pixel-level controls and see citywide impact updates.")
