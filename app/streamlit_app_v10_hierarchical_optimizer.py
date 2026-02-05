import os
import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import branca.colormap as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import hashlib

# NEW (wards)
import requests
import geopandas as gpd
from shapely.geometry import Point

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Kolkata UCD Estimator — SoC years (1 km × 1 km → ward → cityscale) [Setpoint-aware]",
    layout="wide",
)

# NEW baseline + NEW model (XGB)
DATA_PARQUET = "data/pixels_baseline_2019_dt.parquet"
MODEL_FILE   = "models/ucd_model_dt_xgb.joblib"

PIXEL_AREA = 1_000_000.0  # m² (1 km × 1 km)

# WARD LAYER (KML from OpenCity)
WARD_KML_URL = (
    "https://data.opencity.in/dataset/kolkata-wards-information/resource/"
    "b3195f54-6c85-4cb4-afa0-06af482fe9df/download/10f7cc2f-5005-412f-861d-d5e085dd92fb.kml"
)
WARD_KML_PATH = "data/kolkata_wards_2022.kml"

# Full-bleed-ish layout
st.markdown(
    """
    <style>
      .block-container {padding-left: 0.6rem; padding-right: 0.6rem; max-width: 100%;}
      [data-testid="stSidebar"] {min-width: 320px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# STREAMLIT-SAFE SLIDER + BOX (single canonical key; unique widget keys)
# =========================================================
def slider_with_box_int(label, minv, maxv, default, step, key):
    k_slider = f"{key}__slider"
    k_box = f"{key}__box"

    if key not in st.session_state:
        st.session_state[key] = int(default)
    if k_slider not in st.session_state:
        st.session_state[k_slider] = int(st.session_state[key])
    if k_box not in st.session_state:
        st.session_state[k_box] = int(st.session_state[key])

    def _from_slider():
        st.session_state[key] = int(st.session_state[k_slider])
        st.session_state[k_box] = int(st.session_state[key])

    def _from_box():
        st.session_state[key] = int(st.session_state[k_box])
        st.session_state[k_slider] = int(st.session_state[key])

    c1, c2 = st.columns([0.7, 0.3], vertical_alignment="center")
    with c1:
        st.slider(
            label,
            int(minv),
            int(maxv),
            value=int(st.session_state[key]),
            step=int(step),
            key=k_slider,
            on_change=_from_slider,
        )
    with c2:
        st.number_input(
            " ",
            min_value=int(minv),
            max_value=int(maxv),
            value=int(st.session_state[key]),
            step=int(step),
            key=k_box,
            on_change=_from_box,
        )

    return int(st.session_state[key])


def slider_with_box_float(label, minv, maxv, default, step, key):
    """
    Float version (forces float types to avoid StreamlitAPIException)
    """
    k_slider = f"{key}__slider"
    k_box = f"{key}__box"

    minv = float(minv); maxv = float(maxv); default = float(default); step = float(step)

    if key not in st.session_state:
        st.session_state[key] = float(default)
    if k_slider not in st.session_state:
        st.session_state[k_slider] = float(st.session_state[key])
    if k_box not in st.session_state:
        st.session_state[k_box] = float(st.session_state[key])

    def _from_slider():
        st.session_state[key] = float(st.session_state[k_slider])
        st.session_state[k_box] = float(st.session_state[key])

    def _from_box():
        st.session_state[key] = float(st.session_state[k_box])
        st.session_state[k_slider] = float(st.session_state[key])

    c1, c2 = st.columns([0.7, 0.3], vertical_alignment="center")
    with c1:
        st.slider(
            label,
            min_value=minv,
            max_value=maxv,
            value=float(st.session_state[key]),
            step=step,
            key=k_slider,
            on_change=_from_slider,
        )
    with c2:
        st.number_input(
            " ",
            min_value=minv,
            max_value=maxv,
            value=float(st.session_state[key]),
            step=step,
            format="%.2f",
            key=k_box,
            on_change=_from_box,
        )

    return float(st.session_state[key])

# =========================================================
# Setpoint helpers (stable stochastic per pixel)
# =========================================================
def stable_u01(text: str, seed: int = 0) -> float:
    h = hashlib.md5(f"{seed}_{text}".encode("utf-8", errors="ignore")).hexdigest()
    return (int(h[:12], 16) % 10_000_000) / 10_000_000.0

def compute_tset_series(df_pixels: pd.DataFrame, mean_c: float, rng_c: float, mode: str, seed: int) -> pd.Series:
    mean_c = float(mean_c)
    rng_c = float(rng_c)
    if mode.startswith("Fixed") or rng_c <= 0:
        return pd.Series(mean_c, index=df_pixels.index, dtype="float64")
    lo = mean_c - rng_c
    hi = mean_c + rng_c
    u = df_pixels["pixel_group"].astype(str).apply(lambda pg: stable_u01(pg, seed))
    return (lo + (hi - lo) * u.astype("float64")).astype("float64")

# =========================================================
# LOAD
# =========================================================
@st.cache_data
def load_pixels():
    return pd.read_parquet(DATA_PARQUET)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_wards():
    os.makedirs(os.path.dirname(WARD_KML_PATH), exist_ok=True)

    if not os.path.exists(WARD_KML_PATH):
        r = requests.get(WARD_KML_URL, timeout=60)
        r.raise_for_status()
        with open(WARD_KML_PATH, "wb") as f:
            f.write(r.content)

    try:
        wards = gpd.read_file(WARD_KML_PATH, driver="KML")
    except Exception as e:
        raise RuntimeError(
            "Could not read KML with GeoPandas/Fiona in this environment.\n"
            "Fallback: convert the KML to GeoJSON (QGIS: Save Features As… GeoJSON),\n"
            "save it as data/kolkata_wards_2022.geojson, and replace this read_file() call.\n"
            f"Original error: {e}"
        )

    if wards.crs is None:
        wards = wards.set_crs(epsg=4326)
    else:
        wards = wards.to_crs(epsg=4326)

    wards = wards[wards.geometry.notnull()].copy()
    wards["geometry"] = wards.geometry.buffer(0)
    return wards

@st.cache_data
def build_pixel_ward_join(df_pixels: pd.DataFrame, _wards_gdf: gpd.GeoDataFrame, wards_cache_key: float) -> pd.DataFrame:
    pixels_gdf = gpd.GeoDataFrame(
        df_pixels[["pixel_group", "lon", "lat"]].copy(),
        geometry=[Point(xy) for xy in zip(df_pixels["lon"], df_pixels["lat"])],
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        pixels_gdf[["pixel_group", "geometry"]],
        _wards_gdf[["geometry"]].copy(),
        how="left",
        predicate="within",
    ).rename(columns={"index_right": "ward_idx"})

    return pd.DataFrame(joined[["pixel_group", "ward_idx"]])

# =========================================================
# LOAD DATA + MODEL + WARDS
# =========================================================
df0 = load_pixels()
bundle = load_model()
model = bundle["model"]
features = bundle["features"]

# Require these cols in dt-parquet
required_cols = ["pixel_group", "iy", "ix", "lon", "lat", "RH", "pop", "infil", "bvol", "bldh", "ahem", "Tout_C", "dT_cool"]
miss = [c for c in required_cols if c not in df0.columns]
if miss:
    st.error(f"Missing columns in {DATA_PARQUET}: {miss}. Rebuild dt parquet.")
    st.stop()

wards_gdf = load_wards()
wards_cache_key = float(os.path.getmtime(WARD_KML_PATH)) if os.path.exists(WARD_KML_PATH) else 0.0
pix_in_ward = build_pixel_ward_join(df0, wards_gdf, wards_cache_key)

ward_label_candidates = [c for c in wards_gdf.columns if c.lower() not in ["geometry"]]
WARD_LABEL_COL = None
for cand in ["Name", "name", "WARD_NO", "Ward_No", "ward_no", "WARD", "ward"]:
    if cand in wards_gdf.columns:
        WARD_LABEL_COL = cand
        break
if WARD_LABEL_COL is None and len(ward_label_candidates) > 0:
    WARD_LABEL_COL = ward_label_candidates[0]

if "selected_pixels" not in st.session_state:
    st.session_state.selected_pixels = []

if "ward_choice" not in st.session_state:
    st.session_state.ward_choice = 0
if "ward_select_mode" not in st.session_state:
    st.session_state.ward_select_mode = "Replace selection with this ward"

# =========================================================
# APP HEADER
# =========================================================
st.title("Kolkata UCD Estimator — SoC years (1 km × 1 km → ward → cityscale)")
st.caption("Baseline = fixed numerical baseline; scenarios change UCD using outdoor + setpoint + urban factors")

# =========================================================
# SIDEBAR CONTROLS (keep old + add setpoint)
# =========================================================
with st.sidebar:
    st.markdown("<h3 style='color:#7a0019;'>Scenario controls</h3>", unsafe_allow_html=True)

    # Reset EVERYTHING (sliders + widget mirrors + selection)
    if st.button("Reset ALL sliders & selection to baseline (0)", use_container_width=True):
        reset_keys = [
            # new setpoint controls
            "tset_mean", "tset_rng", "tset_seed",
            # old
            "city_bvol", "city_bldh", "city_pop", "city_dTout", "city_dRH", "city_infil", "city_ahem",
            "loc_bvol", "loc_bldh", "loc_pop", "loc_dTout", "loc_dRH", "loc_infil", "loc_ahem",
            "enable_local",
            "tset_mode",
        ]
        for k in reset_keys:
            if k in st.session_state:
                del st.session_state[k]
            ks = f"{k}__slider"
            kb = f"{k}__box"
            if ks in st.session_state:
                del st.session_state[ks]
            if kb in st.session_state:
                del st.session_state[kb]

        st.session_state.selected_pixels = []
        st.rerun()

    st.markdown("---")
    st.markdown("<h4 style='color:#7a0019;'>Indoor setpoint preference</h4>", unsafe_allow_html=True)
    st.caption("Affects ONLY scenario/final (baseline remains fixed).")

    tset_mean = slider_with_box_float("Setpoint mean (°C)", 18.0, 30.0, 24.0, 0.1, key="tset_mean")
    tset_rng  = slider_with_box_float("Setpoint ± range (°C)", 0.0, 6.0, 2.5, 0.1, key="tset_rng")
    tset_mode = st.selectbox("Setpoint mode", ["Fixed (same everywhere)", "Stochastic (per pixel)"], index=0, key="tset_mode")
    tset_seed = slider_with_box_int("Setpoint seed", 0, 9999, 42, 1, key="tset_seed")

    st.markdown("---")
    st.markdown("<h4 style='color:#7a0019;'>City-level scenario</h4>", unsafe_allow_html=True)

    bvol_city = slider_with_box_int("Δ Building volume (%)", -50, 200, 0, 5, key="city_bvol")
    bldh_city = slider_with_box_int("Δ Building height (%)", -50, 200, 0, 5, key="city_bldh")
    pop_city  = slider_with_box_int("Δ Population (%)",      -50, 200, 0, 5, key="city_pop")

    # Renamed: outdoor temperature (°C) instead of T2(K). Delta is identical numerically.
    dTout_city = slider_with_box_float("Δ Outdoor temperature (°C)", -5.0, 8.0, 0.0, 0.1, key="city_dTout")
    dRH_city   = slider_with_box_float("Δ Relative humidity (%)",  -30.0, 30.0, 0.0, 0.5, key="city_dRH")

    infilling_city = slider_with_box_int("Δ Urban infilling (%)",   -100, 300, 0, 10, key="city_infil")
    ahem_city      = slider_with_box_int("Δ Anthropogenic heat (%)", -50, 300, 0, 10, key="city_ahem")

    st.markdown("---")
    st.markdown("<h4 style='color:#7a0019;'>Local project overrides (selected pixels)</h4>", unsafe_allow_html=True)
    st.caption("Applies ONLY to selected pixels (on top of city scenario).")

    enable_local = st.checkbox("Enable local overrides", value=True, key="enable_local")

    if enable_local:
        loc_bvol = slider_with_box_int("Local Δ Building volume (%)", -80, 200, 0, 5, key="loc_bvol")
        loc_bldh = slider_with_box_int("Local Δ Building height (%)", -80, 200, 0, 5, key="loc_bldh")
        loc_pop  = slider_with_box_int("Local Δ Population (%)",      -80, 200, 0, 5, key="loc_pop")

        loc_dTout = slider_with_box_float("Local Δ Outdoor temperature (°C)", -5.0, 8.0, 0.0, 0.1, key="loc_dTout")
        loc_dRH   = slider_with_box_float("Local Δ Relative humidity (%)",  -30.0, 30.0, 0.0, 0.5, key="loc_dRH")

        loc_infil = slider_with_box_int("Local Δ Urban infilling (%)", -100, 300, 0, 10, key="loc_infil")
        loc_ahem  = slider_with_box_int("Local Δ Anthropogenic heat (%)", -50, 300, 0, 10, key="loc_ahem")
    else:
        loc_bvol = loc_bldh = loc_pop = 0
        loc_dTout = loc_dRH = 0.0
        loc_infil = loc_ahem = 0

    # -----------------------------
    # WARD TOOLS
    # -----------------------------
    st.markdown("---")
    st.markdown("<h4 style='color:#7a0019;'>Ward tools (KMC)</h4>", unsafe_allow_html=True)
    st.caption("Select all pixel dots that fall within a ward polygon.")

    if WARD_LABEL_COL:
        ward_labels = wards_gdf[WARD_LABEL_COL].astype(str).fillna("Unknown").tolist()
    else:
        ward_labels = [f"Ward #{i}" for i in range(len(wards_gdf))]

    ward_choice = st.selectbox(
        "Pick a ward",
        options=list(range(len(wards_gdf))),
        index=int(st.session_state.ward_choice),
        format_func=lambda i: ward_labels[i],
        key="ward_choice",
    )

    ward_select_mode = st.radio(
        "Selection action",
        ["Replace selection with this ward", "Add this ward to selection"],
        index=0 if st.session_state.ward_select_mode.startswith("Replace") else 1,
        key="ward_select_mode",
    )

    if st.button("Select all pixels inside chosen ward", use_container_width=True):
        chosen_ward_idx = wards_gdf.index[int(ward_choice)]
        pg_list = pix_in_ward.loc[pix_in_ward["ward_idx"] == chosen_ward_idx, "pixel_group"].tolist()

        if ward_select_mode.startswith("Replace"):
            st.session_state.selected_pixels = []

        for pg in pg_list:
            if pg not in st.session_state.selected_pixels:
                st.session_state.selected_pixels.append(pg)

        st.success(f"Selected {len(pg_list)} pixels in that ward.")
        st.rerun()

# =========================================================
# BASELINE + CITY SCENARIO
# =========================================================
df = df0.copy()

# sanitize baseline (same as old)
df["RH_base"]    = np.clip(df["RH"].astype(float), 0, 100)
df["infil_base"] = np.clip(df["infil"].astype(float), 0, 1.0)
df["ahem_base"]  = np.clip(df["ahem"].astype(float), 0, None)

# ---- BASELINE IS FIXED NUMERICAL BASELINE ----
feature_map_base = {
    "dT_cool": "dT_cool",
    "RH": "RH_base",
    "bvol": "bvol",
    "bldh": "bldh",
    "pop": "pop",
    "infil": "infil_base",
    "ahem": "ahem_base",
}
X_base = np.column_stack([df[feature_map_base[f]].values for f in features])
df["ucd_base"] = model.predict(X_base)

# ---- CITY SCENARIO (CHANGES) ----
df["bvol_city"]  = df["bvol"]  * (1 + bvol_city / 100)
df["bldh_city"]  = df["bldh"]  * (1 + bldh_city / 100)
df["pop_city"]   = df["pop"]   * (1 + pop_city  / 100)

df["Tout_city"]  = df["Tout_C"].astype(float) + float(dTout_city)
df["RH_city"]    = np.clip(df["RH_base"] + float(dRH_city), 0, 100)

df["infil_city"] = np.clip(df["infil_base"] * (1 + infilling_city / 100), 0, 1.0)
df["ahem_city"]  = np.clip(df["ahem_base"]  * (1 + ahem_city / 100), 0, None)

# Setpoint (scenario only)
df["Tset_city"] = compute_tset_series(df, tset_mean, tset_rng, tset_mode, tset_seed)
df["dT_city"] = np.maximum(0.0, df["Tout_city"] - df["Tset_city"])

feature_map_city = {
    "dT_cool": "dT_city",
    "RH": "RH_city",
    "bvol": "bvol_city",
    "bldh": "bldh_city",
    "pop": "pop_city",
    "infil": "infil_city",
    "ahem": "ahem_city",
}
X_city = np.column_stack([df[feature_map_city[f]].values for f in features])
df["ucd_city"] = model.predict(X_city)

# start final as city
for v in ["bvol", "bldh", "pop", "RH", "infil", "ahem"]:
    df[f"{v}_final"] = df[f"{v}_city"]
df["Tout_final"] = df["Tout_city"]
df["Tset_final"] = df["Tset_city"]
df["dT_final"]   = df["dT_city"]
df["ucd_final"]  = df["ucd_city"].copy()

# =========================================================
# APPLY LOCAL OVERRIDES TO SELECTED PIXELS
# =========================================================
selected = set(st.session_state.selected_pixels)
if enable_local and len(selected) > 0:
    mask = df["pixel_group"].isin(selected)

    df.loc[mask, "bvol_final"] = df.loc[mask, "bvol_city"] * (1 + loc_bvol / 100)
    df.loc[mask, "bldh_final"] = df.loc[mask, "bldh_city"] * (1 + loc_bldh / 100)
    df.loc[mask, "pop_final"]  = df.loc[mask, "pop_city"]  * (1 + loc_pop  / 100)

    df.loc[mask, "Tout_final"] = df.loc[mask, "Tout_city"] + float(loc_dTout)
    df.loc[mask, "RH_final"]   = np.clip(df.loc[mask, "RH_city"] + float(loc_dRH), 0, 100)

    df.loc[mask, "infil_final"] = np.clip(df.loc[mask, "infil_city"] * (1 + loc_infil / 100), 0, 1.0)
    df.loc[mask, "ahem_final"]  = np.clip(df.loc[mask, "ahem_city"]  * (1 + loc_ahem  / 100), 0, None)

    df.loc[mask, "dT_final"] = np.maximum(0.0, df.loc[mask, "Tout_final"] - df.loc[mask, "Tset_final"])

    feature_map_final = {
        "dT_cool": "dT_final",
        "RH": "RH_final",
        "bvol": "bvol_final",
        "bldh": "bldh_final",
        "pop": "pop_final",
        "infil": "infil_final",
        "ahem": "ahem_final",
    }
    X_final = np.column_stack([df[feature_map_final[f]].values for f in features])
    df["ucd_final"] = model.predict(X_final)

# =========================================================
# CITY TOTALS
# =========================================================
df["MW_base"]  = df["ucd_base"]  * PIXEL_AREA / 1e6
df["MW_final"] = df["ucd_final"] * PIXEL_AREA / 1e6

city_MW_base  = float(df["MW_base"].sum())
city_MW_final = float(df["MW_final"].sum())
city_MW_delta = city_MW_final - city_MW_base

# =========================================================
# MAP (FINAL) + WARDS LAYER
# =========================================================
m = folium.Map(location=[df["lat"].mean(), df["lon"].mean()], zoom_start=11, tiles=None)

folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="ESRI Satellite",
).add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="ESRI Light Gray",
).add_to(m)

def ward_style(_feat):
    return {"fillOpacity": 0.05, "weight": 1.0}

ward_tooltip = None
if WARD_LABEL_COL:
    ward_tooltip = folium.GeoJsonTooltip(fields=[WARD_LABEL_COL], aliases=["Ward"], sticky=True)

folium.GeoJson(
    wards_gdf,
    name="KMC Wards",
    style_function=ward_style,
    tooltip=ward_tooltip,
).add_to(m)

# Highlight chosen ward
try:
    chosen_ward_idx = wards_gdf.index[int(st.session_state.ward_choice)]
    chosen_poly = wards_gdf.loc[[chosen_ward_idx]]
    folium.GeoJson(
        chosen_poly,
        name="Selected ward",
        style_function=lambda _f: {"fillOpacity": 0.12, "weight": 3.0},
    ).add_to(m)

    b = chosen_poly.total_bounds
    if np.all(np.isfinite(b)) and (b[2] > b[0]) and (b[3] > b[1]):
        m.fit_bounds([[b[1], b[0]], [b[3], b[2]]])
except Exception:
    pass

# UCD colormap (FINAL)
vals = df["ucd_final"].values
cmap = cm.LinearColormap(
    ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#800026"],
    vmin=float(np.nanmin(vals)),
    vmax=float(np.nanmax(vals)),
)
cmap.caption = "Predicted UCD (W m⁻²) — FINAL"

selected = set(st.session_state.selected_pixels)

for _, r in df.iterrows():
    is_sel = r["pixel_group"] in selected
    folium.CircleMarker(
        [float(r["lat"]), float(r["lon"])],
        radius=6 if is_sel else 4,
        fill=True,
        fill_color=cmap(float(r["ucd_final"])),
        color="#000000" if is_sel else cmap(float(r["ucd_final"])),
        weight=2.5 if is_sel else 1.0,
        fill_opacity=0.92 if is_sel else 0.85,
        tooltip=(
            f"Pixel {r['pixel_group']}<br>"
            f"UCD final: {r['ucd_final']:.2f} W m⁻²<br>"
            f"ΔUCD: {(r['ucd_final'] - r['ucd_base']):+.2f} W m⁻²"
        ),
    ).add_to(m)

cmap.add_to(m)
folium.LayerControl(collapsed=True).add_to(m)

# =========================================================
# LAYOUT
# =========================================================
c_map, c_info = st.columns([2.8, 1.2], vertical_alignment="top")

with c_map:
    st.markdown("### Map (click pixels; selected pixels get a black outline)")
    try:
        out = st_folium(m, use_container_width=True, height=720)
    except TypeError:
        out = st_folium(m, width=1600, height=720)

    # =========================================================
    # NEW: Bottom plot (same width) — Baseline vs Scenario (ALL pixels)
    # Baseline = teal blue, Scenario = maroon red
    # =========================================================
    st.markdown("### Baseline vs updated scenario (all pixels)")

    # Sort by baseline for a clean visual comparison
    dplot = df[["ucd_base", "ucd_final"]].copy()
    dplot = dplot.sort_values("ucd_base", ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(12, 3.6))
    ax = fig.add_subplot(111)

    ax.plot(dplot["ucd_base"].values, color="#0aa6b5", linewidth=2.0, label="Baseline (UCD)")
    ax.plot(dplot["ucd_final"].values, color="#7a0019", linewidth=2.0, label="Scenario (Predicted UCD)")

    ax.set_xlabel("Pixels (sorted by baseline)")
    ax.set_ylabel("UCD (W m⁻²)")
    ax.set_title("Citywide UCD comparison: baseline vs scenario prediction")
    ax.legend(loc="best")

    st.pyplot(fig, clear_figure=True)

with c_info:
    st.markdown("### City totals (baseline → final)")
    cA, cB = st.columns(2)
    cA.metric("Baseline total demand (MW)", f"{city_MW_base:,.1f}")
    cB.metric("Final total demand (MW)", f"{city_MW_final:,.1f}", delta=f"{city_MW_delta:+,.1f} MW")

    st.markdown("---")
    st.markdown("### Scenario summary")
    st.write(f"**Setpoint:** {tset_mean:.1f} ± {tset_rng:.1f} °C ({tset_mode}, seed={tset_seed})")
    st.write(f"**Outdoor ΔT (city):** {dTout_city:+.1f} °C")
    st.write(f"**RH Δ (city):** {dRH_city:+.1f} %")

    st.markdown("---")
    st.markdown("### Multi-pixel selection")
    if len(st.session_state.selected_pixels) == 0:
        st.info("No selected pixels yet. Click a pixel, then add it to selection (or use Ward tools).")
    else:
        sel_df = df[df["pixel_group"].isin(st.session_state.selected_pixels)].copy()
        sel_df["ΔUCD (W m⁻²)"] = sel_df["ucd_final"] - sel_df["ucd_base"]
        sel_df["ΔMW"] = sel_df["MW_final"] - sel_df["MW_base"]
        show = sel_df[["pixel_group", "ucd_base", "ucd_final", "ΔUCD (W m⁻²)", "MW_base", "MW_final", "ΔMW"]].copy()
        show = show.sort_values("ΔMW", ascending=True)
        st.dataframe(show, use_container_width=True, height=220)

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Clear selection", use_container_width=True):
                st.session_state.selected_pixels = []
                st.rerun()
        with b2:
            if st.button("Remove last selected", use_container_width=True):
                if len(st.session_state.selected_pixels) > 0:
                    st.session_state.selected_pixels = st.session_state.selected_pixels[:-1]
                    st.rerun()

# =========================================================
# PIXEL CLICK → DETAILS + 3D REPRESENTATION + ADD TO SELECTION
# =========================================================
def nearest_pixel_idx(df_, lat_, lon_):
    return ((df_["lat"] - lat_) ** 2 + (df_["lon"] - lon_) ** 2).idxmin()

if out and out.get("last_clicked"):
    lat = out["last_clicked"]["lat"]
    lon = out["last_clicked"]["lng"]
    idx = nearest_pixel_idx(df, lat, lon)
    row = df.loc[idx].copy()

    with c_info:
        st.markdown("---")
        st.subheader(f"Clicked pixel: {row['pixel_group']}")

        if st.button("Add this pixel to selection", use_container_width=True):
            pg = row["pixel_group"]
            if pg not in st.session_state.selected_pixels:
                st.session_state.selected_pixels.append(pg)
            st.rerun()

        d_ucd = float(row["ucd_final"] - row["ucd_base"])
        d_mw = float(row["MW_final"] - row["MW_base"])

        c1, c2 = st.columns(2)
        c1.metric("UCD (W m⁻²)", f"{row['ucd_final']:.2f}", delta=f"{d_ucd:+.2f}")
        c2.metric("Pixel demand (MW)", f"{row['MW_final']:.2f}", delta=f"{d_mw:+.2f} MW")

        st.caption("FINAL includes city-level + local overrides (if enabled & selected). Deltas are vs baseline (numerical baseline).")

        # Table (baseline vs final)
        footprint_base = float(row["infil_base"] * PIXEL_AREA)
        footprint_final = float(row["infil_final"] * PIXEL_AREA)

        height_base = float(row["bvol"] / max(footprint_base, 1.0))
        height_final = float(row["bvol_final"] / max(footprint_final, 1.0))

        table = pd.DataFrame(
            {
                "Variable": [
                    "Building volume (m³)",
                    "Urban infilling (fraction)",
                    "Effective built area (m²)",
                    "Effective mean height (m)",
                    "Population (people)",
                    "Outdoor temperature Tout (°C)",
                    "Indoor setpoint Tset (°C)",
                    "Cooling driver dT_cool (°C)",
                    "Relative humidity RH (%)",
                    "Anthropogenic heat (AHE)",
                    "UCD (W m⁻²)",
                ],
                "Baseline": [
                    row["bvol"],
                    row["infil_base"],
                    footprint_base,
                    height_base,
                    row["pop"],
                    row["Tout_C"],
                    row.get("Tset_C", np.nan),       # if present in parquet
                    row["dT_cool"],
                    row["RH_base"],
                    row["ahem_base"],
                    row["ucd_base"],
                ],
                "Final": [
                    row["bvol_final"],
                    row["infil_final"],
                    footprint_final,
                    height_final,
                    row["pop_final"],
                    row["Tout_final"],
                    row["Tset_final"],
                    row["dT_final"],
                    row["RH_final"],
                    row["ahem_final"],
                    row["ucd_final"],
                ],
                "Δ (Final−Base)": [
                    row["bvol_final"] - row["bvol"],
                    row["infil_final"] - row["infil_base"],
                    footprint_final - footprint_base,
                    height_final - height_base,
                    row["pop_final"] - row["pop"],
                    row["Tout_final"] - row["Tout_C"],
                    row["Tset_final"] - float(row.get("Tset_C", row["Tset_final"])),
                    row["dT_final"] - row["dT_cool"],
                    row["RH_final"] - row["RH_base"],
                    row["ahem_final"] - row["ahem_base"],
                    row["ucd_final"] - row["ucd_base"],
                ],
            }
        )
        st.dataframe(table, use_container_width=True, height=320)

        # 3D isometric-like representation (unchanged)
        st.markdown("### 3D pixel representation (isometric)")

        base = 1000.0
        side = base * float(np.sqrt(max(row["infil_final"], 0.0)))
        h_vis = min(height_final / 10.0, 800.0)

        fig = plt.figure(figsize=(4.2, 4.2))
        ax = fig.add_subplot(111, projection="3d")

        ax.bar3d(0, 0, 0, side, side, h_vis, shade=True)

        ax.set_xlim(0, base)
        ax.set_ylim(0, base)
        ax.set_zlim(0, max(h_vis * 1.2, 1))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title("1 km pixel — infilling footprint + volume-derived height (FINAL)")

        st.pyplot(fig)
else:
    with c_info:
        st.markdown("---")
        st.info("Click a pixel on the map to view its full details and 3D representation.")
