import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import branca.colormap as cm

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Kolkata UCD Optimizer (1 km)",
    layout="wide"
)

DATA_PARQUET = "data/pixels_baseline_2019.parquet"
MODEL_FILE   = "models/ucd_model.joblib"

@st.cache_data
def load_pixels():
    return pd.read_parquet(DATA_PARQUET)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

# =========================================================
# LOAD DATA + MODEL
# =========================================================
df0     = load_pixels()
bundle  = load_model()
model   = bundle["model"]
features = bundle["features"]

st.title("Urban Cooling Demand Optimizer — Kolkata (1 km × 1 km)")
st.caption("City-scale policy → Local project → City-wide feedback (W m⁻²)")

# =========================================================
# SIDEBAR — CITY LEVEL (MAROON)
# =========================================================
st.sidebar.markdown(
    "<h3 style='color:#7a0019;'>City-level scenario</h3>",
    unsafe_allow_html=True
)

bvol_city = st.sidebar.slider("Δ Building volume (%)", -50, 200, 0, 5)
bldh_city = st.sidebar.slider("Δ Building height (%)", -50, 200, 0, 5)
pop_city  = st.sidebar.slider("Δ Population (%)", -50, 200, 0, 5)

dT2_city  = st.sidebar.slider("Δ Air temperature (K)", -5.0, 8.0, 0.0, 0.1)
dRH_city  = st.sidebar.slider("Δ Relative humidity (%)", -30.0, 30.0, 0.0, 0.5)

infilling_city = st.sidebar.slider("Δ Urban infilling (%)", -100, 300, 0, 10)
ahem_city      = st.sidebar.slider("Δ Anthropogenic heat (%)", -50, 300, 0, 10)

# =========================================================
# BASELINE + CITY SCENARIO
# =========================================================
df = df0.copy()

df["ucd_base"] = model.predict(df[features].values)

df["bvol_adj"]  = df["bvol"]  * (1 + bvol_city/100)
df["bldh_adj"]  = df["bldh"]  * (1 + bldh_city/100)
df["pop_adj"]   = df["pop"]   * (1 + pop_city/100)
df["T2_adj"]    = df["T2"]    + dT2_city
df["RH_adj"]    = np.clip(df["RH"] + dRH_city, 0, 100)
df["infil_adj"] = np.clip(df["infil"] * (1 + infilling_city/100), 0, None)
df["ahem_adj"]  = np.clip(df["ahem"]  * (1 + ahem_city/100),  0, None)

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
df["ucd_city"] = model.predict(X_city)

city_base_mean = df["ucd_base"].mean()
city_city_mean = df["ucd_city"].mean()

# =========================================================
# MAP (FULL BLEED)
# =========================================================
m = folium.Map(
    location=[df["lat"].mean(), df["lon"].mean()],
    zoom_start=11,
    tiles=None
)

folium.TileLayer("OpenStreetMap", name="OSM").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="ESRI Satellite"
).add_to(m)

vals = df["ucd_city"].values
cmap = cm.LinearColormap(
    ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#800026"],
    vmin=float(vals.min()),
    vmax=float(vals.max())
)
cmap.caption = "Predicted UCD (W m⁻²)"

for _, r in df.iterrows():
    folium.CircleMarker(
        [r["lat"], r["lon"]],
        radius=4,
        fill=True,
        fill_color=cmap(r["ucd_city"]),
        color=cmap(r["ucd_city"]),
        fill_opacity=0.85,
        tooltip=(
            f"Pixel {r['pixel_group']}<br>"
            f"Baseline: {r['ucd_base']:.2f} W m⁻²<br>"
            f"City scenario: {r['ucd_city']:.2f} W m⁻²"
        )
    ).add_to(m)

cmap.add_to(m)
folium.LayerControl(collapsed=True).add_to(m)

# =========================================================
# LAYOUT (NO SCROLL)
# =========================================================
c_map, c_info = st.columns([2.6, 1.2])

with c_map:
    out = st_folium(m, width=1150, height=620)

# =========================================================
# PIXEL SELECTION
# =========================================================
pixel_selected = False
if out and out.get("last_clicked"):
    lat = out["last_clicked"]["lat"]
    lon = out["last_clicked"]["lng"]
    idx = ((df["lat"] - lat)**2 + (df["lon"] - lon)**2).idxmin()
    row = df.loc[idx].copy()
    pixel_selected = True

# =========================================================
# SIDEBAR — PIXEL LEVEL (PRUSSIAN BLUE)
# =========================================================
if pixel_selected:
    st.sidebar.markdown(
        "<h3 style='color:#003153;'>Pixel-level intervention</h3>",
        unsafe_allow_html=True
    )

    bvol_px      = st.sidebar.slider("Pixel Δ building volume (%)", -50, 200, 0, 5)
    bldh_px      = st.sidebar.slider("Pixel Δ building height (%)", -50, 200, 0, 5)
    infilling_px = st.sidebar.slider("Pixel Δ infilling (%)", -100, 300, 0, 10)
    ahem_px      = st.sidebar.slider("Pixel Δ anthropogenic heat (%)", -50, 300, 0, 10)

    row["bvol_adj"]  *= (1 + bvol_px/100)
    row["bldh_adj"]  *= (1 + bldh_px/100)
    row["infil_adj"] *= (1 + infilling_px/100)
    row["ahem_adj"]  *= (1 + ahem_px/100)

    X_px = np.array([[row[feature_map[f]] for f in features]])
    ucd_px_new = model.predict(X_px)[0]

    df_after = df.copy()
    df_after.loc[idx, "ucd_city"] = ucd_px_new
    city_after_mean = df_after["ucd_city"].mean()

    # -----------------------------------------------------
    # LOCAL COMPENSATION LOGIC (bvol ↔ infilling)
    # -----------------------------------------------------
    eps = 0.05  # 5% perturbation

    def predict_row(r):
        X = np.array([[r[feature_map[f]] for f in features]])
        return model.predict(X)[0]

    r_bvol = row.copy()
    r_bvol["bvol_adj"] *= (1 + eps)
    dU_dbvol = (predict_row(r_bvol) - ucd_px_new) / (eps * row["bvol_adj"])

    r_inf = row.copy()
    r_inf["infil_adj"] *= (1 + eps)
    dU_dinf = (predict_row(r_inf) - ucd_px_new) / (eps * row["infil_adj"])

    compensation_text = None
    if abs(dU_dinf) > 1e-6:
        infil_needed = -(dU_dbvol / dU_dinf) * (bvol_px / 100)
        compensation_text = (
            f"To offset **+{bvol_px:.0f}% building volume**, "
            f"≈ **{infil_needed*100:+.1f}% infilling change** "
            f"is required locally."
        )

# =========================================================
# RIGHT PANEL — RESULTS + TABLE
# =========================================================
with c_info:
    st.subheader("City-wide impact")

    st.metric("Baseline mean UCD (W m⁻²)", f"{city_base_mean:.2f}")

    st.metric(
        "City-scenario mean UCD (W m⁻²)",
        f"{city_city_mean:.2f}",
        delta=f"{city_city_mean - city_base_mean:+.2f}"
    )

    if pixel_selected:
        st.metric(
            "City mean after local project (W m⁻²)",
            f"{city_after_mean:.2f}",
            delta=f"{city_after_mean - city_city_mean:+.4f}"
        )

        st.markdown("---")
        st.subheader(f"Selected pixel: {row['pixel_group']}")

        pixel_table = pd.DataFrame({
            "Variable": [
                "Air temperature (K)",
                "Relative humidity (%)",
                "Population",
                "Building volume",
                "Building height",
                "Urban infilling",
                "Anthropogenic heat",
                "UCD (city scenario, W m⁻²)",
                "UCD (after project, W m⁻²)",
            ],
            "Value": [
                row["T2_adj"],
                row["RH_adj"],
                row["pop_adj"],
                row["bvol_adj"],
                row["bldh_adj"],
                row["infil_adj"],
                row["ahem_adj"],
                row["ucd_city"],
                ucd_px_new,
            ],
        })

        st.dataframe(pixel_table, use_container_width=True)

        if compensation_text:
            st.info(compensation_text)
