import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import branca.colormap as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="Kolkata UCD Optimizer (1 km)", layout="wide")

DATA_PARQUET = "data/pixels_baseline_2019.parquet"
MODEL_FILE = "models/ucd_model.joblib"

PIXEL_AREA = 1_000_000.0  # m² (1 km × 1 km)

# =========================================================
# STREAMLIT-SAFE SLIDER + BOX (single canonical key; unique widget keys)
# =========================================================
def slider_with_box_int(label, minv, maxv, default, step, key):
    """
    Slider + number_input that stay synced.
    Uses ONE canonical state: st.session_state[key]
    But each widget has its own unique key to avoid StreamlitDuplicateElementKey.
    """
    k_slider = f"{key}__slider"
    k_box = f"{key}__box"

    # initialize canonical + widget states once
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
            minv,
            maxv,
            value=int(st.session_state[key]),
            step=step,
            key=k_slider,
            on_change=_from_slider,
        )
    with c2:
        st.number_input(
            " ",
            min_value=minv,
            max_value=maxv,
            value=int(st.session_state[key]),
            step=step,
            key=k_box,
            on_change=_from_box,
        )

    return int(st.session_state[key])


def slider_with_box_float(label, minv, maxv, default, step, key):
    """
    Float version (same syncing logic).
    """
    k_slider = f"{key}__slider"
    k_box = f"{key}__box"

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
            float(minv),
            float(maxv),
            value=float(st.session_state[key]),
            step=float(step),
            key=k_slider,
            on_change=_from_slider,
        )
    with c2:
        st.number_input(
            " ",
            min_value=float(minv),
            max_value=float(maxv),
            value=float(st.session_state[key]),
            step=float(step),
            format="%.3f",
            key=k_box,
            on_change=_from_box,
        )

    return float(st.session_state[key])


# =========================================================
# LOAD
# =========================================================
@st.cache_data
def load_pixels():
    return pd.read_parquet(DATA_PARQUET)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


df0 = load_pixels()
bundle = load_model()
model = bundle["model"]
features = bundle["features"]

# session state for multi-pixel selection
if "selected_pixels" not in st.session_state:
    st.session_state.selected_pixels = []  # list of pixel_group

st.title("Urban Cooling Demand Optimizer — Kolkata (1 km × 1 km)")
st.caption("City-scale policy → Local project → City-wide feedback (W m⁻²)")

# =========================================================
# SIDEBAR — CITY LEVEL (MAROON)
# =========================================================
st.sidebar.markdown("<h3 style='color:#7a0019;'>City-level scenario</h3>", unsafe_allow_html=True)

bvol_city = slider_with_box_int("Δ Building volume (%)", -50, 200, 0, 5, key="city_bvol")
bldh_city = slider_with_box_int("Δ Building height (%)", -50, 200, 0, 5, key="city_bldh")
pop_city = slider_with_box_int("Δ Population (%)", -50, 200, 0, 5, key="city_pop")

dT2_city = slider_with_box_float("Δ Air temperature (K)", -5.0, 8.0, 0.0, 0.1, key="city_dT2")
dRH_city = slider_with_box_float("Δ Relative humidity (%)", -30.0, 30.0, 0.0, 0.5, key="city_dRH")

infilling_city = slider_with_box_int("Δ Urban infilling (%)", -100, 300, 0, 10, key="city_infil")
ahem_city = slider_with_box_int("Δ Anthropogenic heat (%)", -50, 300, 0, 10, key="city_ahem")

st.sidebar.markdown("---")
st.sidebar.markdown("<h4 style='color:#7a0019;'>Local project overrides (multi-pixel)</h4>", unsafe_allow_html=True)
st.sidebar.caption("Applies ONLY to the selected pixels (on top of city-level scenario).")

enable_local = st.sidebar.checkbox("Enable local overrides", value=True, key="enable_local")

if enable_local:
    loc_bvol = slider_with_box_int("Local Δ Building volume (%)", -80, 200, 0, 5, key="loc_bvol")
    loc_bldh = slider_with_box_int("Local Δ Building height (%)", -80, 200, 0, 5, key="loc_bldh")
    loc_pop = slider_with_box_int("Local Δ Population (%)", -80, 200, 0, 5, key="loc_pop")

    loc_dT2 = slider_with_box_float("Local Δ Air temperature (K)", -5.0, 8.0, 0.0, 0.1, key="loc_dT2")
    loc_dRH = slider_with_box_float("Local Δ Relative humidity (%)", -30.0, 30.0, 0.0, 0.5, key="loc_dRH")

    loc_infil = slider_with_box_int("Local Δ Urban infilling (%)", -100, 300, 0, 10, key="loc_infil")
    loc_ahem = slider_with_box_int("Local Δ Anthropogenic heat (%)", -50, 300, 0, 10, key="loc_ahem")
else:
    loc_bvol = loc_bldh = loc_pop = 0
    loc_dT2 = loc_dRH = 0.0
    loc_infil = loc_ahem = 0

# =========================================================
# BASELINE + CITY SCENARIO
# =========================================================
df = df0.copy()

# baseline prediction
df["ucd_base"] = model.predict(df[features].values)

# apply city-level adjustments
df["bvol_city"] = df["bvol"] * (1 + bvol_city / 100)
df["bldh_city"] = df["bldh"] * (1 + bldh_city / 100)
df["pop_city"] = df["pop"] * (1 + pop_city / 100)
df["T2_city"] = df["T2"] + dT2_city
df["RH_city"] = np.clip(df["RH"] + dRH_city, 0, 100)
df["infil_city"] = np.clip(df["infil"] * (1 + infilling_city / 100), 0, 1.0)
df["ahem_city"] = np.clip(df["ahem"] * (1 + ahem_city / 100), 0, None)

feature_map_city = {
    "T2": "T2_city",
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
for v in ["bvol", "bldh", "pop", "T2", "RH", "infil", "ahem"]:
    df[f"{v}_final"] = df[f"{v}_city"]
df["ucd_final"] = df["ucd_city"].copy()

# =========================================================
# APPLY LOCAL OVERRIDES TO SELECTED PIXELS
# =========================================================
selected = set(st.session_state.selected_pixels)
if enable_local and len(selected) > 0:
    mask = df["pixel_group"].isin(selected)

    df.loc[mask, "bvol_final"] = df.loc[mask, "bvol_city"] * (1 + loc_bvol / 100)
    df.loc[mask, "bldh_final"] = df.loc[mask, "bldh_city"] * (1 + loc_bldh / 100)
    df.loc[mask, "pop_final"] = df.loc[mask, "pop_city"] * (1 + loc_pop / 100)
    df.loc[mask, "T2_final"] = df.loc[mask, "T2_city"] + loc_dT2
    df.loc[mask, "RH_final"] = np.clip(df.loc[mask, "RH_city"] + loc_dRH, 0, 100)
    df.loc[mask, "infil_final"] = np.clip(df.loc[mask, "infil_city"] * (1 + loc_infil / 100), 0, 1.0)
    df.loc[mask, "ahem_final"] = np.clip(df.loc[mask, "ahem_city"] * (1 + loc_ahem / 100), 0, None)

    feature_map_final = {
        "T2": "T2_final",
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
df["MW_base"] = df["ucd_base"] * PIXEL_AREA / 1e6
df["MW_final"] = df["ucd_final"] * PIXEL_AREA / 1e6

city_MW_base = float(df["MW_base"].sum())
city_MW_final = float(df["MW_final"].sum())
city_MW_delta = city_MW_final - city_MW_base

# =========================================================
# MAP (FINAL)
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
        [r["lat"], r["lon"]],
        radius=5 if is_sel else 4,
        fill=True,
        fill_color=cmap(r["ucd_final"]),
        color="#000000" if is_sel else cmap(r["ucd_final"]),
        weight=2.5 if is_sel else 1.0,
        fill_opacity=0.90 if is_sel else 0.85,
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
c_map, c_info = st.columns([2.6, 1.2], vertical_alignment="top")

with c_map:
    st.markdown("### Map (click pixels; selected pixels get a black outline)")
    out = st_folium(m, width=1150, height=620)

with c_info:
    st.markdown("### City totals (baseline → final)")
    cA, cB = st.columns(2)
    cA.metric("Baseline total demand (MW)", f"{city_MW_base:,.1f}")
    cB.metric("Final total demand (MW)", f"{city_MW_final:,.1f}", delta=f"{city_MW_delta:+,.1f} MW")

    st.markdown("---")
    st.markdown("### Multi-pixel selection")
    if len(st.session_state.selected_pixels) == 0:
        st.info("No selected pixels yet. Click a pixel, then add it to selection.")
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

feature_cols_final = {
    "T2": "T2_final",
    "RH": "RH_final",
    "bvol": "bvol_final",
    "bldh": "bldh_final",
    "pop": "pop_final",
    "infil": "infil_final",
    "ahem": "ahem_final",
}
feature_cols_final_list = [feature_cols_final[f] for f in features]

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

        st.caption("FINAL includes city-level + local overrides (if enabled & selected). Deltas are vs baseline (2019).")

        # Table (baseline vs final)
        footprint_base = float(row["infil"] * PIXEL_AREA)
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
                    "Air temperature T2 (K)",
                    "Relative humidity RH (%)",
                    "Anthropogenic heat (AHE)",
                    "UCD (W m⁻²)",
                ],
                "Baseline": [
                    row["bvol"],
                    row["infil"],
                    footprint_base,
                    height_base,
                    row["pop"],
                    row["T2"],
                    row["RH"],
                    row["ahem"],
                    row["ucd_base"],
                ],
                "Final": [
                    row["bvol_final"],
                    row["infil_final"],
                    footprint_final,
                    height_final,
                    row["pop_final"],
                    row["T2_final"],
                    row["RH_final"],
                    row["ahem_final"],
                    row["ucd_final"],
                ],
                "Δ (Final−Base)": [
                    row["bvol_final"] - row["bvol"],
                    row["infil_final"] - row["infil"],
                    footprint_final - footprint_base,
                    height_final - height_base,
                    row["pop_final"] - row["pop"],
                    row["T2_final"] - row["T2"],
                    row["RH_final"] - row["RH"],
                    row["ahem_final"] - row["ahem"],
                    row["ucd_final"] - row["ucd_base"],
                ],
            }
        )
        st.dataframe(table, use_container_width=True, height=300)

        # 3D isometric-like representation
        st.markdown("### 3D pixel representation (isometric)")

        base = 1000.0  # visual base length (m)
        side = base * float(np.sqrt(max(row["infil_final"], 0.0)))  # area proportional
        h_vis = min(height_final / 10.0, 800.0)  # visual scaling

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
