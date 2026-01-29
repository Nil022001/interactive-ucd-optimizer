import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
import branca.colormap as cm

st.set_page_config(page_title="Kolkata UCD Optimizer (1km)", layout="wide")

DATA_PARQUET = "data/pixels_baseline_2019.parquet"
MODEL_FILE = "models/ucd_model.joblib"

@st.cache_data
def load_pixels():
    return pd.read_parquet(DATA_PARQUET)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

st.title("Interactive Urban Cooling Demand (UCD) — Kolkata (1 km × 1 km)")
st.caption("Basemap + 1 km pixels colored by UCD (yellow→red). Click a pixel to inspect it.")

df = load_pixels()
bundle = load_model()

model = bundle["model"]
features = bundle["features"]

# Predict baseline UCD using the saved model
df["ucd_pred"] = model.predict(df[features].values)

# Sidebar controls
st.sidebar.header("Controls")
color_mode = st.sidebar.selectbox(
    "Color pixels by",
    ["Predicted UCD (W m⁻²)", "Simulated UCD (y_ucd_wm2)"],
    index=0
)

val_col = "ucd_pred" if color_mode.startswith("Predicted") else "y_ucd_wm2"

# Build a robust yellow->red colormap using percentiles (prevents outliers dominating)
vals = df[val_col].astype(float).values
vmin = float(np.nanpercentile(vals, 2))
vmax = float(np.nanpercentile(vals, 98))
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals)) if float(np.nanmax(vals)) > vmin else (vmin + 1.0)

colormap = cm.LinearColormap(
    colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],  # yellow -> red
    vmin=vmin,
    vmax=vmax
)
colormap.caption = f"{val_col} (W m⁻²)"

# Map center
center_lat = float(df["lat"].mean())
center_lon = float(df["lon"].mean())

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

# Add pixels as colored circle markers
for _, r in df.iterrows():
    val = float(r[val_col])
    color = colormap(val)

    tooltip = (
        f"pixel: {r['pixel_group']}<br>"
        f"{val_col}: {val:.3f} W/m²<br>"
        f"T2: {r['T2']:.2f} K | RH: {r['RH']:.2f}%<br>"
        f"bvol: {r['bvol']:.1f} | bldh: {r['bldh']:.2f}<br>"
        f"pop: {r['pop']:.0f} | infil: {r['infil']:.3f}<br>"
        f"ahem: {r['ahem']:.1f}"
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

# Add legend/colorbar
colormap.add_to(m)

st.subheader("Kolkata 1 km pixels")
out = st_folium(m, width=1200, height=650)

st.subheader("Clicked pixel (nearest)")

if out and out.get("last_clicked"):
    click_lat = out["last_clicked"]["lat"]
    click_lon = out["last_clicked"]["lng"]

    # nearest pixel by squared distance
    d = (df["lat"] - click_lat) ** 2 + (df["lon"] - click_lon) ** 2
    idx = d.idxmin()
    row = df.loc[idx].copy()

    st.write("Nearest pixel_group:", row["pixel_group"])
    st.dataframe(row.to_frame("value"))
else:
    st.info("Click on a pixel marker to see the nearest pixel details.")
