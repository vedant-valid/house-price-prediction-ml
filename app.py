"""
app.py
------
Streamlit UI for the House Price Prediction project.
Run with:  streamlit run app.py
"""

import os
import pickle
import streamlit as st
import pandas as pd

from inference import predict_property

# ------------------------------------------------------------------ #
# PAGE CONFIG
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ------------------------------------------------------------------ #
# CUSTOM CSS — premium look
# ------------------------------------------------------------------ #
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #63b3ed;
        margin: 0.35rem 0 0 0;
        font-weight: 400;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(99, 179, 237, 0.25);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(99,179,237,0.15);
    }
    div[data-testid="stMetricLabel"] {
        color: #90cdf4 !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #68d391 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid rgba(99,179,237,0.15);
    }
    section[data-testid="stSidebar"] label {
        color: #a0aec0 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    section[data-testid="stSidebar"] .stSlider > div {
        color: #e2e8f0 !important;
    }

    /* Section headers inside sidebar */
    .sidebar-section {
        color: #63b3ed;
        font-weight: 600;
        font-size: 0.95rem;
        margin: 1.2rem 0 0.5rem 0;
        padding-bottom: 4px;
        border-bottom: 1px solid rgba(99,179,237,0.2);
        letter-spacing: 0.02em;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(90deg, #3182ce, #2b6cb0);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        width: 100%;
        cursor: pointer;
        transition: all 0.25s ease;
        letter-spacing: 0.03em;
        box-shadow: 0 4px 14px rgba(49,130,206,0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2b6cb0, #2c5282);
        box-shadow: 0 6px 20px rgba(49,130,206,0.55);
        transform: translateY(-1px);
    }

    /* Info / result cards */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .result-card h4 {
        color: #63b3ed;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.6rem;
    }
    .result-card p, .result-card li {
        color: #cbd5e0;
        font-size: 0.92rem;
        line-height: 1.65;
    }
    .driver-badge {
        display: inline-block;
        background: rgba(49,130,206,0.2);
        border: 1px solid rgba(49,130,206,0.4);
        color: #90cdf4;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px 3px;
    }

    /* Divider */
    hr { border-color: rgba(99,179,237,0.15) !important; }

    /* Streamlit image */
    img { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# LOAD COLUMN REFERENCE (for city / statezip dropdowns)
# ------------------------------------------------------------------ #
@st.cache_resource
def load_column_reference():
    try:
        with open("models/column_reference.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return []

column_reference = load_column_reference()

def _extract_options(prefix: str) -> list:
    """Pull sorted human-readable values from OHE column names."""
    strip = len(prefix)
    options = sorted(
        col[strip:] for col in column_reference if col.startswith(prefix)
    )
    return options if options else ["Unknown"]

city_options    = _extract_options("city_")
statezip_options = _extract_options("statezip_")

# ------------------------------------------------------------------ #
# PAGE HEADER
# ------------------------------------------------------------------ #
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🏠 House Price Predictor</p>
  <p class="hero-subtitle">King County, Washington — ML-Powered Valuation Tool</p>
</div>
""", unsafe_allow_html=True)
st.divider()


# ------------------------------------------------------------------ #
# SIDEBAR — INPUT FORM
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown(
        "<p style='color:#63b3ed;font-size:1.1rem;font-weight:700;"
        "margin-bottom:0.5rem;'>Property Details</p>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Size & Structure ────────────────────────────────
    st.markdown('<p class="sidebar-section">📐 Size & Structure</p>', unsafe_allow_html=True)

    sqft_living   = st.number_input("Living Area (sqft)",   min_value=500,  max_value=10000, value=1800, step=50)
    sqft_lot      = st.number_input("Lot Size (sqft)",      min_value=1000, max_value=50000, value=5000, step=100)
    sqft_above    = st.number_input("Above Ground (sqft)",  min_value=500,  max_value=10000, value=1800, step=50)
    sqft_basement = st.number_input("Basement (sqft)",      min_value=0,    max_value=3000,  value=0,    step=50)
    bedrooms      = st.slider("Bedrooms",   min_value=1, max_value=10, value=3)
    bathrooms     = st.slider("Bathrooms",  min_value=1.0, max_value=8.0, value=2.0, step=0.5)
    floors        = st.selectbox("Floors", options=[1.0, 1.5, 2.0, 2.5, 3.0], index=0)

    # ── Section 2: Location ────────────────────────────────────────
    st.markdown('<p class="sidebar-section">📍 Location</p>', unsafe_allow_html=True)

    default_city = city_options.index("Seattle") if "Seattle" in city_options else 0
    city = st.selectbox("City", options=city_options, index=default_city)

    default_zip = statezip_options.index("WA 98103") if "WA 98103" in statezip_options else 0
    statezip = st.selectbox("State + ZIP", options=statezip_options, index=default_zip)

    # ── Section 3: Property Features ──────────────────────────────
    st.markdown('<p class="sidebar-section">🏗️ Property Features</p>', unsafe_allow_html=True)

    yr_built  = st.number_input("Year Built", min_value=1900, max_value=2014, value=1990, step=1)

    condition_map = {1: "1 — Poor", 2: "2 — Fair", 3: "3 — Average", 4: "4 — Good", 5: "5 — Excellent"}
    condition_label = st.selectbox("Condition", options=list(condition_map.values()), index=2)
    condition = int(condition_label.split(" — ")[0])

    view_map = {0: "0 — None", 1: "1 — Fair", 2: "2 — Average", 3: "3 — Good", 4: "4 — Excellent"}
    view_label = st.selectbox("View Quality", options=list(view_map.values()), index=0)
    view = int(view_label.split(" — ")[0])

    waterfront = 1 if st.checkbox("🌊 Waterfront Property", value=False) else 0

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict Price", type="primary", use_container_width=True)


# ------------------------------------------------------------------ #
# MAIN PANEL — placeholder or results
# ------------------------------------------------------------------ #
if not predict_clicked:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem; color:#4a5568;">
        <p style="font-size:3.5rem; margin:0;">🏡</p>
        <p style="font-size:1.15rem; color:#718096; margin-top:1rem;">
            Fill in the property details in the sidebar and click
            <strong style="color:#63b3ed;">🔍 Predict Price</strong> to get a valuation.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    user_input = {
        "sqft_living":   sqft_living,
        "sqft_lot":      sqft_lot,
        "sqft_above":    sqft_above,
        "sqft_basement": sqft_basement,
        "bedrooms":      bedrooms,
        "bathrooms":     bathrooms,
        "floors":        floors,
        "waterfront":    waterfront,
        "view":          view,
        "condition":     condition,
        "yr_built":      yr_built,
        "city":          city,
        "statezip":      statezip,
        "year_sold":     2014,
        "month_sold":    6,
    }

    with st.spinner("Analyzing property..."):
        result = predict_property(user_input)

    pred   = result["predicted_price"]
    low    = result["price_range"]["low"]
    high   = result["price_range"]["high"]
    conf   = result["confidence"]
    model  = result["model_used"]
    drivers = result["top_price_drivers"]

    # ── ROW 1 — Three metric cards ─────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Predicted Price",
            value="${:,.0f}".format(pred),
        )
    with col2:
        st.metric(
            label="Price Range (±10%)",
            value="${:,.0f} – ${:,.0f}".format(low, high),
        )
    with col3:
        r2_display = "0.733" if model == "Linear Regression" else "—"
        st.metric(
            label="Confidence",
            value=conf,
            delta="R² = {} · {}".format(r2_display, model),
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 2 — Top drivers + property summary ─────────────────────
    col_d, col_s = st.columns([1, 1])

    with col_d:
        driver_badges = " ".join(
            '<span class="driver-badge">{}</span>'.format(d) for d in drivers
        )
        st.markdown("""
        <div class="result-card">
            <h4>🎯 Top Price Drivers</h4>
            <p>{}</p>
        </div>
        """.format(driver_badges), unsafe_allow_html=True)

    with col_s:
        property_age = 2014 - yr_built
        wf_label = "Yes ✅" if waterfront else "No"
        st.markdown("""
        <div class="result-card">
            <h4>📋 Property Summary</h4>
            <p>
                <b>City:</b> {city} &nbsp;|&nbsp; <b>ZIP:</b> {zip}<br>
                <b>Living Area:</b> {:,} sqft &nbsp;|&nbsp; <b>Lot:</b> {:,} sqft<br>
                <b>Beds:</b> {} &nbsp;|&nbsp; <b>Baths:</b> {} &nbsp;|&nbsp; <b>Floors:</b> {}<br>
                <b>Property Age:</b> {} yrs &nbsp;|&nbsp; <b>Waterfront:</b> {}
            </p>
        </div>
        """.format(
            sqft_living, sqft_lot,
            bedrooms, bathrooms, floors,
            property_age, wf_label,
            city=city, zip=statezip,
        ), unsafe_allow_html=True)

    st.divider()

    # ── ROW 3 — Feature importance chart ───────────────────────────
    chart_path = os.path.join("assets", "feature_importance.png")
    if os.path.exists(chart_path):
        st.subheader("📊 Model Feature Importance")
        st.image(chart_path, width='stretch')
    else:
        st.info("Feature importance chart not found. Run `run_training.py` to generate it.")

    st.divider()

    # ── ROW 4 — Model selection summary ────────────────────────────
    st.subheader("📝 Model Selection Analysis")
    try:
        with open("assets/model_summary.txt", "r") as _f:
            st.markdown(_f.read())
    except FileNotFoundError:
        st.info("Run `run_training.py` once to generate the model summary.")

    st.divider()

    # ── ROW 5 — Feature insights ───────────────────────────────────
    st.subheader("💡 Feature Insights")
    try:
        with open("assets/feature_insights.txt", "r") as _f:
            raw = _f.read()
        for bullet in raw.split("\n"):
            if bullet.strip().startswith("*"):
                st.markdown("- " + bullet.strip()[1:].strip())
            elif bullet.strip() and not bullet.strip().startswith("---"):
                st.markdown(bullet)
    except FileNotFoundError:
        st.info("Run `run_training.py` once to generate the feature insights.")

    st.divider()

    # ── Footer ─────────────────────────────────────────────────────
    st.markdown("""
    <p style="text-align:center; color:#4a5568; font-size:0.8rem; margin-top:1rem;">
        House Price Predictor · King County, WA · Project 9 — Milestone 1<br>
        Model: <strong style="color:#63b3ed;">{}</strong> · 
        Confidence: <strong style="color:#68d391;">{}</strong>
    </p>
    """.format(model, conf), unsafe_allow_html=True)
