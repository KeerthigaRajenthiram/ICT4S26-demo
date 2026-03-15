import sys
from pathlib import Path

# Add repo root (experiment-factory) to PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[3]  # app.py -> streamlit_app -> GUI -> apps -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))



import streamlit as st

from ui.styles import apply_global_styles
from ui.state import reset_session
from ui.components import (
    render_home_page,
    render_explore_page,
    render_recommend_page,
    render_about_page,
)
from data.access import db_health


st.set_page_config(
    page_title="AutoML KB Agent",
    page_icon="🤖",
    layout="wide",
)

apply_global_styles()

st.sidebar.title("🤖 AutoML KB Agent")
st.sidebar.caption("Intent-aware model selection from historical AutoML runs")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔎 Explore", "🤖 Recommend", "ℹ️ About"],
    index=0,
)

if st.sidebar.button("Reset session"):
    reset_session()
    st.rerun()

healthy, msg = db_health()
st.sidebar.markdown(f"**DB:** {'✅ Connected' if healthy else '❌ Not connected'}")
st.sidebar.caption(msg)

if page == "🏠 Home":
    render_home_page(healthy)
elif page == "🔎 Explore":
    render_explore_page(healthy)
elif page == "🤖 Recommend":
    render_recommend_page(healthy)
elif page == "ℹ️ About":
    render_about_page()
