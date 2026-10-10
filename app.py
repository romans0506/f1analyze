"""F1 Analyze — run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="F1 Analyze", page_icon="🏁", layout="wide")

MODES = {"simple": "🟢 Simple", "expert": "🔬 Expert"}
# The session remembers the mode across pages; ?mode=expert in a shared link sets it.
current = st.session_state.get("mode") or st.query_params.get("mode", "simple")
if current not in MODES:
    current = "simple"

with st.sidebar:
    picked = st.segmented_control("Mode", list(MODES), format_func=MODES.get, default=current,
                                  help="Simple: the story of the race, explained. Expert: every stat and chart.")
    mode = picked or current  # clicking the active option deselects it; keep the mode
    st.session_state["mode"] = mode
    st.query_params["mode"] = mode

replay = st.Page("views/replay.py", title="Race replay", icon="📺", url_path="replay")

if mode == "simple":
    pages = [
        st.Page("views/simple/weekend.py", title="Race weekend", icon="🏁", default=True),
        replay,
        st.Page("views/simple/recap.py", title="Last race recap", icon="📰", url_path="recap"),
        st.Page("views/simple/championship.py", title="Championship", icon="🏆", url_path="championship"),
    ]
else:
    pages = [
        st.Page("views/prediction.py", title="Next GP prediction", icon="🔮", default=True),
        st.Page("views/season.py", title="Season overview", icon="📊", url_path="season"),
        st.Page("views/grand_prix.py", title="Grand Prix analysis", icon="🏎️", url_path="grand-prix"),
        replay,
        st.Page("views/drivers.py", title="Driver head-to-head", icon="⚔️", url_path="head-to-head"),
        st.Page("views/model_report.py", title="Model accuracy", icon="📐", url_path="model"),
    ]

st.navigation(pages).run()
