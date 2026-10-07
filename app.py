"""F1 Analyze — run with:  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="F1 Analyze", page_icon="🏁", layout="wide")

pages = st.navigation([
    st.Page("views/prediction.py", title="Next GP prediction", icon="🔮", default=True),
    st.Page("views/season.py", title="Season overview", icon="🏆"),
    st.Page("views/grand_prix.py", title="Grand Prix analysis", icon="🏎️"),
    st.Page("views/drivers.py", title="Driver head-to-head", icon="⚔️"),
    st.Page("views/model_report.py", title="Model accuracy", icon="📐"),
])
pages.run()
