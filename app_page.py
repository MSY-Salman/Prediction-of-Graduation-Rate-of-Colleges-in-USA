import streamlit as st


# --- PAGE SETUP ---
home_page = st.Page(
    "App.py",
    title="HOME PAGE",
    icon=":material/account_circle:",
    default=True,
)
project_1_page = st.Page(
    "1Insights using EDA.py",
    title="Insights using EDA",
)
project_2_page = st.Page(
    "2Training ML Model.py",
    title="Machine Learning",
)


# --- NAVIGATION SETUP [WITHOUT SECTIONS] ---
# pg = st.navigation(pages=[home_page, project_1_page, project_2_page])

# --- NAVIGATION SETUP [WITH SECTIONS]---
pg = st.navigation(
    {
        "Info": [home_page],
        "Projects": [project_1_page, project_2_page],
    }
)


# --- SHARED ON ALL PAGES ---
st.markdown("""
    <hr>
    <p style='text-align:center; color:gray; font-size:12px;'>
    Developed with ❤️ by <b>MSY Technologies</b> | Insights Platform 📊
    </p>
""", unsafe_allow_html=True)



# --- RUN NAVIGATION ---
pg.run()
