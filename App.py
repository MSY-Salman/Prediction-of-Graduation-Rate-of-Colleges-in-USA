import math
import time
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt



# Load and preprocess data
data = pd.read_csv(r"College.csv")
data['Admission_ratio'] = (data['Enroll'] / data['Apps']) * 100
data.rename(columns={'Unnamed: 0': 'Colleges'}, inplace=True)

# Sidebar success message
st.sidebar.success("Navigate through options above!")

# Session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Function to check login
def check_user():
    password = st.session_state.get("passwd", "")
    if password == "123456789":  
        st.session_state.logged_in = True
        st.success("Login successful! You can proceed. ✅")
    else:
        st.error("Incorrect password. You can't proceed! ❌")

# Login page
if not st.session_state.logged_in:
    st.title("🔒 Login Page")
    st.markdown("""
        <h4 style='color:#4CAF50;'>Welcome to MSY Insights Platform!</h4>
        <p>Enter your credentials below to explore insights on graduation rates of colleges in the USA.</p>
    """, unsafe_allow_html=True)
    st.text_input("Enter Username", key='uname', label_visibility="collapsed")
    st.text_input("Enter Password (1******9)", key='passwd', type="password", label_visibility="collapsed")
    if st.button("Login"):
        check_user()
else:
    # Main title and introduction
    st.markdown("""
        <h1 style='color:#1E90FF;'>🎓 Insights on Graduation Rates of Colleges in the USA</h1>
        <p style='color:#666;'>Explore data, analyze correlations, and uncover patterns!</p>
    """, unsafe_allow_html=True)

    # Number of records slider
    st.markdown("<h4 style='color:#4CAF50;'>📋 Display Dataset</h4>", unsafe_allow_html=True)
    x = st.slider('Choose Number Of Records To Display:', min_value=1, max_value=len(data), value=10)
    st.write(data.iloc[:x])

    # Function to show correlation information
    def correlation():
        st.markdown("<h3 style='color:#1E90FF;'>📈 Correlation of Graduation Rate</h3>", unsafe_allow_html=True)
        st.markdown("The dependency between response and predictors is shown below:")
        a = data.drop(columns=["Private", "Colleges"], axis=1)
        st.dataframe(a.corr()["Grad.Rate"].style.background_gradient(cmap="coolwarm"))

    # Function to show dataset overview
    def show_data_info():
        st.markdown("<h3 style='color:#32CD32;'>📊 Data Overview</h3>", unsafe_allow_html=True)
        st.markdown(f"**You have {data.shape[0]} rows and {data.shape[1]} columns in your dataset.**")
        st.markdown("### Quantitative Predictors Overview")
        st.write(data.describe().agg(['count', 'mean', 'min', 'max']))
        st.markdown("### Qualitative Predictors Overview")
        st.write(data[["Private", "Colleges"]].describe())

    # Function to display record of a particular college
    def part_college():
        st.markdown("<h4 style='color:#FF6347;'>🔍 Search Record for a Particular College</h4>", unsafe_allow_html=True)
        a = st.selectbox("Select College:", options=data['Colleges'].dropna().tolist())
        if a:
            st.write(f"<h4 style='color:#4CAF50;'>Record for {a}:</h4>", unsafe_allow_html=True)
            st.dataframe(data[data['Colleges'] == a])

    # Display components
    st.write("")
    part_college()
    st.write("") 
    if st.checkbox("Show Data Info", key="data_info_checkbox"):
        show_data_info()
    if st.checkbox("Show Correlation", key="correlation_checkbox"):
        correlation()

    # Logout button
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.success("Logged out successfully. ✅")

# Footer
st.markdown("""
    <hr>
    <p style='text-align:center; color:gray; font-size:12px;'>
    Developed with ❤️ by <b>MSY Technologies</b> | Insights Platform 📊
    </p>
""", unsafe_allow_html=True)
