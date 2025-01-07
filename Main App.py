import math
import time
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Welcome to MSY!",
    page_icon="📊",
    layout="wide",
)

data = pd.read_csv(r"D:\SALMAN\ALL+CSV+FILES+-+2nd+Edition+-+corrected\ALL CSV FILES - 2nd Edition\College.csv")
data['Admission_ratio'] = (data['Enroll'] / data['Apps']) * 100
data.rename(columns={'Unnamed: 0': 'Colleges'}, inplace=True)
st.sidebar.success("Select Work Field From Above")
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
def check_user():
    password = st.session_state.get("passwd", "")
    if password == "123456789":  # Correct password
        st.session_state.logged_in = True
        st.success("Login successful! You can proceed.", icon="✅")
    else:
        st.error("Incorrect password. You can't proceed!", icon="❌")
if not st.session_state.logged_in:
    st.title("Login Page", anchor="login_page")
    st.text_input("Enter Username", key='uname', label_visibility="collapsed")
    st.text_input("Enter Password (1******9)", key='passwd', type="password", label_visibility="collapsed")
    if st.button("Login", key="login_btn"):
        check_user()
else:
    st.title("Insights on Graduation Rate of Colleges in USA", anchor="main_title")
    x = st.slider('Choose Number Of Records To Display:', min_value=1, max_value=len(data), key="records_slider")
    st.write(data.iloc[:x])
    def correlation():
        st.write("<h3 style='color:#1E90FF;'>Correlation of Graduation Rate</h3>", unsafe_allow_html=True)
        st.write("The dependency between response and predictors are shown below:")
        a = data.drop(columns=["Private", "Colleges"], axis=1)
        st.write(a.corr()["Grad.Rate"])
    def show_data_info():
        st.write(f"<h3 style='color:#32CD32;'>Data Overview</h3>", unsafe_allow_html=True)
        st.write(f"You have {data.shape[0]} rows and {data.shape[1]} columns in your dataset.")
        st.write("Some information about quantitative predictors is given below:")
        st.write(data.describe().agg(['count', 'mean', 'min', 'max']))
        st.write("Some information about qualitative predictors is given below:")
        st.write(data[["Private", "Colleges"]].describe())
    def part_college():
        a = st.selectbox("Search Record of Particular College", options=data['Colleges'].dropna().tolist(), index=None)
        if a is None:
            st.write("")
        else:
            st.write(f"<h4 style='color:#FF6347;'>Record of {a}:</h4>", unsafe_allow_html=True)
            st.write(data[data['Colleges'] == a])
    st.write("")
    part_college()
    st.write("") 
    if st.checkbox("Show Data Info", key="data_info_checkbox"):
        show_data_info()
    if st.checkbox("Show Correlation", key="correlation_checkbox"):
        correlation()
    if st.button("Logout", key="logout_btn"):
        st.session_state.logged_in = False
        st.success("Logged out successfully.", icon="✅")
