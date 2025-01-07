import math
import time
import numpy
import pandas
import requests
import matplotlib
import numpy as pp
import pandas as pd
import seaborn as sns
from scipy import stats
from numpy import random
from itertools import combinations
from sklearn import tree 
from sklearn import metrics
import statsmodels.api as sm
from tabulate import tabulate
from functools import partial
from termcolor import colored
from sklearn.base import clone
from numpy.linalg import inv,qr
import matplotlib.pyplot as plt
from sklearn import linear_model
from random import normalvariate
from sklearn.cluster import KMeans
from pandas import Series,DataFrame
from scipy.optimize import minimize
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
import streamlit as st
from statsmodels.formula.api import ols
from sklearn.naive_bayes import GaussianNB
from statsmodels.stats.anova import anova_lm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut as LOOCV
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import (cross_validate,KFold,ShuffleSplit)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.read_csv(r"D:\SALMAN\ALL+CSV+FILES+-+2nd+Edition+-+corrected\ALL CSV FILES - 2nd Edition\College.csv")
data.rename(columns={'Unnamed: 0': 'Colleges'}, inplace=True)

st.set_page_config(
    page_title="Welcome to MSY!",
    page_icon="📊",
    layout="wide",
)
st.title("Prediction of Graduation Rate in College")
st.write("")
st.write("")


# def extract_column_wrt_corr(data,perce):
#     a=data.drop(columns=['Colleges','Private'])
#     a1=a.columns
#     b=a.corr()['Grad.Rate']
#     l=[]
#     for i in range(len(b)):
#         # if b[i] >=0.3 or b[i] <=  -0.3:
#         if (b[i] >=( perce/100) or b[i] <=  -(perce/100)) and b[i]!=1:
#             l.append(b.index[i]) 
#     return l
# # X=(extract_column_wrt_corr(data,40))
# def perform_regression(col,dictt):
#     X = data[col]
#     y = data['Grad.Rate']  
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     r2 = r2_score(y_test, y_pred)
#     mse = mean_squared_error(y_test, y_pred)
#     coefficients = model.coef_
#     a= model.intercept_
#     print(dictt)
#     for i in range(len(coefficients)):
#         input_value = dictt.get(col[i].replace(".", "_"), 0)  # Default to 0 if key not found
#         print(input_value,type(input_value))
#         input_value = float(input_value)  # Ensure it is a float
#         a+=(coefficients[i]*input_value)
#     return a

# st.write("Select how much percent should be dependency of predictors on response?")
# perce = st.number_input('Select Percentage', min_value=0, max_value=50, value=25, step=1)
# if st.button(f"Take Input"):
#     values=extract_column_wrt_corr(data,perce)
#     variable_dict = {var.replace('.', '_'): None for var in values}
#     for i in range(1):
#         if "Apps" in variable_dict:
#             Apps = st.text_input("Apps", placeholder="Number of applications received")
#         else:
#             Apps = 0

#         if "Accept" in variable_dict:
#             Accept = st.text_input("Accept", placeholder="Number of applicants accepted")
#         else:
#             Accept = 0

#         if "Enroll" in variable_dict:
#             Enroll = st.text_input("Enroll", placeholder="Number of new students enrolled")
#         else:
#             Enroll = 0

#         if "Top10perc" in variable_dict:
#             Top10perc = st.text_input("Top10perc", placeholder="New students from top 10 percent of high school class")
#         else:
#             Top10perc = 0

#         if "Top25perc" in variable_dict:
#             Top25perc = st.text_input("Top25perc", placeholder="New students from top 25 percent of high school class")
#         else:
#             Top25perc = 0

#         if "F_Undergrad" in variable_dict:
#             F_Undergrad = st.text_input("F_Undergrad", placeholder="Number of full-time undergraduates")
#         else:
#             F_Undergrad = 0

#         if "P_Undergrad" in variable_dict:
#             P_Undergrad = st.text_input("P_Undergrad", placeholder="Number of part-time undergraduates")
#         else:
#             P_Undergrad = 0

#         if "Outstate" in variable_dict:
#             Outstate = st.text_input("Outstate", placeholder="Out-of-state tuition")
#         else:
#             Outstate = 0

#         if "Room_Board" in variable_dict:
#             Room_Board = st.text_input("Room_Board", placeholder="Room and board costs")
#         else:
#             Room_Board = 0

#         if "Books" in variable_dict:
#             Books = st.text_input("Books", placeholder="Estimated book costs")
#         else:
#             Books = 0

#         if "Personal" in variable_dict:
#             Personal = st.text_input("Personal", placeholder="Estimated personal spending")
#         else:
#             Personal = 0

#         if "PhD" in variable_dict:
#             PhD = st.text_input("PhD", placeholder="Percent of faculty with Ph.D.s")
#         else:
#             PhD = 0

#         if "Terminal" in variable_dict:
#             Terminal = st.text_input("Terminal", placeholder="Percent of faculty with terminal degree")
#         else:
#             Terminal = 0

#         if "S_F_Ratio" in variable_dict:
#             S_F_Ratio = st.text_input("S_F_Ratio", placeholder="Student/faculty ratio")
#         else:
#             S_F_Ratio = 0

#         if "perc_alumni" in variable_dict:
#             perc_alumni = st.text_input("perc_alumni", placeholder="Percent of alumni who donate")
#         else:
#             perc_alumni = 0

#         if "Expend" in variable_dict:
#             Expend = st.text_input("Expend", placeholder="Instructional expenditure per student")
#         else:
#             Expend = 0


#         dictt={'Apps': Apps,'Accept': Accept,'Enroll': Enroll,'Top10perc': Top10perc,'Top25perc': Top25perc,'F_Undergrad': F_Undergrad,'P_Undergrad': P_Undergrad,'Outstate': Outstate,'Room_Board': Room_Board,'Books': Books,'Personal': Personal,'PhD': PhD,'Terminal': Terminal,'S_F_Ratio': S_F_Ratio,'perc_alumni': perc_alumni,'Expend': Expend}

#         if st.button("Calculate Graduation Rate"):
#             grad_rate=perform_regression(values,dictt)
#             print(grad_rate)
#             st.write(grad_rate)







import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
st.markdown("""
    <style>
        .header {
            color: #ffffff;
            background-color: #4CAF50;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .sub-header {
            color: #4CAF50;
            text-align: center;
            font-size: 18px;
            margin-bottom: 20px;
        }
        .input-box {
            margin: 10px 0;
        }
        .result-box {
            padding: 15px;
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 16px;
            font-weight: bold;
            color: #4CAF50;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            font-size: 14px;
            color: #888888;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown('<div class="header">College Graduation Rate Predictor</div>', unsafe_allow_html=True)
def extract_column_wrt_corr(data,perce):
    a=data.drop(columns=['Colleges','Private'])
    b=a.corr()['Grad.Rate']
    l=[]
    for i in range(len(b)):
        if (b[i] >=( perce/100) or b[i] <=  -(perce/100)) and b[i]!=1:
            l.append(b.index[i]) 
    return l
def perform_regression(col, dictt):
    X = data[col]
    y = data['Grad.Rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    a = model.intercept_
    coefficients = model.coef_
    for i in range(len(coefficients)):
        input_value = dictt.get(col[i].replace(".", "_"), 0)
        input_value = float(input_value) if input_value else 0
        a += coefficients[i] * input_value
    return a, r2
st.markdown('<div class="sub-header">Enter Predictor Variables</div>', unsafe_allow_html=True)
Apps = st.text_input("Apps", placeholder="Number of applications received", key="Apps")
Accept = st.text_input("Accept", placeholder="Number of applicants accepted", key="Accept")
Enroll = st.text_input("Enroll", placeholder="Number of new students enrolled", key="Enroll")
Top10perc = st.text_input("Top10perc", placeholder="New students from top 10 percent of high school class", key="Top10perc")
Top25perc = st.text_input("Top25perc", placeholder="New students from top 25 percent of high school class", key="Top25perc")
F_Undergrad = st.text_input("F_Undergrad", placeholder="Number of full-time undergraduates", key="F_Undergrad")
P_Undergrad = st.text_input("P_Undergrad", placeholder="Number of part-time undergraduates", key="P_Undergrad")
Outstate = st.text_input("Outstate", placeholder="Out-of-state tuition", key="Outstate")
Room_Board = st.text_input("Room_Board", placeholder="Room and board costs", key="Room_Board")
Books = st.text_input("Books", placeholder="Estimated book costs", key="Books")
Personal = st.text_input("Personal", placeholder="Estimated personal spending", key="Personal")
PhD = st.text_input("PhD", placeholder="Percent of faculty with Ph.D.s", key="PhD")
Terminal = st.text_input("Terminal", placeholder="Percent of faculty with terminal degree", key="Terminal")
S_F_Ratio = st.text_input("S_F_Ratio", placeholder="Student/faculty ratio", key="S_F_Ratio")
perc_alumni = st.text_input("perc_alumni", placeholder="Percent of alumni who donate", key="perc_alumni")
Expend = st.text_input("Expend", placeholder="Instructional expenditure per student", key="Expend")
dictt = {
    'Apps': Apps,
    'Accept': Accept,
    'Enroll': Enroll,
    'Top10perc': Top10perc,
    'Top25perc': Top25perc,
    'F_Undergrad': F_Undergrad,
    'P_Undergrad': P_Undergrad,
    'Outstate': Outstate,
    'Room_Board': Room_Board,
    'Books': Books,
    'Personal': Personal,
    'PhD': PhD,
    'Terminal': Terminal,
    'S_F_Ratio': S_F_Ratio,
    'perc_alumni': perc_alumni,
    'Expend': Expend,
}
if st.button("Calculate Graduation Rate"):
    values = extract_column_wrt_corr(data, 0)
    grad_rate, accuracy = perform_regression(values, dictt)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="result-box">
            <b>Predicted Graduation Rate:</b> {grad_rate:.2f}% <br>
            <b>Model Accuracy (R² Score):</b> {accuracy:.2f} <br>
        </div>
    """, unsafe_allow_html=True)
st.markdown("""
    <hr>
    <p style='text-align:center; color:gray; font-size:12px;'>
    Developed with ❤️ by <b>MSY Technologies</b> | Insights Platform 📊
    </p>
""", unsafe_allow_html=True)
