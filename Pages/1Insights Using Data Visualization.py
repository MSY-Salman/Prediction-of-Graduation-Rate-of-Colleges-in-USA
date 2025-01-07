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
import seaborn as sns
import streamlit as st
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
@st.cache
def load_data():
    data = pd.read_csv(r"D:\SALMAN\ALL+CSV+FILES+-+2nd+Edition+-+corrected\ALL CSV FILES - 2nd Edition\College.csv")
    data['Admission_ratio'] = (data['Enroll'] / data['Apps']) * 100
    data.rename(columns={'Unnamed: 0': 'Colleges'}, inplace=True)
    data['Private'] = data['Private'].map({"Yes": 1, "No": 0})
    return data
data = load_data()
st.title("🎓 Data Visualization Using Matplotlib And Seaborn")
if st.checkbox("📂 Show Raw Data"):
    st.write(data.head())
private_data = data[data['Private'] == 1].dropna(subset=['PhD', 'Grad.Rate'])
public_data = data[data['Private'] == 0].dropna(subset=['PhD', 'Grad.Rate'])
if st.checkbox("📊 Show Graduation Rate Distribution"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data['Grad.Rate'], kde=True, bins=20, color='skyblue', ax=ax)
    ax.set_title("Distribution of Graduation Rates", fontsize=14, fontweight='bold')
    ax.set_xlabel("Graduation Rate (%)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)
if st.checkbox("🎓 Show PhD Faculty vs Graduation Rate"):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(private_data['PhD'], private_data["Grad.Rate"], color='red', label="Private Colleges", marker='x', edgecolors='black', alpha=0.7)
    ax.scatter(public_data['PhD'], public_data["Grad.Rate"], color='blue', label="Public Colleges", marker='o', edgecolors='black', alpha=0.7)
    ax.set_xlabel("PhD Faculty Count", fontsize=12, fontweight='bold', color='darkblue')
    ax.set_ylabel("Graduation Rate (%)", fontsize=12, fontweight='bold', color='darkblue')
    ax.legend(title="College Type", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
if st.checkbox("💸 Show Tuition Fees vs Graduation Rate"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data['Outstate'], y=data['Grad.Rate'], hue=data['Private'], palette={1: 'red', 0: 'blue'}, ax=ax)
    ax.set_title("Tuition Fees vs Graduation Rate", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tuition Fees (Outstate)", fontsize=12)
    ax.set_ylabel("Graduation Rate (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
if st.checkbox("🔥 Show Correlation Heatmap"):
    fig, ax = plt.subplots(figsize=(12, 8))
    a = data.drop(columns=['Colleges', 'Private'])
    sns.heatmap(a.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap", fontsize=14, fontweight='bold')
    st.pyplot(fig)
if st.checkbox("🏆 Show Graduation Rate vs Top 10 Percent"):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=data['Top10perc'], y=data['Grad.Rate'], hue=data['Private'], palette={1: 'green', 0: 'orange'}, ax=ax)
    ax.set_title("Top 10 Percent vs Graduation Rate", fontsize=14, fontweight='bold')
    ax.set_xlabel("Top 10% of High School Class", fontsize=12)
    ax.set_ylabel("Graduation Rate (%)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)
st.write("💡 Select the checkboxes above to explore data visualizations!")
