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

data = pd.read_csv(r"D:\SALMAN\ALL+CSV+FILES+-+2nd+Edition+-+corrected\ALL CSV FILES - 2nd Edition\College.csv")
data['Admission_ratio']=(data['Enroll']/data['Apps'])*100
st.title("Insights on Graduation Rate of Colleges in USA")
x = st.slider('Choose Number Of Records To Display?',min_value=1,max_value=len(data)) 
st.write(data[0:x])