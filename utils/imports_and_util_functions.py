from xbbg import blp
import pandas as pd
import os
from scipy.stats import skew, kurtosis, bartlett
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from datetime import datetime, timedelta
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import joblib
import shap
import plotly.graph_objects as go
from IPython.display import display, HTML
from xgboost import XGBClassifier
import pickle

def get_test_dates(df, pre_days, post_days_list):
    test_dates_dict = {}
    for post_days in post_days_list:
        test_dates = []
        for date in df['Event Date']:
            start_date = date - timedelta(days=pre_days)
            end_date = date + timedelta(days=post_days)
            test_dates.extend(pd.date_range(start=start_date, end=end_date).tolist())
        test_dates_dict[post_days] = test_dates
    return test_dates_dict