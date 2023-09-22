import warnings
warnings.filterwarnings("ignore")
# from tqdm import TqdmExperimentalWarning
# warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# from xbbg import blp
import pandas as pd
import os
from scipy.stats import skew, kurtosis, bartlett
from statsmodels.tsa.stattools import acf
import statsmodels.api as sm
from datetime import datetime, timedelta
# import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score,recall_score
import time
import joblib
import shap
import plotly.graph_objects as go
from IPython.display import display, HTML
import pickle
import ta
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import h5py
from keras.models import load_model

# Define the file paths and tickers
tickers = ['RIY Index','RTY Index', 'C0A0 Index','H0A0 Index','SPBDAL Index', 'MXEA Index', 'MXEF Index','EMUSTRUU Index', 'SFFRNEWS Index']
readable_names = ['US Large Cap Equities','US Small Cap Equities','US Investment Grade Bonds', 'US High Yield Bonds', 'US Bank Loans', 'Developed Country Equities', 'Emerging Market Equities','Emerging Market Debt']  ##, 'Sentiment Score'

# Color mapping for asset classes
asset_colors = {
    'US Large Cap Equities': '#1f77b4',  # Example color
    'US Small Cap Equities': '#ff7f0e',  # Example color
    'US Investment Grade Bonds': '#2ca02c',  # ... and so on
    'US High Yield Bonds': '#d62728',
    'US Bank Loans': '#9467bd',
    'Developed Country Equities': '#8c564b',
    'Emerging Market Equities': '#e377c2',
    'Emerging Market Debt': '#7f7f7f',
    # Add more asset classes and colors as necessary
}

# List of days after event
days_after_event = [5, 30, 60, 90]

index_prices_path = "data/index_prices.csv"
index_returns_path = "data/index_returns.csv"

date_string = "4/2/2007"
date_format = "%m/%d/%Y"
start_date = datetime.strptime(date_string, date_format)
# end_date = datetime.today()
end_string = "8/25/2023"
end_date = datetime.strptime(end_string, date_format)


asset_columns = ['US Large Cap Equities', 'US Small Cap Equities',
                     'US Investment Grade Bonds', 'US High Yield Bonds', 'US Bank Loans',
                     'Developed Country Equities', 'Emerging Market Equities',
                     'Emerging Market Debt']
data = {
    'Event': ['Lehman Collapse','ECB QE Announcement', 'Brexit Vote', 'COVID-19 Pandemic','Russia-Ukraine/Fed Hikes','SVB Collapse'],
    'Event Date': ['9/15/2008', '1/22/2015', '6/23/2016', '3/11/2020','2/25/2022','3/10/2023']
}

df_events = pd.DataFrame(data)
df_events['Event Date'] = pd.to_datetime(df_events['Event Date'])

# Suppress warnings
pd.set_option('mode.chained_assignment', None)
plt.rcParams["font.family"] = "Arial"