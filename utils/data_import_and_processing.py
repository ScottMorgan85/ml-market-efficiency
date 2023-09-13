# import os
# import pandas as pd
# from datetime import datetime
# from xbbg import blp

from utils.helpers import *

def fetch_data(tickers, start_date, end_date):
    return blp.bdh(tickers=tickers, flds=['Px_Last'], start_date=start_date, end_date=end_date)

def fetch_and_save_data(tickers, start_date, end_date, index_prices_path, index_returns_path):
    index_prices_exists = os.path.exists(index_prices_path)
    index_returns_exists = os.path.exists(index_returns_path)

    if not index_prices_exists or not index_returns_exists:
        print("Fetching data from Bloomberg...")
        data = fetch_data(tickers, start_date, end_date)

        # Processing Index Prices
        if not index_prices_exists:
            print("Processing and saving Index Prices...")
            index_data_raw = data.copy()
            index_data_raw.columns = readable_names
            index_prices = index_data_raw
            index_prices.index = pd.to_datetime(index_prices.index)
            index_prices = index_prices[index_prices.index.weekday < 5]
            index_prices.dropna(inplace=True)
            index_prices.to_csv(index_prices_path)
        
        # Processing Index Returns
        if not index_returns_exists:
            print("Processing and saving Index Returns...")
            index_returns_raw = index_prices.drop(columns=['Sentiment Score'])
            index_returns_raw = index_returns_raw.pct_change().dropna()
            threshold = len(index_returns_raw.columns) - 2
            index_returns = index_returns_raw.dropna(thresh=threshold)
            index_returns.index = pd.to_datetime(index_returns.index)
            index_returns = index_returns[index_returns.index.weekday < 5]
            index_returns.dropna(inplace=True)
            index_returns.to_csv(index_returns_path)
    else:
        # Load CSV files if they already exist
        index_prices = pd.read_csv(index_prices_path, index_col=0, parse_dates=True)
        index_returns = pd.read_csv(index_returns_path, index_col=0, parse_dates=True)
        print("Data loaded from existing CSV files.")

    print("Data import process completed.")
    return index_prices, index_returns

def generate_lagged_returns_and_targets(index_returns):
   
    lagged_returns = index_returns.shift(-1)
    lagged_returns.dropna(inplace=True)
    lagged_returns.columns = [f"{col}" for col in lagged_returns.columns]
    
    for column in lagged_returns.columns:
        lagged_returns[column] = (lagged_returns[column] > 0).astype(int)
    
    return lagged_returns
    print("Generation of lagged return targets completed.")

def MACD(series, short_window, long_window, signal_window):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.rolling(window=signal_window).mean()
    return macd_line, signal_line

def generate_technical_indicators(df):
    indicator_dataframes = []

    for column in df.columns:
        # Skip 'Sentiment Score' for now as you instructed
        if column == 'Sentiment Score':
            continue

        sma_30 = ta.trend.SMAIndicator(close=df[column], window=30).sma_indicator()
        sma_60 = ta.trend.SMAIndicator(close=df[column], window=60).sma_indicator()

        ema_30 = ta.trend.EMAIndicator(close=df[column], window=30).ema_indicator()
        ema_60 = ta.trend.EMAIndicator(close=df[column], window=60).ema_indicator()


        rsi_14 = ta.momentum.RSIIndicator(close=df[column], window=14).rsi()
        macd, macdsignal = MACD(df[column], 12, 26, 9)  # modified the unpacking

        indicators = pd.concat([
            sma_30.rename(f"{column}_SMA_30"),
            sma_60.rename(f"{column}_SMA_60"),
            ema_30.rename(f"{column}_EMA_30"),
            ema_60.rename(f"{column}_EMA_60"),
            rsi_14.rename(f"{column}_RSI_14"),
            macd.rename(f"{column}_MACD"),
            macdsignal.rename(f"{column}_MACD_Signal")
        ], axis=1)

        indicator_dataframes.append(indicators)

    combined_indicators = pd.concat(indicator_dataframes, axis=1)
    combined_indicators = combined_indicators.dropna()

    return combined_indicators

def generate_and_combine(dataframe, lagged_targets):
    technical_features = generate_technical_indicators(dataframe)
    combined_data = pd.concat([technical_features, lagged_targets], axis=1)
    combined_data = combined_data.dropna(inplace=True)
    print("Feature engineering completed.")
    return combined_data