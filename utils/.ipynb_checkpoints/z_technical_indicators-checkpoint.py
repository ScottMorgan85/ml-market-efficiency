import ta 

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

# technical_features = generate_technical_indicators(index_prices)
# combined_data = pd.concat([technical_features, lagged_targets], axis=1)
# combined_data.dropna(inplace=True) # Drop NaN value

def generate_and_combine(dataframe, lagged_targets):
    technical_features = generate_technical_indicators(dataframe)
    combined_data = pd.concat([technical_features, lagged_targets], axis=1)
    combined_data.dropna(inplace=True)
    return combined_data