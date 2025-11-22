import pandas as pd
import ta

def add_indicators(df):
    # Verifica se há dados suficientes
    if df.empty:
        return df

    # --- Médias Móveis (SMA) ---
    # Usando ta.trend.SMAIndicator
    df['SMA_7'] = ta.trend.SMAIndicator(close=df['close'], window=7).sma_indicator()
    df['SMA_14'] = ta.trend.SMAIndicator(close=df['close'], window=14).sma_indicator()
    df['SMA_25'] = ta.trend.SMAIndicator(close=df['close'], window=25).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
    df['SMA_200'] = ta.trend.SMAIndicator(close=df['close'], window=200).sma_indicator()
    
    # --- RSI ---
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    
    # --- MACD ---
    macd = ta.trend.MACD(close=df['close'])
    df['MACD_12_26_9'] = macd.macd()
    df['MACDh_12_26_9'] = macd.macd_diff()
    df['MACDs_12_26_9'] = macd.macd_signal()
    
    # --- Estocástico ---
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['STOCHk_14_3_3'] = stoch.stoch()
    df['STOCHd_14_3_3'] = stoch.stoch_signal()
    
    # --- Bollinger Bands ---
    bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['BBL_20_2.0'] = bb.bollinger_lband()
    df['BBM_20_2.0'] = bb.bollinger_mavg()
    df['BBU_20_2.0'] = bb.bollinger_hband()
    
    # --- ATR (Volatilidade) ---
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    return df
