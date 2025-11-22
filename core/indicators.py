import pandas_ta as ta

def add_indicators(df):
    # Médias Móveis
    df['SMA_7'] = ta.sma(df['close'], length=7)
    df['SMA_14'] = ta.sma(df['close'], length=14)
    df['SMA_25'] = ta.sma(df['close'], length=25)
    df['SMA_50'] = ta.sma(df['close'], length=50)
    df['SMA_200'] = ta.sma(df['close'], length=200)
    
    # RSI
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    
    # MACD
    macd = ta.macd(df['close'])
    df = df.join(macd) # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    
    # Estocástico
    stoch = ta.stoch(df['high'], df['low'], df['close'])
    df = df.join(stoch) # STOCHk_14_3_3, STOCHd_14_3_3
    
    # Bollinger Bands
    bb = ta.bbands(df['close'], length=20)
    df = df.join(bb)
    
    # ATR (Volatilidade)
    df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    return df
