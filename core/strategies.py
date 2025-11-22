import pandas as pd

class StrategyEngine:
    def __init__(self, df):
        self.df = df

    def ma_cross(self, fast_period=9, slow_period=21):
        df = self.df.copy()
        df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
        
        df['signal'] = 0
        # Sinal de compra: Rápida cruza acima da lenta
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1
        # Sinal de venda: Rápida cruza abaixo da lenta
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1
        return df

    def ma_stoch(self, fast_ma=9, slow_ma=21, k_period=14, overbought=80, oversold=20):
        df = self.df.copy()
        # Recalcular indicadores conforme parametros
        import pandas_ta as ta
        df['fast_ma'] = ta.sma(df['close'], length=fast_ma)
        df['slow_ma'] = ta.sma(df['close'], length=slow_ma)
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=k_period)
        df['k'] = stoch[f'STOCHk_{k_period}_3_3']
        
        df['signal'] = 0
        
        # Compra: MA Alta E Stoch saindo de oversold
        condition_buy = (df['fast_ma'] > df['slow_ma']) & (df['k'] < oversold)
        df.loc[condition_buy, 'signal'] = 1
        
        # Venda: MA Baixa E Stoch saindo de overbought
        condition_sell = (df['fast_ma'] < df['slow_ma']) & (df['k'] > overbought)
        df.loc[condition_sell, 'signal'] = -1
        
        return df

    def gru_signal(self, model, scaler, lookback=60):
        # Esta função seria chamada iterativamente ou em batch
        # Retorna um DataFrame com a coluna 'signal' baseada na predição
        pass
