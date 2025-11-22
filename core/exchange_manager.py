import ccxt
import pandas as pd
import streamlit as st

class ExchangeManager:
    def __init__(self, exchange_id, api_key=None, secret=None, testnet=False, market_type='swap'):
        self.exchange_id = exchange_id
        self.testnet = testnet
        self.market_type = market_type
        self.exchange = self._initialize_exchange(api_key, secret)

    def _initialize_exchange(self, api_key, secret):
        exchange_class = getattr(ccxt, self.exchange_id)
        params = {
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True,
            'options': {'defaultType': self.market_type}
        }
        
        exchange = exchange_class(params)
        
        if self.testnet:
            exchange.set_sandbox_mode(True)
            
        return exchange

    def fetch_ohlcv(self, symbol, timeframe, limit=1000):
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            st.error(f"Erro ao baixar dados: {str(e)}")
            return pd.DataFrame()

    def get_balance(self):
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            st.error(f"Erro ao buscar saldo: {str(e)}")
            return None

    def create_order(self, symbol, side, amount, price=None, order_type='market'):
        try:
            if order_type == 'market':
                return self.exchange.create_market_order(symbol, side, amount)
            elif order_type == 'limit':
                return self.exchange.create_limit_order(symbol, side, amount, price)
        except Exception as e:
            st.error(f"Erro na ordem: {str(e)}")
            return None
