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
        try:
            # Mapeamento para lidar com binanceus se necessário
            exchange_class = getattr(ccxt, self.exchange_id)
            
            params = {
                'enableRateLimit': True,
                'options': {'defaultType': self.market_type} # swap = futuros
            }

            if api_key and secret:
                params['apiKey'] = api_key
                params['secret'] = secret
            
            exchange = exchange_class(params)
            
            if self.testnet:
                exchange.set_sandbox_mode(True)
                
            # Tenta carregar os mercados para validar a conexão
            # Se falhar aqui (erro 451), capturamos no except
            if not self.testnet:
                # Otimização: Carregar apenas se necessário para evitar bloqueio imediato em alguns endpoints
                pass 
                
            return exchange
        except Exception as e:
            st.error(f"Erro ao inicializar exchange {self.exchange_id}: {str(e)}")
            return None

    def fetch_ohlcv(self, symbol, timeframe, limit=1000):
        if not self.exchange:
            return pd.DataFrame()

        try:
            # Tenta baixar dados
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except ccxt.NetworkError as e:
            st.error(f"Erro de Rede (Bloqueio de IP ou Falha): {str(e)}")
            return pd.DataFrame()
        except ccxt.ExchangeError as e:
            st.error(f"Erro da Exchange (Símbolo inválido?): {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Erro desconhecido ao baixar dados: {str(e)}")
            return pd.DataFrame()

    def get_balance(self):
        if not self.exchange: return None
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            st.warning(f"Não foi possível ler o saldo (Verifique permissões ou IP): {str(e)}")
            return None

    def create_order(self, symbol, side, amount, price=None, order_type='market'):
        if not self.exchange: return None
        try:
            if order_type == 'market':
                return self.exchange.create_market_order(symbol, side, amount)
            elif order_type == 'limit':
                return self.exchange.create_limit_order(symbol, side, amount, price)
        except Exception as e:
            st.error(f"Erro na ordem: {str(e)}")
            return None
