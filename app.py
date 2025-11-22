import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os

# Imports locais
from core.exchange_manager import ExchangeManager
from core.strategies import StrategyEngine
from core.backtest_engine import run_backtest
from core.optimizer import optimize_strategy
from core.gru_model import GRUModel
from core.indicators import add_indicators

# Configuração da Página
st.set_page_config(page_title="AI Trading Bot Pro", layout="wide", page_icon="📈")

# --- CSS Personalizado ---
st.markdown("""
<style>
    .stButton>button { width: 100%; }
    .metric-card { background-color: #1E1E1E; padding: 15px; border-radius: 10px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Configurações ---
st.sidebar.title("⚙️ Configuração")

# Credenciais
try:
    # Tenta pegar secrets específicos para a exchange selecionada seria o ideal, 
    # mas aqui mantemos simples
    api_key = st.secrets.get("API_KEY", "")
    api_secret = st.secrets.get("API_SECRET", "")
    if api_key:
        st.sidebar.success("Credenciais carregadas!")
except:
    api_key = ""
    api_secret = ""

if not api_key:
    st.sidebar.info("Insira chaves para operar ou deixe em branco para ver dados públicos (se permitido).")
    api_key = st.sidebar.text_input("API Key", type="password")
    api_secret = st.sidebar.text_input("API Secret", type="password")

# Adicionado 'binanceus' para quem roda no servidor dos EUA
exchange_name = st.sidebar.selectbox("Exchange", ["binance", "binanceus", "bingx"])
env_type = st.sidebar.radio("Ambiente", ["Testnet", "Real"], index=0)
is_testnet = True if env_type == "Testnet" else False

# Nota sobre símbolos
if exchange_name == 'binanceus':
    default_symbol = "BTC/USD" # Binance US usa USD, não USDT em muitos pares spot
    market_type_default = 'spot' # Binance US tem restrições em futuros
else:
    default_symbol = "BTC/USDT"
    market_type_default = 'swap' # Futuros perpétuos

symbol = st.sidebar.text_input("Símbolo", default_symbol)
timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)

# Inicialização da Exchange
exchange = ExchangeManager(exchange_name, api_key, api_secret, testnet=is_testnet, market_type=market_type_default)

# --- Tabs ---
# (O restante do código das abas permanece igual...)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trade & Estratégias", "🧠 AI / GRU", "📊 Backtest", "🧪 Otimização", "📡 Execução"])

# Carregar Dados (Cache)
@st.cache_data(ttl=300)
def load_data(sym, tf, limit=500):
    df = exchange.fetch_ohlcv(sym, tf, limit)
    if not df.empty:
        df = add_indicators(df)
    return df

if st.sidebar.button("Atualizar Dados"):
    st.cache_data.clear()

df = load_data(symbol, timeframe)

if df.empty:
    st.error("Não foi possível carregar dados. Verifique o símbolo ou a conexão.")
    st.stop()

# --- TAB 1: Estratégias Clássicas ---
with tab1:
    st.header(f"Análise Técnica: {symbol}")
    
    strat_choice = st.selectbox("Escolha a Estratégia", 
                                ["MA Cross", "MA + Estocástico", "MA + RSI"])
    
    strat_engine = StrategyEngine(df)
    df_strategy = df.copy()
    
    if strat_choice == "MA Cross":
        col1, col2 = st.columns(2)
        fast = col1.number_input("MA Rápida", 5, 100, 9)
        slow = col2.number_input("MA Lenta", 10, 300, 21)
        df_strategy = strat_engine.ma_cross(fast, slow)
        
        # Plot
        fig = go.Figure(data=[go.Candlestick(x=df_strategy['timestamp'],
                        open=df_strategy['open'], high=df_strategy['high'],
                        low=df_strategy['low'], close=df_strategy['close'], name="OHLC")])
        fig.add_trace(go.Scatter(x=df_strategy['timestamp'], y=df_strategy['fast_ma'], line=dict(color='orange', width=1), name='Fast MA'))
        fig.add_trace(go.Scatter(x=df_strategy['timestamp'], y=df_strategy['slow_ma'], line=dict(color='blue', width=1), name='Slow MA'))
        
        # Sinais
        buys = df_strategy[df_strategy['signal'] == 1]
        sells = df_strategy[df_strategy['signal'] == -1]
        fig.add_trace(go.Scatter(x=buys['timestamp'], y=buys['close'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='Buy'))
        fig.add_trace(go.Scatter(x=sells['timestamp'], y=sells['close'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red'), name='Sell'))
        
        fig.update_layout(height=600, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: AI / GRU ---
with tab2:
    st.header("Deep Learning: GRU Model")
    st.info("Este modelo tenta prever o preço de fechamento futuro baseado em sequências passadas.")
    
    col1, col2, col3 = st.columns(3)
    lookback = col1.number_input("Lookback (Candles)", 10, 100, 60)
    epochs = col2.number_input("Épocas de Treino", 1, 50, 5)
    layers = col3.number_input("Camadas GRU", 1, 5, 2)
    
    if st.button("Treinar Modelo GRU"):
        with st.spinner("Treinando Rede Neural... (Isso pode demorar)"):
            gru = GRUModel(lookback=lookback)
            X, y, scaler = gru.prepare_data(df)
            
            # Split treino/teste
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            model = gru.build_model(layers=layers)
            history = gru.train(X_train, y_train, epochs=epochs)
            
            st.success("Modelo Treinado!")
            
            # Previsão
            predicted = gru.predict(X_test)
            predicted_prices = gru.scaler.inverse_transform(predicted)
            real_prices = gru.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Gráfico de Previsão
            fig_gru = go.Figure()
            fig_gru.add_trace(go.Scatter(y=real_prices.flatten(), name="Real", line=dict(color='cyan')))
            fig_gru.add_trace(go.Scatter(y=predicted_prices.flatten(), name="Previsto (GRU)", line=dict(color='yellow')))
            fig_gru.update_layout(title="Predição vs Realidade (Conjunto de Teste)", template="plotly_dark")
            st.plotly_chart(fig_gru, use_container_width=True)
            
            # Sinal Atual (Último candle)
            last_sequence = X[-1].reshape(1, lookback, 1)
            next_pred = gru.predict(last_sequence)
            next_price = gru.scaler.inverse_transform(next_pred)[0][0]
            current_price = df['close'].iloc[-1]
            
            st.metric("Preço Atual", f"{current_price:.2f}")
            st.metric("Previsão Próx. Fechamento", f"{next_price:.2f}", delta=f"{next_price - current_price:.2f}")

# --- TAB 3: Backtest ---
with tab3:
    st.header("Simulação Histórica")
    
    bt_capital = st.number_input("Capital Inicial (USD)", 100, 100000, 1000)
    bt_fee = st.number_input("Taxa (%)", 0.0, 1.0, 0.1) / 100
    
    if st.button("Rodar Backtest (Estratégia Atual)"):
        # Usa a estratégia configurada na Tab 1
        res_df, metrics, trades_df = run_backtest(df_strategy, initial_capital=bt_capital, fee_pct=bt_fee)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Retorno Total", f"{metrics['Total Return %']:.2f}%")
        c2.metric("Capital Final", f"${metrics['Final Equity']:.2f}")
        c3.metric("Total Trades", metrics['Num Trades'])
        
        st.line_chart(res_df['equity'])
        st.dataframe(trades_df)

# --- TAB 4: Otimização ---
with tab4:
    st.header("Otimização de Parâmetros (MA Cross)")
    st.write("Encontra os melhores valores de MA Rápida e Lenta para os dados atuais.")
    
    n_trials = st.slider("Número de Tentativas", 10, 100, 20)
    
    if st.button("Otimizar"):
        with st.spinner("Otimizando com Optuna..."):
            best_params, best_value = optimize_strategy(df, n_trials)
            st.success(f"Melhores Parâmetros Encontrados!")
            st.json(best_params)
            st.metric("Melhor Retorno (%)", f"{best_value:.2f}%")

# --- TAB 5: Execução ---
with tab5:
    st.header("📡 Painel de Execução")
    
    if env_type == "Real":
        st.error("⚠️ CUIDADO: VOCÊ ESTÁ EM MODO REAL. DINHEIRO REAL ESTÁ EM RISCO.")
        confirm = st.checkbox("Eu entendo os riscos e quero prosseguir.")
    else:
        st.success("✅ Modo Testnet Ativo")
        confirm = True
        
    if confirm and api_key:
        balance = exchange.get_balance()
        if balance:
            free_usdt = balance.get('USDT', {}).get('free', 0)
            st.metric("Saldo Disponível (USDT)", f"{free_usdt:.2f}")
        
        c1, c2 = st.columns(2)
        order_side = c1.selectbox("Lado", ["buy", "sell"])
        order_amt = c2.number_input("Quantidade", 0.001, 100.0, 0.01)
        
        if st.button("ENVIAR ORDEM DE MERCADO"):
            with st.spinner("Enviando ordem..."):
                resp = exchange.create_order(symbol, order_side, order_amt)
                if resp:
                    st.success(f"Ordem executada! ID: {resp['id']}")
                    st.json(resp)
    else:
        st.warning("Configure as chaves de API e confirme os riscos para operar.")

# Disclaimer Rodapé
st.markdown("---")
st.caption("⚠️ Aviso Legal: Este software é para fins educacionais. Trading de criptomoedas envolve alto risco. O autor não se responsabiliza por perdas financeiras.")
