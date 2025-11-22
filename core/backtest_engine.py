import pandas as pd
import numpy as np

def run_backtest(df, initial_capital=1000, fee_pct=0.001):
    """
    Executa backtest simples baseado na coluna 'signal' (1, -1, 0).
    Assume execução no preço de fechamento (close).
    """
    capital = initial_capital
    position = 0 # 0: Flat, 1: Long, -1: Short (se suportado)
    equity_curve = []
    trades = []
    
    # Garante que temos sinais limpos
    df = df.dropna()
    
    entry_price = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']
        signal = row['signal']
        date = row['timestamp']
        
        # Lógica simples Long Only para exemplo (pode expandir para Short)
        if position == 0 and signal == 1:
            # Compra
            amount = (capital * (1 - fee_pct)) / price
            position = 1
            entry_price = price
            trades.append({'type': 'buy', 'price': price, 'date': date, 'capital': capital})
            
        elif position == 1 and signal == -1:
            # Venda
            capital = amount * price * (1 - fee_pct)
            position = 0
            trades.append({'type': 'sell', 'price': price, 'date': date, 'capital': capital, 'pnl': (price - entry_price)/entry_price})
            
        # Atualiza curva de capital
        current_equity = capital if position == 0 else amount * price
        equity_curve.append(current_equity)
        
    df_res = df.iloc[:len(equity_curve)].copy()
    df_res['equity'] = equity_curve
    
    # Métricas
    total_return = ((equity_curve[-1] - initial_capital) / initial_capital) * 100
    
    metrics = {
        'Total Return %': total_return,
        'Final Equity': equity_curve[-1],
        'Num Trades': len(trades) // 2
    }
    
    return df_res, metrics, pd.DataFrame(trades)
