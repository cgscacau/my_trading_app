import optuna
from core.strategies import StrategyEngine
from core.backtest_engine import run_backtest

def objective_ma(trial, df):
    # Definir espaço de busca
    fast_ma = trial.suggest_int('fast_ma', 5, 50)
    slow_ma = trial.suggest_int('slow_ma', 51, 200)
    
    if fast_ma >= slow_ma:
        return -1000 # Penalidade
        
    strat = StrategyEngine(df)
    df_sig = strat.ma_cross(fast_period=fast_ma, slow_period=slow_ma)
    
    _, metrics, _ = run_backtest(df_sig)
    return metrics['Total Return %']

def optimize_strategy(df, n_trials=20):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective_ma(trial, df), n_trials=n_trials)
    return study.best_params, study.best_value
