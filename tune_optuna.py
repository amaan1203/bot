#!/usr/bin/env python3
"""
Optuna hyperparameter search for the DAPO-NIFTY50 agent.
Optimized for high-beta sensitivity research.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# ── Dataset / env config ──────────────────────────────────────────────────────
DATASET_DIR   = "./dataset"
TRAIN_PATH    = os.path.join(DATASET_DIR, "nifty50_train_scores.csv")
TRADE_PATH    = os.path.join(DATASET_DIR, "nifty50_trade_scores.csv")
TRAIN_END     = "2022-12-31"
TRADE_START   = "2023-01-01"

INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
              "dx_30", "close_30_sma", "close_60_sma"]

# ── Data helpers ───────────────────────────────────────────────────────────────
def _align_and_index(df: pd.DataFrame, start: str, end: str | None, fixed_tics: list | None = None) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    if start:
        df = df[df["date"] >= start].copy()
    if end:
        df = df[df["date"] <= end].copy()

    all_dates = sorted(df["date"].unique())
    all_tics  = fixed_tics if fixed_tics is not None else sorted(df["tic"].unique())
    full_idx  = pd.MultiIndex.from_product([all_dates, all_tics], names=["date", "tic"])
    df = df.set_index(["date", "tic"]).reindex(full_idx).reset_index()
    df = df.sort_values(["date", "tic"])

    df["llm_sentiment"] = df["llm_sentiment"].fillna(0.0)
    df["llm_risk"]      = df["llm_risk"].fillna(0.5)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["llm_sentiment", "llm_risk"]:
            df[col] = df.groupby("tic")[col].transform(lambda x: x.ffill().bfill())
            if col == "close":
                df[col] = df[col].fillna(1.0)
            else:
                df[col] = df[col].fillna(0.0)

    df["close"] = df["close"].replace(0.0, 1.0)
    date_to_idx  = {d: i for i, d in enumerate(all_dates)}
    df["new_idx"] = df["date"].map(date_to_idx)
    df = df.set_index("new_idx")
    return df

def load_datasets():
    df_raw_train = pd.read_csv(TRAIN_PATH)
    master_tics = sorted(df_raw_train["tic"].unique())
    df_train = _align_and_index(pd.read_csv(TRAIN_PATH), start="", end=TRAIN_END, fixed_tics=master_tics)
    df_trade = _align_and_index(pd.read_csv(TRADE_PATH), start=TRADE_START, end=None, fixed_tics=master_tics)
    return df_train, df_trade

def make_env(df: pd.DataFrame, stock_dim: int):
    from env_stocktrading_llm_risk import StockTradingEnv
    state_space = 1 + 2 * stock_dim + (2 + len(INDICATORS)) * stock_dim
    return StockTradingEnv(
        df=df, stock_dim=stock_dim, hmax=100, initial_amount=1_000_000,
        num_stock_shares=[0] * stock_dim, buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim, reward_scaling=1e-4,
        state_space=state_space, action_space=stock_dim,
        tech_indicator_list=INDICATORS, turbulence_threshold=70, risk_indicator_col="vix",
    )

def run_backtest(ac, df_trade: pd.DataFrame, stock_dim: int) -> float:
    env_gym = make_env(df_trade, stock_dim)
    obs, _ = env_gym.reset()
    obs = np.array(obs, dtype=np.float32)
    done = False
    with torch.no_grad():
        while not done:
            action = ac.act(obs)
            obs, _reward, done, _trunc, _info = env_gym.step(action)
            obs = np.array(obs, dtype=np.float32)

    account_values = env_gym.save_asset_memory()["account_value"]
    daily_returns  = account_values.pct_change().dropna()
    if daily_returns.empty or daily_returns.std() == 0 or np.isnan(daily_returns.std()):
        return -999.0
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    return float(np.nan_to_num(sharpe, nan=-999.0))

# ── Optuna objective ───────────────────────────────────────────────────────────

def objective(trial, df_train, df_trade, stock_dim, args):
    import optuna
    from dapo_algorithm import dapo, MLPActorCritic

    # NEW RANGES: Focus on high beta as requested
    alpha          = trial.suggest_float("alpha", 0.1, 3.0)
    beta           = trial.suggest_float("beta", 1.5, 5.0)
    pi_lr          = trial.suggest_float("pi_lr", 1e-5, 1e-4, log=True)
    epsilon_low    = trial.suggest_float("epsilon_low", 0.1, 0.25)
    epsilon_high   = trial.suggest_float("epsilon_high", 0.25, 0.40)
    gamma          = trial.suggest_float("gamma", 0.99, 0.999)
    train_pi_iters = trial.suggest_int("train_pi_iters", 50, 150, step=25)
    num_samples    = trial.suggest_categorical("num_samples_per_state", [5, 10, 20])

    env_kwargs = {"stock_dim": stock_dim}
    
    try:
        from spinup.utils.run_utils import setup_logger_kwargs
        logger_kwargs = setup_logger_kwargs(f"optuna_v4_{trial.number}", seed=42, data_dir="./optuna_logs")
    except Exception:
        logger_kwargs = {}

    ac = dapo(
        env_fn=lambda: make_env(df_train, stock_dim).get_sb_env()[0],
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[256, 256], activation=torch.nn.ReLU),
        seed=42, steps_per_epoch=args.steps, epochs=args.epochs, gamma=gamma,
        epsilon_low=epsilon_low, epsilon_high=epsilon_high, pi_lr=pi_lr,
        train_pi_iters=train_pi_iters, num_samples_per_state=num_samples,
        adjustment_type="both", alpha=alpha, beta=beta,
        env_kwargs=env_kwargs, logger_kwargs=logger_kwargs, force_cpu=True,
    )

    sharpe = run_backtest(ac, df_trade, stock_dim)
    trial.report(sharpe, step=args.epochs)
    if trial.should_prune():
        raise optuna.TrialPruned()
    return sharpe

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int, default=30,   help="Epochs per trial")
    parser.add_argument("--steps",    type=int, default=5000, help="Steps per epoch")
    parser.add_argument("--n_trials", type=int, default=50,   help="Number of Optuna trials")
    parser.add_argument("--n_jobs",   type=int, default=1,    help="Parallel workers")
    parser.add_argument("--db",       type=str, default="sqlite:///dapo_nifty50.db")
    args = parser.parse_args()

    import optuna
    os.makedirs("optuna_logs", exist_ok=True)
    os.makedirs("results",     exist_ok=True)

    df_train, df_trade = load_datasets()
    stock_dim = df_train["tic"].nunique()

    # V4: Fresh study to avoid categorical distribution mismatch errors
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=10),
        study_name="dapo_nifty50_v4",
        storage=args.db,
        load_if_exists=True,
    )

    # Force a high-beta starting trial
    study.enqueue_trial({
        "alpha": 1.5, "beta": 3.0, "pi_lr": 3e-5,
        "epsilon_low": 0.2, "epsilon_high": 0.35, "gamma": 0.99,
        "train_pi_iters": 100, "num_samples_per_state": 20
    })

    print(f"\n Starting Optuna v4 search ({args.n_trials} trials)")
    study.optimize(
        lambda trial: objective(trial, df_train, df_trade, stock_dim, args),
        n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=True,
    )

    print(f"\nBest Sharpe : {study.best_value:.4f}")
    print(f"Best Params : {study.best_params}")
    with open("best_hyperparams_v4.json", "w") as f:
        json.dump({"sharpe": study.best_value, "params": study.best_params}, f, indent=2)
