#!/usr/bin/env python3
"""
Step 5: Backtest trained DAPO model on Nifty50 trade data (2023-01-01 onwards).
Compares against Nifty 50 Index (^NSEI) as benchmark.

Usage:
    python 5_backtest_nifty.py --model checkpoint/nifty50/nifty50_dapo_both_a1.0_b1.0.pth

Output in ./dapo_results_nifty/:
    DAPO_Nifty50_Comparison.png
    dapo_nifty50_metrics.csv
    dapo_nifty50_returns.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf
from gymnasium.spaces import Box, Discrete

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = "./dataset"
RESULTS_DIR  = "./dapo_results_nifty"
TRADE_START  = "2023-01-01"
TRADE_END    = "2026-04-08"

try:
    from finrl.config import INDICATORS
except ImportError:
    INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
                  "dx_30", "close_30_sma", "close_60_sma"]

from env_stocktrading_llm_risk import StockTradingEnv as StockTradingEnv_llm_risk

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cpu")   # inference on CPU is fine


# ── Model definition ───────────────────────────────────────────────────────────
def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class MLPGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = nn.Parameter(torch.as_tensor(log_std))
        self.mu_net  = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        from torch.distributions.normal import Normal
        mu  = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0],
                                       hidden_sizes, activation)
        self.to(device)

    def step(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(device)
            pi  = self.pi._distribution(obs)
            a   = pi.sample()
        return a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)


# ── Data loading ───────────────────────────────────────────────────────────────
def load_trade_data():
    scores_path = os.path.join(DATASET_DIR, "nifty50_trade_scores.csv")
    base_path   = os.path.join(DATASET_DIR, "nifty50_trade_base.csv")

    if os.path.exists(scores_path):
        print("Loading scored trade data")
        trade = pd.read_csv(scores_path)
    elif os.path.exists(base_path):
        print("  Using base trade data (neutral LLM scores)")
        trade = pd.read_csv(base_path)
        trade["llm_sentiment"] = 0.0
        trade["llm_risk"]      = 0.5
    else:
        raise FileNotFoundError(
            f"No trade data found. Run 2_prepare_dataset.py first."
        )

    # ── Align Tickers (Ensuring same 50 stocks as training) ────────────────────
    print(" Aligning data (ensuring all 50 tickers present on all dates)...")
    
    # 1. Define the 50 tickers used in training
    train_tics = [
        'ADANIENT', 'ADANIPORTS', 'APOLLOHOSP', 'ASIANPAINT', 'AXISBANK', 
        'BAJAJ-AUTO', 'BAJAJFINSV', 'BAJFINANCE', 'BHARTIARTL', 'BPCL', 
        'BRITANNIA', 'CIPLA', 'COALINDIA', 'DIVISLAB', 'DRREDDY', 'EICHERMOT', 
        'GRASIM', 'HCLTECH', 'HDFCBANK', 'HDFCLIFE', 'HEROMOTOCO', 'HINDALCO', 
        'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'INFY', 'ITC', 'JSWSTEEL', 
        'KOTAKBANK', 'LT', 'LTIM', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 
        'ONGC', 'POWERGRID', 'RELIANCE', 'SBILIFE', 'SBIN', 'SHRIRAMFIN', 
        'SUNPHARMA', 'TATACONSUM', 'TATAMOTORS', 'TATASTEEL', 'TCS', 'TECHM', 
        'TITAN', 'ULTRACEMCO', 'WIPRO'
    ]
    
    # 2. Cross-product of unique dates and these 50 tickers
    unique_dates = sorted(trade["date"].unique())
    template = pd.DataFrame(
        [(d, t) for d in unique_dates for t in train_tics],
        columns=["date", "tic"]
    )
    
    # 3. Merge trade data into template
    trade = template.merge(trade, on=["date", "tic"], how="left")
    
    # 4. Filling missing values
    # Technical indicators and prices: Forward fill then back fill
    cols_to_fill = ["open", "high", "low", "close", "volume", "vix", "turbulence"] + INDICATORS
    for col in cols_to_fill:
        if col in trade.columns:
            trade[col] = trade.groupby("tic")[col].ffill().bfill()
            
    # Scores and Volume: Fill remaining NAs with neutral defaults
    trade["volume"] = trade["volume"].fillna(0)
    if "llm_sentiment" not in trade.columns: trade["llm_sentiment"] = 0.0
    if "llm_risk" not in trade.columns: trade["llm_risk"] = 0.5
    trade["llm_sentiment"] = trade["llm_sentiment"].fillna(0.0)
    trade["llm_risk"]      = trade["llm_risk"].fillna(0.5)

    # ── Final Processing ──────────────────────────────────────────────────────
    trade = trade[trade["date"] >= TRADE_START].copy()
    unique_dates = sorted(trade["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    trade["new_idx"] = trade["date"].map(date_to_idx)
    trade = trade.set_index("new_idx")

    return trade


# ── Prediction ─────────────────────────────────────────────────────────────────
def run_prediction(model, environment):
    print("▶  Running backtest...")
    state, _ = environment.reset()
    episode_assets = [environment.initial_amount]
    date_memory    = []

    total_steps = len(environment.df.index.unique())

    with torch.no_grad():
        for i in range(total_steps):
            s_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action   = model.step(s_tensor)[0] if s_tensor.dim() == 2 else model.act(state)

            state, reward, done, _, _ = environment.step(action)

            # Total asset value
            try:
                price_array    = environment.df.loc[environment.day, "close"].values
                stock_holdings = np.array(environment.state[
                    environment.stock_dim + 1 : environment.stock_dim * 2 + 1
                ])
                cash           = environment.state[0]
                total_asset    = cash + (price_array * stock_holdings).sum()
            except Exception:
                total_asset = episode_assets[-1]

            episode_assets.append(total_asset)

            # Date
            try:
                date_memory.append(environment.df.loc[environment.day, "date"].iloc[0])
            except Exception:
                date_memory.append(str(i))

            if done:
                break

    print(f"   Backtest complete: {len(episode_assets)} asset points")
    return episode_assets, date_memory


# ── Benchmark: Nifty 50 ────────────────────────────────────────────────────────
def get_nifty50_benchmark(start, end, initial=1_000_000):
    print("📈 Downloading Nifty 50 benchmark (^NSEI)...")
    try:
        df = yf.download("^NSEI", start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df.rename(columns=str.lower)
        df["date"]  = pd.to_datetime(df.index).strftime("%Y-%m-%d")
        fv          = df["close"].iloc[0]
        df["value"] = df["close"] / fv * initial
        print(f"    {len(df)} data points")
        return df["value"].tolist(), df["date"].tolist(), "Nifty 50 Index"
    except Exception as e:
        print(f"    Failed: {e}")
        return None, None, None


# ── Metrics ────────────────────────────────────────────────────────────────────
def compute_metrics(assets: list, dates: list):
    s = pd.Series(assets[1:], index=pd.to_datetime(dates[: len(assets) - 1]))
    daily = s.pct_change().dropna()

    total_return = (s.iloc[-1] / s.iloc[0] - 1) * 100
    annual_return = ((1 + total_return / 100) ** (252 / len(s)) - 1) * 100
    sharpe = np.sqrt(252) * daily.mean() / daily.std() if daily.std() != 0 else 0
    downside = daily[daily < 0]
    sortino = np.sqrt(252) * daily.mean() / downside.std() if len(downside) > 0 else 0
    max_dd = ((s / s.cummax()) - 1).min() * 100
    volatility = daily.std() * np.sqrt(252) * 100
    var = np.percentile(daily, 5)
    cvar = daily[daily <= var].mean() * 100

    return {
        "Cumulative Return (%)": round(total_return, 2),
        "Annual Return (%)":     round(annual_return, 2),
        "Max Drawdown (%)":      round(max_dd, 2),
        "Sharpe Ratio":          round(sharpe, 3),
        "Sortino Ratio":         round(sortino, 3),
        "Volatility (%)":        round(volatility, 2),
        "CVaR 5% (%)":           round(cvar, 2),
    }


# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_results(all_results, benchmark_assets, benchmark_dates, benchmark_name):
    plt.rcParams.update({"font.size": 14, "font.family": "sans-serif"})
    fig, ax = plt.subplots(figsize=(16, 8))

    colors = plt.cm.tab10.colors
    for i, result in enumerate(all_results):
        assets = result["assets"]
        dates  = result["dates"]
        label  = result["name"]

        s = pd.Series(assets[1:], index=pd.to_datetime(dates[: len(assets) - 1]))
        cum_ret = (s / s.iloc[0] - 1) * 100
        ax.plot(cum_ret, linewidth=2, label=label, color=colors[i % len(colors)])

    # Benchmark
    if benchmark_assets and benchmark_dates:
        bs = pd.Series(benchmark_assets, index=pd.to_datetime(benchmark_dates))
        # Align to trade period
        bs = bs[bs.index >= pd.to_datetime(TRADE_START)]
        bench_cum = (bs / bs.iloc[0] - 1) * 100
        ax.plot(bench_cum, linewidth=2, linestyle="--", label=benchmark_name,
                color="purple", alpha=0.8)

    ax.set_title("DAPO Agent Performance on Nifty 50 Market", fontsize=18, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, "DAPO_Nifty50_Comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Plot saved: {save_path}")
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["checkpoint/nifty50/nifty50_dapo_both_a1.0_b1.0.pth"],
        help="Paths to one or more model checkpoints",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Display names for each model (same order as --models)",
    )
    args = parser.parse_args()

    # ── Load trade data ────────────────────────────────────────────────────────
    print("\n📂 Loading trade data...")
    trade = load_trade_data()
    print(f"   Trade rows: {len(trade)}  "
          f"({trade['date'].min()} → {trade['date'].max()})")
    print(f"   Stocks: {trade['tic'].nunique()}")

    stock_dim      = trade["tic"].nunique()
    state_space    = 1 + 2 * stock_dim + (2 + len(INDICATORS)) * stock_dim
    env_kwargs = {
        "hmax"               : 100,
        "initial_amount"     : 1_000_000,
        "num_stock_shares"   : [0] * stock_dim,
        "buy_cost_pct"       : [0.001] * stock_dim,
        "sell_cost_pct"      : [0.001] * stock_dim,
        "state_space"        : state_space,
        "stock_dim"          : stock_dim,
        "tech_indicator_list": INDICATORS,
        "action_space"       : stock_dim,
        "reward_scaling"     : 1e-4,
    }

    # ── Benchmark ──────────────────────────────────────────────────────────────
    bench_assets, bench_dates, bench_name = get_nifty50_benchmark(
        TRADE_START, TRADE_END, initial=1_000_000
    )

    # ── Run each model ─────────────────────────────────────────────────────────
    labels     = args.labels or [f"Model {i+1}" for i in range(len(args.models))]
    all_results = []
    all_metrics = {}

    for model_path, label in zip(args.models, labels):
        print(f"\n [{label}] Loading from {model_path}...")

        if not os.path.exists(model_path):
            print(f"    File not found: {model_path}")
            continue

        # Fresh environment for each model
        env = StockTradingEnv_llm_risk(
            df=trade.copy(), turbulence_threshold=70,
            risk_indicator_col="vix", **env_kwargs,
        )
        obs_space = env.observation_space
        act_space = env.action_space

        # Load model
        actor = MLPActorCritic(obs_space, act_space, hidden_sizes=(256, 256))
        ckpt  = torch.load(model_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            actor.load_state_dict(ckpt["model_state_dict"])
            epoch = ckpt.get("epoch", "?")
            adj   = ckpt.get("adjustment_type", "?")
            print(f"    Loaded checkpoint (epoch={epoch}, adjustment={adj})")
        else:
            actor.load_state_dict(ckpt)
        actor.eval()

        assets, dates = run_prediction(actor, env)

        metrics = compute_metrics(assets, dates)
        all_metrics[label] = metrics

        all_results.append({"name": label, "assets": assets, "dates": dates})

        print(f"\n    {label} Metrics:")
        for k, v in metrics.items():
            print(f"      {k:30s} {v}")

    # ── Save metrics ───────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_path = os.path.join(RESULTS_DIR, "dapo_nifty50_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\n Metrics saved: {metrics_path}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    if all_results:
        plot_results(all_results, bench_assets, bench_dates, bench_name)
    else:
        print(" No model results to plot.")

    print("\n Backtest complete!")
