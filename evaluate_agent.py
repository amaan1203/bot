#!/usr/bin/env python3
"""
Full evaluation of a trained DAPO-NIFTY50 checkpoint using QuantStats.

Installation (first time):
    pip install quantstats pyfolio-reloaded yfinance

Usage:
    # Evaluate the default final checkpoint:
    python evaluate_agent.py

    # Evaluate a specific checkpoint:
    python evaluate_agent.py --checkpoint checkpoint/nifty50/nifty50_dapo_both_a1.0_b1.0.pth

    # Custom date range:
    python evaluate_agent.py --start 2023-01-01 --end 2025-01-01

Output:
    results/tearsheet.html      — full QuantStats HTML report
    results/returns.csv         — daily returns series
    Console                     — key metrics printed to stdout
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR     = "./dataset"
RESULTS_DIR     = "./results"
TRADE_PATH      = os.path.join(DATASET_DIR, "nifty50_trade_scores.csv")
DEFAULT_CKPT    = "checkpoint/nifty50/nifty50_dapo_both_a1.5_b3.0.pth"
DEFAULT_START   = "2023-01-01"
DEFAULT_END     = "2026-04-08"

try:
    from finrl.config import INDICATORS
except ImportError:
    INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
                  "dx_30", "close_30_sma", "close_60_sma"]

device = torch.device("cpu")   # inference is always on CPU


# ── Data loader ────────────────────────────────────────────────────────────────

def load_trade_data(start: str, fixed_tics: list | None = None) -> pd.DataFrame:
    """Load and align trade data. Mirrors logic from tune_optuna.py."""
    if not os.path.exists(TRADE_PATH):
        raise FileNotFoundError(
            f"Trade data not found: {TRADE_PATH}\n"
            "Run 2_prepare_dataset.py + 3_score_news_ollama.py first."
        )
    trade = pd.read_csv(TRADE_PATH)
    trade["date"] = pd.to_datetime(trade["date"]).dt.strftime("%Y-%m-%d")

    all_dates = sorted(trade["date"].unique())
    all_tics  = fixed_tics if fixed_tics is not None else sorted(trade["tic"].unique())
    # Build template for all ticks on all dates
    template = pd.DataFrame(
        [(d, t) for d in all_dates for t in all_tics],
        columns=["date", "tic"]
    )
    trade = template.merge(trade, on=["date", "tic"], how="left")

    # Fill missing values
    numeric_cols = trade.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["llm_sentiment", "llm_risk"]:
            trade[col] = trade.groupby("tic")[col].transform(lambda x: x.ffill().bfill())
            if col == "close":
                trade[col] = trade[col].fillna(1.0)
            else:
                trade[col] = trade[col].fillna(0.0)

    # Safety: Ensure close price is NEVER zero
    trade["close"] = trade["close"].replace(0.0, 1.0)
    
    trade["llm_sentiment"] = trade["llm_sentiment"].fillna(0.0)
    trade["llm_risk"]      = trade["llm_risk"].fillna(0.5)

    trade = trade[trade["date"] >= start].copy()
    unique_dates = sorted(trade["date"].unique())
    trade["new_idx"] = trade["date"].map({d: i for i, d in enumerate(unique_dates)})
    trade = trade.set_index("new_idx")
    return trade


# ── Agent loader ───────────────────────────────────────────────────────────────

def load_agent(checkpoint_path: str, obs_dim: int, act_dim: int):
    """Load MLPActorCritic from a checkpoint produced by dapo_algorithm.py."""
    from dapo_algorithm import MLPActorCritic
    from gymnasium.spaces import Box

    obs_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    act_space = Box(low=-1, high=1, shape=(act_dim,),  dtype=np.float32)

    ac = MLPActorCritic(obs_space, act_space, hidden_sizes=(256, 256))
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ac.load_state_dict(ckpt["model_state_dict"])
        print(f"   Loaded checkpoint  : epoch={ckpt.get('epoch', '?')}, "
              f"adj={ckpt.get('adjustment_type', '?')}")
    else:
        ac.load_state_dict(ckpt)

    ac.eval()
    return ac


# ── Run agent through environment ──────────────────────────────────────────────

def get_account_values(ac, env) -> pd.Series:
    """Run agent deterministically through env. Returns account-value series indexed by date."""
    obs, _ = env.reset()
    done   = False
    with torch.no_grad():
        while not done:
            action = ac.act(obs)
            obs, _reward, done, _trunc, _info = env.step(action)

    df = env.save_asset_memory()
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["account_value"]


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(checkpoint_path: str, start_date: str, end_date: str):
    try:
        import quantstats as qs
        import yfinance as yf
    except ImportError:
        print("ERROR: quantstats / yfinance not installed.")
        print("       Run: pip install quantstats pyfolio-reloaded yfinance")
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data & build env
    print("\n Loading trade data (aligning with train tickers)...")
    
    # Try to find master list of tickers from training data to ensure obs_dim consistency
    train_path = os.path.join(DATASET_DIR, "nifty50_train_scores.csv")
    master_tics = None
    if os.path.exists(train_path):
        master_tics = sorted(pd.read_csv(train_path)["tic"].unique())
    
    trade    = load_trade_data(start=start_date, fixed_tics=master_tics)
    stock_dim = trade["tic"].nunique()
    state_space = 1 + 2 * stock_dim + (2 + len(INDICATORS)) * stock_dim
    print(f"   Stocks      : {stock_dim}")
    print(f"   State space : {state_space}")
    print(f"   Trade rows  : {len(trade)}")

    from env_stocktrading_llm_risk import StockTradingEnv
    env = StockTradingEnv(
        df=trade,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=1_000_000,
        num_stock_shares=[0] * stock_dim,
        buy_cost_pct=[0.001] * stock_dim,
        sell_cost_pct=[0.001] * stock_dim,
        reward_scaling=1e-4,
        state_space=state_space,
        action_space=stock_dim,
        tech_indicator_list=INDICATORS,
        turbulence_threshold=70,
        risk_indicator_col="vix",
    )

    # Quick state-dim assertion
    _obs, _ = env.reset()
    assert len(_obs) == state_space, (
        f"State dim mismatch in evaluate_agent.py: got {len(_obs)}, expected {state_space}"
    )
    env.reset()  # reset again for clean run

    # Load agent
    print(f"\n Loading checkpoint: {checkpoint_path}")
    ac = load_agent(checkpoint_path, obs_dim=state_space, act_dim=stock_dim)

    # Run agent
    print("\n Running backtest...")
    account_values = get_account_values(ac, env)
    returns = account_values.pct_change().dropna()
    returns.name = "DAPO-NIFTY50"

    # Save returns
    returns_path = os.path.join(RESULTS_DIR, "returns.csv")
    returns.to_csv(returns_path)
    print(f"   Returns saved: {returns_path}")

    # Fetch NIFTY-50 benchmark
    print("\n Downloading NIFTY-50 benchmark (^NSEI)...")
    try:
        nifty     = yf.download("^NSEI", start=start_date, end=end_date,
                                 auto_adjust=True, progress=False)
        if isinstance(nifty.columns, pd.MultiIndex):
            nifty.columns = nifty.columns.droplevel(1)
        benchmark = nifty["Close"].pct_change().dropna()
        benchmark.index = pd.to_datetime(benchmark.index).tz_localize(None)
        benchmark.name  = "NIFTY-50"
    except Exception as e:
        print(f"   Benchmark download failed ({e}). Running without benchmark.")
        benchmark = None

    # Align dates
    if benchmark is not None:
        returns, benchmark = returns.align(benchmark, join="inner")

    # ── Console metrics ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  DAPO-NIFTY50 Performance Metrics")
    print("=" * 55)
    print(f"  Annual return   : {qs.stats.cagr(returns):.2%}")
    print(f"  Sharpe ratio    : {qs.stats.sharpe(returns):.4f}")
    print(f"  Sortino ratio   : {qs.stats.sortino(returns):.4f}")
    print(f"  Calmar ratio    : {qs.stats.calmar(returns):.4f}")
    print(f"  Max drawdown    : {qs.stats.max_drawdown(returns):.2%}")
    print(f"  Win rate        : {qs.stats.win_rate(returns):.2%}")
    print(f"  Ann. volatility : {qs.stats.volatility(returns):.2%}")
    if benchmark is not None:
        print(f"\n  NIFTY-50 Sharpe : {qs.stats.sharpe(benchmark):.4f}")
        print(f"  NIFTY-50 return : {qs.stats.cagr(benchmark):.2%}")
    print("=" * 55)

    # ── Full HTML tearsheet ───────────────────────────────────────────────────
    tearsheet_path = os.path.join(RESULTS_DIR, "tearsheet.html")
    print(f"\n Generating QuantStats tearsheet → {tearsheet_path}")
    if benchmark is not None:
        qs.reports.full(returns, benchmark=benchmark, output=tearsheet_path)
    else:
        qs.reports.full(returns, output=tearsheet_path)
    print(f"   Open in browser: file://{os.path.abspath(tearsheet_path)}")

    return returns, benchmark


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DAPO-NIFTY50 checkpoint with QuantStats")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--start",      type=str, default=DEFAULT_START,
                        help="Trade start date (YYYY-MM-DD)")
    parser.add_argument("--end",        type=str, default=DEFAULT_END,
                        help="Trade end date (YYYY-MM-DD)")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("       Run 4_train_dapo_nifty.py first, then specify the output .pth file.")
        sys.exit(1)

    evaluate(
        checkpoint_path=args.checkpoint,
        start_date=args.start,
        end_date=args.end,
    )
