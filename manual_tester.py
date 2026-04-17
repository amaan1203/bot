#!/usr/bin/env python3
"""
Manual Inference Tester for DAPO-NIFTY50.
Allows users to see daily decisions, sentiment scores, and portfolio states.

Usage:
    python manual_tester.py --days 10 --start 2023-01-01
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import traceback
from gymnasium.spaces import Box

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = "./dataset"
TRADE_PATH   = os.path.join(DATASET_DIR, "nifty50_trade_scores.csv")
BEST_MODEL   = "checkpoint/nifty50/nifty50_dapo_both_a1.5_b3.0.pth"

INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
              "dx_30", "close_30_sma", "close_60_sma"]

DEVICE = torch.device("cpu")

# ── Model Definition (Match dapo_algorithm.py) ────────────────────────────────
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

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32).to(DEVICE)
            pi  = self.pi._distribution(obs)
            a   = pi.mean  # Use mean for deterministic inference
        return a.cpu().numpy()

# ── Manual Loop ────────────────────────────────────────────────────────────────
def run_manual_test(days_to_run, start_date):
    from env_stocktrading_llm_risk import StockTradingEnv

    # 1. Load Data
    print(f" Loading data from {TRADE_PATH}...")
    trade = pd.read_csv(TRADE_PATH)
    trade["date"] = pd.to_datetime(trade["date"]).dt.strftime("%Y-%m-%d")
    
    # MASTER TICKER ALIGNMENT (Essential for 50-stock dim)
    # We must have 50 tickers to match the model. Use training set to get the master list.
    train_path = os.path.join(DATASET_DIR, "nifty50_train_scores.csv")
    train_df = pd.read_csv(train_path)
    all_tics = sorted(train_df["tic"].unique())  # Guaranteed 50 tics
    stock_dim = len(all_tics)
    print(f"   Model expects {stock_dim} tickers. Aligning trade data...")

    # Filter and Align
    trade = trade[trade["date"] >= start_date].copy()
    unique_dates = sorted(trade["date"].unique())
    
    # Reindex trade data to ensure every date has all 50 tickers
    aligned_list = []
    for d in unique_dates:
        day_df = trade[trade["date"] == d]
        # Find missing tics
        missing_tics = set(all_tics) - set(day_df["tic"].unique())
        if missing_tics:
            # Create dummy rows for missing tics
            dummies = pd.DataFrame([
                {
                    "date": d, "tic": t, "close": 0.0, 
                    "llm_sentiment": 0.0, "llm_risk": 0.5,
                    "vix": day_df["vix"].iloc[0] if not day_df.empty else 20.0
                } for t in missing_tics
            ])
            # Fill other indicators with 0
            for col in INDICATORS: dummies[col] = 0.0
            day_df = pd.concat([day_df, dummies], ignore_index=True)
        
        # Sort by tic to match state vector order
        day_df = day_df.sort_values("tic")
        aligned_list.append(day_df)
    
    trade = pd.concat(aligned_list)
    trade["new_idx"] = trade["date"].map({d: i for i, d in enumerate(unique_dates)})
    trade = trade.set_index("new_idx")
    
    state_space = 1 + 2 * stock_dim + (2 + len(INDICATORS)) * stock_dim
    
    # 2. Setup Env
    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1_000_000,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct": [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dim,
        "reward_scaling": 1e-4,
    }
    
    env = StockTradingEnv(df=trade, turbulence_threshold=70, risk_indicator_col="vix", **env_kwargs)
    
    # 3. Load Model
    print(f" Loading model from {BEST_MODEL}...")
    checkpoint = torch.load(BEST_MODEL, map_location=DEVICE)
    ac = MLPActorCritic(env.observation_space, env.action_space, hidden_sizes=(256, 256))
    ac.load_state_dict(checkpoint["model_state_dict"])
    ac.eval()
    
    # 4. Run Loop
    obs, _ = env.reset()
    print("\n" + "="*80)
    print(f"{'DATE':<12} | {'TIC':<10} | {'SENT':<6} | {'RISK':<6} | {'ACTION':<8} | {'PRICE':<10}")
    print("-" * 80)
    
    results_data = []
    current_day = 0
    while current_day < days_to_run:
        date = env._get_date()
        action = ac.act(obs)
        
        # Take step
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Portfolio Value calculation
        cash = env.state[0]
        prices = np.array(env.state[1:stock_dim+1])
        shares = np.array(env.state[stock_dim+1:2*stock_dim+1])
        total_val = cash + np.sum(prices * shares)

        # Get trade details for this day
        sorted_indices = np.argsort(action)
        top_tics = [all_tics[i] for i in sorted_indices[-3:][::-1]]
        bot_tics = [all_tics[i] for i in sorted_indices[:3]]
        representative_tics = top_tics + bot_tics
        
        for tic in representative_tics:
            try:
                idx = all_tics.index(tic)
                a_val = action[idx]
                
                # Get sentiment/risk from the raw data
                row = trade[(trade["date"] == date) & (trade["tic"] == tic)]
                if not row.empty:
                    sent = float(row.iloc[0]["llm_sentiment"])
                    risk = float(row.iloc[0]["llm_risk"])
                    price = float(row.iloc[0]["close"])
                else:
                    sent, risk, price = 0.0, 0.5, 0.0
                
                action_str = "BUY" if a_val > 0.05 else ("SELL" if a_val < -0.05 else "HOLD")
                print(f"{date:<12} | {tic:<10} | {sent:>6.2f} | {risk:>6.2f} | {action_str:<8} | {price:>10.2f}")
                
                results_data.append({
                    "date": date,
                    "ticker": tic,
                    "sentiment": sent,
                    "risk": risk,
                    "action": action_str,
                    "action_value": a_val,
                    "price": price,
                    "portfolio_value": total_val
                })
            except Exception:
                continue
        
        print(f"--- Day {current_day+1} Summary: Portfolio Value: {total_val:,.2f} ---")
        
        obs = next_obs
        current_day += 1
        if done or truncated:
            break

    # 5. Save Results
    if args.output:
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(args.output, index=False)
        print(f"\n[SUCCESS] Results saved to: {args.output}")

    print("="*80)
    print("\nManual Test Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",  type=int, default=5, help="Number of days to simulate")
    parser.add_argument("--start", type=str, default="2023-01-02", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="manual_test_results.csv", help="CSV file to save results")
    args = parser.parse_args()
    
    try:
        run_manual_test(args.days, args.start)
    except Exception:
        traceback.print_exc()
