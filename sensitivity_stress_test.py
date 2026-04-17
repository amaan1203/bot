#!/usr/bin/env python3
"""
Sensitivity Stress Test Framework for DAPO-NIFTY50.

Sweeps alpha/beta values and signal modes at INFERENCE TIME (no retraining).
Measures how much the DeepSeek LLM signal contributes to profit.

Usage:
    python sensitivity_stress_test.py --days 100 --start 2025-01-01
"""

import os
import argparse
import itertools
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR = "./dataset"
TRADE_PATH  = os.path.join(DATASET_DIR, "nifty50_trade_scores.csv")
TRAIN_PATH  = os.path.join(DATASET_DIR, "nifty50_train_scores.csv")
BEST_MODEL  = "checkpoint/nifty50/nifty50_dapo_both_a1.5134_b1.0542.pth"
RESULTS_DIR = "./stress_test_results"
HMAX        = 100
INITIAL     = 1_000_000

INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
              "dx_30", "close_30_sma", "close_60_sma"]

# ── Alpha/Beta Grid ─────────────────────────────────────────────────────────────
ALPHAS  = [0.0, 0.5, 1.0, 1.51, 2.0, 3.0]
BETAS   = [0.0, 0.5, 1.0, 1.05, 2.0, 3.0]

# ── Signal Modes ────────────────────────────────────────────────────────────────
# 'none'           → No LLM signal (pure RL baseline)
# 'sentiment_only' → Only sentiment scales actions
# 'risk_only'      → Only risk scales actions
# 'both'           → Full combined signal (production mode)
MODES = ["none", "sentiment_only", "risk_only", "both"]

DEVICE = torch.device("cpu")

# ── Model Definition ────────────────────────────────────────────────────────────
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
            a   = pi.mean  # Deterministic inference
        return a.cpu().numpy()

# ── Data Loading ────────────────────────────────────────────────────────────────
def load_and_align_data(start_date, days_to_run):
    """Load trade data and align tickers to match model's 50-stock dim."""
    trade = pd.read_csv(TRADE_PATH)
    trade["date"] = pd.to_datetime(trade["date"]).dt.strftime("%Y-%m-%d")

    train_df = pd.read_csv(TRAIN_PATH)
    all_tics = sorted(train_df["tic"].unique())
    stock_dim = len(all_tics)

    trade = trade[trade["date"] >= start_date].copy()

    # Limit to the number of days requested
    unique_dates = sorted(trade["date"].unique())[:days_to_run]
    trade = trade[trade["date"].isin(unique_dates)].copy()

    # Reindex to ensure every date has all 50 tickers
    aligned_list = []
    for d in unique_dates:
        day_df = trade[trade["date"] == d].copy()
        missing = set(all_tics) - set(day_df["tic"].unique())
        if missing:
            dummies = pd.DataFrame([{
                "date": d, "tic": t, "close": 0.0,
                "llm_sentiment": 0.0, "llm_risk": 0.5,
                "vix": day_df["vix"].iloc[0] if not day_df.empty else 20.0,
                **{ind: 0.0 for ind in INDICATORS}
            } for t in missing])
            day_df = pd.concat([day_df, dummies], ignore_index=True)
        day_df = day_df.sort_values("tic")
        aligned_list.append(day_df)

    trade = pd.concat(aligned_list)
    trade["new_idx"] = trade["date"].map({d: i for i, d in enumerate(unique_dates)})
    trade = trade.set_index("new_idx")

    return trade, all_tics, stock_dim, unique_dates

# ── Signal Adjustment ───────────────────────────────────────────────────────────
def compute_adjustment(day_df, all_tics, mode, alpha, beta):
    """
    Compute per-stock action scaling factors based on LLM signals.
    Returns a numpy array of shape (stock_dim,).
    """
    stock_dim = len(all_tics)
    adjustments = np.ones(stock_dim)

    if mode == "none":
        return adjustments  # Identity: no scaling

    # Build sentiment and risk arrays (sorted by tic)
    sentiments = np.zeros(stock_dim)
    risks = np.full(stock_dim, 0.5)
    for i, tic in enumerate(all_tics):
        row = day_df[day_df["tic"] == tic]
        if not row.empty:
            sentiments[i] = float(row.iloc[0]["llm_sentiment"])
            risks[i]      = float(row.iloc[0]["llm_risk"])

    # Shift sentiment from [-1,1] → [0.1, 2.0] so it's always positive for power op
    sent_shifted = np.clip((sentiments + 1.0) / 2.0 * 2.0, 0.1, 2.0)
    risk_safe    = np.clip(risks, 0.1, 2.0)

    if mode == "sentiment_only":
        adjustments = sent_shifted ** alpha
    elif mode == "risk_only":
        adjustments = 1.0 / (risk_safe ** beta + 1e-8)
    elif mode == "both":
        adjustments = (sent_shifted ** alpha) / (risk_safe ** beta + 1e-8)

    # Clip to reasonable range so we don't blow up actions
    adjustments = np.clip(adjustments, 0.1, 5.0)
    return adjustments

# ── Single Run ──────────────────────────────────────────────────────────────────
def run_single(actor, env, trade, all_tics, unique_dates, mode, alpha, beta, stock_dim):
    """Run a full episode with one (mode, alpha, beta) config. Returns metrics dict."""
    obs, _ = env.reset()
    portfolio_values = [INITIAL]
    days_run = 0

    for day_idx, date in enumerate(unique_dates):
        raw_action = actor.act(obs)

        # Get day data for signal lookup
        day_df = trade[trade["date"] == date] if "date" in trade.columns else \
                 trade.loc[[day_idx]] if day_idx in trade.index else pd.DataFrame()

        # Scale actions by LLM signals
        adj = compute_adjustment(day_df, all_tics, mode, alpha, beta)
        scaled_action = raw_action * adj

        try:
            obs, _, done, truncated, _ = env.step(scaled_action)
        except Exception:
            break

        # Record portfolio value
        cash = env.state[0]
        prices = np.array(env.state[1:stock_dim+1])
        shares = np.array(env.state[stock_dim+1:2*stock_dim+1])
        pv = cash + np.sum(prices * shares)
        portfolio_values.append(pv)
        days_run += 1

        if done or truncated:
            break

    # Compute metrics
    pv_arr    = np.array(portfolio_values)
    final_val = pv_arr[-1]
    total_ret = (final_val - INITIAL) / INITIAL * 100.0
    peak      = pv_arr.max()
    drawdown  = ((pv_arr - np.maximum.accumulate(pv_arr)) / np.maximum.accumulate(pv_arr)).min() * 100.0
    win_days  = int(np.sum(pv_arr[1:] > INITIAL))
    win_rate  = win_days / max(days_run, 1) * 100.0

    return {
        "mode": mode,
        "alpha": alpha,
        "beta": beta,
        "days_run": days_run,
        "final_value": round(final_val, 2),
        "net_profit": round(final_val - INITIAL, 2),
        "total_return_pct": round(total_ret, 4),
        "peak_value": round(peak, 2),
        "max_drawdown_pct": round(drawdown, 4),
        "win_rate_pct": round(win_rate, 2),
        "portfolio_series": pv_arr.tolist(),
    }

# ── Plotting ────────────────────────────────────────────────────────────────────
def plot_heatmap(df, output_dir):
    """2D heatmap of Return% for alpha vs beta (mode='both' only)."""
    both_df = df[df["mode"] == "both"].copy()
    if both_df.empty:
        return

    pivot = both_df.pivot_table(
        index="alpha", columns="beta", values="total_return_pct", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=pivot.values.min(), vmax=pivot.values.max())
    plt.colorbar(im, ax=ax, label="Total Return %")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"β={b:.2f}" for b in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"α={a:.2f}" for a in pivot.index])
    ax.set_title("Sensitivity Stress Test — Return% Heatmap\n(alpha vs beta, mode=both)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Beta (Risk Exponent)", fontsize=12)
    ax.set_ylabel("Alpha (Sentiment Exponent)", fontsize=12)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                    fontsize=8, color="black" if -5 < val < 5 else "white")

    plt.tight_layout()
    path = os.path.join(output_dir, "stress_test_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] {path}")

def plot_modes_comparison(df, output_dir):
    """Bar chart comparing best return per mode."""
    best_per_mode = df.groupby("mode")["total_return_pct"].max().reset_index()
    best_per_mode.columns = ["mode", "best_return_pct"]
    best_per_mode = best_per_mode.sort_values("best_return_pct", ascending=True)

    colors = {
        "none": "#e74c3c",
        "sentiment_only": "#3498db",
        "risk_only": "#e67e22",
        "both": "#27ae60",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(best_per_mode["mode"],
                   best_per_mode["best_return_pct"],
                   color=[colors.get(m, "#7f8c8d") for m in best_per_mode["mode"]],
                   edgecolor="white", linewidth=1.5, height=0.5)

    for bar, val in zip(bars, best_per_mode["best_return_pct"]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=11, fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Best Total Return % (over run period)", fontsize=12)
    ax.set_title("Sensitivity Stress Test — Signal Mode Comparison\n(Best Return per Mode)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(left=min(best_per_mode["best_return_pct"].min() - 1, -1))
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "stress_test_modes.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] {path}")

def plot_portfolio_curves(results_list, unique_dates, output_dir):
    """Overlay best portfolio curve per mode."""
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {"none": "#e74c3c", "sentiment_only": "#3498db",
              "risk_only": "#e67e22", "both": "#27ae60"}

    best_per_mode = {}
    for r in results_list:
        m = r["mode"]
        if m not in best_per_mode or r["total_return_pct"] > best_per_mode[m]["total_return_pct"]:
            best_per_mode[m] = r

    x = range(len(unique_dates) + 1)
    for mode, r in best_per_mode.items():
        series = r["portfolio_series"]
        label = f"{mode} (α={r['alpha']:.2f}, β={r['beta']:.2f}) → {r['total_return_pct']:.2f}%"
        ax.plot(list(x)[:len(series)], series,
                label=label, color=colors.get(mode, "gray"), linewidth=2)

    ax.axhline(INITIAL, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Starting Capital")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"₹{x/1e6:.3f}M"))
    ax.set_xlabel("Trading Days", fontsize=12)
    ax.set_ylabel("Portfolio Value", fontsize=12)
    ax.set_title("Sensitivity Stress Test — Best Portfolio Curve per Signal Mode",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "stress_test_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [Saved] {path}")

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",   type=int, default=100, help="Number of trading days to simulate")
    parser.add_argument("--start",  type=str, default="2025-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--alphas", type=float, nargs="+", default=ALPHAS, help="Alpha values to sweep")
    parser.add_argument("--betas",  type=float, nargs="+", default=BETAS,  help="Beta values to sweep")
    parser.add_argument("--modes",  type=str,   nargs="+", default=MODES,  help="Signal modes to test")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "="*70)
    print("  DAPO-NIFTY50 Sensitivity Stress Test Framework")
    print("="*70)
    print(f"  Start Date  : {args.start}")
    print(f"  Days        : {args.days}")
    print(f"  Alpha values: {args.alphas}")
    print(f"  Beta values : {args.betas}")
    print(f"  Modes       : {args.modes}")

    # 1. Load Data
    print("\n[1/4] Loading and aligning trade data...")
    trade, all_tics, stock_dim, unique_dates = load_and_align_data(args.start, args.days)
    print(f"      Tickers: {stock_dim}, Dates: {len(unique_dates)}")

    # 2. Load Model (once)
    print("[2/4] Loading model...")
    from env_stocktrading_llm_risk import StockTradingEnv
    state_space = 1 + 2 * stock_dim + (2 + len(INDICATORS)) * stock_dim

    env_kwargs = {
        "hmax": HMAX, "initial_amount": INITIAL,
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct":  [0.001] * stock_dim,
        "sell_cost_pct": [0.001] * stock_dim,
        "state_space": state_space, "stock_dim": stock_dim,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dim, "reward_scaling": 1e-4,
    }

    env_ref = StockTradingEnv(df=trade.copy(), turbulence_threshold=70,
                               risk_indicator_col="vix", **env_kwargs)

    checkpoint = torch.load(BEST_MODEL, map_location=DEVICE)
    actor = MLPActorCritic(env_ref.observation_space, env_ref.action_space, hidden_sizes=(256, 256))
    actor.load_state_dict(checkpoint["model_state_dict"])
    actor.eval()
    print(f"      Model loaded from {BEST_MODEL}")

    # 3. Build config grid
    configs = []
    for mode in args.modes:
        for alpha, beta in itertools.product(args.alphas, args.betas):
            if mode == "none":
                # For baseline, alpha/beta don't matter — only run once
                if alpha == args.alphas[0] and beta == args.betas[0]:
                    configs.append((mode, 0.0, 0.0))
            elif mode == "sentiment_only":
                # Beta doesn't matter for sentiment_only
                if beta == args.betas[0]:
                    configs.append((mode, alpha, 0.0))
            elif mode == "risk_only":
                # Alpha doesn't matter for risk_only
                if alpha == args.alphas[0]:
                    configs.append((mode, 0.0, beta))
            else:  # 'both'
                configs.append((mode, alpha, beta))

    total = len(configs)
    print(f"\n[3/4] Running {total} configurations...")

    results_list = []
    for mode, alpha, beta in tqdm(configs, desc="Stress Test", ncols=70):
        env = StockTradingEnv(df=trade.copy(), turbulence_threshold=70,
                               risk_indicator_col="vix", **env_kwargs)
        result = run_single(actor, env, trade, all_tics, unique_dates, mode, alpha, beta, stock_dim)
        results_list.append(result)

    # 4. Save results
    print("\n[4/4] Saving results and generating charts...")
    df_results = pd.DataFrame([{k: v for k, v in r.items() if k != "portfolio_series"}
                                for r in results_list])
    df_results = df_results.sort_values("total_return_pct", ascending=False)

    csv_path = os.path.join(RESULTS_DIR, "stress_test_results.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"  [Saved] {csv_path}")

    # Generate plots
    plot_heatmap(df_results, RESULTS_DIR)
    plot_modes_comparison(df_results, RESULTS_DIR)
    plot_portfolio_curves(results_list, unique_dates, RESULTS_DIR)

    # Print summary table
    print("\n" + "="*70)
    print("  RESULTS SUMMARY — Top 10 Configs by Return")
    print("="*70)
    print(df_results[["mode", "alpha", "beta", "total_return_pct",
                       "net_profit", "max_drawdown_pct", "win_rate_pct"]].head(10).to_string(index=False))

    print("\n  MODE COMPARISON — Best Return per Mode")
    print("-"*50)
    for mode in args.modes:
        mode_df = df_results[df_results["mode"] == mode]
        if not mode_df.empty:
            best = mode_df.iloc[0]
            print(f"  {mode:<16} → {best['total_return_pct']:>7.2f}%  "
                  f"(α={best['alpha']:.2f}, β={best['beta']:.2f}, "
                  f"profit=₹{best['net_profit']:,.0f})")

    print("\n" + "="*70)
    print(f"  All files saved to: {os.path.abspath(RESULTS_DIR)}/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
