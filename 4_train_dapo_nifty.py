#!/usr/bin/env python3
"""
Step 4: Train the DAPO agent on Nifty50 data.
Reads from ./dataset/nifty50_train_risk.csv and ./dataset/nifty50_train_sentiment.csv
(or falls back to base data with neutral LLM scores).

Usage:
    python 4_train_dapo_nifty.py --adjustment_type both --alpha 1.0 --beta 1.0

    # For faster testing (1 epoch):
    python 4_train_dapo_nifty.py --epochs 1
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import time
import argparse

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR   = "./dataset"
CHECKPOINT_DIR = "./checkpoint/nifty50"

# TRAIN split: use data up to this date
TRAIN_END = "2022-12-31"

# ── Device: prefer MPS (Apple Silicon) → CUDA → CPU ───────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple Silicon MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


def load_and_merge_train_data():
    """
    Load training data. Prioritizes the combined scores file,
    falls back to base data + individual score merges.
    """
    scores_path    = os.path.join(DATASET_DIR, "nifty50_train_scores.csv")
    risk_path      = os.path.join(DATASET_DIR, "nifty50_news_risk.csv")
    sentiment_path = os.path.join(DATASET_DIR, "nifty50_news_sentiment.csv")
    base_path      = os.path.join(DATASET_DIR, "nifty50_train_base.csv")

    # ── Case 1: Combined scores file available ────────────────────────────────
    if os.path.exists(scores_path):
        print(" Loading pre-scored train data (prices + indicators + LLM scores)")
        train = pd.read_csv(scores_path)

    # ── Case 2: Merge from base + individual score files ──────────────────────
    else:
        if not os.path.exists(base_path):
            raise FileNotFoundError(
                f"No training data found. Run 2_prepare_dataset.py first.\n"
                f"Expected: {base_path}"
            )
        
        print(f" Loading base data and merging individual scores...")
        train = pd.read_csv(base_path)
        
        # Merge Sentiment
        if os.path.exists(sentiment_path):
            print("   Merging sentiment scores...")
            sent = pd.read_csv(sentiment_path)
            # Take daily average if multiple headlines exist
            daily_sent = sent.groupby(["date", "tic"])["llm_sentiment"].mean().reset_index()
            train = train.merge(daily_sent, on=["date", "tic"], how="left")
        
        # Merge Risk
        if os.path.exists(risk_path):
            print("   Merging risk scores...")
            risk = pd.read_csv(risk_path)
            # Take daily average if multiple headlines exist
            daily_risk = risk.groupby(["date", "tic"])["llm_risk"].mean().reset_index()
            train = train.merge(daily_risk, on=["date", "tic"], how="left")

        # Fill missing scores with neutral defaults
        if "llm_sentiment" not in train.columns:
            train["llm_sentiment"] = 0.0
        if "llm_risk" not in train.columns:
            train["llm_risk"] = 0.5
        
        train["llm_sentiment"] = train["llm_sentiment"].fillna(0.0)
        train["llm_risk"]      = train["llm_risk"].fillna(0.5)

    # ── Filter to train period ────────────────────────────────────────────────
    train["date"] = pd.to_datetime(train["date"]).dt.strftime("%Y-%m-%d")
    train = train[train["date"] <= TRAIN_END].copy()

    # ── Build integer date index ──────────────────────────────────────────────
    unique_dates = sorted(train["date"].unique())
    date_to_idx  = {d: i for i, d in enumerate(unique_dates)}
    train["new_idx"] = train["date"].map(date_to_idx)
    train = train.set_index("new_idx")

    print(f"   Training rows  : {len(train)}")
    print(f"   Date range     : {train['date'].min()} → {train['date'].max()}")
    print(f"   Unique stocks  : {train['tic'].nunique()}")
    
    # Summary of scores
    print(f"   Sentiment mean : {train['llm_sentiment'].mean():.3f}")
    print(f"   Risk mean      : {train['llm_risk'].mean():.3f}")

    return train


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DAPO agent on Nifty50 data")
    parser.add_argument("--hid",            type=int,   default=256)
    parser.add_argument("--l",              type=int,   default=2,     help="Number of hidden layers")
    parser.add_argument("--seed",     "-s", type=int,   default=42)
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--steps",          type=int,   default=32000, help="Steps per epoch")
    parser.add_argument("--exp_name",       type=str,   default="dapo_nifty50_full_fixed")
    parser.add_argument(
        "--adjustment_type", type=str, default="both",
        choices=["both", "sentiment", "risk", "none"],
    )
    parser.add_argument("--alpha", type=float, default=1.5, help="Sentiment exponent")
    parser.add_argument("--beta",  type=float, default=3.0, help="Risk exponent")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n Loading training data...")
    train = load_and_merge_train_data()

    # ── Data Alignment: Ensure every date has every ticker ────────────────────
    # This prevents observation shape mismatches in the environment
    print(" Aligning data (ensuring all tickers present on all dates)...")
    all_dates = sorted(train["date"].unique())
    all_tics  = sorted(train["tic"].unique())
    
    full_idx = pd.MultiIndex.from_product([all_dates, all_tics], names=["date", "tic"])
    train = train.reset_index().set_index(["date", "tic"]).reindex(full_idx).reset_index()
    
    # Sort by date then ticker (critical for the environment's observation logic)
    train = train.sort_values(["date", "tic"])
    
    # Forward-fill prices and indicators per ticker, then fill remaining with 0
    # LLM scores get defaults
    print("   Filling missing values...")
    train["llm_sentiment"] = train["llm_sentiment"].fillna(0.0)
    train["llm_risk"]      = train["llm_risk"].fillna(0.5)
    
    # Identify numeric columns to fill
    numeric_cols = train.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ["llm_sentiment", "llm_risk"]:
            train[col] = train.groupby("tic")[col].transform(lambda x: x.ffill().bfill()).fillna(0.0)

    # Re-build integer date index
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    train["new_idx"] = train["date"].map(date_to_idx)
    train = train.set_index("new_idx")

    # ── Environment setup ─────────────────────────────────────────────────────
    # Import here to avoid top-level FinRL dependency issues
    try:
        from finrl.config import INDICATORS
    except ImportError:
        # Fallback: match the exact indicators used in the original pipeline
        INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30",
                      "dx_30", "close_30_sma", "close_60_sma"]
        print(f"FinRL not found, using default INDICATORS: {INDICATORS}")

    # Verify all indicator columns exist
    missing = [ind for ind in INDICATORS if ind not in train.columns]
    if missing:
        print(f"Missing indicator columns: {missing}")
        print(f"   Available: {list(train.columns)}")
        sys.exit(1)

    from env_stocktrading_llm_risk import StockTradingEnv

    stock_dim   = train["tic"].nunique()
    state_space = 1 + 2 * stock_dim + (2 + len(INDICATORS)) * stock_dim  # +sentiment +risk
    print(f"\n Stock dim: {stock_dim}, State space: {state_space}")

    env_kwargs = {
        "hmax"              : 100,
        "initial_amount"    : 1000000,
        "num_stock_shares"  : [0] * stock_dim,
        "buy_cost_pct"      : [0.001] * stock_dim,   # 0.1% transaction cost
        "sell_cost_pct"     : [0.001] * stock_dim,
        "state_space"       : state_space,
        "stock_dim"         : stock_dim,
        "tech_indicator_list": INDICATORS,
        "action_space"      : stock_dim,
        "reward_scaling"    : 1e-4,
    }

    print("  Creating training environment...")
    e_train_gym = StockTradingEnv(df=train, turbulence_threshold=70,
                                  risk_indicator_col="vix", **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    # ── State-dim sanity check ────────────────────────────────────────────────
    _obs, _ = e_train_gym.reset()
    _expected = e_train_gym.observation_space.shape[0]
    assert len(_obs) == _expected, (
        f"State dim mismatch after reset(): got {len(_obs)}, expected {_expected}. "
        "Check _initiate_state() — LLM columns must be present in BOTH the "
        "initial=True AND initial=False (warm-start) branches."
    )
    print(f"  State-dim check passed: obs shape = {len(_obs)}")

    # ── Training ──────────────────────────────────────────────────────────────
    from dapo_algorithm import dapo, MLPActorCritic

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Patch dapo_algorithm: checkpoint path + device (MPS/CUDA/CPU)
    import dapo_algorithm as dapo_mod
    dapo_mod.checkpoint_dir = CHECKPOINT_DIR   # override module-level constant
    dapo_mod.device = DEVICE                   # propagate MPS/CUDA/CPU device

    # Also expose checkpoint dir via env var so dapo() inner function picks it up
    os.environ["DAPO_CHECKPOINT_DIR"] = CHECKPOINT_DIR

    try:
        from spinup.utils.run_utils import setup_logger_kwargs
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
    except ImportError:
        logger_kwargs = {}
        print("  spinup not found — logging disabled")

    print(f"\n  Starting DAPO training ({args.epochs} epochs, {args.steps} steps/epoch)")
    print(f"   Adjustment: {args.adjustment_type}  α={args.alpha}  β={args.beta}")
    print(f"   Checkpoint: {CHECKPOINT_DIR}")

    trained_model = dapo(
        lambda: env_train,
        actor_critic=MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        seed=args.seed,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
        num_samples_per_state=20,
        env_kwargs=env_kwargs,
        epsilon_low=0.20,
        epsilon_high=0.35,
        pi_lr=3.0e-05,
        gamma=0.99,
        train_pi_iters=100,
        adjustment_type=args.adjustment_type,
        alpha=args.alpha,
        beta=args.beta,
        force_cpu=(DEVICE.type == "cpu"),
    )

    # ── Save final model ───────────────────────────────────────────────────────
    if args.adjustment_type == "both":
        model_name = f"nifty50_dapo_{args.adjustment_type}_a{args.alpha}_b{args.beta}.pth"
    elif args.adjustment_type == "sentiment":
        model_name = f"nifty50_dapo_sentiment_a{args.alpha}.pth"
    elif args.adjustment_type == "risk":
        model_name = f"nifty50_dapo_risk_b{args.beta}.pth"
    else:
        model_name = "nifty50_dapo_no_adjustment.pth"

    final_path = os.path.join(CHECKPOINT_DIR, model_name)
    torch.save({
        "epoch"            : args.epochs - 1,
        "model_state_dict" : trained_model.state_dict(),
        "adjustment_type"  : args.adjustment_type,
        "alpha"            : args.alpha,
        "beta"             : args.beta,
        "stock_dim"        : stock_dim,
        "state_space"      : state_space,
        "indicators"       : INDICATORS,
    }, final_path)

    print(f"\n Training complete! Model saved to: {final_path}")
