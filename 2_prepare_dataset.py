#!/usr/bin/env python3
"""
Step 2: Build the complete Nifty50 price dataset.

Strategy:
  1. Load per-ticker CSVs from ./indian-market-data/ (data up to 2021-04-30)
  2. Identify the gap: 2021-05-01 → today
  3. Fetch only that gap from Yahoo Finance for current Nifty50 tickers
  4. Concatenate local + fetched data
  5. Compute technical indicators, India VIX, turbulence
  6. Split into train (≤2022-12-31) and trade (≥2023-01-01)
"""

import pandas as pd
import numpy as np
import os
import time
import glob
import warnings
warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
LOCAL_DIR    = "./indian-market-data"
OUTPUT_DIR   = "./dataset"
TRAIN_START  = "2016-01-01"
TRAIN_END    = "2022-12-31"
TRADE_START  = "2023-01-01"
GAP_START    = "2021-05-01"          # fetch from Yahoo Finance from this date
FETCH_END    = "2026-04-09"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Updated Nifty 50 tickers & Yahoo Finance Mapping ──────────────────────────
# Mapping based on user-provided definitive list
YF_MAP = {
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "AXISBANK": "AXISBANK.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "BEL": "BEL.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "CIPLA": "CIPLA.NS",
    "COALINDIA": "COALINDIA.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "DRREDDY": "DRREDDY.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "GRASIM": "GRASIM.NS",
    "HCLTECH": "HCLTECH.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "HDFCLIFE": "HDFCLIFE.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "HINDALCO": "HINDALCO.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "INFY": "INFY.NS",
    "ITC": "ITC.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS",
    "LTIM": "LTIM.NS",
    "M&M": "M&M.NS",
    "MARUTI": "MARUTI.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "POWERGRID": "POWERGRID.NS",
    "RELIANCE": "RELIANCE.NS",
    "SBILIFE": "SBILIFE.NS",
    "SBIN": "SBIN.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "TCS": "TCS.NS",
    "TECHM": "TECHM.NS",
    "TITAN": "TITAN.NS",
    "TRENT": "TRENT.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "WIPRO": "WIPRO.NS"
}

CURRENT_NIFTY50 = list(YF_MAP.keys())

# ── Filter tickers by presence in News Data ───────────────────────────────────
NEWS_CSV = os.path.join(OUTPUT_DIR, "nifty50_news_combined.csv")
if os.path.exists(NEWS_CSV):
    try:
        news_df = pd.read_csv(NEWS_CSV, usecols=["tic"])
        news_tickers = set(news_df["tic"].unique())
        print(f"\n📂 Found {len(news_tickers)} tickers in news data.")
        
        # Keep only tickers that are in both the news data AND our master list
        old_count = len(YF_MAP)
        YF_MAP = {k: v for k, v in YF_MAP.items() if k in news_tickers}
        CURRENT_NIFTY50 = list(YF_MAP.keys())
        
        removed = old_count - len(YF_MAP)
        if removed > 0:
            print(f"🧹 Removed {removed} tickers with no news data.")
    except Exception as e:
        print(f"⚠️ Warning: Could not read news tickers for filtering: {e}")
else:
    print(f"⚠️ Warning: {NEWS_CSV} not found. Proceeding with all tickers.")

# Tickers in local data that have NSE symbol mismatches
LOCAL_ALIAS = {
    "MM": "M&M",
    "MM.NS": "M&M",
    "M&M.NS": "M&M",
}

# ── 1. Load local data ─────────────────────────────────────────────────────────
def load_local_data():
    files = [f for f in glob.glob(os.path.join(LOCAL_DIR, "*.csv"))
             if not any(x in f for x in ["NIFTY50_all", "stock_metadata", "INFRATEL"])]

    all_dfs = []
    for fpath in sorted(files):
        fname = os.path.basename(fpath).replace(".csv", "")
        # Resolve ticker alias (MM -> M&M, etc.) or just use uppercase name
        tic = LOCAL_ALIAS.get(fname, fname.upper())

        try:
            df = pd.read_csv(fpath)
            if df.empty or "Date" not in df.columns:
                continue

            df.columns = [c.strip() for c in df.columns]
            df = df.rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume"
            })
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            df = df.dropna(subset=["date", "close"])
            df = df[["date", "open", "high", "low", "close", "volume"]].copy()
            df["tic"] = tic
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            all_dfs.append(df)
            print(f"  {tic:15s}: {len(df):5d} rows  {df['date'].min()} → {df['date'].max()}")
        except Exception:
            pass

    return pd.concat(all_dfs, ignore_index=True)


# ── 2. Fetch gap data from Yahoo Finance ──────────────────────────────────────
def fetch_yf_gap(tickers_to_fetch, start, end, skip_yf=False):
    """Batch download with per-ticker fallback and polite delays."""
    if skip_yf:
        print("  --skip-yf: skipping Yahoo Finance")
        return pd.DataFrame()
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed.")
        return pd.DataFrame()

    yf_tickers = [YF_MAP[t] for t in tickers_to_fetch]
    yf_to_raw  = {yf_t: raw_t for raw_t, yf_t in YF_MAP.items() if raw_t in tickers_to_fetch}

    print(f"  Batch-downloading {len(yf_tickers)} tickers ({start} → {end})...")
    batch_results = {}
    try:
        raw = yf.download(
            yf_tickers, start=start, end=end,
            auto_adjust=True, progress=True, group_by="ticker"
        )
        for yf_tic, raw_tic in yf_to_raw.items():
            try:
                df = raw[yf_tic].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
                if df.empty or df.dropna(how="all").empty:
                    continue
                df.columns = [c.lower() for c in df.columns]
                avail = [c for c in ["open","high","low","close","volume"] if c in df.columns]
                df = df[avail].dropna(how="all")
                df.index = pd.to_datetime(df.index)
                df["date"] = df.index.strftime("%Y-%m-%d")
                df["tic"]  = raw_tic
                batch_results[raw_tic] = df.reset_index(drop=True)
            except Exception:
                pass
    except Exception as e:
        print(f"  Batch failed ({e}), falling back to per-ticker...")

    failed = [t for t in tickers_to_fetch if t not in batch_results]
    if failed:
        print(f"  Retrying {len(failed)} tickers individually (2s delay each)...")
    per_ticker = []
    for raw_tic in failed:
        yf_tic = YF_MAP[raw_tic]
        time.sleep(2)
        try:
            df = yf.download(yf_tic, start=start, end=end,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if df.empty:
                print(f"    ⚠️  {raw_tic}: still no data")
                continue
            df.columns = [c.lower() for c in df.columns]
            avail = [c for c in ["open","high","low","close","volume"] if c in df.columns]
            df = df[avail].dropna(how="all")
            df.index = pd.to_datetime(df.index)
            df["date"] = df.index.strftime("%Y-%m-%d")
            df["tic"] = raw_tic
            per_ticker.append(df.reset_index(drop=True))
            print(f"    ✅ {raw_tic}: {len(df)} rows")
        except Exception as e:
            print(f"    ❌ {raw_tic}: {e}")

    all_dfs = list(batch_results.values()) + per_ticker
    if not all_dfs:
        print("  ⚠️ Yahoo Finance rate-limited. Local data only will be used.")
        return pd.DataFrame()

    result = pd.concat(all_dfs, ignore_index=True)
    print(f"  ✅ Yahoo Finance: {len(result)} rows, {result['tic'].nunique()} tickers")
    return result


# ── 3. Technical Indicators ────────────────────────────────────────────────────
def add_tech_indicators(df):
    df = df.sort_values("date").copy()
    c, h, l = df["close"], df["high"], df["low"]
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    sma30 = c.rolling(30).mean(); std30 = c.rolling(30).std()
    df["boll_ub"] = sma30 + 2 * std30; df["boll_lb"] = sma30 - 2 * std30
    delta = c.diff(); gain = delta.clip(lower=0).rolling(30).mean(); loss = (-delta.clip(upper=0)).rolling(30).mean()
    df["rsi_30"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    tp = (h + l + c) / 3; sma = tp.rolling(30).mean(); mad = tp.rolling(30).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df["cci_30"] = (tp - sma) / (0.015 * mad + 1e-9)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1); atr = tr.rolling(30).mean()
    dm_pos = (h - h.shift()).clip(lower=0); dm_neg = (l.shift() - l).clip(lower=0)
    di_pos = 100 * dm_pos.rolling(30).mean() / (atr + 1e-9); di_neg = 100 * dm_neg.rolling(30).mean() / (atr + 1e-9)
    df["dx_30"] = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg + 1e-9)
    df["close_30_sma"] = c.rolling(30).mean(); df["close_60_sma"] = c.rolling(60).mean()
    df["day"] = pd.to_datetime(df["date"]).dt.dayofweek
    return df


# ── 4. India VIX ──────────────────────────────────────────────────────────────
def fetch_india_vix(start, end):
    try:
        import yfinance as yf
        vix = yf.download("^INDIAVIX", start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.droplevel(1)
        vix.columns = [c.lower() for c in vix.columns]; vix = vix[["close"]].rename(columns={"close": "vix"})
        vix.index = pd.to_datetime(vix.index); vix["date"] = vix.index.strftime("%Y-%m-%d")
        return vix.reset_index(drop=True)[["date", "vix"]]
    except Exception: return None

# ── 5. Turbulence ─────────────────────────────────────────────────────────────
def compute_turbulence(price_df, lookback=252):
    pivot = price_df.pivot(index="date", columns="tic", values="close").sort_index()
    returns = pivot.pct_change().dropna(); dates = returns.index.tolist(); turb = []
    for i in range(len(dates)):
        if i < lookback: turb.append(0.0); continue
        hist = returns.iloc[i - lookback:i]; today = returns.iloc[i].values
        try: cov_inv = np.linalg.pinv(hist.cov().values); diff = today - hist.mean().values; t_val = float(diff @ cov_inv @ diff.T)
        except Exception: t_val = 0.0
        turb.append(t_val)
    return pd.DataFrame({"date": dates, "turbulence": turb})

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-yf", action="store_true", help="Skip Yahoo Finance fetch")
    parser.add_argument("--train-start", default=TRAIN_START)
    parser.add_argument("--train-end",   default=TRAIN_END)
    parser.add_argument("--trade-start", default=TRADE_START)
    args = parser.parse_args()
    TRAIN_START = args.train_start; TRAIN_END = args.train_end; TRADE_START = args.trade_start

    print("\n Loading local historical data...")
    local_df = load_local_data()
    print(f"\n   Local total: {len(local_df)} rows, {local_df['tic'].nunique()} tickers")

    print(f"\n Checking coverage gaps (need data from {GAP_START})...")
    local_tickers = set(local_df["tic"].unique())
    need_gap = [t for t in CURRENT_NIFTY50 if t in local_tickers]
    need_full = [t for t in CURRENT_NIFTY50 if t not in local_tickers]

    all_yf_to_fetch = need_gap + need_full
    if all_yf_to_fetch:
        print(f"\n Fetching gap data from Yahoo Finance...")
        yf_df = fetch_yf_gap(all_yf_to_fetch, GAP_START, FETCH_END, skip_yf=args.skip_yf)
    else: yf_df = pd.DataFrame()

    print("\n Combining local and fetched data...")
    frames = [local_df]
    if not yf_df.empty: frames.append(yf_df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["tic", "date"]).drop_duplicates(subset=["date", "tic"], keep="first")
    combined = combined[combined["tic"].isin(CURRENT_NIFTY50)].copy()

    print("\n Computing technical indicators...")
    groups = []
    for tic, grp in combined.groupby("tic"): groups.append(add_tech_indicators(grp))
    combined = pd.concat(groups, ignore_index=True)

    print("\n Fetching India VIX...")
    vix_df = fetch_india_vix(combined["date"].min(), combined["date"].max())
    if vix_df is not None:
        combined = combined.merge(vix_df, on="date", how="left")
        combined["vix"] = combined.groupby("tic")["vix"].transform(lambda s: s.ffill().bfill()).fillna(0.0)
    else: combined["vix"] = 0.0

    print("\n Computing turbulence...")
    turb = compute_turbulence(combined[["date", "tic", "close"]].dropna())
    combined = combined.merge(turb, on="date", how="left")
    combined["turbulence"] = combined["turbulence"].fillna(0.0)

    INDICATOR_COLS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
    combined = combined.dropna(subset=INDICATOR_COLS)
    combined = combined[combined["date"] >= TRAIN_START].copy()
    
    full_path = os.path.join(OUTPUT_DIR, "nifty50_full.csv")
    combined.to_csv(full_path, index=False)
    print(f"\n💾 Full dataset (filtered from {TRAIN_START}) saved: {full_path}")

    train = combined[combined["date"] <= TRAIN_END].copy()
    trade = combined[combined["date"] >= TRADE_START].copy()
    train.to_csv(os.path.join(OUTPUT_DIR, "nifty50_train_base.csv"), index=False)
    trade.to_csv(os.path.join(OUTPUT_DIR, "nifty50_trade_base.csv"), index=False)

    print(f"\n✅ Train: {len(train):,} rows | Trade: {len(trade):,} rows")
    print(f"📊 Final Tickers: {combined['tic'].nunique()} | Columns: {list(combined.columns)}")
