#!/usr/bin/env python3
"""
Step 1: Combine raw Nifty50 news CSV chunks into a single clean dataset.
Reads all news_dataset_*.csv files from ./nifty50_news/,
parses dates, deduplicates, and saves to ./dataset/nifty50_news_combined.csv
"""

import pandas as pd
import glob
import os
from datetime import datetime

NEWS_DIR = "./nifty50_news"
OUTPUT_DIR = "./dataset"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nifty50_news_combined.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(" Combining Nifty50 news chunks...")
files = sorted(glob.glob(os.path.join(NEWS_DIR, "news_dataset_*.csv")))
print(f"  Found {len(files)} files: {[os.path.basename(f) for f in files]}")

dfs = []
for f in files:
    df = pd.read_csv(f)
    print(f"  {os.path.basename(f)}: {len(df)} rows")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
print(f"\nCombined total rows: {len(combined)}")

# ── Clean & normalize ──────────────────────────────────────────────────────────
# Rename columns to match pipeline conventions
combined = combined.rename(columns={
    "Date": "date_raw",
    "Article Title": "article_title",
    "Stock Symbol": "tic",
    "URL": "url",
    "Publisher": "publisher",
})

# Parse date: 'Sun, 08 May 2016 07:00:00 GMT' → YYYY-MM-DD
print("\n Parsing dates...")
combined["date"] = pd.to_datetime(
    combined["date_raw"], format="%a, %d %b %Y %H:%M:%S %Z", errors="coerce"
)
# Fallback for any that failed
mask = combined["date"].isna()
if mask.any():
    combined.loc[mask, "date"] = pd.to_datetime(combined.loc[mask, "date_raw"], errors="coerce")

# Drop rows where date couldn't be parsed
before = len(combined)
combined = combined.dropna(subset=["date"])
combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
print(f"  Dropped {before - len(combined)} rows with unparseable dates")

# Deduplicate on (date, tic, article_title)
before = len(combined)
combined = combined.drop_duplicates(subset=["date", "tic", "article_title"])
print(f"  Removed {before - len(combined)} duplicate rows")

# Keep only relevant columns
combined = combined[["date", "tic", "article_title", "publisher"]].sort_values(
    ["tic", "date"]
).reset_index(drop=True)

print(f"\n Final rows: {len(combined)}")
print(f"   Date range: {combined['date'].min()} → {combined['date'].max()}")
print(f"   Unique tickers: {combined['tic'].nunique()}")
print(f"   Tickers: {sorted(combined['tic'].unique().tolist())}")

combined.to_csv(OUTPUT_FILE, index=False)
print(f"\n💾 Saved to: {OUTPUT_FILE}")
