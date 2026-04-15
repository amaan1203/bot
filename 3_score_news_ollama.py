#!/usr/bin/env python3
"""
Step 3: Score Nifty50 news headlines using a local LLM (Qwen2.5-7B).
Generates both llm_sentiment and llm_risk scores via local Ollama.

Usage:
    # First ensure Ollama is running: ollama serve
    # Then pull the model: ollama pull qwen2.5:7b
    python 3_score_news_ollama.py
"""

import pandas as pd
import numpy as np
import os
import time
import json
import re
import argparse
import ollama  # Using the official ollama library for stability

# ── Config ─────────────────────────────────────────────────────────────────────
OLLAMA_HOST      = "http://localhost:11434"
MODEL_NAME       = "qwen2.5:3b"       # Balanced speed and intelligence
BATCH_SIZE       = 100             # headlines per batch save
RETRY_DELAY      = 2              # seconds between retries
MAX_RETRIES      = 3

INPUT_FILE   = "./dataset/nifty50_news_combined.csv"
TRAIN_END    = "2022-12-31"
TRADE_START  = "2023-01-01"
OUTPUT_DIR   = "./dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Prompts ────────────────────────────────────────────────────────────────────
SENTIMENT_PROMPT = """You are a professional financial analyst.

Evaluate how the following news impacts the stock price of "{ticker}".

Consider:
- earnings impact
- macroeconomic signals
- regulatory changes
- investor reaction

Score the EXPECTED MARKET IMPACT (not tone) on a scale of 1 to 5:

1 = Strong Bearish (price likely to drop significantly)
2 = Moderately Bearish
3 = Neutral / No clear impact
4 = Moderately Bullish
5 = Strong Bullish (price likely to rise significantly)

Be consistent across different headlines.
Use extreme values (1 or 5) only when strongly justified.

Headline: "{headline}"

Reply with ONLY a single integer (1,2,3,4,5)."""

RISK_PROMPT = """You are a financial risk analyst.

Evaluate how much uncertainty or volatility this news introduces for stock "{ticker}".

Score the RISK LEVEL on a scale of 1 to 5:

1 = Very Low Risk (stable, predictable)
2 = Low Risk
3 = Moderate / Uncertain
4 = High Risk
5 = Very High Risk (major uncertainty or crisis)

Be consistent across different headlines.
Focus on uncertainty and volatility, not sentiment.

Headline: "{headline}"

Reply with ONLY a single integer (1,2,3,4,5)."""


def extract_score(text: str) -> int:
    """Extract integer score 1-5 from model response."""
    text = text.strip()

    match = re.search(r"\b([1-5])\b", text)
    if match:
        return int(match.group(1))

    match = re.search(r"([1-5])", text)
    if match:
        return int(match.group(1))

    return 3  # neutral fallback (raw scale only)

def normalize_sentiment(score: int) -> float:
    """Map 1-5 → -1 to +1"""
    return (score - 3) / 2

def normalize_risk(score: int) -> float:
    """Map 1-5 → 0 to 1"""
    return (score - 1) / 4


def score_headline(headline: str, ticker: str, prompt_template: str) -> float:
    prompt = prompt_template.format(ticker=ticker, headline=headline[:500])
    
    for attempt in range(MAX_RETRIES):
        try:
            response = ollama.chat(
                model=MODEL_NAME,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1, 'num_predict': 10}
            )
            raw = response['message']['content']
            score = extract_score(raw)

            if "risk" in prompt_template.lower():
                return normalize_risk(score)
            else:
                return normalize_sentiment(score)

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"\n  Retry {attempt+1}/{MAX_RETRIES}: {e}")
                time.sleep(RETRY_DELAY)
            else:
                print(f"\n  Failed after {MAX_RETRIES} attempts: {e}")

                if "risk" in prompt_template.lower():
                    return 0.5
                else:
                    return 0.0


def score_batch(df: pd.DataFrame, prompt_template: str, score_col: str,
                checkpoint_path: str) -> pd.DataFrame:
    """Score all headlines, resuming from checkpoint if available."""
    if os.path.exists(checkpoint_path):
        checkpoint = pd.read_csv(checkpoint_path)
        scored_ids = set(checkpoint["orig_idx"].tolist()) if "orig_idx" in checkpoint.columns else set()
        results = checkpoint.to_dict("records")
        remaining = df[~df.index.isin(scored_ids)].copy()
        remaining["orig_idx"] = remaining.index
        print(f"  Resuming from checkpoint: {len(results)} already scored, {len(remaining)} remaining")
    else:
        results = []
        remaining = df.copy()
        remaining["orig_idx"] = remaining.index
        scored_ids = set()

    total = len(df)
    done = len(results)

    for i, (_, row) in enumerate(remaining.iterrows()):
        score = score_headline(row["article_title"], row["tic"], prompt_template)
        rec = row.to_dict()
        rec[score_col] = score
        results.append(rec)
        done += 1

        if done % 10 == 0 or done == total:
            pct = done / total * 100
            print(f"\r  Progress: {done}/{total} ({pct:.1f}%) | Last: {row['tic']} → {score}", end="", flush=True)

        if done % BATCH_SIZE == 0:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)

    print()
    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
    return pd.DataFrame(results)


def merge_scores_with_prices(price_df: pd.DataFrame, news_scored: pd.DataFrame,
                             score_col: str, default_val: int = 0.0) -> pd.DataFrame:
    """Merge LLM scores into price dataframe by taking the daily mean score."""
    daily_scores = (news_scored.groupby(["date", "tic"])[score_col]
                    .mean().reset_index())

    merged = price_df.merge(daily_scores, on=["date", "tic"], how="left")
    merged[score_col] = merged[score_col].fillna(default_val)
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    MODEL_NAME = args.model
    BATCH_SIZE = args.batch_size

    print(f"📰 Loading news from {INPUT_FILE}...")
    if not os.path.exists(INPUT_FILE):
        print("   News file not found. Run 1_combine_news.py first.")
        exit(1)

    news = pd.read_csv(INPUT_FILE)
    news["date"] = pd.to_datetime(news["date"]).dt.strftime("%Y-%m-%d")
    print(f"  Loaded {len(news)} headlines for {news['tic'].nunique()} tickers")

    if args.sample:
        news = news.sample(n=args.sample, random_state=42)
        print(f"  Using sample of {args.sample} headlines")

    print(f"\n Connecting to local model: {MODEL_NAME}")
    try:
        # Connectivity test
        test = score_headline("Company reports growth", "RELIANCE", SENTIMENT_PROMPT)
        print(f"  Model responding. Test score: {test}")
    except Exception as e:
        print(f"   Cannot reach Ollama: {e}")
        exit(1)

    # 1. Sentiment
    print(f"\n Scoring SENTIMENT ({len(news)} headlines)...")
    cp_s = os.path.join(OUTPUT_DIR, "_checkpoint_sentiment.csv")
    news_sentiment = score_batch(news, SENTIMENT_PROMPT, "llm_sentiment", cp_s)
    
    out_s = os.path.join(OUTPUT_DIR, "nifty50_news_sentiment.csv")
    news_sentiment[["date", "tic", "article_title", "llm_sentiment"]].to_csv(out_s, index=False)
    print(f"  💾 Saved: {out_s}")

    # 2. Risk
    print(f"\n  Scoring RISK ({len(news)} headlines)...")
    cp_r = os.path.join(OUTPUT_DIR, "_checkpoint_risk.csv")
    news_risk = score_batch(news, RISK_PROMPT, "llm_risk", cp_r)
    
    out_r = os.path.join(OUTPUT_DIR, "nifty50_news_risk.csv")
    news_risk[["date", "tic", "article_title", "llm_risk"]].to_csv(out_r, index=False)
    print(f"  💾 Saved: {out_r}")

    # 3. Final Merge
    print("\n🔗 Merging scores with price files...")
    train_b = os.path.join(OUTPUT_DIR, "nifty50_train_base.csv")
    trade_b = os.path.join(OUTPUT_DIR, "nifty50_trade_base.csv")

    if os.path.exists(train_b) and os.path.exists(trade_b):
        train_p = pd.read_csv(train_b)
        trade_p = pd.read_csv(trade_b)
        
        # Merge Sentiment
        train_p = merge_scores_with_prices(train_p, news_sentiment, "llm_sentiment", default_val=0.0)
        trade_p = merge_scores_with_prices(trade_p, news_sentiment, "llm_sentiment", default_val=0.0)

        # Merge Risk
        train_p = merge_scores_with_prices(train_p, news_risk, "llm_risk", default_val=0.5)
        trade_p = merge_scores_with_prices(trade_p, news_risk, "llm_risk", default_val=0.5)
        
        train_p.to_csv(os.path.join(OUTPUT_DIR, "nifty50_train_scores.csv"), index=False)
        trade_p.to_csv(os.path.join(OUTPUT_DIR, "nifty50_trade_scores.csv"), index=False)
        print("  ✅ Final datasets saved with LLM scores!")
    else:
        print("   Price files not found. Only news scores saved.")

    print("\n✅ All scoring complete!")
