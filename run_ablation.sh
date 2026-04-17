#!/usr/bin/env bash
# =============================================================================
# run_ablation.sh — DAPO-NIFTY50 ablation study
#
# Runs all four adjustment configurations (none / sentiment / risk / both)
# then sweeps alpha/beta exponents for the best config.
#
# Usage:
#   chmod +x run_ablation.sh
#   ./run_ablation.sh                          # full run (default params)
#   EPOCHS=5 STEPS=2000 ./run_ablation.sh      # quick smoke-test
#   SKIP_SWEEP=1 ./run_ablation.sh             # skip alpha/beta sweep
#
# Results land in:
#   checkpoint/nifty50/          — trained .pth files
#   data/                        — spinup progress logs
#   results/ablation_summary.csv — aggregated EpRet across experiments
# =============================================================================

set -euo pipefail

# ── Configurable defaults ─────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-100}"
STEPS="${STEPS:-20000}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-1.0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"

PYTHON="${PYTHON:-python}"   # override with: PYTHON=python3 ./run_ablation.sh

echo "=============================================="
echo "  DAPO-NIFTY50 Ablation Study"
echo "  Epochs : ${EPOCHS} | Steps/epoch : ${STEPS}"
echo "=============================================="
echo ""

# ── Phase 1: Four core configurations ─────────────────────────────────────────

echo "[1/4] Baseline — no LLM signals (adjustment_type=none)"
${PYTHON} 4_train_dapo_nifty.py \
  --adjustment_type none \
  --alpha 1.0 --beta 1.0 \
  --epochs "${EPOCHS}" --steps "${STEPS}" \
  --exp_name dapo_none
echo ""

echo "[2/4] Sentiment only (adjustment_type=sentiment)"
${PYTHON} 4_train_dapo_nifty.py \
  --adjustment_type sentiment \
  --alpha "${ALPHA}" --beta 1.0 \
  --epochs "${EPOCHS}" --steps "${STEPS}" \
  --exp_name "dapo_sentiment_a${ALPHA}"
echo ""

echo "[3/4] Risk only (adjustment_type=risk)"
${PYTHON} 4_train_dapo_nifty.py \
  --adjustment_type risk \
  --alpha 1.0 --beta "${BETA}" \
  --epochs "${EPOCHS}" --steps "${STEPS}" \
  --exp_name "dapo_risk_b${BETA}"
echo ""

echo "[4/4] Both signals (adjustment_type=both)"
${PYTHON} 4_train_dapo_nifty.py \
  --adjustment_type both \
  --alpha "${ALPHA}" --beta "${BETA}" \
  --epochs "${EPOCHS}" --steps "${STEPS}" \
  --exp_name "dapo_both_a${ALPHA}_b${BETA}"
echo ""

echo "=============================================="
echo "  Phase 1 complete. Core ablation done."
echo "=============================================="
echo ""

# ── Phase 2: Alpha/beta sensitivity sweep (skippable) ─────────────────────────
if [ "${SKIP_SWEEP}" -eq 1 ]; then
  echo "Skipping alpha/beta sweep (SKIP_SWEEP=1)."
else
  echo "Starting alpha/beta sensitivity sweep..."
  echo "(Set SKIP_SWEEP=1 to skip this phase)"
  echo ""

  for alpha in 0.5 1.0 1.5 2.0; do
    for beta in 0.5 1.0 1.5 2.0; do
      echo "  → α=${alpha}  β=${beta}"
      ${PYTHON} 4_train_dapo_nifty.py \
        --adjustment_type both \
        --alpha "${alpha}" \
        --beta  "${beta}"  \
        --epochs "${EPOCHS}" --steps "${STEPS}" \
        --exp_name "dapo_a${alpha}_b${beta}"
      echo ""
    done
  done

  echo "=============================================="
  echo "  Phase 2 complete. Sweep done (16 configs)."
  echo "=============================================="
  echo ""
fi

# ── Phase 3: Aggregate results ────────────────────────────────────────────────
echo "Aggregating results into results/ablation_summary.csv ..."

${PYTHON} - << 'PYEOF'
import glob, os
import pandas as pd

records = []
for progress_file in glob.glob("data/*/progress.txt"):
    exp_name = os.path.basename(os.path.dirname(progress_file))
    try:
        df = pd.read_table(progress_file)
        if "AverageEpRet" in df.columns:
            records.append({
                "experiment":    exp_name,
                "final_EpRet":   round(df["AverageEpRet"].iloc[-1], 4),
                "max_EpRet":     round(df["AverageEpRet"].max(), 4),
                "final_Sharpe":  round(df.get("Sharpe", pd.Series([float("nan")])).iloc[-1], 4)
                                 if "Sharpe" in df.columns else float("nan"),
                "epochs_run":    len(df),
            })
    except Exception as e:
        print(f"  Warning: could not parse {progress_file}: {e}")

if records:
    os.makedirs("results", exist_ok=True)
    result_df = pd.DataFrame(records).sort_values("final_EpRet", ascending=False)
    out_path  = "results/ablation_summary.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print(result_df.to_string(index=False))
else:
    print("  No progress.txt files found — run training first.")
PYEOF

echo ""
echo "=============================================="
echo "  Ablation study complete!"
echo "  Results: results/ablation_summary.csv"
echo "=============================================="
