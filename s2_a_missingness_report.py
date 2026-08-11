"""
STEP 2a: Missingness report.

WHAT THIS SCRIPT DOES:
  Loads train_merged.csv (from Step 1) and, for every single column,
  computes what % of rows are missing. This is purely informational --
  it doesn't change or drop anything yet. You'll look at this report and
  decide, column by column (or by threshold), what to do in Step 2b.

WHY WE DO THIS ON TRAIN ONLY:
  Per Step 2.3 of the guide: never make decisions using statistics from
  the full dataset (train + test combined). Missingness stats, imputers,
  thresholds -- all fit on train only, then applied to test.

HOW TO RUN IN PYCHARM:
  Right-click -> Run 'step2a_missingness_report'.
  Output prints to the console AND saves a full CSV report you can open
  in Excel/PyCharm to scroll through comfortably.
"""

import pandas as pd
import os

# Anchor all paths to THIS FILE's location, not PyCharm's "working directory"
# (which can silently differ between runs/configs and causes FileNotFoundError).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
train_path = os.path.join(INTERIM_DIR, "train_merged.csv")

print(f"Loading {train_path} ...")
train = pd.read_csv(train_path)
print(f"  -> {train.shape[0]:,} rows, {train.shape[1]} columns\n")

# ----------------------------------------------------------------------------
# Compute % missing per column
# ----------------------------------------------------------------------------

missing_count = train.isna().sum()
missing_pct = (missing_count / len(train) * 100).round(2)

report = pd.DataFrame({
    "column": train.columns,
    "missing_count": missing_count.values,
    "missing_pct": missing_pct.values,
    "dtype": train.dtypes.astype(str).values,
}).sort_values("missing_pct", ascending=False).reset_index(drop=True)

# ----------------------------------------------------------------------------
# Print a readable summary to the console
# ----------------------------------------------------------------------------

print("=" * 70)
print("MISSINGNESS REPORT (sorted worst -> best)")
print("=" * 70)

# Show every column with ANY missing values (most useful view)
with_missing = report[report["missing_pct"] > 0]
print(f"\n{len(with_missing)} of {train.shape[1]} columns have at least some missing values.\n")

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 120)
print(with_missing.to_string(index=False))

# Quick threshold buckets, since the guide suggests 40-60% as a common
# "consider dropping" range -- this just shows you how many columns fall
# into each bucket so you can pick a threshold with actual numbers in front of you.
print("\n" + "=" * 70)
print("COLUMN COUNTS BY MISSINGNESS BUCKET")
print("=" * 70)
buckets = [
    (">80%", report[report["missing_pct"] > 80]),
    ("60-80%", report[(report["missing_pct"] > 60) & (report["missing_pct"] <= 80)]),
    ("40-60%", report[(report["missing_pct"] > 40) & (report["missing_pct"] <= 60)]),
    ("20-40%", report[(report["missing_pct"] > 20) & (report["missing_pct"] <= 40)]),
    ("0-20%", report[(report["missing_pct"] > 0) & (report["missing_pct"] <= 20)]),
    ("0% (complete)", report[report["missing_pct"] == 0]),
]
for label, subset in buckets:
    print(f"  {label:>15}: {len(subset)} columns")

# ----------------------------------------------------------------------------
# Save full report to CSV so you can browse it comfortably / share it
# ----------------------------------------------------------------------------

report_path = os.path.join(INTERIM_DIR, "missingness_report.csv")
report.to_csv(report_path, index=False)
print(f"\nFull report saved to: {report_path}")