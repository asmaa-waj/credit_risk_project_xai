"""
STEP 3a: Outlier boxplots (visual EDA).

WHAT THIS SCRIPT DOES:

  PART 1 -- Broad scan:
    Boxplots for EVERY continuous numeric column, using train_clean.csv
    (Step 2 output -- BEFORE any outlier treatment). This is the visual
    check the guide's Step 3.1 calls for ("visualize numeric features to
    spot implausible values") applied properly: looking at the full set of
    continuous columns rather than only the 6 already known to be
    problematic. Columns are batched into grids of 12 per image (too many
    to fit legibly on one plot).

  PART 2 -- Before / after comparison:
    For the 6 columns already winsorized in Step 3 (AMT_INCOME_TOTAL,
    AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE, DAYS_EMPLOYED, OWN_CAR_AGE),
    a side-by-side "before" (train_clean.csv) vs "after"
    (train_outliers_handled.csv) boxplot, showing the actual effect of
    capping. DAYS_EMPLOYED's "before" panel will look dominated by the
    365243 placeholder cluster -- that's expected, and is itself the
    visual justification for why that column needed fixing.

WHICH COLUMNS COUNT AS "CONTINUOUS" (for Part 1):
  Numeric dtype with more than 10 unique values in train_clean.csv. This
  excludes binary/flag columns (HAS_*, *_MISSING, FLAG_*) and low-count
  fields, while keeping every genuinely continuous or wide-count column,
  including ones never examined before (bureau/previous/POS/credit-card
  aggregates, building/apartment fields, etc.).

OUTPUT (in outputs/figures/):
  - outlier_scan_broad_batch_01.png, _02.png, ... (Part 1)
  - outlier_before_after_treated_columns.png (Part 2)

REQUIRES: matplotlib.
    pip install matplotlib
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

train_clean = pd.read_csv(os.path.join(INTERIM_DIR, "train_clean.csv"))
print(f"Loaded train_clean (pre-outlier-treatment): {train_clean.shape}")

# ----------------------------------------------------------------------------
# PART 1: Broad scan across all continuous columns
# ----------------------------------------------------------------------------

id_col = "SK_ID_CURR"
target_col = "TARGET"

numeric_cols = train_clean.select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c not in (id_col, target_col)]
continuous_cols = [c for c in numeric_cols if train_clean[c].nunique() > 10]

print(f"\n{len(continuous_cols)} continuous columns identified for the broad scan "
      f"(numeric, >10 unique values, excluding ID/TARGET)")

BATCH_SIZE = 12  # 4x3 grid per image
n_batches = (len(continuous_cols) + BATCH_SIZE - 1) // BATCH_SIZE

for batch_idx in range(n_batches):
    batch_cols = continuous_cols[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
    n_cols_grid = 4
    n_rows_grid = (len(batch_cols) + n_cols_grid - 1) // n_cols_grid

    fig, axes = plt.subplots(n_rows_grid, n_cols_grid, figsize=(n_cols_grid * 4, n_rows_grid * 3.5))
    axes = np.array(axes).reshape(-1)  # flatten regardless of grid shape

    for i, col in enumerate(batch_cols):
        axes[i].boxplot(train_clean[col].dropna(), orientation='vertical')
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(axis="x", bottom=False, labelbottom=False)

    # Hide any unused subplot slots in the last batch
    for j in range(len(batch_cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Outlier scan (pre-treatment) — batch {batch_idx + 1} of {n_batches}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(FIGURES_DIR, f"outlier_scan_broad_batch_{batch_idx + 1:02d}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}  ({len(batch_cols)} columns: {batch_cols})")

# ----------------------------------------------------------------------------
# PART 2: Before / after comparison for the 6 already-treated columns
# ----------------------------------------------------------------------------

TREATED_COLS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
    "AMT_GOODS_PRICE", "DAYS_EMPLOYED", "OWN_CAR_AGE",
    "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE",
    "BUREAU_LOAN_COUNT", "BUREAU_AVG_CREDIT_SUM", "BUREAU_MAX_CREDIT_SUM",
    "BUREAU_SUM_CREDIT_SUM_DEBT", "BUREAU_TOTAL_LATE_MONTHS",
    "PREV_LOAN_COUNT", "PREV_AVG_CREDIT_AMOUNT", "PREV_AVG_ANNUITY",
    "POS_MAX_DPD", "CC_AVG_BALANCE", "CC_MAX_BALANCE", "CC_AVG_CREDIT_LIMIT",
    "INSTALL_AVG_PAYMENT_DELAY",
]

train_after = pd.read_csv(os.path.join(INTERIM_DIR, "train_outliers_handled.csv"))
print(f"\nLoaded train_outliers_handled (post-treatment): {train_after.shape}")

fig, axes = plt.subplots(len(TREATED_COLS), 2, figsize=(9, len(TREATED_COLS) * 3))

for i, col in enumerate(TREATED_COLS):
    if col not in train_clean.columns:
        print(f"  Skipping {col}: not found")
        continue

    axes[i, 0].boxplot(train_clean[col].dropna(), orientation='vertical')
    axes[i, 0].set_title(f"{col} — BEFORE", fontsize=10)
    axes[i, 0].tick_params(axis="x", bottom=False, labelbottom=False)

    axes[i, 1].boxplot(train_after[col].dropna(), orientation='vertical')
    axes[i, 1].set_title(f"{col} — AFTER", fontsize=10)
    axes[i, 1].tick_params(axis="x", bottom=False, labelbottom=False)

fig.suptitle("Outlier treatment: before vs. after (Step 3)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])

out_path = os.path.join(FIGURES_DIR, "outlier_before_after_treated_columns.png")
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"Saved: {out_path}")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print("Review the broad-scan batches for any additional columns worth capping")
print("beyond the 6 already treated in Step 3 -- if you find any, let me know")
print("and I'll extend the Step 3 script to cover them.")