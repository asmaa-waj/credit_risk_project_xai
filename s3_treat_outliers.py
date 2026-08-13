"""
STEP 3: Treat outliers.

WHAT THIS SCRIPT DOES:

  PART A -- Fix the DAYS_EMPLOYED placeholder:
    Home Credit uses 365243 (about 1000 years) as a placeholder in
    DAYS_EMPLOYED, standing in for "not currently employed" (retirees,
    pensioners, unemployed applicants). Left as-is, this value would
    wreck any calculation involving DAYS_EMPLOYED (e.g. the
    EMPLOYED_TO_AGE_RATIO feature planned for Step 5). We:
      1. Replace 365243 with NaN.
      2. Add a DAYS_EMPLOYED_MISSING flag (1 = was the placeholder,
         0 = a real value) -- this is informative on its own, since it
         usually signals unemployed/retired.
      3. Impute the resulting NaNs with the TRAIN median.

  PART B -- Winsorize (cap, don't delete) selected columns at the
  1st / 99th percentile:
      AMT_INCOME_TOTAL, AMT_CREDIT, AMT_ANNUITY, AMT_GOODS_PRICE,
      DAYS_EMPLOYED (after Part A fixes it), OWN_CAR_AGE.
    Any value below the 1st percentile is pushed up to it; any value
    above the 99th percentile is pushed down to it. Rows are never
    deleted -- this neutralizes extreme/erroneous values (e.g. the known
    AMT_INCOME_TOTAL data-entry outlier in this dataset) while keeping
    every applicant in the data.

CRITICAL RULE FOLLOWED THROUGHOUT (same as Steps 1-2):
  Percentile bounds and medians are computed from TRAIN ONLY, then the
  exact same bounds are applied to TEST. Test is never used to compute
  anything.

OUTPUT:
  - data/interim/train_outliers_handled.csv
  - data/interim/test_outliers_handled.csv
  - data/interim/outlier_treatment_log.csv (bounds used, for your writeup)
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

train = pd.read_csv(os.path.join(INTERIM_DIR, "train_clean.csv"))
test = pd.read_csv(os.path.join(INTERIM_DIR, "test_clean.csv"))
print(f"Loaded train_clean: {train.shape}")
print(f"Loaded test_clean:  {test.shape}")

log_rows = []


def log(column, action, detail):
    log_rows.append({"column": column, "action": action, "detail": detail})


# ----------------------------------------------------------------------------
# PART A: Fix the DAYS_EMPLOYED placeholder (365243 -> NaN -> flag -> median)
# ----------------------------------------------------------------------------

PLACEHOLDER = 365243

n_placeholder_train = (train["DAYS_EMPLOYED"] == PLACEHOLDER).sum()
n_placeholder_test = (test["DAYS_EMPLOYED"] == PLACEHOLDER).sum()
print(f"\nDAYS_EMPLOYED placeholder ({PLACEHOLDER}) found in "
      f"{n_placeholder_train:,} train rows, {n_placeholder_test:,} test rows")

# Step 1: flag BEFORE replacing, so we capture exactly which rows had the placeholder
train["DAYS_EMPLOYED_MISSING"] = (train["DAYS_EMPLOYED"] == PLACEHOLDER).astype(int)
test["DAYS_EMPLOYED_MISSING"] = (test["DAYS_EMPLOYED"] == PLACEHOLDER).astype(int)

# Step 2: replace placeholder with NaN
train["DAYS_EMPLOYED"] = train["DAYS_EMPLOYED"].replace(PLACEHOLDER, np.nan)
test["DAYS_EMPLOYED"] = test["DAYS_EMPLOYED"].replace(PLACEHOLDER, np.nan)

# Step 3: impute with TRAIN median (computed AFTER removing the placeholder,
# so the median isn't itself distorted by 365243 values)
days_employed_median = train["DAYS_EMPLOYED"].median()
train["DAYS_EMPLOYED"] = train["DAYS_EMPLOYED"].fillna(days_employed_median)
test["DAYS_EMPLOYED"] = test["DAYS_EMPLOYED"].fillna(days_employed_median)

log("DAYS_EMPLOYED", "placeholder fix", f"365243 -> NaN -> filled with train median ({days_employed_median:.1f})")
log("DAYS_EMPLOYED_MISSING", "derived flag", "1 if DAYS_EMPLOYED was the 365243 placeholder (not employed), else 0")

print(f"DAYS_EMPLOYED median used for imputation: {days_employed_median:.1f}")

# ----------------------------------------------------------------------------
# PART B: Winsorize selected columns at 1st / 99th percentile
# ----------------------------------------------------------------------------

WINSORIZE_COLS = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED",
    "OWN_CAR_AGE",
]

print("\nWinsorizing (bounds computed on TRAIN only):")
for col in WINSORIZE_COLS:
    if col not in train.columns:
        print(f"  Skipping {col} -- not found in dataset")
        continue

    lower = train[col].quantile(0.01)
    upper = train[col].quantile(0.99)

    n_capped_low_train = (train[col] < lower).sum()
    n_capped_high_train = (train[col] > upper).sum()
    n_capped_low_test = (test[col] < lower).sum()
    n_capped_high_test = (test[col] > upper).sum()

    train[col] = train[col].clip(lower=lower, upper=upper)
    test[col] = test[col].clip(lower=lower, upper=upper)  # same train-derived bounds

    print(f"  {col}: [{lower:,.1f}, {upper:,.1f}]  "
          f"capped {n_capped_low_train + n_capped_high_train:,} train rows, "
          f"{n_capped_low_test + n_capped_high_test:,} test rows")

    log(col, "winsorize 1st/99th pct",
        f"bounds [{lower:.2f}, {upper:.2f}] from train; "
        f"capped {n_capped_low_train + n_capped_high_train} train rows, "
        f"{n_capped_low_test + n_capped_high_test} test rows")

# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------

train_out = os.path.join(INTERIM_DIR, "train_outliers_handled.csv")
test_out = os.path.join(INTERIM_DIR, "test_outliers_handled.csv")
log_out = os.path.join(INTERIM_DIR, "outlier_treatment_log.csv")

train.to_csv(train_out, index=False)
test.to_csv(test_out, index=False)
pd.DataFrame(log_rows).to_csv(log_out, index=False)

print(f"\ntrain shape: {train.shape}")
print(f"test shape:  {test.shape}")
print(f"\nSaved: {train_out}")
print(f"Saved: {test_out}")
print(f"Saved: {log_out}")
print("\nStep 3 complete. Next: Step 4 -- Encode Categorical Variables.")
print("(Note: your guide flags Step 3.3 -- re-check outliers again after Step 5's")
print(" engineered ratio features, since division can produce new extreme values.)")