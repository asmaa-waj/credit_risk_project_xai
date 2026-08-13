"""
STEP 5: Feature engineering.

WHAT THIS SCRIPT DOES:

  PART A -- Convert negative day-counts to human-readable years:
    DAYS_BIRTH, DAYS_EMPLOYED, DAYS_REGISTRATION, DAYS_ID_PUBLISH, and
    DAYS_LAST_PHONE_CHANGE are stored as negative integers counting back
    from the application date (e.g. DAYS_BIRTH = -14652 means the
    applicant was born 14,652 days before applying). This isn't a data
    error -- it's just Home Credit's storage convention, and tree models
    handle it fine as-is. But it's meaningless in a SHAP/LIME explanation
    shown to a loan officer, so we add readable _YEARS versions alongside
    the originals (which we keep, since EMPLOYED_TO_AGE_RATIO below uses
    the original negative-day columns directly).

  PART B -- Core ratio features (as listed in the guide):
    - DEBT_TO_INCOME        = AMT_CREDIT / AMT_INCOME_TOTAL
    - ANNUITY_TO_INCOME     = AMT_ANNUITY / AMT_INCOME_TOTAL
    - CREDIT_TO_GOODS_RATIO = AMT_CREDIT / AMT_GOODS_PRICE
    - EMPLOYED_TO_AGE_RATIO = DAYS_EMPLOYED / DAYS_BIRTH
        (both negative, so this comes out as a clean positive fraction --
        how much of the applicant's life has been spent employed)
    - BUREAU_LATE_MONTHS_RATIO = BUREAU_TOTAL_LATE_MONTHS / BUREAU_LOAN_COUNT
        (share of bureau-reported loan-months that were late; a bureau-history
        version of "ratio of previous loans overdue" from the guide's Step 5
        suggestions -- BUREAU_LOAN_COUNT and BUREAU_TOTAL_LATE_MONTHS were
        already aggregated back in Step 1)
    Note: INSTALL_AVG_PAYMENT_DELAY and INSTALL_UNDERPAID_RATIO (also
    suggested in the guide) were already built during Step 1's installments
    aggregation, so they aren't recreated here.

  PART C -- Handle division-by-zero / infinite values:
    Any ratio with a zero or missing denominator produces inf or NaN.
    Per Step 5's guideline, these are replaced with NaN and then imputed
    with the TRAIN median (same rule as Step 2).

  PART D -- Re-check outliers on the newly created ratios (Step 3.3):
    Ratios can produce new extreme values (e.g. a near-zero income
    denominator blowing up DEBT_TO_INCOME). Winsorize the 5 new ratio
    columns at the 1st/99th percentile, bounds computed on TRAIN ONLY.

CRITICAL RULE FOLLOWED THROUGHOUT:
  All medians and percentile bounds are computed on TRAIN ONLY, then the
  same values are applied to TEST.

OUTPUT:
  - data/interim/train_features.csv, data/interim/test_features.csv
  - data/interim/feature_engineering_log.csv (what was built and how)
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

train = pd.read_csv(os.path.join(INTERIM_DIR, "train_encoded.csv"))
test = pd.read_csv(os.path.join(INTERIM_DIR, "test_encoded.csv"))
print(f"Loaded train_encoded: {train.shape}")
print(f"Loaded test_encoded:  {test.shape}")

log_rows = []


def log(column, formula, note):
    log_rows.append({"column": column, "formula": formula, "note": note})


# ----------------------------------------------------------------------------
# PART A: Days -> years conversions (originals kept)
# ----------------------------------------------------------------------------

DAYS_TO_YEARS = {
    "DAYS_BIRTH": "AGE_YEARS",
    "DAYS_EMPLOYED": "YEARS_EMPLOYED",
    "DAYS_REGISTRATION": "YEARS_REGISTRATION",
    "DAYS_ID_PUBLISH": "YEARS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE": "YEARS_LAST_PHONE_CHANGE",
}

for days_col, years_col in DAYS_TO_YEARS.items():
    if days_col not in train.columns:
        print(f"  Skipping {days_col} -> {years_col}: column not found")
        continue
    train[years_col] = -train[days_col] / 365
    test[years_col] = -test[days_col] / 365
    log(years_col, f"-{days_col} / 365", "human-readable years, for interpretability (SHAP/LIME/dashboard)")

print(f"\nPart A done -- added {len(DAYS_TO_YEARS)} human-readable year columns")

# ----------------------------------------------------------------------------
# PART B: Core ratio features
# ----------------------------------------------------------------------------

def safe_ratio(df, numerator, denominator):
    """Compute numerator/denominator, letting divide-by-zero produce inf
    (handled explicitly in Part C, not silently)."""
    return df[numerator] / df[denominator]


ratio_definitions = [
    ("DEBT_TO_INCOME", "AMT_CREDIT", "AMT_INCOME_TOTAL", "credit amount relative to income"),
    ("ANNUITY_TO_INCOME", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "loan annuity relative to income"),
    ("CREDIT_TO_GOODS_RATIO", "AMT_CREDIT", "AMT_GOODS_PRICE", "credit amount relative to the goods it's financing"),
    ("EMPLOYED_TO_AGE_RATIO", "DAYS_EMPLOYED", "DAYS_BIRTH", "share of life spent employed (both negative -> positive fraction)"),
    ("BUREAU_LATE_MONTHS_RATIO", "BUREAU_TOTAL_LATE_MONTHS", "BUREAU_LOAN_COUNT", "share of bureau-reported loan-months that were late"),
]

new_ratio_cols = []
for new_col, num, denom, note in ratio_definitions:
    if num not in train.columns or denom not in train.columns:
        print(f"  Skipping {new_col}: {num} or {denom} not found")
        continue
    train[new_col] = safe_ratio(train, num, denom)
    test[new_col] = safe_ratio(test, num, denom)
    new_ratio_cols.append(new_col)
    log(new_col, f"{num} / {denom}", note)

print(f"Part B done -- added {len(new_ratio_cols)} ratio columns: {new_ratio_cols}")

# ----------------------------------------------------------------------------
# PART C: Handle division-by-zero / infinite / NaN values in the new ratios
# ----------------------------------------------------------------------------

print("\nPart C -- cleaning inf/NaN in new ratio columns:")
for col in new_ratio_cols:
    train[col] = train[col].replace([np.inf, -np.inf], np.nan)
    test[col] = test[col].replace([np.inf, -np.inf], np.nan)

    n_bad_train = train[col].isna().sum()
    n_bad_test = test[col].isna().sum()

    median_val = train[col].median()  # TRAIN ONLY, computed after removing inf
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)

    print(f"  {col}: {n_bad_train:,} train / {n_bad_test:,} test values were inf or NaN "
          f"-> filled with train median ({median_val:.4f})")
    log(col, "cleanup", f"inf/NaN (from zero denominators) filled with train median {median_val:.4f}")

# ----------------------------------------------------------------------------
# PART D: Re-check outliers on the new ratios (winsorize at 1st/99th pct)
# ----------------------------------------------------------------------------

print("\nPart D -- winsorizing new ratio columns (bounds from TRAIN only):")
for col in new_ratio_cols:
    lower = train[col].quantile(0.01)
    upper = train[col].quantile(0.99)

    n_capped_train = ((train[col] < lower) | (train[col] > upper)).sum()
    n_capped_test = ((test[col] < lower) | (test[col] > upper)).sum()

    train[col] = train[col].clip(lower=lower, upper=upper)
    test[col] = test[col].clip(lower=lower, upper=upper)

    print(f"  {col}: [{lower:.4f}, {upper:.4f}]  capped {n_capped_train:,} train rows, {n_capped_test:,} test rows")
    log(col, "winsorize 1st/99th pct", f"bounds [{lower:.4f}, {upper:.4f}] from train")

# Defragment after many column additions
train = train.copy()
test = test.copy()

# ----------------------------------------------------------------------------
# Final checks
# ----------------------------------------------------------------------------

assert list(train.columns) == list(test.columns), "Column mismatch between train and test after feature engineering!"
assert train.isna().sum().sum() == 0, "Missing values introduced during feature engineering!"
assert test.isna().sum().sum() == 0, "Missing values introduced during feature engineering!"
assert not np.isinf(train.select_dtypes(include=[np.number])).any().any(), "Infinite values remain in train!"
assert not np.isinf(test.select_dtypes(include=[np.number])).any().any(), "Infinite values remain in test!"

print(f"\ntrain shape: {train.shape}")
print(f"test shape:  {test.shape}")
print("Confirmed: identical columns, no missing values, no infinite values.")

# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------

train_out = os.path.join(INTERIM_DIR, "train_features.csv")
test_out = os.path.join(INTERIM_DIR, "test_features.csv")
log_out = os.path.join(INTERIM_DIR, "feature_engineering_log.csv")

train.to_csv(train_out, index=False)
test.to_csv(test_out, index=False)
pd.DataFrame(log_rows).to_csv(log_out, index=False)

print(f"\nSaved: {train_out}")
print(f"Saved: {test_out}")
print(f"Saved: {log_out}  ({len(log_rows)} entries)")
print("\nStep 5 complete. Next: Step 6 -- Address Class Imbalance.")