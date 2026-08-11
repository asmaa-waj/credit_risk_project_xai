"""
STEP 2b: Handle missing values.

WHAT THIS SCRIPT DOES:
  Takes train_merged.csv / test_merged.csv (from Step 1) and fills in every
  missing value, using a different strategy depending on WHY that column is
  missing. No columns are dropped -- the missingness report showed nothing
  above 80%, and even the highest-missing columns (EXT_SOURCE_1, credit card
  aggregates) carry real signal, so dropping them would throw away
  information rather than noise.

THE 4 GROUPS OF COLUMNS, AND WHY EACH IS HANDLED DIFFERENTLY:

  1. STRUCTURAL groups (CC_*, BUREAU_*, PREV_*, POS_*, INSTALL_*):
     These are OUR OWN aggregated columns from Step 1. A NaN here doesn't
     mean "unknown" -- it means the applicant genuinely has zero rows in
     that source table (e.g. never had a Home Credit credit card). So we
     fill with 0 AND add a HAS_X binary flag (1 = had history, 0 = didn't),
     so the model can still distinguish "no credit card" from "had a
     credit card with $0 balance".

  2. OWN_CAR_AGE:
     Missing almost always means the applicant doesn't own a car (there's
     a separate FLAG_OWN_CAR column that already captures this). Filled
     with 0 -- no extra flag needed since FLAG_OWN_CAR already exists.

  3. EXT_SOURCE_1 / EXT_SOURCE_2 / EXT_SOURCE_3:
     These are normalized scores from external credit bureaus and are
     historically the most predictive features in this dataset. Missing
     means that particular bureau had no score for this applicant --
     genuinely unknown, not zero. We add a MISSING flag for each (since
     "no external score available" can itself be informative) and impute
     with the TRAIN median (robust to outliers, per Step 2.2 of the guide).

  4. Everything else still missing (building/apartment details, occupation
     type, credit bureau inquiry counts, social circle counts, etc.):
     Genuinely unknown values with no structural meaning. Numeric columns
     get the TRAIN median; categorical (text) columns get filled with the
     string "Unknown" so it's treated as its own category.

CRITICAL RULE FOLLOWED THROUGHOUT (Step 2.3 of the guide):
  Every statistic used to fill values (medians, etc.) is computed from
  TRAIN ONLY, then the exact same values are applied to fill TEST. Test
  is never used to compute anything -- this avoids data leakage.

OUTPUT:
  - data/interim/train_clean.csv, data/interim/test_clean.csv
    (fully imputed -- Step 8's checklist item "no missing values remain"
    will be satisfied after this)
  - data/interim/imputation_log.csv
    (a column-by-column record of what was done and why -- this is exactly
    the documentation Step 2.4 of the guide asks you to keep)
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

train = pd.read_csv(os.path.join(INTERIM_DIR, "train_merged.csv"))
test = pd.read_csv(os.path.join(INTERIM_DIR, "test_merged.csv"))
print(f"Loaded train_merged: {train.shape}")
print(f"Loaded test_merged:  {test.shape}")

# This log collects one row per decision, to save as imputation_log.csv
log_rows = []


def log(column, group, method, reasoning):
    log_rows.append({"column": column, "group": group, "method": method, "reasoning": reasoning})


# ----------------------------------------------------------------------------
# GROUP 1: Structural aggregates from Step 1 (CC_, BUREAU_, PREV_, POS_, INSTALL_)
# ----------------------------------------------------------------------------
# For each prefix: add a HAS_X flag BEFORE filling (flag = 1 if the applicant
# had any record in that source table, 0 if not), then fill the actual
# columns with 0.

structural_prefixes = {
    "CC_": ("HAS_CREDIT_CARD", "no rows in credit_card_balance -> applicant never had a Home Credit credit card"),
    "BUREAU_": ("HAS_BUREAU_HISTORY", "no rows in bureau -> applicant has no external credit history on record"),
    "PREV_": ("HAS_PREV_APPLICATION", "no rows in previous_application -> first-time Home Credit applicant"),
    "POS_": ("HAS_POS_HISTORY", "no rows in POS_CASH_balance -> no prior point-of-sale/cash loans"),
    "INSTALL_": ("HAS_INSTALLMENT_HISTORY", "no rows in installments_payments -> no prior installment history"),
}

for prefix, (flag_name, reasoning) in structural_prefixes.items():
    cols = [c for c in train.columns if c.startswith(prefix)]
    if not cols:
        continue
    # All columns in a group are NaN together (they come from the same
    # source table), so checking the first one tells us about the whole group.
    train[flag_name] = train[cols[0]].notna().astype(int)
    test[flag_name] = test[cols[0]].notna().astype(int)

    for c in cols:
        train[c] = train[c].fillna(0)
        test[c] = test[c].fillna(0)
        log(c, "structural", "fill 0 + flag", reasoning)
    log(flag_name, "structural", "derived flag", f"1 if applicant had rows in source table, else 0")

print(f"\nGroup 1 done -- structural flags added: {[v[0] for v in structural_prefixes.values()]}")

# ----------------------------------------------------------------------------
# GROUP 2: OWN_CAR_AGE
# ----------------------------------------------------------------------------

if "OWN_CAR_AGE" in train.columns:
    train["OWN_CAR_AGE"] = train["OWN_CAR_AGE"].fillna(0)
    test["OWN_CAR_AGE"] = test["OWN_CAR_AGE"].fillna(0)
    log("OWN_CAR_AGE", "structural", "fill 0", "missing almost always means no car owned (see FLAG_OWN_CAR)")

print("Group 2 done -- OWN_CAR_AGE filled with 0")

# ----------------------------------------------------------------------------
# GROUP 3: EXT_SOURCE_1 / 2 / 3 -- missing flag + train-median imputation
# ----------------------------------------------------------------------------

ext_source_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
ext_source_medians = {}

for c in ext_source_cols:
    if c not in train.columns:
        continue
    flag_name = f"{c}_MISSING"
    train[flag_name] = train[c].isna().astype(int)
    test[flag_name] = test[c].isna().astype(int)

    median_val = train[c].median()  # computed on TRAIN ONLY
    ext_source_medians[c] = median_val

    train[c] = train[c].fillna(median_val)
    test[c] = test[c].fillna(median_val)  # same train-derived value applied to test

    log(c, "ext_source", "fill train median", "external bureau score unavailable for this applicant")
    log(flag_name, "ext_source", "derived flag", "1 if the external score was missing, else 0")

print(f"Group 3 done -- EXT_SOURCE medians (from train): {ext_source_medians}")

# ----------------------------------------------------------------------------
# GROUP 4: Everything else still missing -- genuinely unknown values
# ----------------------------------------------------------------------------
# Numeric -> train median. Categorical (text) -> "Unknown" category.
# This automatically covers the building/apartment block, OCCUPATION_TYPE,
# AMT_REQ_CREDIT_BUREAU_*, social circle counts, AMT_GOODS_PRICE, etc. --
# whatever is left after Groups 1-3 already handled the structural columns.

remaining_missing_cols = [c for c in train.columns if train[c].isna().sum() > 0]
print(f"\n{len(remaining_missing_cols)} columns remain with missing values -> Group 4 (genuinely unknown)")

for c in remaining_missing_cols:
    # Use pandas' own numeric-dtype check instead of comparing dtype to
    # "object" -- some pandas versions store text as a "string" dtype
    # rather than "object", which the old check missed.
    if pd.api.types.is_numeric_dtype(train[c]):
        median_val = train[c].median()  # TRAIN ONLY
        train[c] = train[c].fillna(median_val)
        test[c] = test[c].fillna(median_val)
        log(c, "genuinely_unknown_numeric", "fill train median", "no reported value; median is robust to outliers")
    else:
        train[c] = train[c].fillna("Unknown")
        test[c] = test[c].fillna("Unknown")
        log(c, "genuinely_unknown_categorical", "fill 'Unknown'", "no reported value; treated as its own category")

# Defragment the dataframes (we added many columns one at a time above,
# which is fine correctness-wise but triggers pandas' PerformanceWarning).
train = train.copy()
test = test.copy()

# ----------------------------------------------------------------------------
# Final check: confirm nothing is missing anymore
# ----------------------------------------------------------------------------

train_missing_left = train.isna().sum().sum()
test_missing_left = test.isna().sum().sum()
assert train_missing_left == 0, f"{train_missing_left} missing values remain in train -- check the log!"
assert test_missing_left == 0, f"{test_missing_left} missing values remain in test -- check the log!"
print("\nConfirmed: zero missing values remain in both train and test.")

print(f"train shape after imputation: {train.shape}  (added {train.shape[1] - 138} flag columns)")
print(f"test shape after imputation:  {test.shape}")

# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------

train_out = os.path.join(INTERIM_DIR, "train_clean.csv")
test_out = os.path.join(INTERIM_DIR, "test_clean.csv")
log_out = os.path.join(INTERIM_DIR, "imputation_log.csv")

train.to_csv(train_out, index=False)
test.to_csv(test_out, index=False)
pd.DataFrame(log_rows).to_csv(log_out, index=False)

print(f"\nSaved: {train_out}")
print(f"Saved: {test_out}")
print(f"Saved: {log_out}  ({len(log_rows)} logged decisions)")
print("\nStep 2 complete. Next: Step 3 -- Treat Outliers (e.g. the DAYS_EMPLOYED 365243 placeholder).")