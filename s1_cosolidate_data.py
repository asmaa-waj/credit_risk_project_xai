import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# STEP 0: Setup & ground rules
# ----------------------------------------------------------------------------
"""
STEP 1: Consolidate the Home Credit tables into one applicant-level dataset,
then create your own train/test split (since only application_train exists).

WHAT THIS SCRIPT DOES (in plain English):
  1. Loads HC_application_train.csv -- your base table, one row per applicant.
  2. Loads each of the other 6 auxiliary tables. These have MULTIPLE rows per
     applicant (e.g. a person can have 5 previous loans), so they can't be
     joined directly.
  3. Aggregates each auxiliary table down to ONE row per applicant
     (e.g. "average previous credit amount", "number of past loans").
  4. Left-joins everything onto the base table.
  5. Splits the result into train/test (80/20, stratified on TARGET) --
     this stands in for the missing application_test.csv, and matches
     Step 0.3 in your guide (split BEFORE any imputing/encoding/scaling).
  6. Saves train_merged.csv and test_merged.csv into a new "interim" folder,
     so your /raw files stay untouched.

HOW TO RUN THIS IN PYCHARM:
  1. This file should sit next to your "raw" folder (containing the HC_ CSVs).
  2. Install packages if needed: File > Settings > Project > Python
     Interpreter > + > install "pandas", "numpy", "scikit-learn".
  3. Right-click in the editor -> "Run 'step1_consolidate_data'".
  4. Watch the output panel at the bottom for progress.
  5. When done, check the new "interim" folder for train_merged.csv / test_merged.csv.
"""
RANDOM_STATE = 42  # used everywhere for reproducibility -- splitting here,
                    # and later for imputation/resampling/model training too.

# Anchor all paths to THIS FILE's location, not PyCharm's "working directory"
# (which can silently differ between runs/configs and causes FileNotFoundError).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
os.makedirs(INTERIM_DIR, exist_ok=True)


def load_csv(filename):
    """Load a CSV from the raw folder and print its shape so you can see
    progress as the script runs."""
    path = os.path.join(RAW_DIR, filename)
    print(f"Loading {filename} ...")
    df = pd.read_csv(path)
    print(f"  -> {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


# ----------------------------------------------------------------------------
# STEP 1a: Load the base table
# ----------------------------------------------------------------------------

app = load_csv("HC_application_train.csv")

# ----------------------------------------------------------------------------
# STEP 1b: Bureau + bureau_balance (external credit history)
# ----------------------------------------------------------------------------
# HC_bureau.csv          = one row per PAST loan the applicant had at OTHER banks
# HC_bureau_balance.csv  = one row per MONTH of history for each of those loans
#
# Two-level aggregation:
#   bureau_balance (many rows per loan) -> one row per loan
#   then merge into bureau
#   bureau (many rows per applicant)    -> one row per applicant

bureau = load_csv("HC_bureau.csv")
bureau_balance = load_csv("HC_bureau_balance.csv")

# STATUS: "0" = paid on time, "1"-"5" = days past due buckets, "C" = closed, "X" = unknown
bureau_balance["IS_LATE"] = bureau_balance["STATUS"].isin(["1", "2", "3", "4", "5"]).astype(int)

bureau_balance_agg = bureau_balance.groupby("SK_ID_BUREAU").agg(
    BB_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
    BB_LATE_MONTHS_COUNT=("IS_LATE", "sum"),
).reset_index()

bureau = bureau.merge(bureau_balance_agg, on="SK_ID_BUREAU", how="left")

bureau_agg = bureau.groupby("SK_ID_CURR").agg(
    BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
    BUREAU_AVG_CREDIT_SUM=("AMT_CREDIT_SUM", "mean"),
    BUREAU_MAX_CREDIT_SUM=("AMT_CREDIT_SUM", "max"),
    BUREAU_SUM_CREDIT_SUM_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
    BUREAU_AVG_DAYS_CREDIT=("DAYS_CREDIT", "mean"),
    BUREAU_TOTAL_LATE_MONTHS=("BB_LATE_MONTHS_COUNT", "sum"),
).reset_index()

print(f"Bureau aggregated -> {bureau_agg.shape[0]:,} applicants")

# ----------------------------------------------------------------------------
# STEP 1c: previous_application + POS_CASH_balance (prior loans with Home Credit)
# ----------------------------------------------------------------------------

previous_application = load_csv("HC_previous_application.csv")
pos_cash_balance = load_csv("HC_POS_CASH_balance.csv")

prev_agg = previous_application.groupby("SK_ID_CURR").agg(
    PREV_LOAN_COUNT=("SK_ID_PREV", "count"),
    PREV_AVG_CREDIT_AMOUNT=("AMT_CREDIT", "mean"),
    PREV_AVG_ANNUITY=("AMT_ANNUITY", "mean"),
).reset_index()

pos_agg = pos_cash_balance.groupby("SK_ID_CURR").agg(
    POS_AVG_CNT_INSTALMENT_FUTURE=("CNT_INSTALMENT_FUTURE", "mean"),
    POS_MAX_DPD=("SK_DPD", "max"),  # DPD = Days Past Due
).reset_index()

print(f"Previous application aggregated -> {prev_agg.shape[0]:,} applicants")
print(f"POS_CASH_balance aggregated -> {pos_agg.shape[0]:,} applicants")

# ----------------------------------------------------------------------------
# STEP 1d: credit_card_balance + installments_payments (repayment behavior)
# ----------------------------------------------------------------------------

credit_card_balance = load_csv("HC_credit_card_balance.csv")
installments_payments = load_csv("HC_installments_payments.csv")

cc_agg = credit_card_balance.groupby("SK_ID_CURR").agg(
    CC_AVG_BALANCE=("AMT_BALANCE", "mean"),
    CC_MAX_BALANCE=("AMT_BALANCE", "max"),
    CC_AVG_CREDIT_LIMIT=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
).reset_index()

installments_payments["IS_UNDERPAID"] = (
    installments_payments["AMT_PAYMENT"] < installments_payments["AMT_INSTALMENT"]
).astype(int)
installments_payments["PAYMENT_DELAY_DAYS"] = (
    installments_payments["DAYS_ENTRY_PAYMENT"] - installments_payments["DAYS_INSTALMENT"]
)

install_agg = installments_payments.groupby("SK_ID_CURR").agg(
    INSTALL_AVG_PAYMENT_DELAY=("PAYMENT_DELAY_DAYS", "mean"),
    INSTALL_UNDERPAID_RATIO=("IS_UNDERPAID", "mean"),
).reset_index()

print(f"Credit card balance aggregated -> {cc_agg.shape[0]:,} applicants")
print(f"Installments payments aggregated -> {install_agg.shape[0]:,} applicants")

# ----------------------------------------------------------------------------
# STEP 1e: Left-join everything onto the base table
# ----------------------------------------------------------------------------
# "left" join = keep EVERY applicant, even with no bureau/previous-loan
# history. Those columns become NaN for them -- expected, handled in Step 2.

all_aggs = [bureau_agg, prev_agg, pos_agg, cc_agg, install_agg]

starting_rows = app.shape[0]
merged = app.copy()
for agg_df in all_aggs:
    merged = merged.merge(agg_df, on="SK_ID_CURR", how="left")

assert merged.shape[0] == starting_rows, "Row count changed after merge -- check for duplicate keys!"
print(f"\nMerged applicant-level dataset shape: {merged.shape}")

# ----------------------------------------------------------------------------
# STEP 1f: Create your own train/test split (80/20, stratified on TARGET)
# ----------------------------------------------------------------------------
# Since there's no usable application_test file, we split here -- BEFORE any
# imputing, encoding, or scaling -- per Step 0.3 of the guide. Everything in
# Steps 2 onward must be *fit* on train_merged only, then *applied* to
# test_merged, to avoid data leakage.

train_merged, test_merged = train_test_split(
    merged,
    test_size=0.2,
    stratify=merged["TARGET"],   # keeps the ~8% default rate the same in both sets
    random_state=RANDOM_STATE,
)

print(f"train_merged shape: {train_merged.shape}")
print(f"test_merged shape:  {test_merged.shape}")
print(f"Train default rate: {train_merged['TARGET'].mean():.4f}")
print(f"Test default rate:  {test_merged['TARGET'].mean():.4f}")

# ----------------------------------------------------------------------------
# STEP 1g: Save results
# ----------------------------------------------------------------------------

train_out_path = os.path.join(INTERIM_DIR, "train_merged.csv")
test_out_path = os.path.join(INTERIM_DIR, "test_merged.csv")

train_merged.to_csv(train_out_path, index=False)
test_merged.to_csv(test_out_path, index=False)

print(f"\nSaved: {train_out_path}")
print(f"Saved: {test_out_path}")
print("\nStep 1 complete. Next: Step 2 -- Handle Missing Values.")