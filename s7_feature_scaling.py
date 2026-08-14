"""
STEP 7: Feature scaling.

WHAT THIS SCRIPT DOES:
  Applies StandardScaler (mean 0, std 1) to the CONTINUOUS numeric columns
  only. Binary/flag columns (HAS_*, *_MISSING, one-hot dummies, etc.) are
  left alone -- they're already 0/1, and scaling a binary column doesn't
  meaningfully change anything but does make it harder to read.

  Not required for Decision Tree/Random Forest/XGBoost (they split on
  thresholds, not distances), but produced here for later use with LIME's
  local linear surrogate models, which can weight features unevenly across
  wildly different scales (e.g. AMT_CREDIT in the hundreds of thousands vs.
  a ratio column between 0 and 1).

  WHICH COLUMNS COUNT AS "CONTINUOUS":
    Any numeric column, excluding SK_ID_CURR (an ID, not a feature),
    TARGET (the label), and any column with only 2 unique values in train
    (binary flags and one-hot dummies -- scaling these adds no value).

CRITICAL RULE FOLLOWED:
  The scaler is FIT on train_features.csv only, then the exact same
  fitted transform is applied to test_features.csv and to
  train_resampled_smote.csv. Nothing is fit on test or on the resampled
  (synthetic) data.

OUTPUT:
  - data/interim/train_scaled.csv
  - data/interim/test_scaled.csv
  - data/interim/train_resampled_smote_scaled.csv
  - data/interim/scaling_log.csv (which columns were scaled vs left alone)
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

train = pd.read_csv(os.path.join(INTERIM_DIR, "train_features.csv"))
test = pd.read_csv(os.path.join(INTERIM_DIR, "test_features.csv"))
train_smote = pd.read_csv(os.path.join(INTERIM_DIR, "train_resampled_smote.csv"))

print(f"Loaded train_features: {train.shape}")
print(f"Loaded test_features:  {test.shape}")
print(f"Loaded train_resampled_smote: {train_smote.shape}")

log_rows = []

# ----------------------------------------------------------------------------
# Identify which columns to scale
# ----------------------------------------------------------------------------

id_col = "SK_ID_CURR"
target_col = "TARGET"

candidate_cols = [c for c in train.columns if c not in (id_col, target_col)]
binary_cols = [c for c in candidate_cols if train[c].nunique() <= 2]
continuous_cols = [c for c in candidate_cols if c not in binary_cols]

print(f"\n{len(continuous_cols)} continuous columns will be scaled")
print(f"{len(binary_cols)} binary/flag/one-hot columns left unscaled")

for c in binary_cols:
    log_rows.append({"column": c, "scaled": False, "reason": "binary (<=2 unique values in train)"})
for c in continuous_cols:
    log_rows.append({"column": c, "scaled": True, "reason": "continuous numeric feature"})

# ----------------------------------------------------------------------------
# Fit StandardScaler on TRAIN ONLY, apply to train/test/smote
# ----------------------------------------------------------------------------

scaler = StandardScaler()
train_scaled_values = scaler.fit_transform(train[continuous_cols])  # fit + transform on train
test_scaled_values = scaler.transform(test[continuous_cols])         # transform only, using train's fit
smote_scaled_values = scaler.transform(train_smote[continuous_cols])  # same train-fit scaler

train_scaled = train.copy()
test_scaled = test.copy()
train_smote_scaled = train_smote.copy()

train_scaled[continuous_cols] = train_scaled_values
test_scaled[continuous_cols] = test_scaled_values
train_smote_scaled[continuous_cols] = smote_scaled_values

print("\nScaling applied (fit on train, same transform applied to test and SMOTE set).")

# ----------------------------------------------------------------------------
# Final checks
# ----------------------------------------------------------------------------

assert list(train_scaled.columns) == list(test_scaled.columns), "Column mismatch train vs test after scaling!"
assert train_scaled.isna().sum().sum() == 0, "Missing values introduced during scaling!"
assert test_scaled.isna().sum().sum() == 0, "Missing values introduced during scaling!"

# Sanity check: scaled train columns should have ~mean 0, ~std 1
sample_check = train_scaled[continuous_cols].mean().abs().max()
print(f"Sanity check -- max abs mean across scaled train columns: {sample_check:.6f} (should be ~0)")

# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------

train_out = os.path.join(INTERIM_DIR, "train_scaled.csv")
test_out = os.path.join(INTERIM_DIR, "test_scaled.csv")
smote_out = os.path.join(INTERIM_DIR, "train_resampled_smote_scaled.csv")
log_out = os.path.join(INTERIM_DIR, "scaling_log.csv")

train_scaled.to_csv(train_out, index=False)
test_scaled.to_csv(test_out, index=False)
train_smote_scaled.to_csv(smote_out, index=False)
pd.DataFrame(log_rows).to_csv(log_out, index=False)

print(f"\ntrain_scaled shape: {train_scaled.shape}")
print(f"test_scaled shape:  {test_scaled.shape}")
print(f"train_resampled_smote_scaled shape: {train_smote_scaled.shape}")
print(f"\nSaved: {train_out}")
print(f"Saved: {test_out}")
print(f"Saved: {smote_out}")
print(f"Saved: {log_out}")
print("\nStep 7 complete.")
print("Unscaled versions (train_features.csv, test_features.csv, train_resampled_smote.csv)")
print("remain available too -- use those for Decision Tree/RF/XGBoost, use the _scaled")
print("versions only where LIME or distance-based analysis needs comparable feature ranges.")
print("\nNext: Step 8 -- Final Validation Checklist.")