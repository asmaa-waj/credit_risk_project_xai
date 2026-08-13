"""
STEP 6: Address class imbalance.

WHAT THIS SCRIPT DOES:

  1. Confirms the exact imbalance ratio in train (should be ~8%, matching
     the EDA from Section 3.2.2 and the split check back in Step 1).

  2. Documents the class-weighting decision. This doesn't touch the data
     at all -- it's a parameter (class_weight='balanced' for Decision
     Tree/Random Forest, scale_pos_weight for XGBoost) passed at MODEL
     TRAINING time, not something applied here. train_features.csv from
     Step 5 IS the "weighted" version -- nothing new to save for it.

  3. Generates a SMOTE-resampled ALTERNATIVE training set, so the modeling
     stage (Section 3.2.4) can compare class-weighting vs. SMOTE directly,
     per Step 6.4 of the guide. Since every column was fully one-hot/
     ordinal/frequency encoded in Step 4 (nothing categorical remains),
     plain SMOTE is appropriate -- SMOTENC (which guide mentions as the
     alternative) is only needed if raw categorical columns were kept.

CRITICAL RULES FOLLOWED:
  - SMOTE is fit and applied to TRAIN ONLY. The test set is never
    resampled, loaded read-only here, and re-saved completely unchanged --
    it must keep the real ~8% distribution to give honest metrics later.
  - SK_ID_CURR and TARGET are excluded from the resampling input (an ID
    column has no business being synthesized), then reattached after.
    Synthetic rows get negative placeholder IDs so they're clearly
    distinguishable from real applicants in any later inspection.

REQUIRES: the imbalanced-learn package.
    pip install imbalanced-learn

OUTPUT:
  - data/interim/train_resampled_smote.csv  (SMOTE-balanced alternative)
  - data/interim/test_features.csv is NOT touched/duplicated here --
    Step 7 continues to read directly from Step 5's output for test.
  - data/interim/class_imbalance_log.csv (decision documentation)
"""

import pandas as pd
import numpy as np
import os

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    raise ImportError(
        "This script needs imbalanced-learn. In PyCharm's terminal, run:\n"
        "  pip install imbalanced-learn"
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

RANDOM_STATE = 42

train = pd.read_csv(os.path.join(INTERIM_DIR, "train_features.csv"))
print(f"Loaded train_features: {train.shape}")

log_rows = []


def log(item, detail):
    log_rows.append({"item": item, "detail": detail})


# ----------------------------------------------------------------------------
# STEP 1: Confirm the exact imbalance ratio
# ----------------------------------------------------------------------------

class_counts = train["TARGET"].value_counts()
default_rate = train["TARGET"].mean()

print(f"\nClass distribution in train:")
print(f"  TARGET=0 (repaid):  {class_counts[0]:,}")
print(f"  TARGET=1 (default): {class_counts[1]:,}")
print(f"  Default rate: {default_rate:.4f} ({default_rate*100:.2f}%)")

log("imbalance_ratio", f"TARGET=0: {class_counts[0]}, TARGET=1: {class_counts[1]}, default rate: {default_rate:.4f}")

# ----------------------------------------------------------------------------
# STEP 2: Document the class-weighting decision (no data change)
# ----------------------------------------------------------------------------

log("class_weighting", "train_features.csv (Step 5 output) used as-is with "
    "class_weight='balanced' (Decision Tree/Random Forest) or scale_pos_weight "
    f"={class_counts[0] / class_counts[1]:.2f} (XGBoost) at model training time. "
    "No resampling applied for this option.")

print(f"\nFor XGBoost's scale_pos_weight, use: {class_counts[0] / class_counts[1]:.2f}")

# ----------------------------------------------------------------------------
# STEP 3: Generate a SMOTE-resampled alternative training set
# ----------------------------------------------------------------------------

id_col = "SK_ID_CURR"
target_col = "TARGET"

X_train = train.drop(columns=[id_col, target_col])
y_train = train[target_col]

print(f"\nApplying SMOTE (random_state={RANDOM_STATE})...")
smote = SMOTE(random_state=RANDOM_STATE)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

n_synthetic = len(X_resampled) - len(X_train)
print(f"SMOTE added {n_synthetic:,} synthetic minority-class rows")
print(f"  New class distribution: TARGET=0: {(y_resampled == 0).sum():,}, TARGET=1: {(y_resampled == 1).sum():,}")

# Reattach TARGET
train_resampled = X_resampled.copy()
train_resampled[target_col] = y_resampled

# Reattach SK_ID_CURR: real rows keep their original ID, synthetic rows get
# sequential negative placeholder IDs (clearly not real applicants).
original_ids = train[id_col].values
synthetic_ids = -np.arange(1, n_synthetic + 1)  # -1, -2, -3, ...
all_ids = np.concatenate([original_ids, synthetic_ids])
train_resampled[id_col] = all_ids

# Put ID and TARGET back at the front for readability
cols = [id_col, target_col] + [c for c in train_resampled.columns if c not in (id_col, target_col)]
train_resampled = train_resampled[cols]

log("smote", f"Applied SMOTE to train only (fit_resample on X_train, y_train). "
    f"Added {n_synthetic} synthetic rows, given negative SK_ID_CURR values "
    f"({synthetic_ids.min()} to -1) to distinguish them from real applicants. "
    "Test set NOT resampled -- must keep real-world class distribution.")

# ----------------------------------------------------------------------------
# Final checks
# ----------------------------------------------------------------------------

assert train_resampled[target_col].value_counts()[0] == train_resampled[target_col].value_counts()[1], \
    "SMOTE output isn't balanced 50/50 -- check imblearn version/parameters"
assert train_resampled.isna().sum().sum() == 0, "Missing values found in SMOTE output!"

print(f"\ntrain_resampled shape: {train_resampled.shape}")
print("Confirmed: classes are balanced 50/50, no missing values.")

# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------

resampled_out = os.path.join(INTERIM_DIR, "train_resampled_smote.csv")
log_out = os.path.join(INTERIM_DIR, "class_imbalance_log.csv")

train_resampled.to_csv(resampled_out, index=False)
pd.DataFrame(log_rows).to_csv(log_out, index=False)

print(f"\nSaved: {resampled_out}")
print(f"Saved: {log_out}")
print("\nStep 6 complete.")
print("Two training options now exist for the modeling stage to compare:")
print("  1. train_features.csv        -> use with class_weight='balanced' / scale_pos_weight")
print("  2. train_resampled_smote.csv -> use as-is, already balanced 50/50")
print("Either way, test_features.csv (untouched, real ~8% distribution) is the evaluation set.")
print("\nNext: Step 7 -- Feature Scaling.")