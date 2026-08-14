"""
STEP 8: Final validation before handing off to modeling.

WHAT THIS SCRIPT DOES:
  Runs through the guide's Step 8 checklist programmatically wherever
  possible, and produces the two "final" artifacts a modeling stage
  actually needs: a consolidated data dictionary and versioned copies of
  the finished datasets in data/processed/.

CHECKLIST ITEMS AND HOW EACH IS VERIFIED:
  [ ] No missing values remain
      -> checked directly: .isna().sum().sum() == 0 on every final file.
  [ ] Train and test have identical columns, same order
      -> checked directly: column list comparison.
  [ ] No target leakage
      -> can't be fully automated (leakage is about HOW a feature was
         built, not a pattern in the data), so this documents the manual
         reasoning: every engineered feature (Step 5) was built from
         AMT_*/DAYS_*/BUREAU_* columns, never from TARGET. As a lightweight
         sanity check, this script also flags any feature suspiciously
         highly correlated with TARGET (|corr| > 0.95) for manual review --
         a real feature is very unlikely to hit that by chance.
  [ ] Class imbalance handling is decided and documented
      -> checked directly: confirms class_imbalance_log.csv exists.
  [ ] Data dictionary maps every final column back to its meaning
      -> built directly: consolidates every log from Steps 2, 4, 5, 6, 7
         (imputation, encoding, feature engineering, class imbalance,
         scaling) into one file, with "original application field,
         unchanged" for any column no step touched.
  [ ] Processed dataset saved with a version tag
      -> done directly: copies the final CSVs into data/processed/ with
         a version prefix (bump PROCESSED_VERSION below for future runs).

OUTPUT (in data/processed/):
  - v1_train.csv, v1_test.csv                       (unscaled, for tree models)
  - v1_train_resampled_smote.csv                    (SMOTE alternative)
  - v1_train_scaled.csv, v1_test_scaled.csv         (for LIME/distance-based use)
  - v1_train_resampled_smote_scaled.csv
  - v1_data_dictionary.csv
  - v1_validation_report.txt (plain-text summary of every checklist result)
"""

import pandas as pd
import numpy as np
import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

PROCESSED_VERSION = "v1"  # bump this (v2, v3, ...) if you rerun the pipeline later with changes

report_lines = []


def report(line):
    print(line)
    report_lines.append(line)


report("=" * 70)
report("STEP 8: FINAL VALIDATION CHECKLIST")
report("=" * 70)

# ----------------------------------------------------------------------------
# Load every final-stage file
# ----------------------------------------------------------------------------

files = {
    "train": "train_features.csv",
    "test": "test_features.csv",
    "train_resampled_smote": "train_resampled_smote.csv",
    "train_scaled": "train_scaled.csv",
    "test_scaled": "test_scaled.csv",
    "train_resampled_smote_scaled": "train_resampled_smote_scaled.csv",
}

data = {}
for key, fname in files.items():
    path = os.path.join(INTERIM_DIR, fname)
    data[key] = pd.read_csv(path)
    report(f"Loaded {fname}: {data[key].shape}")

# ----------------------------------------------------------------------------
# CHECK 1: No missing values remain
# ----------------------------------------------------------------------------

report("\n[CHECK 1] No missing values remain")
check1_pass = True
for key, df in data.items():
    n_missing = df.isna().sum().sum()
    status = "OK" if n_missing == 0 else "FAIL"
    if n_missing != 0:
        check1_pass = False
    report(f"  {key}: {n_missing} missing values -> {status}")

# ----------------------------------------------------------------------------
# CHECK 2: Train and test have identical columns, same order
# ----------------------------------------------------------------------------

report("\n[CHECK 2] Train/test column alignment")
pairs = [("train", "test"), ("train_scaled", "test_scaled")]
check2_pass = True
for train_key, test_key in pairs:
    train_cols = list(data[train_key].columns)
    test_cols = list(data[test_key].columns)
    # test won't have exactly train's columns if TARGET differs -- but both
    # were built from the same split, so TARGET exists in both here.
    match = train_cols == test_cols
    if not match:
        check2_pass = False
    report(f"  {train_key} vs {test_key}: {'identical' if match else 'MISMATCH'}")

# ----------------------------------------------------------------------------
# CHECK 3: No target leakage (manual reasoning + automated correlation flag)
# ----------------------------------------------------------------------------

report("\n[CHECK 3] Target leakage check")
report("  Manual confirmation: every engineered feature (Step 5) was built from")
report("  AMT_*, DAYS_*, and BUREAU_* source columns only -- TARGET was never used")
report("  as an input to any imputation, encoding, or feature-engineering step.")

numeric_cols = data["train"].select_dtypes(include=[np.number]).columns
numeric_cols = [c for c in numeric_cols if c not in ("SK_ID_CURR", "TARGET")]
correlations = data["train"][numeric_cols].corrwith(data["train"]["TARGET"]).abs().sort_values(ascending=False)
suspicious = correlations[correlations > 0.95]

if len(suspicious) > 0:
    report(f"  WARNING: {len(suspicious)} column(s) correlate with TARGET above 0.95 -- review manually:")
    for col, corr in suspicious.items():
        report(f"    {col}: {corr:.4f}")
    check3_pass = False
else:
    report(f"  No column exceeds 0.95 correlation with TARGET (highest: {correlations.iloc[0]:.4f},"
           f" {correlations.index[0]}) -> no automated red flags")
    check3_pass = True

# ----------------------------------------------------------------------------
# CHECK 4: Class imbalance handling documented
# ----------------------------------------------------------------------------

report("\n[CHECK 4] Class imbalance handling documented")
imbalance_log_path = os.path.join(INTERIM_DIR, "class_imbalance_log.csv")
check4_pass = os.path.exists(imbalance_log_path)
report(f"  class_imbalance_log.csv exists: {check4_pass}")

# ----------------------------------------------------------------------------
# CHECK 5: Build the consolidated data dictionary
# ----------------------------------------------------------------------------

report("\n[CHECK 5] Building consolidated data dictionary")

log_files = {
    "imputation": ("imputation_log.csv", "column"),
    "outlier_treatment": ("outlier_treatment_log.csv", "column"),
    "encoding": ("encoding_data_dictionary.csv", "encoded_column"),
    "feature_engineering": ("feature_engineering_log.csv", "column"),
    "scaling": ("scaling_log.csv", "column"),
}

notes_by_column = {}
for step_name, (fname, col_key) in log_files.items():
    path = os.path.join(INTERIM_DIR, fname)
    if not os.path.exists(path):
        continue
    log_df = pd.read_csv(path)
    for _, row in log_df.iterrows():
        col_name = row[col_key]
        note = " | ".join(f"{k}={v}" for k, v in row.items() if k != col_key)
        notes_by_column.setdefault(col_name, []).append(f"[{step_name}] {note}")

final_columns = [c for c in data["train"].columns if c not in ("SK_ID_CURR", "TARGET")]
dictionary_rows = []
for col in final_columns:
    if col in notes_by_column:
        meaning = " ; ".join(notes_by_column[col])
    else:
        meaning = "Original Home Credit application field, unchanged by preprocessing"
    dictionary_rows.append({"column": col, "meaning": meaning})

data_dictionary = pd.DataFrame(dictionary_rows)
report(f"  Built dictionary for {len(data_dictionary)} columns "
       f"({sum(1 for r in dictionary_rows if 'unchanged' not in r['meaning'])} documented by a preprocessing step, "
       f"{sum(1 for r in dictionary_rows if 'unchanged' in r['meaning'])} original fields)")

# ----------------------------------------------------------------------------
# CHECK 6: Save versioned processed datasets
# ----------------------------------------------------------------------------

report(f"\n[CHECK 6] Saving versioned datasets to data/processed/ (version={PROCESSED_VERSION})")

save_map = {
    "train": f"{PROCESSED_VERSION}_train.csv",
    "test": f"{PROCESSED_VERSION}_test.csv",
    "train_resampled_smote": f"{PROCESSED_VERSION}_train_resampled_smote.csv",
    "train_scaled": f"{PROCESSED_VERSION}_train_scaled.csv",
    "test_scaled": f"{PROCESSED_VERSION}_test_scaled.csv",
    "train_resampled_smote_scaled": f"{PROCESSED_VERSION}_train_resampled_smote_scaled.csv",
}

for key, out_name in save_map.items():
    out_path = os.path.join(PROCESSED_DIR, out_name)
    data[key].to_csv(out_path, index=False)
    report(f"  Saved: {out_path}")

dict_out_path = os.path.join(PROCESSED_DIR, f"{PROCESSED_VERSION}_data_dictionary.csv")
data_dictionary.to_csv(dict_out_path, index=False)
report(f"  Saved: {dict_out_path}")

# ----------------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------------

report("\n" + "=" * 70)
report("CHECKLIST SUMMARY")
report("=" * 70)
report(f"  [{'x' if check1_pass else ' '}] No missing values remain")
report(f"  [{'x' if check2_pass else ' '}] Train/test identical columns")
report(f"  [{'x' if check3_pass else '!'}] No target leakage (manual + automated flag)")
report(f"  [{'x' if check4_pass else ' '}] Class imbalance handling documented")
report(f"  [x] Data dictionary built ({len(data_dictionary)} columns)")
report(f"  [x] Processed dataset saved with version tag '{PROCESSED_VERSION}'")

all_pass = check1_pass and check2_pass and check3_pass and check4_pass
report(f"\nOverall: {'ALL CHECKS PASSED' if all_pass else 'REVIEW ITEMS FLAGGED ABOVE BEFORE HANDING OFF'}")

report_out_path = os.path.join(PROCESSED_DIR, f"{PROCESSED_VERSION}_validation_report.txt")
with open(report_out_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"\nSaved: {report_out_path}")

print("\nStep 8 complete -- preprocessing stage (Section 3.2.3) finished.")
print("Hand-off point for Section 3.2.4 (Model Development) is data/processed/.")