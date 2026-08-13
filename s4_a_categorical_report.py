"""
STEP 4a: Categorical column report.

WHAT THIS SCRIPT DOES:
  Loads train_outliers_handled.csv (from Step 3) and, for every categorical
  (text) column, reports how many unique values it has and what they are.
  This is purely informational -- nothing is changed yet. We use this to
  decide, per column, whether it should be one-hot encoded (low-cardinality
  nominal), ordinal encoded (has a natural order), or frequency/target
  encoded (high-cardinality).

HOW TO RUN IN PYCHARM:
  Right-click -> Run 'step4a_categorical_report'.
"""

import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

train = pd.read_csv(os.path.join(INTERIM_DIR, "train_outliers_handled.csv"))
print(f"Loaded train_outliers_handled: {train.shape}\n")

# A column counts as "categorical" here if it's non-numeric (object/string dtype).
categorical_cols = [c for c in train.columns if not pd.api.types.is_numeric_dtype(train[c])]

print(f"{len(categorical_cols)} categorical columns found.\n")
print("=" * 80)

rows = []
for c in categorical_cols:
    n_unique = train[c].nunique()
    values_preview = train[c].value_counts().head(10)
    rows.append({"column": c, "n_unique": n_unique})

    print(f"\n{c}  ({n_unique} unique values)")
    print(values_preview.to_string())

print("\n" + "=" * 80)
summary = pd.DataFrame(rows).sort_values("n_unique")
print("\nSUMMARY (sorted by cardinality):")
print(summary.to_string(index=False))

report_path = os.path.join(INTERIM_DIR, "categorical_report.csv")
summary.to_csv(report_path, index=False)
print(f"\nSaved: {report_path}")