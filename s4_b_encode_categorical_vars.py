"""
STEP 4b: Encode categorical variables.

WHAT THIS SCRIPT DOES, AND WHY EACH COLUMN GETS A DIFFERENT TREATMENT:

  1. ONE-HOT ENCODING (13 low-cardinality nominal columns):
     No natural order (e.g. NAME_HOUSING_TYPE: "House / apartment" isn't
     "more" or "less" than "Rented apartment"), and few enough categories
     that one column per category doesn't blow up the feature space.
     NAME_CONTRACT_TYPE, CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY,
     NAME_TYPE_SUITE, NAME_INCOME_TYPE, NAME_FAMILY_STATUS,
     NAME_HOUSING_TYPE, WEEKDAY_APPR_PROCESS_START, FONDKAPREMONT_MODE,
     HOUSETYPE_MODE, WALLSMATERIAL_MODE, EMERGENCYSTATE_MODE.

  2. ORDINAL ENCODING (1 column with a genuine order):
     NAME_EDUCATION_TYPE has a real rank (Lower secondary < Secondary <
     Incomplete higher < Higher education < Academic degree). Encoding it
     as a single 0-4 number preserves that order, which a model can use
     directly -- one-hot would throw the ordering away.

  3. FREQUENCY ENCODING (2 high-cardinality columns):
     OCCUPATION_TYPE (19 categories) and ORGANIZATION_TYPE (58 categories)
     would add 75 one-hot columns between them for a single applicant
     attribute each. Instead, each category is replaced by how common it
     was IN TRAIN (e.g. "Laborers" -> 0.18 if 18% of train applicants were
     laborers). We deliberately avoid TARGET encoding here -- encoding by
     default rate would leak the outcome into the features unless done
     with cross-validation folds, which this script doesn't set up.

CRITICAL RULES FOLLOWED THROUGHOUT:
  - All encodings (one-hot categories, ordinal mapping, frequency values)
    are DEFINED FROM TRAIN ONLY, then applied to test -- test is never
    used to decide what the encoding looks like.
  - After one-hot encoding, train and test are reindexed to have IDENTICAL
    columns in the same order (Step 4.3 of the guide) -- any category that
    only appears in test gets dropped (a model can't learn from a category
    it never saw in training), any train category missing from a test row
    becomes 0.

OUTPUT:
  - data/interim/train_encoded.csv, data/interim/test_encoded.csv
  - data/interim/encoding_data_dictionary.csv
    (maps every encoded column back to its original category/meaning --
    this is what Step 4.2 of the guide asks you to keep for later
    SHAP/LIME interpretability work)
"""

import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")

train = pd.read_csv(os.path.join(INTERIM_DIR, "train_outliers_handled.csv"))
test = pd.read_csv(os.path.join(INTERIM_DIR, "test_outliers_handled.csv"))
print(f"Loaded train_outliers_handled: {train.shape}")
print(f"Loaded test_outliers_handled:  {test.shape}")

dictionary_rows = []


def log_dict(encoded_column, original_column, method, meaning):
    dictionary_rows.append({
        "encoded_column": encoded_column,
        "original_column": original_column,
        "method": method,
        "meaning": meaning,
    })


# ----------------------------------------------------------------------------
# STEP A: Ordinal encode NAME_EDUCATION_TYPE
# ----------------------------------------------------------------------------

education_order = {
    "Lower secondary": 0,
    "Secondary / secondary special": 1,
    "Incomplete higher": 2,
    "Higher education": 3,
    "Academic degree": 4,
}

train["NAME_EDUCATION_TYPE_ORDINAL"] = train["NAME_EDUCATION_TYPE"].map(education_order)
test["NAME_EDUCATION_TYPE_ORDINAL"] = test["NAME_EDUCATION_TYPE"].map(education_order)

# Safety net: if test somehow has a category not in the mapping, don't silently
# produce NaN -- fill with the train median rank instead.
if test["NAME_EDUCATION_TYPE_ORDINAL"].isna().any():
    fallback = train["NAME_EDUCATION_TYPE_ORDINAL"].median()
    n_fallback = test["NAME_EDUCATION_TYPE_ORDINAL"].isna().sum()
    test["NAME_EDUCATION_TYPE_ORDINAL"] = test["NAME_EDUCATION_TYPE_ORDINAL"].fillna(fallback)
    print(f"NOTE: {n_fallback} test rows had an unrecognized education category -> filled with train median rank")

train = train.drop(columns=["NAME_EDUCATION_TYPE"])
test = test.drop(columns=["NAME_EDUCATION_TYPE"])

for category, rank in education_order.items():
    log_dict("NAME_EDUCATION_TYPE_ORDINAL", "NAME_EDUCATION_TYPE", "ordinal", f"{category} -> {rank}")

print("\nStep A done -- NAME_EDUCATION_TYPE ordinal-encoded (0-4)")

# ----------------------------------------------------------------------------
# STEP B: Frequency encode OCCUPATION_TYPE and ORGANIZATION_TYPE
# ----------------------------------------------------------------------------

FREQUENCY_ENCODE_COLS = ["OCCUPATION_TYPE", "ORGANIZATION_TYPE"]

for col in FREQUENCY_ENCODE_COLS:
    freq_map = train[col].value_counts(normalize=True)  # computed on TRAIN ONLY

    new_col = f"{col}_FREQ"
    train[new_col] = train[col].map(freq_map)
    test[new_col] = test[col].map(freq_map)

    # Any category in test that never appeared in train -> 0 (model has no
    # basis to treat it as anything but "unseen/rare").
    n_unseen = test[new_col].isna().sum()
    test[new_col] = test[new_col].fillna(0)
    if n_unseen > 0:
        print(f"NOTE: {n_unseen} test rows had a {col} category not seen in train -> frequency set to 0")

    train = train.drop(columns=[col])
    test = test.drop(columns=[col])

    for category, freq in freq_map.items():
        log_dict(new_col, col, "frequency", f"{category} -> {freq:.4f} (share of train rows)")

    print(f"Step B done -- {col} frequency-encoded -> {new_col}")

# ----------------------------------------------------------------------------
# STEP C: One-hot encode the 13 low-cardinality nominal columns
# ----------------------------------------------------------------------------

ONEHOT_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "WEEKDAY_APPR_PROCESS_START", "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
]

train_encoded = pd.get_dummies(train, columns=ONEHOT_COLS, prefix=ONEHOT_COLS)
test_encoded = pd.get_dummies(test, columns=ONEHOT_COLS, prefix=ONEHOT_COLS)

# Reindex test to match train's columns EXACTLY (Step 4.3 of the guide):
# - a category only seen in test gets dropped (model never learned it)
# - a train category missing from test becomes a column of 0s
train_cols = train_encoded.columns
n_test_only = len(set(test_encoded.columns) - set(train_cols))
if n_test_only > 0:
    print(f"NOTE: {n_test_only} one-hot columns existed only in test -> dropped (unseen in train)")

test_encoded = test_encoded.reindex(columns=train_cols, fill_value=0)

for col in ONEHOT_COLS:
    dummy_cols = [c for c in train_cols if c.startswith(f"{col}_")]
    for dc in dummy_cols:
        category = dc[len(col) + 1:]
        log_dict(dc, col, "one-hot", f"1 if original value was '{category}', else 0")

print(f"\nStep C done -- one-hot encoded {len(ONEHOT_COLS)} columns")

# ----------------------------------------------------------------------------
# Final checks
# ----------------------------------------------------------------------------

assert list(train_encoded.columns) == list(test_encoded.columns), "Column mismatch between train and test after encoding!"
remaining_categorical = [c for c in train_encoded.columns if not pd.api.types.is_numeric_dtype(train_encoded[c])]
assert len(remaining_categorical) == 0, f"Still-unencoded columns remain: {remaining_categorical}"

print(f"\ntrain_encoded shape: {train_encoded.shape}")
print(f"test_encoded shape:  {test_encoded.shape}")
print("Confirmed: train and test have identical columns, and no text columns remain.")

# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------

train_out = os.path.join(INTERIM_DIR, "train_encoded.csv")
test_out = os.path.join(INTERIM_DIR, "test_encoded.csv")
dict_out = os.path.join(INTERIM_DIR, "encoding_data_dictionary.csv")

train_encoded.to_csv(train_out, index=False)
test_encoded.to_csv(test_out, index=False)
pd.DataFrame(dictionary_rows).to_csv(dict_out, index=False)

print(f"\nSaved: {train_out}")
print(f"Saved: {test_out}")
print(f"Saved: {dict_out}  ({len(dictionary_rows)} entries)")
print("\nStep 4 complete. Next: Step 5 -- Feature Engineering.")