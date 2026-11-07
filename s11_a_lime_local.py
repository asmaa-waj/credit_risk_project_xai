"""
STEP 11a: LIME Local Explanations
Corrected version: scaled LIME neighbourhood + unscaled XGBoost predictions.

WHAT THIS SCRIPT DOES:
  Generates LIME explanations for the same three applicants selected
  in s10_b_shap_local.py:
  LOW_RISK, HIGH_RISK, and BORDERLINE.

IMPORTANT:
  XGBoost was trained on unscaled data.
  LIME uses scaled data for its neighbourhood calculations.

  Therefore, predict_fn converts LIME's scaled perturbed samples
  back to their original scale before passing them to XGBoost.

OUTPUT:
  - outputs/figures/lime_low_risk.png
  - outputs/figures/lime_high_risk.png
  - outputs/figures/lime_borderline.png
  - outputs/results/lime_local_examples.csv
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    raise ImportError(
        "This script needs lime. Run:\n"
        "pip install lime"
    )

# ----------------------------------------------------------------------------
# PATHS AND SETTINGS
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
INTERIM_DIR = os.path.join(BASE_DIR, "data", "interim")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")

for directory in (RESULTS_DIR, FIGURES_DIR):
    os.makedirs(directory, exist_ok=True)

RANDOM_STATE = 42
N_BACKGROUND = 2000
TOP_N_REASONS = 5
NUM_LIME_FEATURES = 15

# ----------------------------------------------------------------------------
# LOAD MODEL AND DATA
# ----------------------------------------------------------------------------

model_path = os.path.join(MODELS_DIR, "xgboost.joblib")
model = joblib.load(model_path)

print(f"Loaded model: {model_path}")

train_scaled = pd.read_csv(
    os.path.join(DATA_DIR, "v1_train_scaled.csv")
)

test_scaled = pd.read_csv(
    os.path.join(DATA_DIR, "v1_test_scaled.csv")
)

train_original = pd.read_csv(
    os.path.join(INTERIM_DIR, "train_features.csv")
)

scaling_log = pd.read_csv(
    os.path.join(INTERIM_DIR, "scaling_log.csv")
)

print(f"Scaled training data: {train_scaled.shape}")
print(f"Scaled test data: {test_scaled.shape}")
print(f"Original training data: {train_original.shape}")

# ----------------------------------------------------------------------------
# IDENTIFY SCALED FEATURES AND RECONSTRUCT STANDARD SCALER
# ----------------------------------------------------------------------------

id_col = "SK_ID_CURR"
target_col = "TARGET"

feature_cols = [
    column for column in test_scaled.columns
    if column not in (id_col, target_col)
]

continuous_cols = scaling_log.loc[
    scaling_log["scaled"].astype(str).str.strip().str.lower() == "true",
    "column"
].tolist()

# Keep only columns that exist in the model's feature set
continuous_cols = [
    column for column in continuous_cols
    if column in feature_cols
]

if not continuous_cols:
    raise ValueError(
        "No scaled continuous columns were found in scaling_log.csv."
    )

# Verify that the original training data contains all required columns
missing_columns = [
    column for column in continuous_cols
    if column not in train_original.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in train_features.csv: {missing_columns}"
    )

# Reconstruct the same StandardScaler used in Step 7.
# It was originally fitted on the training data only.
scaler = StandardScaler()

scaler.fit(train_original[continuous_cols])

print(
    f"\nReconstructed StandardScaler for "
    f"{len(continuous_cols)} continuous features."
)

# ----------------------------------------------------------------------------
# VERIFY FEATURE CONSISTENCY
# ----------------------------------------------------------------------------

if feature_cols != [
    column for column in train_scaled.columns
    if column not in (id_col, target_col)
]:
    raise ValueError("Feature order mismatch in scaled training data.")

if not set(feature_cols).issubset(train_original.columns):
    raise ValueError(
        "Some model features are missing from train_features.csv."
    )

print("Feature consistency checks passed.")

# ----------------------------------------------------------------------------
# LOAD THE SAME THREE APPLICANTS SELECTED BY SHAP
# ----------------------------------------------------------------------------

shap_examples_path = os.path.join(
    RESULTS_DIR,
    "shap_local_examples.csv"
)

shap_examples = pd.read_csv(shap_examples_path)

selected_ids = (
    shap_examples[
        ["risk_category", "SK_ID_CURR", "predicted_proba"]
    ]
    .drop_duplicates()
    .set_index("risk_category")
)

print("\nSelected applicants from SHAP:")
print(selected_ids.to_string())

# ----------------------------------------------------------------------------
# PREPARE LIME BACKGROUND DATA
# ----------------------------------------------------------------------------

X_background = (
    train_scaled[feature_cols]
    .sample(N_BACKGROUND, random_state=RANDOM_STATE)
    .values
)

# ----------------------------------------------------------------------------
# BUILD LIME EXPLAINER
# ----------------------------------------------------------------------------

print("\nBuilding LimeTabularExplainer...")

explainer = LimeTabularExplainer(
    training_data=X_background,
    feature_names=feature_cols,
    class_names=["Repaid", "Default"],
    mode="classification",
    random_state=RANDOM_STATE,
    discretize_continuous=True
)

# ----------------------------------------------------------------------------
# CORRECTED PREDICTION FUNCTION
# ----------------------------------------------------------------------------

def predict_fn(x):
    """
    LIME supplies scaled perturbed samples.

    Convert continuous columns back to their original scale
    before passing the samples to the unscaled-trained XGBoost model.
    """

    x_unscaled = pd.DataFrame(
        x,
        columns=feature_cols
    )

    # Inverse-transform continuous features only.
    # Binary and one-hot features remain unchanged.
    x_unscaled[continuous_cols] = scaler.inverse_transform(
        x_unscaled[continuous_cols]
    )

    # Return probabilities for both classes.
    return model.predict_proba(x_unscaled)

# ----------------------------------------------------------------------------
# GENERATE LIME EXPLANATIONS FOR THE THREE APPLICANTS
# ----------------------------------------------------------------------------

records = []

for label, row in selected_ids.iterrows():

    sk_id = row["SK_ID_CURR"]

    applicant_row = test_scaled[
        test_scaled[id_col] == sk_id
    ]

    if applicant_row.empty:
        print(
            f"\n[WARNING] Applicant {sk_id} ({label}) "
            "not found. Skipping."
        )
        continue

    x_instance = applicant_row[feature_cols].values[0]

    print(
        f"\nExplaining {label} "
        f"(SK_ID_CURR={sk_id}) with LIME..."
    )

    explanation = explainer.explain_instance(
        x_instance,
        predict_fn,
        num_features=NUM_LIME_FEATURES,
        labels=(1,)
    )

    # Save LIME plot
    fig = explanation.as_pyplot_figure(label=1)

    fig.tight_layout()

    fig_path = os.path.join(
        FIGURES_DIR,
        f"lime_{label.lower()}.png"
    )

    fig.savefig(
        fig_path,
        dpi=120,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {fig_path}")

    # Extract the top five features for the Default class
    top_features = sorted(
        explanation.as_list(label=1),
        key=lambda item: abs(item[1]),
        reverse=True
    )[:TOP_N_REASONS]

    for feature_desc, weight in top_features:

        records.append({
            "risk_category": label,
            "SK_ID_CURR": sk_id,
            "predicted_proba": row["predicted_proba"],
            "feature_condition": feature_desc,
            "lime_weight": weight,
            "direction": (
                "increases default risk"
                if weight > 0
                else "decreases default risk"
            )
        })

# ----------------------------------------------------------------------------
# SAVE RESULTS
# ----------------------------------------------------------------------------

examples_df = pd.DataFrame(records)

print("\nTop LIME reasons per applicant:")
print(examples_df.to_string(index=False))

examples_path = os.path.join(
    RESULTS_DIR,
    "lime_local_examples.csv"
)

examples_df.to_csv(
    examples_path,
    index=False
)

print(f"\nSaved: {examples_path}")

print("\nStep 11a (Corrected LIME) complete.")
print(
    "Next: s12_compare_shap_lime.py "
    "-- compare SHAP vs LIME explanations."
)
