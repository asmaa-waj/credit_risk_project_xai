"""
STEP 11a: LIME Local Explanations — Section 3.2.6, Explainability Integration.

WHAT THIS SCRIPT DOES:
  Generates LIME (Local Interpretable Model-agnostic Explanations) for
  the SAME three applicants selected in s10_b_shap_local.py (LOW_RISK,
  HIGH_RISK, BORDERLINE -- identified by SK_ID_CURR), so their SHAP and
  LIME explanations can be directly compared in s12_compare_shap_lime.py.

WHY v1_test_scaled.csv (NOT v1_test.csv, UNLIKE s10_a/s10_b):
  LIME works by perturbing an applicant's feature values (sampling nearby
  points) and fitting a simple local linear model on the perturbed
  neighborhood. Because it measures "nearby" using raw feature distances,
  features on large scales (e.g. AMT_INCOME_TOTAL, in the hundreds of
  thousands) would dominate that distance calculation over features on
  small scales (e.g. ratios between 0 and 1), regardless of which
  features actually matter more. Standardized data (mean 0, std 1 --
  v1_test_scaled.csv, prepared for exactly this purpose in Step 7 of
  preprocessing) puts every feature on equal footing for that distance
  calculation. SHAP's TreeExplainer has no such issue since it doesn't
  rely on distance -- which is why s10_a/s10_b use the unscaled file.

HOW THE SAME APPLICANTS ARE FOUND IN THE SCALED FILE:
  v1_test.csv and v1_test_scaled.csv contain the same rows in the same
  order (both are the Step 8 validated output, differing only in whether
  continuous features are standardized) -- so the three SK_ID_CURR values
  saved by s10_b (in shap_local_examples.csv) are looked up directly by
  ID in v1_test_scaled.csv, rather than re-sampling.

WHAT LIME NEEDS THAT SHAP DOESN'T: A TRAINING BACKGROUND SET
  Unlike TreeExplainer (which only needs the model), LimeTabularExplainer
  needs a background sample of training data to learn the overall
  feature distributions (ranges, typical values) it perturbs around --
  v1_train_scaled.csv is used for this, at a manageable sample size for
  speed.

OUTPUT:
  - outputs/figures/lime_low_risk.png
  - outputs/figures/lime_high_risk.png
  - outputs/figures/lime_borderline.png
  - outputs/results/lime_local_examples.csv   (same shape as
                                                 shap_local_examples.csv --
                                                 SK_ID_CURR, risk category,
                                                 top LIME features + weights --
                                                 so s12_compare_shap_lime.py
                                                 can join the two directly)

REQUIRES: lime
    pip install lime
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    raise ImportError(
        "This script needs lime. In PyCharm's terminal, run:\n"
        "  pip install lime"
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
for d in (RESULTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
N_BACKGROUND = 2000     # training rows used as LIME's background distribution
TOP_N_REASONS = 5       # how many top LIME features to log per applicant
NUM_LIME_FEATURES = 15  # how many features LIME's local model considers

# ----------------------------------------------------------------------------
# Load model, scaled data, and the SAME applicants selected in s10_b
# ----------------------------------------------------------------------------

model_path = os.path.join(MODELS_DIR, "xgboost.joblib")
model = joblib.load(model_path)
print(f"Loaded model: {model_path}")

train_scaled = pd.read_csv(os.path.join(DATA_DIR, "v1_train_scaled.csv"))
test_scaled = pd.read_csv(os.path.join(DATA_DIR, "v1_test_scaled.csv"))
print(f"Loaded v1_train_scaled: {train_scaled.shape}")
print(f"Loaded v1_test_scaled:  {test_scaled.shape}")

shap_examples_path = os.path.join(RESULTS_DIR, "shap_local_examples.csv")
shap_examples = pd.read_csv(shap_examples_path)
selected_ids = (
    shap_examples[["risk_category", "SK_ID_CURR", "predicted_proba"]]
    .drop_duplicates()
    .set_index("risk_category")
)
print(f"\nLoaded selected applicants from s10_b ({shap_examples_path}):")
print(selected_ids.to_string())

id_col = "SK_ID_CURR"
target_col = "TARGET"
feature_cols = [c for c in test_scaled.columns if c not in (id_col, target_col)]

X_background = train_scaled[feature_cols].sample(N_BACKGROUND, random_state=RANDOM_STATE).values

# ----------------------------------------------------------------------------
# Build the LIME explainer
# ----------------------------------------------------------------------------

print(f"\nBuilding LimeTabularExplainer (background sample size={N_BACKGROUND})...")
explainer = LimeTabularExplainer(
    training_data=X_background,
    feature_names=feature_cols,
    class_names=["Repaid", "Default"],
    mode="classification",
    random_state=RANDOM_STATE,
)

def predict_fn(x):
    return model.predict_proba(pd.DataFrame(x, columns=feature_cols))

# ----------------------------------------------------------------------------
# Generate a LIME explanation + plot for each of the same three applicants
# ----------------------------------------------------------------------------

records = []

for label, row in selected_ids.iterrows():
    sk_id = row["SK_ID_CURR"]
    applicant_row = test_scaled[test_scaled[id_col] == sk_id]
    if applicant_row.empty:
        print(f"\n[WARNING] SK_ID_CURR={sk_id} ({label}) not found in v1_test_scaled.csv -- skipping.")
        continue

    x_instance = applicant_row[feature_cols].values[0]

    print(f"\nExplaining {label} (SK_ID_CURR={sk_id}) with LIME...")
    explanation = explainer.explain_instance(
        x_instance,
        predict_fn,
        num_features=NUM_LIME_FEATURES,
    )

    fig = explanation.as_pyplot_figure()
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"lime_{label.lower()}.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Log the top N LIME features (already sorted by |weight| in as_list())
    top_features = sorted(explanation.as_list(), key=lambda t: abs(t[1]), reverse=True)[:TOP_N_REASONS]
    for feature_desc, weight in top_features:
        records.append({
            "risk_category": label,
            "SK_ID_CURR": sk_id,
            "predicted_proba": row["predicted_proba"],
            "feature_condition": feature_desc,
            "lime_weight": weight,
            "direction": "increases default risk" if weight > 0 else "decreases default risk",
        })

# ----------------------------------------------------------------------------
# Save the combined table (used by s12 for SHAP-vs-LIME comparison)
# ----------------------------------------------------------------------------

examples_df = pd.DataFrame(records)
print(f"\nTop {TOP_N_REASONS} LIME reasons per applicant:")
print(examples_df.to_string(index=False))

examples_path = os.path.join(RESULTS_DIR, "lime_local_examples.csv")
examples_df.to_csv(examples_path, index=False)
print(f"\nSaved: {examples_path}")

print("\nStep 11a (LIME local) complete.")
print("Next: s12_compare_shap_lime.py -- compare SHAP vs LIME top reasons")
print("for these same three applicants side by side.")
