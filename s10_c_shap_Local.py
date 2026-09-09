"""
STEP 10c: SHAP Local Explanations — Section 3.2.6, Explainability Integration.

WHAT THIS SCRIPT DOES:
  Picks a small set of representative applicants from the held-out test
  set (v1_test.csv) -- one clearly-approved case, one clearly-rejected
  case, and one borderline case -- and generates an individual SHAP
  waterfall plot for each, showing exactly which features pushed THAT
  applicant's prediction up or down. This is the local counterpart to
  s10_a_shap_global.py's global summary, and is what the dashboard
  (Section 3.2.7) will show end users for a single prediction, rather
  than the aggregate pattern across all applicants.

HOW THE THREE APPLICANTS ARE SELECTED:
  Applicants are chosen by their predicted probability of default
  (model.predict_proba), not the true label -- this reflects what the
  dashboard will actually do at inference time (a real user's true label
  is unknown when they apply):
    - LOW_RISK:  lowest predicted probability of default in the sample
    - HIGH_RISK: highest predicted probability of default in the sample
    - BORDERLINE: probability closest to 0.5 (the decision boundary)
  Selecting from a sample (not the full 61,503-row test set) keeps this
  fast and reproducible; the sample itself is fixed with RANDOM_STATE=42.

WHY WATERFALL PLOTS (NOT THE BEESWARM FROM s10_a):
  A beeswarm plot summarizes patterns across many predictions at once --
  it isn't designed to show a single prediction's breakdown. A waterfall
  plot starts at the model's average (base) prediction and shows each
  feature pushing the output up or down, ending at that one applicant's
  final predicted probability -- this is the standard SHAP visualization
  for explaining one decision, and maps directly onto the "reason codes"
  the dashboard needs to generate later (Section 3.2.7).

OUTPUT:
  - outputs/figures/shap_waterfall_low_risk.png
  - outputs/figures/shap_waterfall_high_risk.png
  - outputs/figures/shap_waterfall_borderline.png
  - outputs/results/shap_local_examples.csv   (SK_ID_CURR, predicted probability,
                                                 risk category, and top 5 SHAP
                                                 features + values for each of
                                                 the three applicants -- this table
                                                 is also what s11_a_lime_local.py
                                                 and s12_compare_shap_lime.py will
                                                 use, so the same three applicants
                                                 stay consistent across all three
                                                 scripts)

REQUIRES: shap
    pip install shap
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    raise ImportError(
        "This script needs shap. In PyCharm's terminal, run:\n"
        "  pip install shap"
    )

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
for d in (RESULTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
N_SAMPLE = 1000  # same sample size as s10_a, for consistency
TOP_N_REASONS = 5  # how many top SHAP features to log per applicant

# ----------------------------------------------------------------------------
# Load model and data
# ----------------------------------------------------------------------------

model_path = os.path.join(MODELS_DIR, "xgboost.joblib")
model = joblib.load(model_path)
print(f"Loaded model: {model_path}")

test = pd.read_csv(os.path.join(DATA_DIR, "v1_test.csv"))
print(f"Loaded v1_test: {test.shape}")

id_col = "SK_ID_CURR"
target_col = "TARGET"
feature_cols = [c for c in test.columns if c not in (id_col, target_col)]

sample = test.sample(N_SAMPLE, random_state=RANDOM_STATE).reset_index(drop=True)
X_sample = sample[feature_cols]
ids_sample = sample[id_col]

print(f"Sampled {N_SAMPLE} rows (random_state={RANDOM_STATE}) -- same sample as s10_a")

# ----------------------------------------------------------------------------
# Predict probabilities and select 3 representative applicants
# ----------------------------------------------------------------------------

proba = model.predict_proba(X_sample)[:, 1]
sample = sample.copy()
sample["predicted_proba"] = proba

low_idx = proba.argmin()
high_idx = proba.argmax()
borderline_idx = np.abs(proba - 0.5).argmin()

selected = {
    "LOW_RISK": low_idx,
    "HIGH_RISK": high_idx,
    "BORDERLINE": borderline_idx,
}

print("\nSelected applicants:")
for label, idx in selected.items():
    print(f"  {label}: SK_ID_CURR={ids_sample.iloc[idx]}, "
          f"predicted_proba={proba[idx]:.4f}")

# ----------------------------------------------------------------------------
# Compute SHAP values for the sample (needed for waterfall plots)
# ----------------------------------------------------------------------------

print("\nComputing SHAP values with TreeExplainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)
print("SHAP computation complete.")

# ----------------------------------------------------------------------------
# Generate a waterfall plot + log top reasons for each selected applicant
# ----------------------------------------------------------------------------

records = []

for label, idx in selected.items():
    print(f"\nGenerating waterfall plot for {label} (SK_ID_CURR={ids_sample.iloc[idx]})...")

    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"shap_waterfall_{label.lower()}.png")
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Log the top N features driving this specific prediction
    row_shap = shap_values[idx].values
    row_features = pd.DataFrame({
        "feature": feature_cols,
        "shap_value": row_shap,
        "feature_value": X_sample.iloc[idx].values,
    }).reindex(np.abs(row_shap).argsort()[::-1]).head(TOP_N_REASONS)

    for _, r in row_features.iterrows():
        records.append({
            "risk_category": label,
            "SK_ID_CURR": ids_sample.iloc[idx],
            "predicted_proba": proba[idx],
            "feature": r["feature"],
            "feature_value": r["feature_value"],
            "shap_value": r["shap_value"],
            "direction": "increases default risk" if r["shap_value"] > 0 else "decreases default risk",
        })

# ----------------------------------------------------------------------------
# Save the combined table (used by s11_a and s12 for SHAP-vs-LIME comparison)
# ----------------------------------------------------------------------------

examples_df = pd.DataFrame(records)
print(f"\nTop {TOP_N_REASONS} SHAP reasons per applicant:")
print(examples_df.to_string(index=False))

examples_path = os.path.join(RESULTS_DIR, "shap_local_examples.csv")
examples_df.to_csv(examples_path, index=False)
print(f"\nSaved: {examples_path}")

print("\nStep 10b (SHAP local) complete.")
print("Next: s11_a_lime_local.py -- LIME explanations for the SAME three applicants,")
print("using v1_test_scaled.csv (LIME is sensitive to feature scale).")
