"""
STEP 10b: SHAP Global Explanation — Section 3.2.6, Explainability Integration.

WHAT THIS SCRIPT DOES:
  Loads the best-performing model (XGBoost, selected in Section 3.2.5 —
  Comparative Evaluation) and computes SHAP (SHapley Additive exPlanations)
  values on a sample of the held-out test set (v1_test.csv), then produces
  a global summary plot showing which features matter most across all
  predictions, and in which direction.

WHY SHAP FOR THIS MODEL:
  XGBoost is a tree ensemble, so TreeExplainer applies directly — it
  computes EXACT Shapley values (not an approximation) and is fast even
  on hundreds of features, unlike KernelSHAP which is model-agnostic but
  much slower and only approximate. There's no reason to use anything
  other than TreeExplainer here.

WHY A SAMPLE, NOT THE FULL TEST SET:
  v1_test.csv has 61,503 rows. TreeExplainer is fast, but computing and
  plotting SHAP values for all of them adds runtime and clutters the
  summary plot without adding insight — a random sample of N_SAMPLE rows
  (fixed with RANDOM_STATE=42 for reproducibility) is standard practice
  for a global summary plot and is what gets reported in the thesis.

WHY UNSCALED DATA (v1_test.csv, NOT v1_test_scaled.csv):
  XGBoost splits on raw thresholds, not distances, so scaling doesn't
  change its predictions or SHAP values, and using unscaled data keeps
  the SHAP plot's x-axis and feature values in their original, readable
  units (e.g. actual income amounts, not standardized scores). Contrast
  this with Step 11 (LIME), which DOES need v1_test_scaled.csv.

SANITY CHECK BUILT INTO THIS SCRIPT:
  Per HANDOFF_modeling.md, EXT_SOURCE_2/1/3 are the strongest predictors
  in this dataset (highest raw correlation with TARGET). This script
  checks whether they appear in the top 5 features by mean |SHAP value|
  and prints a clear warning if they don't — a real result either way,
  but one that should be double-checked before trusting the rest of the
  output.

OUTPUT:
  - outputs/figures/shap_summary_global.png       (beeswarm summary plot)
  - outputs/figures/shap_bar_global.png            (mean |SHAP value| bar chart)
  - outputs/results/shap_global_importance.csv     (ranked feature importance table)

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
N_SAMPLE = 1000  # rows sampled from v1_test.csv for the summary plot

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

X_test = test[feature_cols]

X_sample = X_test.sample(N_SAMPLE, random_state=RANDOM_STATE)
print(f"Sampled {N_SAMPLE} rows for SHAP global summary (random_state={RANDOM_STATE})")

# ----------------------------------------------------------------------------
# Compute SHAP values (TreeExplainer -- exact, fast for tree ensembles)
# ----------------------------------------------------------------------------

print("\nComputing SHAP values with TreeExplainer...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)
print("SHAP computation complete.")

# ----------------------------------------------------------------------------
# Global importance table (mean |SHAP value| per feature)
# ----------------------------------------------------------------------------

mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap": mean_abs_shap,
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 20 features by mean |SHAP value|:")
print(importance_df.head(20).to_string(index=False))

importance_path = os.path.join(RESULTS_DIR, "shap_global_importance.csv")
importance_df.to_csv(importance_path, index=False)
print(f"\nSaved: {importance_path}")

# ----------------------------------------------------------------------------
# Sanity check: EXT_SOURCE_2/1/3 should be prominent (per HANDOFF_modeling.md)
# ----------------------------------------------------------------------------

top5_features = set(importance_df.head(5)["feature"])
ext_source_cols = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}
found = ext_source_cols & top5_features

if found:
    print(f"\n[SANITY CHECK PASSED] EXT_SOURCE columns in top 5: {sorted(found)}")
else:
    print("\n[SANITY CHECK WARNING] None of EXT_SOURCE_1/2/3 appear in the top 5 "
          "features by SHAP importance. Per HANDOFF_modeling.md, these are expected "
          "to be strong predictors -- double-check the model and data before trusting "
          "the rest of this output.")

# ----------------------------------------------------------------------------
# Global summary plot (beeswarm)
# ----------------------------------------------------------------------------

print("\nGenerating SHAP beeswarm summary plot...")
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
summary_path = os.path.join(FIGURES_DIR, "shap_summary_global.png")
plt.savefig(summary_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {summary_path}")

# ----------------------------------------------------------------------------
# Bar plot (mean |SHAP value|) -- often clearer for a thesis figure than the beeswarm
# ----------------------------------------------------------------------------

print("Generating SHAP mean |value| bar plot...")
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
plt.tight_layout()
bar_path = os.path.join(FIGURES_DIR, "shap_bar_global.png")
plt.savefig(bar_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"Saved: {bar_path}")

print("\nStep 10a (SHAP global) complete.")
print("Next: s10_b_shap_local.py -- individual applicant waterfall plots.")
