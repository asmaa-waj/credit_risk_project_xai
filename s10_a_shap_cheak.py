import pandas as pd
import joblib
import shap
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #project root
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "xgboost.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "v1_test.csv")

# Load model and data
model = joblib.load(MODEL_PATH)
test = pd.read_csv(DATA_PATH)

id_col = "SK_ID_CURR"
target_col = "TARGET"
feature_cols = [c for c in test.columns if c not in (id_col, target_col)]

X_test = test[feature_cols]

# For speed while testing, use a sample first (SHAP on 61k rows can be slow)
X_sample = X_test.sample(500, random_state=42)

# Tree SHAP — fast, exact for tree-based models like XGBoost
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# Sanity check: EXT_SOURCE_2/1/3 should be near the top
print("Generating global summary plot...")
shap.summary_plot(shap_values, X_sample, show=False)

import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "outputs", "figures", "shap_summary.png"), dpi=120)
plt.close()
print("Saved: outputs/figures/shap_summary.png")
