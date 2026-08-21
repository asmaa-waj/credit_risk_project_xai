"""
MODEL 3: XGBoost — Section 3.2.4, Model Development.

WHAT THIS SCRIPT DOES:
  Trains an XGBoost classifier on the same preprocessed data as the other
  two models (v1_train.csv / v1_test.csv), evaluated with the same
  metrics for a direct 3-way comparison: accuracy, precision, recall,
  F1-score, ROC-AUC, and PR-AUC.

WHY XGBOOST THIRD:
  Decision Tree and Random Forest both BAG (train trees independently,
  average their votes). XGBoost BOOSTS instead — it trains trees
  sequentially, where each new tree specifically targets the previous
  trees' mistakes. This usually captures more signal than bagging alone,
  but is also more prone to overfitting if left unchecked, since errors
  can compound across boosting rounds -- which is why its hyperparameters
  below are deliberately more conservative than Random Forest's, not
  matched to them.

CLASS IMBALANCE HANDLING:
  XGBoost doesn't have a class_weight='balanced' option -- its equivalent
  is scale_pos_weight, a single number applied to the positive (minority,
  default) class. Per Step 6 of preprocessing, this is computed the same
  way as documented there: negative_count / positive_count, calculated
  fresh from THIS training set rather than hardcoded, so it stays correct
  even if the data changes upstream.

WHY THESE HYPERPARAMETERS (DELIBERATELY DIFFERENT FROM RANDOM FOREST):
  MAX_DEPTH = 6: XGBoost's own documentation recommends shallow trees
    (typically 3-10) specifically because boosting compounds each tree's
    errors into the next one -- a deep first tree's mistakes get
    amplified down the line. This is NOT matched to Random Forest's
    max_depth=15 on purpose; matching them would unfairly handicap
    XGBoost, since deep trees interact very differently with boosting
    than with bagging.
  N_ESTIMATORS = 300: same round count as Random Forest, for a roughly
    comparable training budget -- though because boosting is sequential
    (not parallel across trees like bagging), this will still take longer
    to train than the Random Forest despite using the same core count.
  LEARNING_RATE = 0.1: standard default; each tree's contribution is
    scaled down by this factor so no single round dominates the model.

A NOTE ON WHAT WASN'T DONE:
  No early stopping was used here, even though XGBoost supports it,
  because early stopping requires a validation set to monitor -- and
  using the TEST set for that would mean test data influenced how long
  training ran, which is a leakage path. A proper early-stopping setup
  would carve a separate validation split out of TRAIN; that's a
  reasonable next iteration if training time becomes a concern, but was
  left out here to keep this run's methodology unambiguous.

OUTPUT:
  - outputs/models/xgboost.joblib
  - outputs/results/xgboost_metrics.csv
  - outputs/results/xgboost_feature_importances.csv
  - outputs/figures/xgboost_confusion_matrix.png

REQUIRES: xgboost
    pip install xgboost
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBClassifier
except ImportError:
    raise ImportError(
        "This script needs xgboost. In PyCharm's terminal, run:\n"
        "  pip install xgboost"
    )

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
N_ESTIMATORS = 300
MAX_DEPTH = 6          # intentionally shallower than Random Forest's 15 -- see docstring
LEARNING_RATE = 0.1
N_JOBS = -1

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------

train = pd.read_csv(os.path.join(DATA_DIR, "v1_train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "v1_test.csv"))
print(f"Loaded v1_train: {train.shape}")
print(f"Loaded v1_test:  {test.shape}")

id_col = "SK_ID_CURR"
target_col = "TARGET"

feature_cols = [c for c in train.columns if c not in (id_col, target_col)]

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

print(f"\n{len(feature_cols)} features, {len(X_train):,} train rows, {len(X_test):,} test rows")

# ----------------------------------------------------------------------------
# Compute scale_pos_weight from THIS training set (not hardcoded)
# ----------------------------------------------------------------------------

n_negative = (y_train == 0).sum()
n_positive = (y_train == 1).sum()
scale_pos_weight = n_negative / n_positive

print(f"\nClass counts in train -- TARGET=0: {n_negative:,}, TARGET=1: {n_positive:,}")
print(f"scale_pos_weight = {scale_pos_weight:.4f}")

# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------

print(f"\nTraining XGBoost (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
      f"learning_rate={LEARNING_RATE}, scale_pos_weight={scale_pos_weight:.2f}, "
      f"random_state={RANDOM_STATE})...")

model = XGBClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    learning_rate=LEARNING_RATE,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
eval_metric="aucpr", 
)
model.fit(X_train, y_train)

print("Training complete.")

# ----------------------------------------------------------------------------
# Evaluate on TRAIN (to check the overfitting gap) and TEST (the real metric)
# ----------------------------------------------------------------------------

def evaluate(model, X, y, label):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
    }

    print(f"\n{label} metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return metrics, y_pred, y_proba


train_metrics, _, _ = evaluate(model, X_train, y_train, "TRAIN")
test_metrics, y_test_pred, y_test_proba = evaluate(model, X_test, y_test, "TEST")

overfit_gap = train_metrics["roc_auc"] - test_metrics["roc_auc"]
print(f"\nTrain-test ROC-AUC gap: {overfit_gap:.4f} "
      f"({'still large' if overfit_gap > 0.05 else 'modest'})")
print("Compare this, and the raw test ROC-AUC/PR-AUC, against both the Decision Tree")
print("and Random Forest results to see which model actually generalizes best.")

print("\nFull classification report (TEST):")
print(classification_report(y_test, y_test_pred, target_names=["Repaid (0)", "Default (1)"]))

# ----------------------------------------------------------------------------
# Confusion matrix figure
# ----------------------------------------------------------------------------

cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["Repaid", "Default"]).plot(ax=ax, cmap="Oranges", colorbar=False)
ax.set_title("XGBoost — Confusion Matrix (Test)")
fig.tight_layout()
cm_path = os.path.join(FIGURES_DIR, "xgboost_confusion_matrix.png")
fig.savefig(cm_path, dpi=120)
plt.close(fig)
print(f"\nSaved: {cm_path}")

# ----------------------------------------------------------------------------
# Feature importances
# ----------------------------------------------------------------------------

importances = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nTop 20 most important features:")
print(importances.head(20).to_string(index=False))

# ----------------------------------------------------------------------------
# Save outputs
# ----------------------------------------------------------------------------

model_path = os.path.join(MODELS_DIR, "xgboost.joblib")
joblib.dump(model, model_path)

metrics_df = pd.DataFrame([
    {"set": "train", **train_metrics},
    {"set": "test", **test_metrics},
])
metrics_path = os.path.join(RESULTS_DIR, "xgboost_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

importances_path = os.path.join(RESULTS_DIR, "xgboost_feature_importances.csv")
importances.to_csv(importances_path, index=False)

print(f"\nSaved: {model_path}")
print(f"Saved: {metrics_path}")
print(f"Saved: {importances_path}")

print("\nXGBoost complete.")
print("All three models are now trained. Next: build the 3-model comparison table")
print("(Section 3.2.5 — Comparative Evaluation) from the three *_metrics.csv files,")
print("then select the best performer to carry into SHAP/LIME (Section 3.2.6).")
