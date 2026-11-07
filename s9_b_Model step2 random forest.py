"""
MODEL 2: Random Forest — Section 3.2.4, Model Development.

WHAT THIS SCRIPT DOES:
  Trains a Random Forest classifier on the same preprocessed data as the
  Decision Tree baseline (v1_train.csv / v1_test.csv), evaluated with the
  same metrics for a direct comparison: accuracy, precision, recall,
  F1-score, ROC-AUC, and PR-AUC (precision-recall AUC — included because
  ROC-AUC alone can look inflated on an ~8% imbalanced positive class;
  PR-AUC doesn't have that problem and is arguably the more honest number
  here).

WHY RANDOM FOREST SECOND:
  It's an ensemble of many trees (each trained on a bootstrapped sample
  with a random feature subset), averaged together. This is specifically
  what should narrow the train/test overfitting gap seen in the Decision
  Tree baseline — a single deep tree memorizes training rows, but
  averaging hundreds of them cancels out most of that memorization. If
  this run's train/test gap isn't meaningfully smaller than the Decision
  Tree's, that itself is a real finding worth reporting.

CLASS IMBALANCE HANDLING:
  Same choice as the Decision Tree: class_weight='balanced' rather than
  the SMOTE-resampled training set, for the same reasons (Step 6 of
  preprocessing) — simplest, most defensible first choice, no synthetic
  data to justify in the methodology. Keeping this identical between
  models is what makes the comparison table in Section 3.2.5 fair — only
  the model itself changes, not the imbalance-handling strategy.

WHY THESE HYPERPARAMETERS:
  N_ESTIMATORS = 300: enough trees for the ensemble average to stabilize
    without excessive runtime on 246k rows x 206 features.
  MAX_DEPTH = 15: matches whatever depth the Decision Tree baseline
    ultimately used, so any performance difference is attributable to
    "one tree vs many trees," not "shallow tree vs deep tree." Change
    this if the Decision Tree script's MAX_DEPTH changes.
  N_JOBS = -1: uses all available CPU cores — Random Forest trains each
    tree independently, so this parallelizes cleanly and meaningfully
    speeds up training on a dataset this size.

OUTPUT:
  - outputs/models/random_forest.joblib
  - outputs/results/random_forest_metrics.csv
  - outputs/results/random_forest_feature_importances.csv
  - outputs/figures/random_forest_confusion_matrix.png
  """

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
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
N_ESTIMATORS = 200
MAX_DEPTH = 15  # keep this matched to the Decision Tree script's MAX_DEPTH for a fair comparison
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
# Train
# ----------------------------------------------------------------------------

print(f"\nTraining Random Forest (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, "
      f"class_weight='balanced', random_state={RANDOM_STATE})...")
print("This will take longer than the Decision Tree — training 300 trees, not 1.")

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
)
model.fit(X_train, y_train)

print("Training complete.")
avg_depth = np.mean([est.get_depth() for est in model.estimators_])
avg_leaves = np.mean([est.get_n_leaves() for est in model.estimators_])
print(f"Average depth across {N_ESTIMATORS} trees: {avg_depth:.1f}, average leaf count: {avg_leaves:.0f}")

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
      f"({'still large' if overfit_gap > 0.05 else 'modest — the ensemble is generalizing better than the single tree'})")
print("Compare this gap against the Decision Tree baseline's to see whether")
print("averaging across trees actually reduced overfitting as expected.")

print("\nFull classification report (TEST):")
print(classification_report(y_test, y_test_pred, target_names=["Repaid (0)", "Default (1)"]))

# ----------------------------------------------------------------------------
# Confusion matrix figure
# ----------------------------------------------------------------------------

cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["Repaid", "Default"]).plot(ax=ax, cmap="Greens", colorbar=False)
ax.set_title("Random Forest — Confusion Matrix (Test)")
fig.tight_layout()
cm_path = os.path.join(FIGURES_DIR, "random_forest_confusion_matrix.png")
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

model_path = os.path.join(MODELS_DIR, "random_forest.joblib")
joblib.dump(model, model_path)

metrics_df = pd.DataFrame([
    {"set": "train", **train_metrics},
    {"set": "test", **test_metrics},
])
metrics_path = os.path.join(RESULTS_DIR, "random_forest_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

importances_path = os.path.join(RESULTS_DIR, "random_forest_feature_importances.csv")
importances.to_csv(importances_path, index=False)

print(f"\nSaved: {model_path}")
print(f"Saved: {metrics_path}")
print(f"Saved: {importances_path}")

print("\nRandom Forest complete.")
print("Next: train XGBoost the same way, then build the 3-model comparison table")
print("(Section 3.2.5 — Comparative Evaluation) from the three *_metrics.csv files.")
