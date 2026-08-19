"""
MODEL 1: Decision Tree (baseline) Model Development.

WHAT THIS SCRIPT DOES:
  Trains a Decision Tree classifier on the preprocessed Home Credit data
  (v1_train.csv from Step 8 of preprocessing) and evaluates it on the
  held-out test set (v1_test.csv), reporting the metrics your guide's
  Section 3.2.5 (Comparative Evaluation) asks for: accuracy, precision,
  recall, F1-score, and ROC-AUC.

WHY DECISION TREE FIRST:
  It's the simplest of the three models (Decision Tree, Random Forest,
  XGBoost) and serves as the baseline everything else gets compared
  against later. No scaling needed (unscaled v1_train.csv/v1_test.csv are
  used) since trees split on thresholds, not distances.

CLASS IMBALANCE HANDLING:
  Uses class_weight='balanced' rather than the SMOTE-resampled training
  set — per Step 6 of preprocessing, this is the simplest, most defensible
  first choice, and doesn't require justifying synthetic data in your
  methodology. class_weight='balanced' automatically weights each class
  inversely proportional to its frequency (no manual weight calculation
  needed).

WHY NO max_depth LIMIT (BY DEFAULT):
  Left unconstrained on purpose, since this is meant as a naive baseline
  for comparison — Random Forest and XGBoost (with proper regularization)
  are expected to outperform it. An unconstrained tree WILL overfit
  (very high train accuracy, noticeably lower test accuracy) — this script
  prints both so that gap is visible and documented, not hidden. If you
  want a fairer standalone baseline, try MAX_DEPTH = 10 or 15 below.

OUTPUT:
  - outputs/models/decision_tree_baseline.joblib   (the trained model)
  - outputs/results/decision_tree_baseline_metrics.csv
  - outputs/results/decision_tree_baseline_feature_importances.csv
  - outputs/figures/decision_tree_confusion_matrix.png
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
MAX_DEPTH = 15

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

print(f"\nTraining Decision Tree (class_weight='balanced', max_depth={MAX_DEPTH}, random_state={RANDOM_STATE})...")

model = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=MAX_DEPTH,
    random_state=RANDOM_STATE,
)
model.fit(X_train, y_train)

print("Training complete.")
print(f"Tree depth reached: {model.get_depth()}, leaf count: {model.get_n_leaves()}")

# ----------------------------------------------------------------------------
# Evaluate on TRAIN (to show the overfitting gap) and TEST (the real metric)
# ----------------------------------------------------------------------------

def evaluate(model, X, y, label):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
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
      f"({'large — expected for an unconstrained baseline tree' if overfit_gap > 0.05 else 'modest'})")

print("\nFull classification report (TEST):")
print(classification_report(y_test, y_test_pred, target_names=["Repaid (0)", "Default (1)"]))

# ----------------------------------------------------------------------------
# Confusion matrix figure
# ----------------------------------------------------------------------------

cm = confusion_matrix(y_test, y_test_pred)
fig, ax = plt.subplots(figsize=(5, 5))
ConfusionMatrixDisplay(cm, display_labels=["Repaid", "Default"]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Decision Tree — Confusion Matrix (Test)")
fig.tight_layout()
cm_path = os.path.join(FIGURES_DIR, "decision_tree_confusion_matrix.png")
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

model_path = os.path.join(MODELS_DIR, "decision_tree_baseline.joblib")
joblib.dump(model, model_path)

metrics_df = pd.DataFrame([
    {"set": "train", **train_metrics},
    {"set": "test", **test_metrics},
])
metrics_path = os.path.join(RESULTS_DIR, "decision_tree_baseline_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

importances_path = os.path.join(RESULTS_DIR, "decision_tree_baseline_feature_importances.csv")
importances.to_csv(importances_path, index=False)

print(f"\nSaved: {model_path}")
print(f"Saved: {metrics_path}")
print(f"Saved: {importances_path}")

print("\nDecision Tree baseline complete.")
print("Next: train Random Forest and XGBoost the same way, then compare all three")
print("in a single table (Section 3.2.5 — Comparative Evaluation).")
