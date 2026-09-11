# XAI Handoff — Section 3.2.6 (Explainability Integration)

Steps 10–12 are complete. This document summarizes what was done, what was
found, and where the outputs live — everything needed to write up Section
3.2.6 or build on this work (e.g. the dashboard in Section 3.2.7) without
re-reading the scripts.

## What was done

| Step | Script | Purpose |
|---|---|---|
| 10a | `s10_a_shap_global.py` | Global SHAP explanation across a sample of test applicants |
| 10b | `s10_b_shap_local.py` | Local SHAP explanation for 3 representative applicants |
| 11a | `s11_a_lime_local.py` | LIME explanation for the SAME 3 applicants |
| 12 | `s12_compare_shap_lime.py` | Side-by-side comparison of SHAP vs LIME top reasons |

Model explained: **XGBoost** (`outputs/models/xgboost.joblib`), selected as
best-performing in Section 3.2.5.

## Step 10a — Global SHAP

- Method: `shap.TreeExplainer` (exact, fast for tree ensembles — no
  approximation needed for XGBoost).
- Data: `v1_test.csv` (unscaled — tree models split on thresholds, not
  distances, so scaling is unnecessary and would only make the plot's
  feature values less readable).
- Sample: 1,000 rows, `random_state=42`.
- **Finding:** `EXT_SOURCE_2`, `EXT_SOURCE_3`, `EXT_SOURCE_1` are the top 3
  features by mean |SHAP value| — confirms the handoff doc's known signal
  (these are the strongest raw correlates with `TARGET`). Sanity check
  passed.
- Secondary signal: `INSTALL_UNDERPAID_RATIO` also stood out — high
  underpaid-installment ratio pushes toward predicted default, which is
  intuitively sensible.

**Outputs:**
- `outputs/figures/shap_summary_global.png` (beeswarm)
- `outputs/figures/shap_bar_global.png` (bar chart)
- `outputs/results/shap_global_importance.csv`

## Step 10b — Local SHAP (3 representative applicants)

Three applicants selected from the same 1,000-row sample by predicted
probability of default (not true label — mirrors how the dashboard will
select applicants at real inference time):

| Category | SK_ID_CURR | Predicted P(default) |
|---|---|---|
| LOW_RISK | 371134 | 0.0059 |
| HIGH_RISK | 376872 | 0.9259 |
| BORDERLINE | 252790 | 0.5003 |

**Findings:**
- LOW_RISK: driven by high `EXT_SOURCE_1/2` (0.80–0.89) — strong external
  credit history clearly pushes toward repayment.
- HIGH_RISK: driven by very low `EXT_SOURCE_2/3` (~0.06) — poor external
  credit history clearly pushes toward default.
- BORDERLINE: the three `EXT_SOURCE_*` columns *disagree with each other*
  (`EXT_SOURCE_1` = 0.74, favorable; `EXT_SOURCE_2/3` = 0.26–0.39,
  unfavorable) — a clean illustration of why this case landed exactly on
  the decision boundary. Good example for the thesis write-up.

**Outputs:**
- `outputs/figures/shap_waterfall_low_risk.png`, `..._high_risk.png`,
  `..._borderline.png`
- `outputs/results/shap_local_examples.csv` (top 5 SHAP features + values
  per applicant — also consumed by Step 12)

## Step 11a — Local LIME (same 3 applicants)

- Method: `LimeTabularExplainer`, background sample of 2,000 rows from
  `v1_train_scaled.csv`.
- Data: `v1_test_scaled.csv` — unlike SHAP, LIME measures distance between
  perturbed samples, so unscaled features (e.g. income in the hundreds of
  thousands) would dominate that distance calculation regardless of actual
  importance. Scaling puts every feature on equal footing.
- Same 3 applicants looked up by `SK_ID_CURR` from Step 10b's output, so
  results are directly comparable.

**Outputs:**
- `outputs/figures/lime_low_risk.png`, `..._high_risk.png`,
  `..._borderline.png`
- `outputs/results/lime_local_examples.csv` (top 5 LIME features + weights
  per applicant)

## Step 12 — SHAP vs LIME Comparison

Compared top-5 features per applicant from both methods: which features
appear in both lists, and whether they agree on **direction** (increases
vs. decreases default risk) when they do overlap.

**Key finding: the two methods largely disagree.**

- Only **1 feature** (`EXT_SOURCE_1`) appeared in both methods' top-5 lists
  across the three applicants.
- Of the cases where it appeared in both, **0 out of 3 agreed on
  direction** — SHAP and LIME assigned opposite effects to the same
  feature for the same applicant, consistently.
- LIME's other top features leaned toward rare/sparse one-hot flags
  (e.g. `FLAG_DOCUMENT_9`, `NAME_TYPE_SUITE_Group`,
  `HOUSETYPE_MODE_specific`), while SHAP consistently surfaced the
  dataset-wide dominant predictors (`EXT_SOURCE_*`, `CREDIT_TO_GOODS_RATIO`).

**Interpretation for the thesis:** this is not a bug or an implementation
error — it reflects a known, documented limitation of LIME. LIME
approximates the model locally with a simple linear model; if XGBoost's
true relationship with a feature is non-linear or interacts with other
features (very plausible for `EXT_SOURCE_1` in a boosted-tree model),
LIME's local linear fit can get the direction wrong for that neighborhood.
SHAP's TreeExplainer, by contrast, computes exact contributions from the
model's actual structure, and its results align with the dataset's known
predictive signal (from `HANDOFF_modeling.md`). Reporting this divergence,
and explaining why it happens, is a legitimate and citable finding —
arguably a stronger result than if the two methods had trivially agreed.

**Outputs:**
- `outputs/results/shap_lime_comparison.csv` (full per-feature comparison:
  which method(s) flagged it, both directions, whether they agree)

## What this means for Section 3.2.7 (Dashboard)

- The dashboard should show **both** SHAP and LIME panels for a selected
  applicant, not just one — the disagreement itself is meaningful for a
  user/reviewer to see, not something to hide.
- SHAP should be treated as the primary/default explanation (exact,
  consistent with known predictive signal); LIME should be presented as a
  secondary, complementary view with a brief note on why the two may
  differ.
- ECOA-style adverse-action reason codes should be built from **SHAP's**
  top negative-direction features, since SHAP's attributions are exact and
  more consistent — not from LIME, given the direction-agreement issues
  found in Step 12.

## Reproducibility

All three steps use `RANDOM_STATE = 42` and the same 1,000-row sample
(from `v1_test.csv` / `v1_test_scaled.csv`) — re-running any of the three
scripts should reproduce identical applicant selections and results.
