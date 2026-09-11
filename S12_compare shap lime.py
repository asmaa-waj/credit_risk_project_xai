"""
STEP 12: SHAP vs LIME Comparison — Section 3.2.6, Explainability Integration.

WHAT THIS SCRIPT DOES:
  Loads the SHAP top-5 reasons (s10_b_shap_local.py) and LIME top-5
  reasons (s11_a_lime_local.py) for the same three applicants (LOW_RISK,
  HIGH_RISK, BORDERLINE), and puts them side by side per applicant so you
  can see where the two explanation methods agree and where they diverge.
  This comparison is itself a result worth reporting in the thesis --
  agreement builds confidence in the explanations; disagreement is a
  legitimate finding about the limits of local surrogate models like
  LIME versus exact game-theoretic attributions like SHAP.

WHY THIS COMPARISON IS NOT A DIRECT NUMBER-FOR-NUMBER MATCH:
  SHAP values and LIME weights are NOT on the same scale and are not
  expected to be -- SHAP values are exact Shapley contributions to the
  model's raw output, while LIME weights come from a local linear
  regression fit on perturbed samples, scaled by that regression's own
  units. Comparing them numerically would be misleading. What IS
  meaningful to compare is:
    (a) whether the SAME features show up as top drivers in both, and
    (b) whether they agree on DIRECTION (increases vs decreases risk)
        for those shared features.
  This script reports exactly those two things, not a numeric delta.

OUTPUT:
  - outputs/results/shap_lime_comparison.csv   (per applicant: which
                                                  features appear in
                                                  BOTH top-5 lists, which
                                                  are SHAP-only, which are
                                                  LIME-only, and whether
                                                  shared features agree
                                                  on direction)
  - Printed summary per applicant, readable directly for the thesis
    results section.
"""

import pandas as pd
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

# ----------------------------------------------------------------------------
# Load both explanation tables
# ----------------------------------------------------------------------------

shap_path = os.path.join(RESULTS_DIR, "shap_local_examples.csv")
lime_path = os.path.join(RESULTS_DIR, "lime_local_examples.csv")

shap_df = pd.read_csv(shap_path)
lime_df = pd.read_csv(lime_path)
print(f"Loaded: {shap_path} ({len(shap_df)} rows)")
print(f"Loaded: {lime_path} ({len(lime_df)} rows)")

# LIME's feature descriptions are conditions like "EXT_SOURCE_2 <= -0.43" --
# extract the base feature name so it can be matched against SHAP's plain
# feature column names.
FEATURE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_]+")

def extract_feature_name(condition: str) -> str:
    match = FEATURE_NAME_PATTERN.match(condition.strip())
    return match.group(0) if match else condition

lime_df = lime_df.copy()
lime_df["feature"] = lime_df["feature_condition"].apply(extract_feature_name)

# ----------------------------------------------------------------------------
# Compare per applicant
# ----------------------------------------------------------------------------

comparison_records = []
risk_categories = shap_df["risk_category"].unique()

for category in risk_categories:
    shap_sub = shap_df[shap_df["risk_category"] == category]
    lime_sub = lime_df[lime_df["risk_category"] == category]

    if lime_sub.empty:
        print(f"\n[WARNING] No LIME results found for {category} -- skipping comparison.")
        continue

    sk_id = shap_sub["SK_ID_CURR"].iloc[0]
    proba = shap_sub["predicted_proba"].iloc[0]

    shap_features = dict(zip(shap_sub["feature"], shap_sub["direction"]))
    lime_features = dict(zip(lime_sub["feature"], lime_sub["direction"]))

    shared = set(shap_features) & set(lime_features)
    shap_only = set(shap_features) - set(lime_features)
    lime_only = set(lime_features) - set(shap_features)

    print(f"\n{'=' * 70}")
    print(f"{category}  (SK_ID_CURR={sk_id}, predicted_proba={proba:.4f})")
    print(f"{'=' * 70}")
    print(f"Shared top features (SHAP AND LIME): {sorted(shared) if shared else 'none'}")
    print(f"SHAP-only top features:              {sorted(shap_only) if shap_only else 'none'}")
    print(f"LIME-only top features:               {sorted(lime_only) if lime_only else 'none'}")

    for feature in shared:
        shap_dir = shap_features[feature]
        lime_dir = lime_features[feature]
        agree = shap_dir == lime_dir
        print(f"  - {feature}: SHAP says '{shap_dir}', LIME says '{lime_dir}' "
              f"-> {'AGREE' if agree else 'DISAGREE'}")

        comparison_records.append({
            "risk_category": category,
            "SK_ID_CURR": sk_id,
            "predicted_proba": proba,
            "feature": feature,
            "in_shap_top5": True,
            "in_lime_top5": True,
            "shap_direction": shap_dir,
            "lime_direction": lime_dir,
            "agree_on_direction": agree,
        })

    for feature in shap_only:
        comparison_records.append({
            "risk_category": category,
            "SK_ID_CURR": sk_id,
            "predicted_proba": proba,
            "feature": feature,
            "in_shap_top5": True,
            "in_lime_top5": False,
            "shap_direction": shap_features[feature],
            "lime_direction": None,
            "agree_on_direction": None,
        })

    for feature in lime_only:
        comparison_records.append({
            "risk_category": category,
            "SK_ID_CURR": sk_id,
            "predicted_proba": proba,
            "feature": feature,
            "in_shap_top5": False,
            "in_lime_top5": True,
            "shap_direction": None,
            "lime_direction": lime_features[feature],
            "agree_on_direction": None,
        })

# ----------------------------------------------------------------------------
# Save combined comparison table
# ----------------------------------------------------------------------------

comparison_df = pd.DataFrame(comparison_records)
comparison_path = os.path.join(RESULTS_DIR, "shap_lime_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)
print(f"\n\nSaved: {comparison_path}")

# ----------------------------------------------------------------------------
# Overall summary across all three applicants -- useful one-paragraph stat
# for the thesis results section
# ----------------------------------------------------------------------------

shared_rows = comparison_df[comparison_df["in_shap_top5"] & comparison_df["in_lime_top5"]]
n_shared = len(shared_rows)
n_agree = shared_rows["agree_on_direction"].sum() if n_shared else 0

print(f"\n{'=' * 70}")
print("OVERALL SUMMARY")
print(f"{'=' * 70}")
print(f"Across all {len(risk_categories)} applicants: {n_shared} feature(s) appeared in "
      f"both SHAP and LIME top-{5} lists.")
if n_shared:
    print(f"Of those, {n_agree}/{n_shared} agreed on direction "
          f"({n_agree / n_shared:.0%}).")
print("\nStep 12 (SHAP vs LIME comparison) complete.")
print("This table and summary are ready to cite directly in the thesis")
print("results section (Section 3.2.6 -- Explainability Integration).")