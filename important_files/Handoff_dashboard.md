# Dashboard Handoff — Section 3.2.7 (Explainable AI Platform Dashboard)

This document summarizes what was built in `dashboard/app.py`, why each
piece was designed the way it was, and what's still open — written so
the team (or future-you) doesn't need to re-read the code to understand
what the dashboard does.

## What it is

A Streamlit web app that lets a user select an applicant from the test
set and see: the model's prediction, plain-English reasons for that
prediction, and a detailed SHAP vs LIME breakdown. Run with:

```
streamlit run dashboard/app.py
```

Opens automatically at `http://localhost:8501`. `Ctrl+C` in the terminal
stops it. Editing the file while it's running shows a "Rerun" button in
the browser instead of needing a restart.

## What it does, screen by screen

1. **Applicant selector** — dropdown of `SK_ID_CURR` values from
   `v1_test.csv`.
2. **Predict button** — runs the trained XGBoost model
   (`outputs/models/xgboost.joblib`) on the selected applicant and shows:
   - Predicted probability of default
   - A "High risk — likely to default" / "Low risk — likely to repay"
     label at the 50% threshold
3. **Reason codes** (plain-English, directional) — see below.
4. **SHAP vs LIME, side by side** — a live waterfall chart (SHAP) and a
   live local explanation chart (LIME) for the same applicant, in two
   columns, with a caption noting the two methods don't always agree
   (per the Step 12 finding) and that this is expected, not an error.
5. **Raw applicant data** — collapsed by default, expandable for anyone
   who wants to inspect the underlying row.

## Reason codes: design decisions

This was built and revised in several passes — worth documenting why it
ended up the way it did:

**Built from SHAP, not LIME.** Per `s12_compare_shap_lime.py`, SHAP and
LIME frequently disagreed on both which features mattered and which
direction they pushed a prediction. SHAP's TreeExplainer computes exact
contributions and consistently matched the dataset's known predictive
signal (`EXT_SOURCE_*`), making it the more defensible basis for a
reason a real applicant would see.

**Shown for every applicant, not just declines.** Real ECOA
adverse-action rules only require lenders to explain rejections, so the
first version only showed reasons when predicted risk was ≥50%. This
looked broken in a demo when a reviewer picked an approved applicant
first, so it was changed to show something either way, framed
differently:
  - High risk (≥50%): **"Main reasons for this decision"** — top SHAP
    features pushing risk UP.
  - Low risk (<50%): **"Main factors supporting this decision"** — top
    SHAP features pushing risk DOWN.
  - A caption under each explains the ECOA distinction on-screen, so the
    design choice is visible to a viewer, not just documented in code.

**Directional wording.** Earlier versions used one neutral phrase per
feature (e.g. "External credit bureau score"), which read identically
whether the feature helped or hurt the applicant. `FEATURE_DESCRIPTIONS`
now maps each feature to a pair of phrases:
```python
"EXT_SOURCE_2": {"increase": "low external credit bureau score (source 2)",
                  "decrease": "high external credit bureau score (source 2)"}
```
`describe_feature(name, direction)` picks the correct phrase depending
on whether that feature pushed risk up or down for this applicant.

**Coverage and fallback.** `FEATURE_DESCRIPTIONS` explicitly covers
~90 of the dataset's 206 features (prioritized using
`shap_global_importance.csv` from Step 10a, so the highest-impact
features are covered first). All 21 `FLAG_DOCUMENT_*` columns share one
generic rule. Anything else falls back to a generated phrase — "a
higher/lower-than-typical value for `<column name>`" — so the dashboard
never crashes or shows a blank reason, even on an unmapped feature.
Coverage can be extended by adding entries to `FEATURE_DESCRIPTIONS`;
check `shap_global_importance.csv` rows beyond what's already covered to
prioritize which ones matter most.

## Technical notes

- **Unscaled (`v1_test.csv`) for prediction and SHAP**, **scaled
  (`v1_train_scaled.csv` / `v1_test_scaled.csv`) for LIME** — same
  reasoning as Steps 10/11: XGBoost splits on thresholds so scaling is
  irrelevant for it, but LIME measures distance between perturbed
  samples and needs every feature on the same scale.
- **Caching**: the model, both explainers, and the datasets are all
  loaded/built once via `@st.cache_resource` / `@st.cache_data`, not on
  every click — without this the app would reload everything each time
  a button is pressed.
- **LIME is noticeably slower than SHAP per click** (it perturbs and
  fits a local model live) — a spinner shows while it computes.

## Known limitations / possible next steps

- `FEATURE_DESCRIPTIONS` doesn't cover all 206 features explicitly
  (mainly untouched: most `_AVG`/`_MODE`/`_MEDI` housing-quality
  variants, `WEEKDAY_APPR_PROCESS_START_*`, most `WALLSMATERIAL_MODE_*`
  / `FONDKAPREMONT_MODE_*` one-hot columns) — the generic fallback
  handles these, but with less natural phrasing.
- Applicant selection is currently limited to existing test-set rows by
  ID; there's no manual "enter your own values" form yet, which would
  make for a stronger live demo if time allows.
- No packaging/deployment step yet (e.g. Streamlit Community Cloud) —
  currently runs locally only.

## Where this fits in the thesis

This is the practical deliverable for Section 3.2.7. Combined with
`HANDOFF_xai.md` (Steps 10–12: SHAP global/local, LIME, and the SHAP vs
LIME comparison), this covers the full technical basis for the
Explainability Integration and Dashboard chapters — methodology,
findings, and a working implementation, ready for the reviewer
walkthrough and thesis write-up.
