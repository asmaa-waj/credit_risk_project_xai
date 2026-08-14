# Preprocessing Handoff — Section 3.2.4 (Model Development)

Preprocessing (Section 3.2.3) is complete. This document is everything you need to start training models without having to read the preprocessing scripts.

## Where the data is

Everything you need is in `data/processed/`, prefixed `v1_`:

| File | Rows | Use it for |
|---|---|---|
| `v1_train.csv` | 246,008 | Training data, **unscaled**. Use this with `class_weight='balanced'` (Decision Tree/Random Forest) or `scale_pos_weight=11.39` (XGBoost). |
| `v1_test.csv` | 61,503 | Evaluation data. **Never train on this.** Real-world ~8.07% default rate — don't resample or reweight it. |
| `v1_train_resampled_smote.csv` | 452,296 | Alternative training data, already balanced 50/50 via SMOTE. Use as-is, no `class_weight` needed — comparing this against `v1_train.csv` + class weighting is a natural sensitivity analysis for your results chapter. |
| `v1_train_scaled.csv` / `v1_test_scaled.csv` | same as above | Same data, but continuous features standardized (mean 0, std 1). Not needed for tree models — kept for later LIME work, which is sensitive to feature scale. |
| `v1_train_resampled_smote_scaled.csv` | 452,296 | SMOTE + scaled combined, same reasoning as above. |
| `v1_data_dictionary.csv` | 206 rows | One row per feature column, explaining what it is and (if engineered/encoded) how it was built. Check this before treating any unfamiliar column name as a mystery. |
| `v1_validation_report.txt` | — | Plain-text log of the Step 8 checklist that confirmed this data is clean — useful to cite in your methodology chapter. |

**Do not use anything in `data/interim/`** for modeling — those are intermediate step-by-step outputs. `data/processed/` is the final, validated handoff point.

## Key columns

- **`SK_ID_CURR`** — applicant ID. Not a feature — exclude it from `X` before training. Note: in the SMOTE file, synthetic rows have negative IDs (`-1`, `-2`, ...) so you can always tell real applicants from synthesized ones if needed.
- **`TARGET`** — the label. `1` = defaulted, `0` = repaid.
- **206 feature columns** — everything else. All numeric already (categoricals were encoded in Step 4), no missing values, outliers capped.

## Recommended starting approach

1. Start with `v1_train.csv` + `class_weight='balanced'` for your Decision Tree baseline — it's the simplest, most defensible first choice per the guide, and doesn't require justifying synthetic data in your methodology.
2. Once you have a baseline, compare against `v1_train_resampled_smote.csv` to see if SMOTE actually helps — it might not, and that's a legitimate, useful result to report either way.
3. Evaluate everything on `v1_test.csv` (or its scaled counterpart if the model needs it) — never touch it during training or tuning.

## Known signal

`EXT_SOURCE_2` is the single most predictive feature (0.16 correlation with `TARGET` — the highest of any column, well short of anything that would suggest leakage). `EXT_SOURCE_1` and `EXT_SOURCE_3` are typically strong too in this dataset. If your model's feature importances don't show the `EXT_SOURCE_*` columns near the top, that's worth double-checking rather than assuming it's fine.

## Reproducibility

Every random operation in preprocessing (train/test split, SMOTE) used `RANDOM_STATE = 42`. If you need to regenerate anything from scratch (e.g. a different feature engineering choice), keep using 42 so results stay comparable across the team.

## Questions about a specific column or preprocessing decision

Check `v1_data_dictionary.csv` first — it covers all 206 engineered/encoded/imputed columns with the exact method used. For anything not in there (an untouched original Home Credit field), the [Home Credit column descriptions](https://www.kaggle.com/c/home-credit-default-risk/data) (also in your `raw/HomeCredit_columns_description.csv`) has the original definitions.
