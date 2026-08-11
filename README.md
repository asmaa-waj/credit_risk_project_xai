# Explainable AI Platform for Credit Risk Assessment

BSc thesis project — Department of Information Technology, Faculty of Mathematical Sciences and Informatics.

Three credit scoring models (Decision Tree baseline, Random Forest, XGBoost) trained on the Home Credit dataset and compared on predictive performance, with SHAP and LIME explanations delivered through an interactive dashboard for non-technical stakeholders (loan officers, compliance officers, applicants).

## Team

| Name | Role |
|---|---|
| Areej Almusalami Abdallah Bllal | Data & Preprocessing |
| Asmaa Wajdi Ata-elmanan Ali | Model & Evaluation |
| Moram Hussein Ibrahim Salih | XAI & Dashboard |

Supervised by Dr. Mohammed Abdallah.

## Implementation Pipeline

1. Dataset Selection — Home Credit dataset (9 files) from Kaggle
2. Exploratory Data Analysis — target/predictor identification, class balance, missing values, correlations
3. Data Cleaning & Feature Engineering — imputation, outlier treatment, encoding, class imbalance handling, derived features (e.g. debt-to-income ratio)
4. Model Development & Comparison — Decision Tree (baseline), Random Forest, XGBoost
5. Comparative Evaluation — side-by-side metrics table across all three models
6. Explainability Integration — SHAP + LIME applied to the best-performing model
7. Dashboard Development — real-time predictions with SHAP/LIME explanations

## Project Structure

```
credit-risk-xai/
├── data/
│   ├── raw/          # Home Credit CSVs (gitignored — download from Kaggle, see below)
│   └── processed/    # cleaned/joined data (gitignored)
├── notebooks/         # EDA and experimentation notebooks
├── src/
│   ├── preprocessing.py   # data joining, cleaning, missing values
│   ├── features.py        # feature engineering
│   ├── model.py            # Decision Tree, Random Forest, XGBoost + comparison
│   └── explain.py          # SHAP + LIME explanations
├── dashboard/
│   └── app.py          # Streamlit dashboard
├── outputs/             # trained models, figures (gitignored)
└── requirements.txt
```

## Setup

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd credit-risk-xai
   ```

2. **Create a virtual environment** (Python 3.12+)
   - In PyCharm: `Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Virtualenv`
   - Or via terminal:
     ```bash
     python -m venv .venv
     .venv\Scripts\Activate.ps1      # Windows
     source .venv/bin/activate       # Mac/Linux
     ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Get the [Home Credit Default Risk dataset](https://www.kaggle.com/c/home-credit-default-risk/data) from Kaggle (requires a free Kaggle account)
   - Unzip it so the CSVs sit directly in `data/raw/` (e.g. `data/raw/application_train.csv`, not nested in a subfolder)

5. **Verify it works**
   ```python
   import pandas as pd
   df = pd.read_csv("data/raw/application_train.csv")
   print(df.shape)  # should print (307511, 122)
   ```

## Team Workflow

Each person works on their own branch and opens a pull request into `main`:

```bash
git checkout -b feature/preprocessing   # or feature/modeling, feature/dashboard
git add .
git commit -m "describe your change"
git push -u origin feature/preprocessing
```

Then open a pull request on GitHub for review before merging.

## Task Breakdown

**Person A — Data & Preprocessing** (`src/preprocessing.py`, `src/features.py`, `notebooks/01_eda.ipynb`)
- Join auxiliary tables (bureau, previous_application, installments) to the main table
- Run EDA: target/predictor identification, class imbalance, missingness, correlations
- Handle missing values, outliers, categorical encoding, class imbalance treatment
- Engineer derived features (debt-to-income ratio, aggregated payment history indicators)
- Hand off a clean train/val/test split

**Person B — Model & Evaluation** (`src/model.py`, `notebooks/02_modeling.ipynb`)
- Train Decision Tree (baseline), Random Forest, and XGBoost
- Evaluate all three on accuracy, precision, recall, F1-score, and ROC-AUC
- Consolidate results into a comparison table and select the best-performing model
- Hand off the best model for SHAP/LIME integration

**Person C — XAI & Dashboard** (`src/explain.py`, `dashboard/app.py`)
- Implement SHAP (global + local explanations) on the best-performing model
- Implement LIME (local surrogate explanations) as a complement to SHAP
- Compare explanation stability/consistency between SHAP and LIME
- Build the Streamlit dashboard: prediction + confidence score, SHAP feature contribution chart, LIME explanation, adverse-action-style reason codes (ECOA-aligned)
- Run the informal reviewer walkthrough once the dashboard is functional

**All together** — testing, evaluation, and writing up the thesis chapters.

## Tools

| Phase | Tool |
|---|---|
| Data processing | Pandas, NumPy |
| EDA | Pandas, Matplotlib, Plotly |
| Machine learning | Scikit-learn (Decision Tree, Random Forest), XGBoost |
| Explainable AI | SHAP, LIME |
| Dashboard | Streamlit |
| Version control | Git, GitHub |
