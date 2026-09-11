"""
Streamlit Dashboard — Section 3.2.7, Explainable AI Platform for Credit Risk.

THIS VERSION changes the reason-codes section so EVERY applicant sees an
explanation, not just high-risk ones:
  - HIGH RISK (>=50%): "Main reasons for this decision" -- top SHAP
    features INCREASING default risk. This matches real ECOA
    adverse-action notices, which are legally required for declines.
  - LOW RISK (<50%): "Main factors supporting this decision" -- top SHAP
    features DECREASING default risk. Real lenders aren't required to
    justify approvals, but showing this keeps the dashboard demo
    consistent and avoids it looking broken when a reviewer picks an
    approved applicant first. A caption below makes the ECOA distinction
    explicit either way, so the design choice is documented on-screen,
    not just in code comments.

Everything else is unchanged from the previous version.

HOW TO RUN THIS FILE:
  streamlit run dashboard/app.py
  (Ctrl+C in the terminal to stop it)

REQUIRES: streamlit, shap, lime
    pip install streamlit shap lime
"""

import pandas as pd
import joblib
import os
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    st.error("This dashboard needs shap. In a terminal, run: pip install shap")
    st.stop()

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    st.error("This dashboard needs lime. In a terminal, run: pip install lime")
    st.stop()

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

RANDOM_STATE = 42
N_BACKGROUND = 2000
NUM_LIME_FEATURES = 10
N_REASON_CODES = 3

# ----------------------------------------------------------------------------
# Plain-English feature descriptions (ECOA-style reason codes)
# Built from v1_data_dictionary.csv. Extend further as needed.
# ----------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_1": "external credit bureau score (source 1)",
    "EXT_SOURCE_2": "external credit bureau score (source 2)",
    "EXT_SOURCE_3": "external credit bureau score (source 3)",
    "EXT_SOURCE_1_MISSING": "no external credit score available (source 1)",
    "EXT_SOURCE_2_MISSING": "no external credit score available (source 2)",
    "EXT_SOURCE_3_MISSING": "no external credit score available (source 3)",

    "AMT_INCOME_TOTAL": "declared income level",
    "AMT_CREDIT": "size of the requested credit",
    "AMT_ANNUITY": "loan repayment (annuity) amount",
    "AMT_GOODS_PRICE": "price of the goods being financed",
    "CNT_CHILDREN": "number of dependent children",
    "CNT_FAM_MEMBERS": "household size",

    "DEBT_TO_INCOME": "requested credit relative to income",
    "ANNUITY_TO_INCOME": "loan repayment amount relative to income",
    "CREDIT_TO_GOODS_RATIO": "requested credit amount relative to the goods' price",
    "EMPLOYED_TO_AGE_RATIO": "share of life spent in stable employment",
    "BUREAU_LATE_MONTHS_RATIO": "history of late payments on past credit accounts",

    "BUREAU_LOAN_COUNT": "number of loans on record with the credit bureau",
    "BUREAU_AVG_CREDIT_SUM": "average size of previously reported credit accounts",
    "BUREAU_MAX_CREDIT_SUM": "largest previously reported credit account",
    "BUREAU_SUM_CREDIT_SUM_DEBT": "outstanding debt on record with the credit bureau",
    "BUREAU_AVG_DAYS_CREDIT": "length of credit history on record",
    "BUREAU_TOTAL_LATE_MONTHS": "months of late payment reported by the credit bureau",
    "HAS_BUREAU_HISTORY": "external credit history on file",

    "PREV_LOAN_COUNT": "number of previous loan applications",
    "PREV_AVG_CREDIT_AMOUNT": "average amount of previous loan applications",
    "PREV_AVG_ANNUITY": "repayment amount on previous loan applications",
    "HAS_PREV_APPLICATION": "previous loan application on record",

    "POS_AVG_CNT_INSTALMENT_FUTURE": "number of remaining installments on existing loans",
    "POS_MAX_DPD": "history of overdue payments on point-of-sale/cash loans",
    "HAS_POS_HISTORY": "prior point-of-sale or cash loan history",

    "CC_AVG_BALANCE": "average credit card balance",
    "CC_MAX_BALANCE": "maximum credit card balance on record",
    "CC_AVG_CREDIT_LIMIT": "credit card limit relative to profile",
    "HAS_CREDIT_CARD": "prior credit card history with the lender",

    "INSTALL_AVG_PAYMENT_DELAY": "history of delayed installment payments",
    "INSTALL_UNDERPAID_RATIO": "history of underpaying past installments",
    "HAS_INSTALLMENT_HISTORY": "prior installment payment history",

    "AGE_YEARS": "applicant's age",
    "YEARS_EMPLOYED": "length of current employment",
    "YEARS_REGISTRATION": "time since registration update",
    "YEARS_ID_PUBLISH": "time since identity document was issued",
    "YEARS_LAST_PHONE_CHANGE": "time since last phone number change",
    "DAYS_EMPLOYED_MISSING": "employment status",
    "REGION_POPULATION_RELATIVE": "population density of applicant's region",
    "REGION_RATING_CLIENT": "regional risk rating",
    "REGION_RATING_CLIENT_W_CITY": "regional risk rating (city-adjusted)",
    "OWN_CAR_AGE": "age of owned vehicle",
    "FLAG_OWN_CAR_N": "car ownership",
    "FLAG_OWN_CAR_Y": "car ownership",
    "FLAG_OWN_REALTY_N": "property ownership",
    "FLAG_OWN_REALTY_Y": "property ownership",

    "NAME_EDUCATION_TYPE_ORDINAL": "education level",
    "OCCUPATION_TYPE_FREQ": "occupation category",
    "ORGANIZATION_TYPE_FREQ": "type of employer organization",

    "NAME_CONTRACT_TYPE_Cash loans": "loan type requested (cash loan)",
    "NAME_CONTRACT_TYPE_Revolving loans": "loan type requested (revolving credit)",

    "CODE_GENDER_F": "gender",
    "CODE_GENDER_M": "gender",

    "NAME_FAMILY_STATUS_Married": "marital status",
    "NAME_FAMILY_STATUS_Civil marriage": "marital status",
    "NAME_FAMILY_STATUS_Separated": "marital status",
    "NAME_FAMILY_STATUS_Single / not married": "marital status",
    "NAME_FAMILY_STATUS_Widow": "marital status",

    "NAME_HOUSING_TYPE_House / apartment": "housing situation (owns house/apartment)",
    "NAME_HOUSING_TYPE_With parents": "housing situation (living with parents)",
    "NAME_HOUSING_TYPE_Rented apartment": "housing situation (renting)",
    "NAME_HOUSING_TYPE_Municipal apartment": "housing situation (municipal housing)",
    "NAME_HOUSING_TYPE_Office apartment": "housing situation (employer-provided housing)",
    "NAME_HOUSING_TYPE_Co-op apartment": "housing situation (co-op housing)",

    "NAME_INCOME_TYPE_Working": "income source (employed)",
    "NAME_INCOME_TYPE_Commercial associate": "income source (commercial associate)",
    "NAME_INCOME_TYPE_Pensioner": "income source (pension)",
    "NAME_INCOME_TYPE_State servant": "income source (public sector employee)",
    "NAME_INCOME_TYPE_Unemployed": "employment status (unemployed)",
    "NAME_INCOME_TYPE_Student": "income source (student)",
    "NAME_INCOME_TYPE_Businessman": "income source (business owner)",
    "NAME_INCOME_TYPE_Maternity leave": "employment status (maternity leave)",

    "NAME_TYPE_SUITE_Unaccompanied": "application was submitted unaccompanied",
    "NAME_TYPE_SUITE_Family": "applied accompanied by family",
    "NAME_TYPE_SUITE_Spouse, partner": "applied accompanied by spouse/partner",
    "NAME_TYPE_SUITE_Children": "applied accompanied by children",

    "OBS_30_CNT_SOCIAL_CIRCLE": "number of social circle contacts observed",
    "DEF_30_CNT_SOCIAL_CIRCLE": "defaults observed within social circle (30-day)",
    "OBS_60_CNT_SOCIAL_CIRCLE": "number of social circle contacts observed (60-day)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "defaults observed within social circle (60-day)",

    "AMT_REQ_CREDIT_BUREAU_MON": "recent credit bureau inquiries (monthly)",
    "AMT_REQ_CREDIT_BUREAU_QRT": "recent credit bureau inquiries (quarterly)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "recent credit bureau inquiries (yearly)",

    "TOTALAREA_MODE": "size of residence",
    "APARTMENTS_AVG": "apartment size indicator",
    "ELEVATORS_AVG": "building amenities (elevators)",
    "ENTRANCES_AVG": "building layout indicator",
    "FLOORSMAX_AVG": "building height indicator",
    "YEARS_BEGINEXPLUATATION_AVG": "age of the building",
    "YEARS_BUILD_MODE": "age of the building",

    "FLAG_DOCUMENT_3": "supporting document provided",
    "FLAG_MOBIL": "mobile phone provided as contact",
    "FLAG_EMP_PHONE": "work phone provided as contact",
    "FLAG_WORK_PHONE": "work phone provided as contact",
    "FLAG_CONT_MOBILE": "mobile phone reachable",
    "FLAG_PHONE": "home phone provided as contact",
    "FLAG_EMAIL": "email provided as contact",
}

def describe_feature(feature_name: str) -> str:
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name]
    if feature_name.startswith("FLAG_DOCUMENT_"):
        return "a supporting document requirement"
    return feature_name.replace("_", " ").lower()

def get_reason_codes(shap_row_values, feature_names, direction="risk_increasing", top_n=N_REASON_CODES):
    """Return the top N features either INCREASING or DECREASING default
    risk, translated into plain English.
      direction="risk_increasing" -> positive SHAP values (used for declines)
      direction="risk_decreasing" -> negative SHAP values (used for approvals)
    """
    pairs = list(zip(feature_names, shap_row_values))
    if direction == "risk_increasing":
        filtered = [p for p in pairs if p[1] > 0]
        filtered.sort(key=lambda p: p[1], reverse=True)
    else:
        filtered = [p for p in pairs if p[1] < 0]
        filtered.sort(key=lambda p: p[1])  # most negative first = strongest support
    top = filtered[:top_n]
    return [describe_feature(name) for name, _ in top]

# ----------------------------------------------------------------------------
# Load model, data, and build both explainers -- ONCE, cached
# ----------------------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODELS_DIR, "xgboost.joblib"))

@st.cache_data
def load_test_data():
    return pd.read_csv(os.path.join(DATA_DIR, "v1_test.csv"))

@st.cache_data
def load_scaled_data():
    train_scaled = pd.read_csv(os.path.join(DATA_DIR, "v1_train_scaled.csv"))
    test_scaled = pd.read_csv(os.path.join(DATA_DIR, "v1_test_scaled.csv"))
    return train_scaled, test_scaled

@st.cache_resource
def build_shap_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_resource
def build_lime_explainer(_train_scaled, feature_cols):
    background = _train_scaled[feature_cols].sample(N_BACKGROUND, random_state=RANDOM_STATE).values
    return LimeTabularExplainer(
        training_data=background,
        feature_names=feature_cols,
        class_names=["Repaid", "Default"],
        mode="classification",
        random_state=RANDOM_STATE,
    )

model = load_model()
test = load_test_data()
train_scaled, test_scaled = load_scaled_data()

id_col = "SK_ID_CURR"
target_col = "TARGET"
feature_cols = [c for c in test.columns if c not in (id_col, target_col)]

shap_explainer = build_shap_explainer(model)
lime_explainer = build_lime_explainer(train_scaled, feature_cols)

def lime_predict_fn(x):
    return model.predict_proba(pd.DataFrame(x, columns=feature_cols))

# ----------------------------------------------------------------------------
# Page layout
# ----------------------------------------------------------------------------

st.set_page_config(page_title="Credit Risk XAI Dashboard", layout="wide")
st.title("Explainable AI Platform for Credit Risk Assessment")
st.write(
    "Select an applicant to see the model's prediction, two explanation "
    "methods (SHAP and LIME), and the main plain-English reasons behind "
    "the decision."
)

# ----------------------------------------------------------------------------
# Applicant selector
# ----------------------------------------------------------------------------

applicant_ids = test[id_col].tolist()
selected_id = st.selectbox("Select Applicant ID (SK_ID_CURR):", applicant_ids)

applicant_row = test[test[id_col] == selected_id]
X_applicant = applicant_row[feature_cols]

applicant_row_scaled = test_scaled[test_scaled[id_col] == selected_id]
x_applicant_scaled = applicant_row_scaled[feature_cols].values[0]

# ----------------------------------------------------------------------------
# Predict, explain, and display
# ----------------------------------------------------------------------------

if st.button("Predict"):
    proba = model.predict_proba(X_applicant)[0, 1]

    st.subheader("Prediction")
    st.metric(label="Predicted probability of default", value=f"{proba:.1%}")

    if proba >= 0.5:
        st.error("High risk — likely to default")
    else:
        st.success("Low risk — likely to repay")

    shap_values = shap_explainer(X_applicant)

    # --- Reason codes: framed differently depending on outcome ---
    if proba >= 0.5:
        st.subheader("Main reasons for this decision")
        reasons = get_reason_codes(shap_values[0].values, feature_cols, direction="risk_increasing")
        if reasons:
            for reason in reasons:
                st.write(f"- {reason.capitalize()}")
        else:
            st.write("No single feature strongly increased this applicant's risk.")
        st.caption(
            "Generated from the top SHAP contributors increasing predicted "
            "default risk, in the style of ECOA adverse-action notices "
            "(legally required for declines)."
        )
    else:
        st.subheader("Main factors supporting this decision")
        reasons = get_reason_codes(shap_values[0].values, feature_cols, direction="risk_decreasing")
        if reasons:
            for reason in reasons:
                st.write(f"- {reason.capitalize()}")
        else:
            st.write("No single feature strongly decreased this applicant's risk.")
        st.caption(
            "Generated from the top SHAP contributors decreasing predicted "
            "default risk. Real lenders are not required to justify "
            "approvals under ECOA — this is shown here for transparency "
            "and consistency across all applicants."
        )

    st.subheader("Detailed explanation (SHAP vs LIME)")
    st.caption(
        "SHAP and LIME are two different explanation methods. They don't "
        "always agree — where they differ is itself informative, not an error."
    )

    col_shap, col_lime = st.columns(2)

    with col_shap:
        st.markdown("**SHAP** (exact, based on the model's actual structure)")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)
        plt.close(fig)

    with col_lime:
        st.markdown("**LIME** (local approximation)")
        with st.spinner("Computing LIME explanation..."):
            explanation = lime_explainer.explain_instance(
                x_applicant_scaled,
                lime_predict_fn,
                num_features=NUM_LIME_FEATURES,
            )
        fig = explanation.as_pyplot_figure()
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Show raw applicant data"):
        st.dataframe(applicant_row)