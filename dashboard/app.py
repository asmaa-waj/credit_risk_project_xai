"""
Streamlit Dashboard — Section 3.2.7, Explainable AI Platform for Credit Risk.

THIS VERSION makes every reason code DIRECTIONAL: instead of just naming
a feature ("External credit bureau score"), each entry now states
which way it pointed ("Low external credit bureau score" for a risk-
increasing reason, "High external credit bureau score" for a risk-
decreasing one). Previously the same neutral phrase was shown regardless
of direction, which made high-risk "reasons" and low-risk "supporting
factors" read identically.

HOW THIS WORKS:
  FEATURE_DESCRIPTIONS now maps each feature to a pair of phrases:
    "increase" -> phrasing to use when this feature pushed risk UP
    "decrease" -> phrasing to use when this feature pushed risk DOWN
  describe_feature() picks the right phrase for the direction it's
  called with. Features not explicitly listed fall back to a generic
  "higher-than-typical" / "lower-than-typical" template built from the
  raw column name, so nothing ever crashes or shows a blank reason.

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
# Plain-English, DIRECTIONAL feature descriptions.
# Each entry: feature -> {"increase": "<phrase when pushing risk UP>",
#                          "decrease": "<phrase when pushing risk DOWN>"}
# Built from v1_data_dictionary.csv + domain reasoning about which
# direction of each feature is favorable/unfavorable for credit risk.
# ----------------------------------------------------------------------------

FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_1": {"increase": "low external credit bureau score (source 1)",
                      "decrease": "high external credit bureau score (source 1)"},
    "EXT_SOURCE_2": {"increase": "low external credit bureau score (source 2)",
                      "decrease": "high external credit bureau score (source 2)"},
    "EXT_SOURCE_3": {"increase": "low external credit bureau score (source 3)",
                      "decrease": "high external credit bureau score (source 3)"},
    "EXT_SOURCE_1_MISSING": {"increase": "no external credit score available (source 1)",
                              "decrease": "external credit score available (source 1)"},
    "EXT_SOURCE_2_MISSING": {"increase": "no external credit score available (source 2)",
                              "decrease": "external credit score available (source 2)"},
    "EXT_SOURCE_3_MISSING": {"increase": "no external credit score available (source 3)",
                              "decrease": "external credit score available (source 3)"},

    "AMT_INCOME_TOTAL": {"increase": "low declared income", "decrease": "high declared income"},
    "AMT_CREDIT": {"increase": "high requested credit amount", "decrease": "low requested credit amount"},
    "AMT_ANNUITY": {"increase": "high loan repayment (annuity) amount", "decrease": "low loan repayment (annuity) amount"},
    "AMT_GOODS_PRICE": {"increase": "high price of the goods being financed", "decrease": "low price of the goods being financed"},
    "CNT_CHILDREN": {"increase": "higher number of dependent children", "decrease": "lower number of dependent children"},
    "CNT_FAM_MEMBERS": {"increase": "larger household size", "decrease": "smaller household size"},

    "DEBT_TO_INCOME": {"increase": "high requested credit relative to income", "decrease": "low requested credit relative to income"},
    "ANNUITY_TO_INCOME": {"increase": "high loan repayment relative to income", "decrease": "low loan repayment relative to income"},
    "CREDIT_TO_GOODS_RATIO": {"increase": "requested credit amount higher than the goods' price", "decrease": "requested credit amount close to or below the goods' price"},
    "EMPLOYED_TO_AGE_RATIO": {"increase": "small share of life spent in stable employment", "decrease": "large share of life spent in stable employment"},
    "BUREAU_LATE_MONTHS_RATIO": {"increase": "high share of bureau-reported months with late payments", "decrease": "little or no history of late payments"},

    "BUREAU_LOAN_COUNT": {"increase": "high number of loans on record with the credit bureau", "decrease": "low number of loans on record with the credit bureau"},
    "BUREAU_AVG_CREDIT_SUM": {"increase": "high average size of previously reported credit accounts", "decrease": "low average size of previously reported credit accounts"},
    "BUREAU_MAX_CREDIT_SUM": {"increase": "large previously reported credit account", "decrease": "modest previously reported credit account"},
    "BUREAU_SUM_CREDIT_SUM_DEBT": {"increase": "high outstanding debt on record with the credit bureau", "decrease": "low outstanding debt on record with the credit bureau"},
    "BUREAU_AVG_DAYS_CREDIT": {"increase": "shorter credit history on record", "decrease": "longer, more established credit history on record"},
    "BUREAU_TOTAL_LATE_MONTHS": {"increase": "history of multiple late-payment months reported by the credit bureau", "decrease": "little to no late-payment history reported by the credit bureau"},
    "HAS_BUREAU_HISTORY": {"increase": "no external credit history on file", "decrease": "established external credit history on file"},

    "PREV_LOAN_COUNT": {"increase": "few or no previous loan applications on record", "decrease": "an established history of previous loan applications"},
    "PREV_AVG_CREDIT_AMOUNT": {"increase": "high average amount on previous loan applications", "decrease": "modest average amount on previous loan applications"},
    "PREV_AVG_ANNUITY": {"increase": "high repayment amount on previous loan applications", "decrease": "low repayment amount on previous loan applications"},
    "HAS_PREV_APPLICATION": {"increase": "no previous loan application on record (first-time applicant)", "decrease": "an existing history of previous loan applications"},

    "POS_AVG_CNT_INSTALMENT_FUTURE": {"increase": "many remaining installments on existing loans", "decrease": "few remaining installments on existing loans"},
    "POS_MAX_DPD": {"increase": "history of significantly overdue payments on point-of-sale/cash loans", "decrease": "no significant history of overdue point-of-sale/cash payments"},
    "HAS_POS_HISTORY": {"increase": "no prior point-of-sale or cash loan history", "decrease": "an existing point-of-sale or cash loan history"},

    "CC_AVG_BALANCE": {"increase": "high average credit card balance", "decrease": "low average credit card balance"},
    "CC_MAX_BALANCE": {"increase": "high maximum credit card balance on record", "decrease": "low maximum credit card balance on record"},
    "CC_AVG_CREDIT_LIMIT": {"increase": "low credit card limit relative to profile", "decrease": "high credit card limit relative to profile"},
    "HAS_CREDIT_CARD": {"increase": "no prior credit card history with the lender", "decrease": "an existing credit card history with the lender"},

    "INSTALL_AVG_PAYMENT_DELAY": {"increase": "history of delayed installment payments", "decrease": "history of on-time or early installment payments"},
    "INSTALL_UNDERPAID_RATIO": {"increase": "history of underpaying past installments", "decrease": "history of fully paying past installments"},
    "HAS_INSTALLMENT_HISTORY": {"increase": "no prior installment payment history", "decrease": "an existing installment payment history"},

    "AGE_YEARS": {"increase": "younger applicant age", "decrease": "older, more established applicant age"},
    "YEARS_EMPLOYED": {"increase": "shorter length of current employment", "decrease": "longer length of current employment"},
    "YEARS_REGISTRATION": {"increase": "more recent registration update", "decrease": "longer-standing registration record"},
    "YEARS_ID_PUBLISH": {"increase": "more recently issued identity document", "decrease": "longer-standing identity document"},
    "YEARS_LAST_PHONE_CHANGE": {"increase": "recent phone number change", "decrease": "long-standing phone number"},
    "DAYS_EMPLOYED_MISSING": {"increase": "not currently employed", "decrease": "currently employed"},
    "REGION_POPULATION_RELATIVE": {"increase": "lower population density region", "decrease": "higher population density region"},
    "REGION_RATING_CLIENT": {"increase": "higher-risk regional rating", "decrease": "lower-risk regional rating"},
    "REGION_RATING_CLIENT_W_CITY": {"increase": "higher-risk regional rating (city-adjusted)", "decrease": "lower-risk regional rating (city-adjusted)"},
    "OWN_CAR_AGE": {"increase": "older owned vehicle", "decrease": "newer owned vehicle"},
    "FLAG_OWN_CAR_N": {"increase": "does not own a car", "decrease": "owns a car"},
    "FLAG_OWN_CAR_Y": {"increase": "owns a car", "decrease": "does not own a car"},
    "FLAG_OWN_REALTY_N": {"increase": "does not own property", "decrease": "owns property"},
    "FLAG_OWN_REALTY_Y": {"increase": "owns property", "decrease": "does not own property"},

    "NAME_EDUCATION_TYPE_ORDINAL": {"increase": "lower education level", "decrease": "higher education level"},
    "OCCUPATION_TYPE_FREQ": {"increase": "less common occupation category", "decrease": "more common, established occupation category"},
    "ORGANIZATION_TYPE_FREQ": {"increase": "less common type of employer organization", "decrease": "more common, established type of employer organization"},

    "NAME_CONTRACT_TYPE_Cash loans": {"increase": "loan type requested (cash loan)", "decrease": "loan type not a cash loan"},
    "NAME_CONTRACT_TYPE_Revolving loans": {"increase": "loan type requested (revolving credit)", "decrease": "loan type not revolving credit"},

    "CODE_GENDER_F": {"increase": "gender (female)", "decrease": "gender (not female)"},
    "CODE_GENDER_M": {"increase": "gender (male)", "decrease": "gender (not male)"},

    "NAME_FAMILY_STATUS_Married": {"increase": "marital status (married)", "decrease": "marital status (not married)"},
    "NAME_FAMILY_STATUS_Civil marriage": {"increase": "marital status (civil marriage)", "decrease": "marital status (not civil marriage)"},
    "NAME_FAMILY_STATUS_Separated": {"increase": "marital status (separated)", "decrease": "marital status (not separated)"},
    "NAME_FAMILY_STATUS_Single / not married": {"increase": "marital status (single)", "decrease": "marital status (not single)"},
    "NAME_FAMILY_STATUS_Widow": {"increase": "marital status (widowed)", "decrease": "marital status (not widowed)"},

    "NAME_HOUSING_TYPE_House / apartment": {"increase": "housing situation (owns house/apartment)", "decrease": "different housing situation"},
    "NAME_HOUSING_TYPE_With parents": {"increase": "housing situation (living with parents)", "decrease": "different housing situation"},
    "NAME_HOUSING_TYPE_Rented apartment": {"increase": "housing situation (renting)", "decrease": "different housing situation"},
    "NAME_HOUSING_TYPE_Municipal apartment": {"increase": "housing situation (municipal housing)", "decrease": "different housing situation"},
    "NAME_HOUSING_TYPE_Office apartment": {"increase": "housing situation (employer-provided housing)", "decrease": "different housing situation"},
    "NAME_HOUSING_TYPE_Co-op apartment": {"increase": "housing situation (co-op housing)", "decrease": "different housing situation"},

    "NAME_INCOME_TYPE_Working": {"increase": "income source (employed)", "decrease": "different income source"},
    "NAME_INCOME_TYPE_Commercial associate": {"increase": "income source (commercial associate)", "decrease": "different income source"},
    "NAME_INCOME_TYPE_Pensioner": {"increase": "income source (pension)", "decrease": "different income source"},
    "NAME_INCOME_TYPE_State servant": {"increase": "income source (public sector employee)", "decrease": "different income source"},
    "NAME_INCOME_TYPE_Unemployed": {"increase": "currently unemployed", "decrease": "currently employed"},
    "NAME_INCOME_TYPE_Student": {"increase": "income source (student)", "decrease": "different income source"},
    "NAME_INCOME_TYPE_Businessman": {"increase": "income source (business owner)", "decrease": "different income source"},
    "NAME_INCOME_TYPE_Maternity leave": {"increase": "currently on maternity leave", "decrease": "not on maternity leave"},

    "NAME_TYPE_SUITE_Unaccompanied": {"increase": "application submitted unaccompanied", "decrease": "application submitted with an accompanying person"},
    "NAME_TYPE_SUITE_Family": {"increase": "applied accompanied by family", "decrease": "applied without family accompaniment"},
    "NAME_TYPE_SUITE_Spouse, partner": {"increase": "applied accompanied by spouse/partner", "decrease": "applied without spouse/partner"},
    "NAME_TYPE_SUITE_Children": {"increase": "applied accompanied by children", "decrease": "applied without children present"},

    "OBS_30_CNT_SOCIAL_CIRCLE": {"increase": "large observed social circle", "decrease": "small observed social circle"},
    "DEF_30_CNT_SOCIAL_CIRCLE": {"increase": "defaults observed within social circle (30-day)", "decrease": "no defaults observed within social circle (30-day)"},
    "OBS_60_CNT_SOCIAL_CIRCLE": {"increase": "large observed social circle (60-day)", "decrease": "small observed social circle (60-day)"},
    "DEF_60_CNT_SOCIAL_CIRCLE": {"increase": "defaults observed within social circle (60-day)", "decrease": "no defaults observed within social circle (60-day)"},

    "AMT_REQ_CREDIT_BUREAU_MON": {"increase": "high number of recent credit bureau inquiries (monthly)", "decrease": "low number of recent credit bureau inquiries (monthly)"},
    "AMT_REQ_CREDIT_BUREAU_QRT": {"increase": "high number of recent credit bureau inquiries (quarterly)", "decrease": "low number of recent credit bureau inquiries (quarterly)"},
    "AMT_REQ_CREDIT_BUREAU_YEAR": {"increase": "high number of recent credit bureau inquiries (yearly)", "decrease": "low number of recent credit bureau inquiries (yearly)"},

    "TOTALAREA_MODE": {"increase": "smaller residence size", "decrease": "larger residence size"},
    "APARTMENTS_AVG": {"increase": "below-average apartment size indicator", "decrease": "above-average apartment size indicator"},
    "ELEVATORS_AVG": {"increase": "fewer building amenities (elevators)", "decrease": "more building amenities (elevators)"},
    "ENTRANCES_AVG": {"increase": "less favorable building layout indicator", "decrease": "more favorable building layout indicator"},
    "FLOORSMAX_AVG": {"increase": "lower building height indicator", "decrease": "higher building height indicator"},
    "YEARS_BEGINEXPLUATATION_AVG": {"increase": "older building", "decrease": "newer building"},
    "YEARS_BUILD_MODE": {"increase": "older building", "decrease": "newer building"},

    "FLAG_DOCUMENT_3": {"increase": "supporting document not provided", "decrease": "supporting document provided"},
    "FLAG_MOBIL": {"increase": "no mobile phone provided as contact", "decrease": "mobile phone provided as contact"},
    "FLAG_EMP_PHONE": {"increase": "no work phone provided as contact", "decrease": "work phone provided as contact"},
    "FLAG_WORK_PHONE": {"increase": "no work phone provided as contact", "decrease": "work phone provided as contact"},
    "FLAG_CONT_MOBILE": {"increase": "mobile phone not reachable", "decrease": "mobile phone reachable"},
    "FLAG_PHONE": {"increase": "no home phone provided as contact", "decrease": "home phone provided as contact"},
    "FLAG_EMAIL": {"increase": "no email provided as contact", "decrease": "email provided as contact"},
}

def describe_feature(feature_name: str, direction: str) -> str:
    """direction is either 'increase' (feature pushed risk up) or
    'decrease' (feature pushed risk down)."""
    if feature_name in FEATURE_DESCRIPTIONS:
        return FEATURE_DESCRIPTIONS[feature_name][direction]
    if feature_name.startswith("FLAG_DOCUMENT_"):
        return ("a missing supporting document" if direction == "increase"
                else "a supporting document provided")
    # Generic fallback for anything not explicitly mapped
    readable = feature_name.replace("_", " ").lower()
    if direction == "increase":
        return f"a higher-than-typical value for {readable}"
    else:
        return f"a lower-than-typical value for {readable}"

def get_reason_codes(shap_row_values, feature_names, direction="increase", top_n=N_REASON_CODES):
    """Return the top N features either INCREASING or DECREASING default
    risk, translated into plain, DIRECTIONAL English."""
    pairs = list(zip(feature_names, shap_row_values))
    if direction == "increase":
        filtered = [p for p in pairs if p[1] > 0]
        filtered.sort(key=lambda p: p[1], reverse=True)
    else:
        filtered = [p for p in pairs if p[1] < 0]
        filtered.sort(key=lambda p: p[1])  # most negative first = strongest support
    top = filtered[:top_n]
    return [describe_feature(name, direction) for name, _ in top]

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

    # --- Reason codes: now directional, framed differently by outcome ---
    if proba >= 0.5:
        st.subheader("Main reasons for this decision")
        reasons = get_reason_codes(shap_values[0].values, feature_cols, direction="increase")
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
        reasons = get_reason_codes(shap_values[0].values, feature_cols, direction="decrease")
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
