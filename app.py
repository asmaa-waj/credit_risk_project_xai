import os
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 1. PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Credit Risk XAI Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# 2. PROJECT PATHS AND SETTINGS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "xgboost.joblib")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "v1_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "v1_test.csv")

THRESHOLD = 0.50          # scores at or above this are flagged as higher risk
NEAR_CUTOFF = 0.10        # "close to the cut-off" if within this distance
BACKGROUND_SIZE = 1000    # training rows LIME uses as its reference
TOP_REASONS = 3           # reasons shown per group
AGREEMENT_TOP = 5         # top factors compared between the two methods
STRONG_SHARE = 0.10       # a factor with >= 10% of total influence is "Strong"
MODERATE_SHARE = 0.04     # >= 4% is "Moderate", otherwise "Small"


# ============================================================
# 3. PLAIN-LANGUAGE WORDING FOR MODEL FEATURES
# ============================================================

FEATURE_LABELS = {
    # External credit scores
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    # This loan
    "AMT_CREDIT": "Loan amount requested",
    "AMT_ANNUITY": "Regular repayment amount on this loan",
    "AMT_GOODS_PRICE": "Price of the item being financed",
    "AMT_INCOME_TOTAL": "Declared income",
    "CREDIT_TO_GOODS_RATIO": "Loan amount compared with the price of the item",
    "ANNUITY_TO_INCOME": "Regular repayment compared with income",
    "DEBT_TO_INCOME": "Loan amount compared with income",
    # Applicant history
    "EMPLOYED_TO_AGE_RATIO": "Share of life spent in the current job",
    "BUREAU_LATE_MONTHS_RATIO": "Share of past months with late payments to other lenders",
    "BUREAU_SUM_CREDIT_SUM_DEBT": "Total outstanding debt with other lenders",
    "PREV_AVG_ANNUITY": "Typical repayment amount on earlier applications",
    "INSTALL_UNDERPAID_RATIO": "Share of earlier installments that were paid short",
    "INSTALL_AVG_PAYMENT_DELAY": "Average delay in paying earlier installments",
    "POS_AVG_CNT_INSTALMENT_FUTURE": "Installments still to be paid on earlier loans",
    "CC_AVG_BALANCE": "Average balance on earlier credit cards",
    "CC_AVG_CREDIT_LIMIT": "Average limit on earlier credit cards",
    "HAS_BUREAU_HISTORY": "Has records with other lenders",
    "HAS_PREV_APPLICATION": "Has earlier applications with this lender",
    "HAS_CREDIT_CARD": "Has an earlier credit card with this lender",
    "HAS_POS_HISTORY": "Has earlier point-of-sale loans",
    "HAS_INSTALLMENT_HISTORY": "Has earlier installment records",
    # Personal details
    "AGE_YEARS": "Applicant's age",
    "DAYS_BIRTH": "Applicant's age",
    "YEARS_EMPLOYED": "Time in the current job",
    "DAYS_EMPLOYED": "Time in the current job",
    "DAYS_EMPLOYED_MISSING": "No current employment recorded",
    "YEARS_REGISTRATION": "Time since the applicant last changed their registration",
    "YEARS_ID_PUBLISH": "Time since the identity document was last changed",
    "YEARS_LAST_PHONE_CHANGE": "Time since the applicant last changed phone number",
    "NAME_EDUCATION_TYPE": "Education level",
    "EDUCATION_TYPE_ORDINAL": "Education level",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Type of employer",
    "CODE_GENDER_M": "Applicant's gender",
    "CODE_GENDER_F": "Applicant's gender",
    # Home and region
    "REGION_POPULATION_RELATIVE": "How densely populated the applicant's region is",
    "YEARS_BUILD_MODE": "Age of the applicant's building",
    "LANDAREA_AVG": "Size of the land around the applicant's home",
    "FLAG_WORK_PHONE": "Has a work phone number",
    "REG_REGION_NOT_LIVE_REGION": "Registered region differs from region of residence",
    "REG_REGION_NOT_WORK_REGION": "Registered region differs from region of work",
    "LIVE_REGION_NOT_WORK_REGION": "Region of residence differs from region of work",
    "REG_CITY_NOT_LIVE_CITY": "Registered city differs from city of residence",
    "REG_CITY_NOT_WORK_CITY": "Registered city differs from city of work",
    "LIVE_CITY_NOT_WORK_CITY": "City of residence differs from city of work",
}

# Prefixes of one-hot columns such as NAME_HOUSING_TYPE_Municipal apartment.
PREFIX_LABELS = {
    "NAME_CONTRACT_TYPE_": "Loan type",
    "NAME_INCOME_TYPE_": "Income source",
    "NAME_FAMILY_STATUS_": "Family status",
    "NAME_HOUSING_TYPE_": "Housing",
    "NAME_TYPE_SUITE_": "Accompanied by",
    "FLAG_OWN_CAR_": "Owns a car",
    "FLAG_OWN_REALTY_": "Owns property",
    "WALLSMATERIAL_MODE_": "Building wall material",
    "HOUSETYPE_MODE_": "House type",
    "FONDKAPREMONT_MODE_": "Building renovation fund",
    "EMERGENCYSTATE_MODE_": "Emergency state of the building",
    "WEEKDAY_APPR_PROCESS_START_": "Day of the week of the application",
}

# Short explanations shown in smaller text for features people may not know.
FEATURE_HELP = {
    "EXT_SOURCE_1": "Scores from outside credit agencies. A higher score generally means lower risk.",
    "EXT_SOURCE_2": "Scores from outside credit agencies. A higher score generally means lower risk.",
    "EXT_SOURCE_3": "Scores from outside credit agencies. A higher score generally means lower risk.",
}

# Wording for yes/no columns where "yes" or "no" alone would read badly.
BINARY_PHRASES = {
    "CODE_GENDER_M": ("Applicant is male", "Applicant is not male"),
    "CODE_GENDER_F": ("Applicant is female", "Applicant is not female"),
    "DAYS_EMPLOYED_MISSING": (
        "No current employment is recorded",
        "A current employment is recorded",
    ),
}

# Categories stored as how common they are, so "higher or lower than typical"
# would be meaningless for them.
FREQUENCY_ENCODED = {"OCCUPATION_TYPE", "ORGANIZATION_TYPE"}

# Personal characteristics that should not be the basis of a lending decision.
SENSITIVE_PREFIXES = ("CODE_GENDER",)
SENSITIVE_FEATURES = {"AGE_YEARS", "DAYS_BIRTH"}


def is_sensitive(feature):
    return feature in SENSITIVE_FEATURES or feature.startswith(SENSITIVE_PREFIXES)


def readable_feature_name(feature):
    """Convert a model feature name into a short, plain-language label."""

    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    for prefix, text in PREFIX_LABELS.items():
        if feature.startswith(prefix):
            category = feature[len(prefix):].replace("_", " ").lower()
            return f"{text}: {category}"

    if feature.startswith("FLAG_DOCUMENT_"):
        number = feature.replace("FLAG_DOCUMENT_", "")
        return f"Supporting document {number} provided"

    if feature.startswith("EXT_SOURCE_"):
        return f"External credit score {feature.replace('EXT_SOURCE_', '')}"

    if feature.startswith("BUREAU_"):
        return "Information from other lenders' credit records"

    if feature.startswith("CC_"):
        return "Information on earlier credit cards"

    if feature.startswith("POS_"):
        return "Information on earlier point-of-sale loans"

    if feature.startswith("INSTALL_"):
        return "Information on earlier installment payments"

    if feature.startswith("PREV_"):
        return "Information on earlier applications"

    if feature.startswith("AMT_"):
        return "Loan or financial amount"

    if feature.startswith("FLAG_"):
        return "Application detail"

    return feature.replace("_", " ").lower().capitalize()


# ============================================================
# 4. PLAIN-LANGUAGE SENTENCES FOR REASONS
# ============================================================

def describe_value(feature, value, median):
    """Say where the applicant's value sits compared with a typical applicant.

    "Typical" is the median of the training data.
    """

    if feature in FREQUENCY_ENCODED:
        return "was taken into account"

    if pd.isna(value):
        return "was not available"

    if pd.isna(median) or np.isclose(value, median):
        return "is about typical for applicants"

    if value > median:
        return "is higher than typical for applicants"

    return "is lower than typical for applicants"


def strength_label(share):
    """Turn a factor's share of the total influence into a plain word."""

    if share >= STRONG_SHARE:
        return "Strong"
    if share >= MODERATE_SHARE:
        return "Moderate"
    return "Small"


def reason_line(feature, applicant_value, train_median, raises_risk,
                share=None):
    """One plain-language bullet explaining a single factor."""

    label = readable_feature_name(feature)
    effect = "raised" if raises_risk else "lowered"

    if feature in binary_features and feature not in FREQUENCY_ENCODED:
        if pd.isna(applicant_value):
            core = f"{label}: not available"
        elif feature in BINARY_PHRASES:
            yes_text, no_text = BINARY_PHRASES[feature]
            core = yes_text if applicant_value >= 0.5 else no_text
        else:
            answer = "yes" if applicant_value >= 0.5 else "no"
            separator = " \u2014 " if ":" in label else ": "
            core = f"{label}{separator}{answer}"
    else:
        core = f"{label} {describe_value(feature, applicant_value, train_median)}"

    text = f"{core}, which {effect} the risk score."

    if share is not None:
        text = f"**{strength_label(share)} influence:** {text}"

    if feature in FEATURE_HELP:
        text += f" *({FEATURE_HELP[feature]})*"

    if is_sensitive(feature):
        text += (
            " ⚠️ *This is a personal characteristic. It should not be used "
            "to justify a lending decision.*"
        )

    return "- " + text


def show_factor_group(title, indices, applicant_row, medians, feature_names,
                      top_n, raises_risk, shares=None):
    """Show one group of factors under a plain heading."""

    st.markdown(f"**{title}**")

    if not indices:
        st.write("No strong factors were found in this group.")
        return

    for i in indices[:top_n]:
        feature = feature_names[i]
        share = None if shares is None else shares[i]
        st.markdown(
            reason_line(
                feature,
                applicant_row[feature],
                medians[feature],
                raises_risk,
                share
            )
        )


def lime_feature_from_condition(condition, feature_names):
    """Find the original feature name inside a LIME condition."""

    # Longest-first avoids matching a shorter feature name by mistake.
    ordered = sorted(feature_names, key=len, reverse=True)

    for feature in ordered:
        if condition.startswith(feature):
            return feature

    # Range conditions look like "0.3 < FEATURE <= 0.5".
    for feature in ordered:
        if feature in condition:
            return feature
    return None


# ============================================================
# 5. LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


@st.cache_data
def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


for required_path in [MODEL_PATH, TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(required_path):
        st.error(
            "A required project file was not found:\n\n"
            f"{required_path}\n\n"
            "Check that app.py is inside the dashboard folder and "
            "that the model and CSV files are in the expected folders."
        )
        st.stop()

try:
    model = load_model(MODEL_PATH)
    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)
except Exception as error:
    st.error(f"Could not load the model or data: {error}")
    st.stop()


# ============================================================
# 6. PREPARE MODEL FEATURES
# ============================================================

excluded_columns = ["SK_ID_CURR", "TARGET"]

feature_columns = [
    column for column in train_df.columns
    if column not in excluded_columns
]

missing_in_test = [
    column for column in feature_columns
    if column not in test_df.columns
]

if missing_in_test:
    st.error(f"These training features are missing from test data: {missing_in_test}")
    st.stop()

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

# Ensure identical column order.
X_test = X_test[feature_columns]

# The XGBoost model was trained on the unscaled v1_train.csv features,
# so the dashboard uses these same raw features.
try:
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
except (ValueError, TypeError) as error:
    st.error(
        "The model input data contains non-numeric values. "
        f"Please check the processed CSV files. Details: {error}"
    )
    st.stop()

# Training-data medians describe what is "typical" for an applicant.
train_medians = X_train.median()

# 0/1 columns (flags and one-hot columns) are described as yes / no.
binary_features = {
    column for column in feature_columns
    if set(X_train[column].dropna().unique()) <= {0.0, 1.0}
}

if hasattr(model, "n_features_in_"):
    if model.n_features_in_ != len(feature_columns):
        st.error(
            f"Model expects {model.n_features_in_} features, "
            f"but the CSV contains {len(feature_columns)}."
        )
        st.stop()

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != feature_columns:
        st.warning(
            "The model's saved feature names do not exactly match "
            "the CSV feature names/order. Please verify the training pipeline."
        )


# ============================================================
# 7. DASHBOARD INTRODUCTION
# ============================================================

st.title("Credit Risk XAI Dashboard")

st.write(
    "This page shows how risky the model thinks an applicant is, and "
    "explains the main reasons behind that score in plain language."
)

st.info(
    "**How to read this page**\n\n"
    "- The **risk score** runs from 0% to 100%. The higher it is, the more "
    "the applicant looks like past applicants who had payment difficulties.\n"
    "- A score at or above **50%** is flagged as higher risk.\n"
    "- The reasons below show **what the model looked at**, not what "
    "caused the risk. A person should always make the final decision."
)

st.caption(
    "Research prototype using the Home Credit Default Risk dataset. "
    "It is not an automatic lending decision."
)

st.divider()


# ============================================================
# 8. SELECT APPLICANT
# ============================================================

st.header("Applicant Assessment")

if "SK_ID_CURR" in test_df.columns:
    test_ids = pd.to_numeric(test_df["SK_ID_CURR"], errors="coerce")
    id_options = [int(x) for x in test_ids.dropna().unique()]

    try:
        # Dropdown that also lets you type any ID (newer Streamlit).
        chosen_id = st.selectbox(
            "Select or type Applicant ID",
            options=id_options,
            index=0,
            accept_new_options=True
        )
    except TypeError:
        # Older Streamlit: dropdown with search-by-typing only.
        chosen_id = st.selectbox(
            "Select or type Applicant ID",
            options=id_options,
            index=0
        )

    try:
        selected_id = int(chosen_id)
    except (ValueError, TypeError):
        st.error("Please enter a valid numeric Applicant ID.")
        st.stop()

    matching_rows = test_df.index[test_ids == selected_id].tolist()

    if not matching_rows:
        st.error(
            f"Applicant ID {selected_id} was not found in the test dataset."
        )
        st.stop()

    applicant_index = matching_rows[0]
    X_applicant = X_test.loc[[applicant_index]]
else:
    st.error("SK_ID_CURR was not found in the test dataset.")
    st.stop()


# ============================================================
# 9. RISK SCORE
# ============================================================

try:
    probabilities = model.predict_proba(X_applicant)
    risk_score = float(probabilities[0][1])
except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

is_higher_risk = risk_score >= THRESHOLD

st.subheader("Risk Score")

col1, col2 = st.columns([1, 2])

col1.metric("Estimated risk score", f"{risk_score:.1%}")
col2.progress(min(max(risk_score, 0.0), 1.0))
col2.caption("0% = lowest risk, 100% = highest risk. The flag line is at 50%.")

if is_higher_risk:
    st.error(
        "**Higher risk.** The score is at or above 50%. A careful manual "
        "review of this application is recommended."
    )
else:
    st.success(
        "**Lower risk.** The score is below 50%. This is the model's view "
        "only, and normal checks still apply."
    )

if abs(risk_score - THRESHOLD) <= NEAR_CUTOFF:
    st.info(
        "This score is close to the 50% line. A small change in the "
        "applicant's information could change the result, so a closer "
        "review is advisable."
    )

st.caption(
    "The risk score is the model's own estimate. It is not the exact chance "
    "that this applicant will fail to repay: the model was set up to catch "
    "more risky applicants, which pushes scores upward."
)

st.divider()


# ============================================================
# 10. SHAP EXPLANATION
# ============================================================

st.header("Why Did the Model Give This Score?")
st.caption("Main explanation (SHAP method)")

shap_by_feature = {}
shap_explanation = None

try:
    shap_explainer = shap.TreeExplainer(model)
    shap_explanation = shap_explainer(X_applicant)

    shap_contributions = np.asarray(shap_explanation.values)

    # For binary XGBoost, expected shape is (1, number_of_features).
    # Handle an extra output dimension if returned by a SHAP version.
    if shap_contributions.ndim == 3:
        shap_contributions = shap_contributions[:, :, -1]

    shap_contributions = shap_contributions.reshape(-1)

    if len(shap_contributions) != len(feature_columns):
        raise ValueError(
            "SHAP returned a different number of contributions "
            "than the number of model features."
        )

    shap_by_feature = dict(zip(feature_columns, shap_contributions))

    # Each factor's share of the total influence on the score.
    total_influence = float(np.abs(shap_contributions).sum())
    if total_influence == 0:
        total_influence = 1.0
    shares = np.abs(shap_contributions) / total_influence

    raising_indices = sorted(
        [i for i, v in enumerate(shap_contributions) if v > 0],
        key=lambda i: abs(shap_contributions[i]),
        reverse=True
    )
    lowering_indices = sorted(
        [i for i, v in enumerate(shap_contributions) if v < 0],
        key=lambda i: abs(shap_contributions[i]),
        reverse=True
    )

    # One-sentence summary of the biggest drivers.
    top_drivers = sorted(
        range(len(shap_contributions)),
        key=lambda i: abs(shap_contributions[i]),
        reverse=True
    )[:TOP_REASONS]
    driver_names = []
    for i in top_drivers:
        name = readable_feature_name(feature_columns[i]).lower()
        if name not in driver_names:
            driver_names.append(name)

    st.markdown(
        "**In short:** the score is mainly driven by "
        + ", ".join(driver_names[:-1])
        + (" and " if len(driver_names) > 1 else "")
        + driver_names[-1] + "."
    )

    st.caption(
        "\"Typical\" means the middle value among past applicants. "
        "Yes/no details are shown as yes or no. The strength labels show "
        "how much each factor mattered compared with the others."
    )

    applicant_row = X_applicant.iloc[0]

    raising = (
        "What raised the risk score", raising_indices, True
    )
    lowering = (
        "What lowered the risk score", lowering_indices, False
    )

    # Show the group that matches the model's result first.
    groups = [raising, lowering] if is_higher_risk else [lowering, raising]

    for title, indices, raises in groups:
        show_factor_group(
            title,
            indices,
            applicant_row,
            train_medians,
            feature_columns,
            TOP_REASONS,
            raises,
            shares
        )

    with st.expander("For technical users: detailed SHAP chart and table"):

        st.caption(
            "SHAP values are measured on the model's internal scale "
            "(log-odds). They show how much each factor pushed the score "
            "up or down, but they are not percentage points."
        )

        try:
            shap.plots.waterfall(
                shap_explanation[0],
                max_display=12,
                show=False
            )
            st.pyplot(plt.gcf(), clear_figure=True)
            plt.close("all")
        except Exception as chart_error:
            st.warning(f"Could not display the SHAP chart: {chart_error}")

        shap_table = pd.DataFrame({
            "Feature": feature_columns,
            "Applicant value": X_applicant.iloc[0].values,
            "Effect on score (technical units)": shap_contributions
        })

        shap_table["Absolute effect"] = (
            shap_table["Effect on score (technical units)"].abs()
        )

        shap_table = shap_table.sort_values(
            "Absolute effect",
            ascending=False
        )

        st.dataframe(
            shap_table.drop(columns=["Absolute effect"]),
            use_container_width=True,
            hide_index=True
        )

except Exception as error:
    st.error(f"The explanation could not be generated: {error}")
    shap_explanation = None
    shap_by_feature = {}


st.divider()


# ============================================================
# 11. LIME EXPLANATION (SECOND OPINION)
# ============================================================

st.header("A Second Opinion")
st.caption("Cross-check using a different method (LIME)")

lime_by_feature = {}

try:
    background = X_train.sample(
        n=min(BACKGROUND_SIZE, len(X_train)),
        random_state=42
    )

    lime_explainer = LimeTabularExplainer(
        training_data=background.to_numpy(),
        feature_names=feature_columns,
        class_names=["Repaid", "Payment difficulties"],
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    def lime_predict_function(samples):
        samples_df = pd.DataFrame(
            samples,
            columns=feature_columns
        )
        return model.predict_proba(samples_df)

    lime_explanation = lime_explainer.explain_instance(
        data_row=X_applicant.iloc[0].to_numpy(),
        predict_fn=lime_predict_function,
        num_features=min(10, len(feature_columns)),
        labels=(1,)
    )

    lime_results = lime_explanation.as_list(label=1)

    lime_results = sorted(
        lime_results,
        key=lambda item: abs(item[1]),
        reverse=True
    )

    st.write(
        "A second method looks at the same applicant in a different way. "
        "It is a useful check, but it may point to different factors."
    )

    lime_lowering, lime_raising = [], []

    for condition, weight in lime_results:
        feature = lime_feature_from_condition(condition, feature_columns)

        if feature is None or feature in lime_by_feature or weight == 0:
            continue

        lime_by_feature[feature] = weight
        feature_index = feature_columns.index(feature)

        if weight > 0:
            lime_raising.append(feature_index)
        else:
            lime_lowering.append(feature_index)

    raising = ("What raised the risk score", lime_raising, True)
    lowering = ("What lowered the risk score", lime_lowering, False)

    groups = [raising, lowering] if is_higher_risk else [lowering, raising]

    for title, indices, raises in groups:
        show_factor_group(
            title,
            indices,
            X_applicant.iloc[0],
            train_medians,
            feature_columns,
            TOP_REASONS,
            raises
        )

    with st.expander("For technical users: detailed LIME table and chart"):

        st.caption(
            "A positive weight pushes toward higher risk and a negative "
            "weight toward lower risk. The weights are not percentages."
        )

        lime_table = pd.DataFrame(
            lime_results,
            columns=["Condition used by the method", "Weight"]
        )

        lime_table["Direction"] = lime_table["Weight"].apply(
            lambda value: (
                "Toward higher risk"
                if value > 0
                else "Toward lower risk"
                if value < 0
                else "Little influence"
            )
        )

        st.dataframe(
            lime_table,
            use_container_width=True,
            hide_index=True
        )

        try:
            fig = lime_explanation.as_pyplot_figure(label=1)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as chart_error:
            st.warning(f"Could not display the LIME chart: {chart_error}")

except Exception as error:
    st.error(f"The second opinion could not be generated: {error}")


# ============================================================
# 12. DO THE TWO METHODS AGREE?
# ============================================================

if shap_by_feature and lime_by_feature:

    st.subheader("Do the Two Methods Agree?")

    shap_top = sorted(
        shap_by_feature, key=lambda f: abs(shap_by_feature[f]), reverse=True
    )[:AGREEMENT_TOP]

    lime_top = sorted(
        lime_by_feature, key=lambda f: abs(lime_by_feature[f]), reverse=True
    )[:AGREEMENT_TOP]

    agreed = [
        f for f in shap_top
        if f in lime_top
        and np.sign(shap_by_feature[f]) == np.sign(lime_by_feature[f])
    ]

    agreed_names = []
    for f in agreed:
        name = readable_feature_name(f).lower()
        if name not in agreed_names:
            agreed_names.append(name)

    if agreed_names:
        st.success(
            "Both methods point to the same main factors: "
            + ", ".join(agreed_names)
            + ". Explanations that two methods agree on are more reliable."
        )
    else:
        st.warning(
            "The two methods did not highlight the same main factors for "
            "this applicant. Treat these explanations with extra care."
        )


st.divider()


# ============================================================
# 13. ACTUAL TEST-DATA OUTCOME
# ============================================================

st.header("What Actually Happened")

st.caption(
    "Shown only to check the model against past data. A real application "
    "would not have this information."
)

if "TARGET" in test_df.columns:
    actual_target = test_df.loc[applicant_index, "TARGET"]

    if pd.notna(actual_target):
        if int(actual_target) == 1:
            st.warning(
                "This applicant did have payment difficulties."
            )
        else:
            st.success(
                "This applicant did not have payment difficulties."
            )
    else:
        st.write("The actual outcome is not available.")
else:
    st.write("The actual outcome is not available in this dataset.")


# ============================================================
# 14. ABOUT THE DASHBOARD
# ============================================================

with st.expander("About this assessment"):

    st.write(
        """
        **What this is.** A research prototype built on a public kaggle dataset of
        past loan applicants from one lender (Home Credit). It estimates how
        similar an applicant is to past applicants who had payment
        difficulties.

        **What the risk score means.** It is the model's own score on a
        0-100% scale, not the exact chance of non-payment. The model was
        deliberately set up to catch more risky applicants, which pushes the
        scores upward, so use the 50% line as a guide to when extra review
        is wise.

        **What the reasons mean.** They show which details the model looked
        at and in which direction each one moved the score. They do not
        prove that a detail caused payment difficulties. Two different
        methods are shown so that you can see where they agree.

        **Personal characteristics.** Details such as age or gender may
        appear among the factors because they exist in the dataset. They
        should not be used to justify a lending decision, and they are
        marked with a warning when they appear.

        **Limits.** The model has only been tested on past data from one
        lender. It has not been checked on real lending decisions and should
        not be the only basis for any real decision. A trained person must
        make the final decision and be able to override the model.
        """
    )
