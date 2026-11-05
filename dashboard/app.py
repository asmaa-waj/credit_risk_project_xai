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
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "xgboost.joblib")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "v1_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "v1_test.csv")

THRESHOLD = 0.50
BACKGROUND_SIZE = 1000
TOP_REASONS = 3


# ============================================================
# 3. FEATURE NAMES FOR PLAIN-LANGUAGE DISPLAY
# ============================================================

FEATURE_LABELS = {
    "EXT_SOURCE_1": "External credit score (source 1)",
    "EXT_SOURCE_2": "External credit score (source 2)",
    "EXT_SOURCE_3": "External credit score (source 3)",
    "AMT_CREDIT": "Requested loan amount",
    "AMT_ANNUITY": "Scheduled loan repayment amount",
    "AMT_GOODS_PRICE": "Price of the financed goods",
    "CREDIT_TO_GOODS_RATIO": "Loan amount compared with goods price",
    "PREV_AVG_ANNUITY": "Repayment amount on previous loan applications",
    "ANNUITY_TO_INCOME": "Loan repayment amount compared with income",
    "INSTALL_UNDERPAID_RATIO": "Previous installment payment information",
    "BUREAU_SUM_CREDIT_SUM_DEBT": "Outstanding debt from previous credit records",
    "POS_AVG_CNT_INSTALMENT_FUTURE": "Previous loan installment information",
    "CC_AVG_BALANCE": "Previous credit card balance",
    "CC_AVG_CREDIT_LIMIT": "Previous credit card limit information",
    "REGION_POPULATION_RELATIVE": "Population information for the applicant's region",
    "YEARS_BUILD_MODE": "Building age information",
    "LANDAREA_AVG": "Property land-area information",
    "WALLSMATERIAL_MODE_Monolithic": "Building wall-material information",
    "NAME_TYPE_SUITE_Group of people": "Who accompanied the applicant",
    "NAME_HOUSING_TYPE_Co-op apartment": "Applicant's housing information",
    "NAME_HOUSING_TYPE_Municipal apartment": "Applicant's housing information",
    "FLAG_WORK_PHONE": "Work phone information",
    "REG_REGION_NOT_LIVE_REGION": "Mismatch between registered region and region of residence",
    "REG_REGION_NOT_WORK_REGION": "Mismatch between registered region and region of work",
    "LIVE_REGION_NOT_WORK_REGION": "Mismatch between region of residence and region of work",
    "REG_CITY_NOT_LIVE_CITY": "Mismatch between registered city and city of residence",
    "REG_CITY_NOT_WORK_CITY": "Mismatch between registered city and city of work",
    "LIVE_CITY_NOT_WORK_CITY": "Mismatch between city of residence and city of work",
    "NAME_EDUCATION_TYPE": "Applicant's education information",
    "EDUCATION_TYPE_ORDINAL": "Recorded education category code",
}


def readable_feature_name(feature):
    """Convert a model feature name into a short employee-friendly label."""

    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    if feature.startswith("FLAG_DOCUMENT_"):
        number = feature.replace("FLAG_DOCUMENT_", "")
        return f"Application document information (document {number})"

    if feature.startswith("EXT_SOURCE_"):
        number = feature.replace("EXT_SOURCE_", "")
        return f"External credit score (source {number})"

    if feature.startswith("BUREAU_"):
        return "Previous credit bureau information"

    if feature.startswith("CC_"):
        return "Previous credit card information"

    if feature.startswith("POS_"):
        return "Previous loan installment information"

    if feature.startswith("INSTALL_"):
        return "Previous installment payment information"

    if feature.startswith("AMT_"):
        return "Loan or financial amount"

    if feature.startswith("FLAG_"):
        return "Application information"

    if feature.startswith("NAME_HOUSING_TYPE_"):
        return "Applicant's housing information"

    if feature.startswith("NAME_TYPE_SUITE_"):
        return "Who accompanied the applicant"

    if feature.startswith("YEARS_BUILD"):
        return "Building age information"

    if feature.startswith("WALLSMATERIAL"):
        return "Building wall-material information"

    if feature.startswith("LANDAREA"):
        return "Property land-area information"

    return feature.replace("_", " ").lower().capitalize()


def shap_reason_text(feature, applicant_value, train_median):
    """Describe where the applicant's value sits compared with typical.

    "Typical" is the training-data median. The effect on risk is shown
    by the heading the line is placed under, so it is not repeated here.
    """

    label = readable_feature_name(feature)

    if pd.isna(applicant_value):
        position = "is missing"
    elif feature in binary_features:
        # 0/1 flag: describe presence instead of above/below typical.
        position = "is present" if applicant_value >= 0.5 else "is not present"
    elif pd.isna(train_median) or np.isclose(applicant_value, train_median):
        position = "is typical"
    elif applicant_value > train_median:
        position = "is above typical"
    else:
        position = "is below typical"

    return f"{label} {position}."


def show_shap_group(title, indices, shap_values, applicant_row, medians,
                    feature_names, top_n):
    """Show one group of SHAP factors under a heading."""

    st.markdown(f"**{title}**")

    if not indices:
        st.write("No strong factors were identified in this group.")
        return

    for i in indices[:top_n]:
        feature = feature_names[i]
        st.markdown(
            "- " + shap_reason_text(
                feature,
                applicant_row[feature],
                medians[feature]
            )
        )


def lime_feature_from_condition(condition, feature_names):
    """Find the original feature name at the start of a LIME condition."""

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


def plain_lime_reason(condition, weight, feature_names):
    """Create a short LIME reason without exposing technical conditions."""

    feature = lime_feature_from_condition(condition, feature_names)

    if feature is None:
        label = "Other application information"
    else:
        label = readable_feature_name(feature)

    if weight > 0:
        direction = "supported a higher estimated risk"
    elif weight < 0:
        direction = "supported a lower estimated risk"
    else:
        direction = "had little influence in this local explanation"

    return label, direction


# ============================================================
# 4. LOAD MODEL AND DATA
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
# 5. PREPARE MODEL FEATURES
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

# The XGBoost model in this project was trained on the unscaled
# v1_train.csv features, so the dashboard uses these same raw features.
try:
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
except (ValueError, TypeError) as error:
    st.error(
        "The model input data contains non-numeric values. "
        f"Please check the processed CSV files. Details: {error}"
    )
    st.stop()

# Training-data medians used to describe whether an applicant's value
# is higher or lower than typical.
train_medians = X_train.median()

# 0/1 flag features (including one-hot columns) are described as
# "present" / "not present" instead of above / below typical.
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
# 6. DASHBOARD INTRODUCTION
# ============================================================

st.title("Credit Risk XAI Dashboard")

st.write(
    "Assess an applicant's estimated likelihood of payment difficulties "
    "and review the main factors behind the model's prediction."
)

st.caption(
    "Research prototype using the Home Credit Default Risk dataset. "
    "The prediction is an estimate, not an automatic lending decision."
)

st.divider()


# ============================================================
# 7. SELECT APPLICANT
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
# 8. PREDICTION
# ============================================================

try:
    probabilities = model.predict_proba(X_applicant)
    default_probability = float(probabilities[0][1])
    repayment_probability = float(probabilities[0][0])
except Exception as error:
    st.error(f"Prediction failed: {error}")
    st.stop()

if default_probability >= THRESHOLD:
    risk_level = "High Risk"
    risk_message = (
        "The model estimates a higher likelihood of payment difficulties."
    )
else:
    risk_level = "Low Risk"
    risk_message = (
        "The model estimates a lower likelihood of payment difficulties."
    )

st.subheader("Prediction Results")

col1, col2, col3 = st.columns(3)

col1.metric("Default Probability", f"{default_probability:.2%}")
col2.metric("Repayment Probability", f"{repayment_probability:.2%}")
col3.metric("Risk Classification", risk_level)

if default_probability >= THRESHOLD:
    st.error(risk_message)    # red for high risk
else:
    st.success(risk_message)  # green for low risk

st.caption(
    "The classification uses a fixed 0.50 threshold. "
    "The probability is the model's estimate for the dataset's TARGET class."
)

st.divider()


# ============================================================
# 9. CREATE SHAP EXPLANATION
# ============================================================

st.header("Main Reasons for This Decision")
st.caption("SHAP explanation")

try:
    shap_explainer = shap.TreeExplainer(model)
    shap_explanation = shap_explainer(X_applicant)

    shap_contributions = np.asarray(
        shap_explanation.values
    )

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

    # Keep the three strongest factors pushing toward the
    # model's final risk classification.
    if default_probability >= THRESHOLD:
        relevant_shap = [
            i for i, value in enumerate(shap_contributions)
            if value > 0
        ]
        opposite_shap = [
            i for i, value in enumerate(shap_contributions)
            if value < 0
        ]
    else:
        relevant_shap = [
            i for i, value in enumerate(shap_contributions)
            if value < 0
        ]
        opposite_shap = [
            i for i, value in enumerate(shap_contributions)
            if value > 0
        ]

    relevant_shap = sorted(
        relevant_shap,
        key=lambda i: abs(shap_contributions[i]),
        reverse=True
    )

    opposite_shap = sorted(
        opposite_shap,
        key=lambda i: abs(shap_contributions[i]),
        reverse=True
    )

    st.caption(
        "Above / below typical compares the applicant's value with the "
        "training-data median. Yes/no application flags are shown as "
        "present or not present. Education is shown as the recorded "
        "category code, not as a level of education."
    )

    lower_risk_indices = [
        i for i in (relevant_shap + opposite_shap)
        if shap_contributions[i] < 0
    ]
    higher_risk_indices = [
        i for i in (relevant_shap + opposite_shap)
        if shap_contributions[i] > 0
    ]
    lower_risk_indices = sorted(
        lower_risk_indices, key=lambda i: abs(shap_contributions[i]),
        reverse=True
    )
    higher_risk_indices = sorted(
        higher_risk_indices, key=lambda i: abs(shap_contributions[i]),
        reverse=True
    )

    applicant_row = X_applicant.iloc[0]

    lower_group = (
        "Factors supporting lower risk",
        lower_risk_indices
    )
    higher_group = (
        "Factors raising risk",
        higher_risk_indices
    )

    # Show the group that matches the model's classification first.
    if default_probability >= THRESHOLD:
        groups = [higher_group, lower_group]
    else:
        groups = [lower_group, higher_group]

    for title, indices in groups:
        show_shap_group(
            title,
            indices,
            shap_contributions,
            applicant_row,
            train_medians,
            feature_columns,
            TOP_REASONS
        )

    # Detailed SHAP chart and table are optional for technical users.
    with st.expander("View detailed SHAP explanation"):

        st.caption(
            "The SHAP values explain contributions to the model output "
            "(raw log-odds for this XGBoost model), not direct percentage "
            "changes in probability."
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
            st.warning(f"Could not display the SHAP waterfall chart: {chart_error}")

        shap_table = pd.DataFrame({
            "Feature": feature_columns,
            "Applicant value": X_applicant.iloc[0].values,
            "SHAP contribution": shap_contributions
        })

        shap_table["Absolute contribution"] = (
            shap_table["SHAP contribution"].abs()
        )

        shap_table = shap_table.sort_values(
            "Absolute contribution",
            ascending=False
        )

        st.dataframe(
            shap_table.drop(columns=["Absolute contribution"]),
            use_container_width=True,
            hide_index=True
        )

except Exception as error:
    st.error(f"SHAP explanation could not be generated: {error}")
    shap_explanation = None
    shap_contributions = None


st.divider()


# ============================================================
# 10. CREATE LIME EXPLANATION
# ============================================================

st.header("Additional Factors Identified by LIME")
st.caption("LIME explanation")

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
        "LIME examines the selected applicant locally and may identify "
        "different factors from SHAP."
    )

    lime_lower, lime_higher = [], []
    seen_features = set()

    for condition, weight in lime_results:
        feature = lime_feature_from_condition(condition, feature_columns)

        if feature is None or feature in seen_features or weight == 0:
            continue

        seen_features.add(feature)
        feature_index = feature_columns.index(feature)

        if weight > 0:
            lime_higher.append(feature_index)
        else:
            lime_lower.append(feature_index)

    lime_lower_group = ("Factors supporting lower risk", lime_lower)
    lime_higher_group = ("Factors raising risk", lime_higher)

    # Show the group that matches the model's classification first.
    if default_probability >= THRESHOLD:
        lime_groups = [lime_higher_group, lime_lower_group]
    else:
        lime_groups = [lime_lower_group, lime_higher_group]

    for title, indices in lime_groups:
        show_shap_group(
            title,
            indices,
            None,
            X_applicant.iloc[0],
            train_medians,
            feature_columns,
            TOP_REASONS
        )

    with st.expander("View detailed LIME explanation"):

        st.caption(
            "Positive LIME weights support class 1 "
            "(payment difficulties); negative weights support class 0 "
            "(no payment difficulties). These weights are not probabilities."
        )

        lime_table = pd.DataFrame(
            lime_results,
            columns=["Feature condition", "LIME contribution"]
        )

        lime_table["Interpretation"] = lime_table[
            "LIME contribution"
        ].apply(
            lambda value: (
                "Supports higher risk"
                if value > 0
                else "Supports lower risk"
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
    st.error(f"LIME explanation could not be generated: {error}")


st.divider()


# ============================================================
# 11. ACTUAL TEST-DATA OUTCOME
# ============================================================

st.header("Actual Applicant Outcome")

st.caption(
    "Shown only for retrospective evaluation of the held-out test data."
)

if "TARGET" in test_df.columns:
    actual_target = test_df.loc[applicant_index, "TARGET"]

    if pd.notna(actual_target):
        if int(actual_target) == 1:
            st.warning(
                "Actual outcome: Payment difficulties were recorded."
            )
        else:
            st.success(
                "Actual outcome: No payment difficulties were recorded."
            )
    else:
        st.write("The actual outcome is not available.")
else:
    st.write("The TARGET column is not available in this dataset.")


# ============================================================
# 12. ABOUT THE DASHBOARD
# ============================================================

with st.expander("About this assessment"):

    st.write(
        """
        This dashboard is a research prototype based on the Home Credit
        Default Risk dataset.

        The XGBoost model estimates the likelihood of the dataset's
        TARGET outcome. In the original dataset, TARGET = 1 represents
        a client with payment difficulties, while TARGET = 0 represents
        other cases.

        SHAP explains feature contributions using the trained model.
        LIME provides a local approximation around the selected applicant.
        The two methods may identify different factors.

        Explanations describe how the model used the available data.
        They do not prove that a factor caused payment difficulties.
        The dashboard should not be used as the sole basis for a real
        lending decision.
        """
)
