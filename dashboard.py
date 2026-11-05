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

# Expected structure:
# project/
#   dashboard/
#       app.py
#   outputs/models/xgboost.joblib
#   data/processed/v1_train.csv
#   data/processed/v1_test.csv

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(APP_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "xgboost.joblib")
TRAIN_PATH = os.path.join(BASE_DIR, "data", "processed", "v1_train.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "processed", "v1_test.csv")

THRESHOLD = 0.50
BACKGROUND_SIZE = 1000
TOP_REASONS = 3
RANDOM_STATE = 42

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
    """Convert a model feature name into a short, employee-friendly label."""
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


def describe_applicant_value(feature, applicant_value, train_median,
                             binary_features):
    """Describe an applicant's value relative to the training data."""
    label = readable_feature_name(feature)

    if pd.isna(applicant_value):
        position = "is missing"
    elif feature in binary_features:
        position = "is present" if applicant_value >= 0.5 else "is not present"
    elif pd.isna(train_median) or np.isclose(applicant_value, train_median):
        position = "is typical"
    elif applicant_value > train_median:
        position = "is above typical"
    else:
        position = "is below typical"

    return f"{label} {position}."


def show_shap_group(title, indices, shap_values, applicant_row, medians,
                    feature_names, binary_features, top_n):
    """Display SHAP factors, whose signs refer to class 1 (payment difficulties)."""
    st.markdown(f"**{title}**")

    if not indices:
        st.write("No factors were identified in this group.")
        return

    for i in indices[:top_n]:
        feature = feature_names[i]
        st.markdown(
            "- " + describe_applicant_value(
                feature,
                applicant_row[feature],
                medians[feature],
                binary_features
            )
        )


def lime_feature_from_condition(condition, feature_names):
    """Find the original feature name within a LIME condition string."""
    ordered = sorted(feature_names, key=len, reverse=True)

    for feature in ordered:
        if condition.startswith(feature):
            return feature

    # LIME may put a numeric range before the feature name.
    for feature in ordered:
        if feature in condition:
            return feature

    return None


def show_lime_group(title, items, feature_names, top_n):
    """
    Display LIME's own direction, not the applicant's position relative
    to the training median. Positive weights support class 1; negative
    weights support class 0.
    """
    st.markdown(f"**{title}**")

    if not items:
        st.write("No factors were identified in this group.")
        return

    for feature, weight in items[:top_n]:
        label = readable_feature_name(feature)

        if weight > 0:
            direction = "supported higher estimated risk"
        else:
            direction = "supported lower estimated risk"

        st.markdown(f"- {label} — {direction}.")


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


required_files = {
    "XGBoost model": MODEL_PATH,
    "Processed training data": TRAIN_PATH,
    "Processed test data": TEST_PATH,
}

missing_files = [
    f"{description}: {path}"
    for description, path in required_files.items()
    if not os.path.exists(path)
]

if missing_files:
    st.error(
        "One or more required project files could not be found:\n\n"
        + "\n\n".join(missing_files)
        + "\n\nCheck that app.py is inside the dashboard folder and that "
          "the model and CSV files are in the expected project folders."
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
    st.error(
        "These training features are missing from the test data: "
        f"{missing_in_test}"
    )
    st.stop()

X_train = train_df[feature_columns].copy()
X_test = test_df[feature_columns].copy()

# Keep exactly the same feature order used by the training data.
X_test = X_test[feature_columns]

try:
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
except (ValueError, TypeError) as error:
    st.error(
        "The processed data contains non-numeric feature values. "
        f"Check the preprocessing output. Details: {error}"
    )
    st.stop()

if X_train.empty or X_test.empty:
    st.error("The processed training or test dataset is empty.")
    st.stop()

train_medians = X_train.median()

# Binary features are described as present/not present.
binary_features = {
    column for column in feature_columns
    if set(X_train[column].dropna().unique()).issubset({0.0, 1.0})
}

if hasattr(model, "n_features_in_"):
    if model.n_features_in_ != len(feature_columns):
        st.error(
            f"The model expects {model.n_features_in_} features, "
            f"but the processed data contains {len(feature_columns)}."
        )
        st.stop()

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != feature_columns:
        st.error(
            "The saved model's feature names or order do not match the "
            "processed training data. Use the same feature preparation "
            "and column order used when the model was trained."
        )
        st.stop()


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

    input_method = st.radio(
        "How would you like to select the applicant?",
        ["Choose from dropdown", "Enter Applicant ID"],
        horizontal=True
    )

    if input_method == "Choose from dropdown":

        applicant_ids = test_df["SK_ID_CURR"].dropna().tolist()

        selected_id = st.selectbox(
            "Select Applicant ID",
            applicant_ids
        )

    else:

        typed_id = st.text_input(
            "Enter Applicant ID",
            placeholder="For example: 371134"
        ).strip()

        if not typed_id:
            st.info("Enter an Applicant ID to continue.")
            st.stop()

        try:
            selected_id = int(typed_id)

        except ValueError:
            st.error("Please enter a valid numeric Applicant ID.")
            st.stop()

    matching_rows = test_df.index[
        test_df["SK_ID_CURR"] == selected_id
    ].tolist()

    if not matching_rows:
        st.error(
            f"Applicant ID {selected_id} was not found "
            "in the test dataset."
        )
        st.stop()

    applicant_index = matching_rows[0]

    X_applicant = X_test.loc[[applicant_index]]

    st.caption(f"Selected Applicant ID: {selected_id}")

else:
    st.error("SK_ID_CURR was not found in the test dataset.")
    st.stop()
# ============================================================
# 8. PREDICTION
# ============================================================

try:
    probabilities = model.predict_proba(X_applicant)

    if probabilities.shape[1] < 2:
        raise ValueError("The model did not return two class probabilities.")

    # This project uses TARGET=1 for payment difficulties.
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
col1.metric("Payment Difficulties Probability", f"{default_probability:.2%}")
col2.metric("No Payment Difficulties Probability", f"{repayment_probability:.2%}")
col3.metric("Risk Classification", risk_level)

if default_probability >= THRESHOLD:
    st.error(risk_message)
else:
    st.success(risk_message)

st.progress(
    min(max(default_probability, 0.0), 1.0),
    text=f"Estimated payment-difficulties probability: {default_probability:.1%}"
)

st.caption(
    "The classification uses a fixed 0.50 threshold. The probability is "
    "the model's estimate for TARGET=1 (payment difficulties)."
)

st.divider()


# ============================================================
# 9. CREATE SHAP EXPLANATION
# ============================================================

st.header("Main Reasons for This Prediction")
st.caption("SHAP local explanation")

shap_explanation = None
shap_contributions = None

try:
    shap_explainer = shap.TreeExplainer(model)
    shap_explanation = shap_explainer(X_applicant)

    shap_contributions = np.asarray(shap_explanation.values)

    # Handle SHAP versions that return an output dimension.
    if shap_contributions.ndim == 3:
        shap_contributions = shap_contributions[:, :, -1]

    shap_contributions = shap_contributions.reshape(-1)

    if len(shap_contributions) != len(feature_columns):
        raise ValueError(
            "SHAP returned a different number of feature contributions "
            "than the model input contains."
        )

    # For this binary classifier, positive SHAP values support class 1
    # (payment difficulties); negative values support class 0.
    higher_risk_indices = [
        i for i, value in enumerate(shap_contributions) if value > 0
    ]
    lower_risk_indices = [
        i for i, value in enumerate(shap_contributions) if value < 0
    ]

    higher_risk_indices.sort(
        key=lambda i: abs(shap_contributions[i]),
        reverse=True
    )
    lower_risk_indices.sort(
        key=lambda i: abs(shap_contributions[i]),
        reverse=True
    )

    st.caption(
        "SHAP direction: positive contributions support payment difficulties "
        "(higher risk), while negative contributions support no payment "
        "difficulties (lower risk). The value descriptions compare the "
        "applicant with the training-data median; they do not describe "
        "the direction of the SHAP effect."
    )

    applicant_row = X_applicant.iloc[0]

    shap_groups = [
        ("Factors raising estimated risk", higher_risk_indices),
        ("Factors supporting lower estimated risk", lower_risk_indices),
    ]

    if default_probability < THRESHOLD:
        shap_groups.reverse()

    for title, indices in shap_groups:
        show_shap_group(
            title=title,
            indices=indices,
            shap_values=shap_contributions,
            applicant_row=applicant_row,
            medians=train_medians,
            feature_names=feature_columns,
            binary_features=binary_features,
            top_n=TOP_REASONS
        )

    with st.expander("View detailed SHAP explanation"):
        st.caption(
            "For this XGBoost model, SHAP values may be expressed in raw "
            "model-output units (log-odds), not direct percentage changes "
            "in probability."
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
            st.warning(
                f"Could not display the SHAP waterfall chart: {chart_error}"
            )

        shap_table = pd.DataFrame({
            "Feature": feature_columns,
            "Applicant value": X_applicant.iloc[0].values,
            "SHAP contribution": shap_contributions
        })

        shap_table["Readable feature"] = shap_table["Feature"].apply(
            readable_feature_name
        )
        shap_table["Absolute contribution"] = (
            shap_table["SHAP contribution"].abs()
        )

        shap_table = shap_table.sort_values(
            "Absolute contribution",
            ascending=False
        )

        st.dataframe(
            shap_table[
                [
                    "Readable feature",
                    "Feature",
                    "Applicant value",
                    "SHAP contribution"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

except Exception as error:
    st.error(f"SHAP explanation could not be generated: {error}")


st.divider()


# ============================================================
# 10. CREATE LIME EXPLANATION
# ============================================================

st.header("Additional Factors Identified by LIME")
st.caption("LIME local explanation")

try:
    background = X_train.sample(
        n=min(BACKGROUND_SIZE, len(X_train)),
        random_state=RANDOM_STATE
    )

    lime_explainer = LimeTabularExplainer(
        training_data=background.to_numpy(),
        feature_names=feature_columns,
        class_names=["No payment difficulties", "Payment difficulties"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    def lime_predict_function(samples):
        """Return model probabilities in the same class order as LIME."""
        samples_df = pd.DataFrame(samples, columns=feature_columns)
        return model.predict_proba(samples_df)

    lime_explanation = lime_explainer.explain_instance(
        data_row=X_applicant.iloc[0].to_numpy(),
        predict_fn=lime_predict_function,
        num_features=min(10, len(feature_columns)),
        labels=(1,)
    )

    lime_results = lime_explanation.as_list(label=1)

    # Sort by absolute local contribution.
    lime_results = sorted(
        lime_results,
        key=lambda item: abs(item[1]),
        reverse=True
    )

    st.write(
        "LIME builds a local approximation around the selected applicant. "
        "It may identify different factors from SHAP."
    )

    lime_higher = []
    lime_lower = []
    seen_features = set()

    for condition, weight in lime_results:
        feature = lime_feature_from_condition(condition, feature_columns)

        if feature is None or feature in seen_features or weight == 0:
            continue

        seen_features.add(feature)

        if weight > 0:
            lime_higher.append((feature, float(weight)))
        else:
            lime_lower.append((feature, float(weight)))

    lime_higher.sort(key=lambda item: abs(item[1]), reverse=True)
    lime_lower.sort(key=lambda item: abs(item[1]), reverse=True)

    lime_groups = [
        ("Factors raising estimated risk", lime_higher),
        ("Factors supporting lower estimated risk", lime_lower),
    ]

    if default_probability < THRESHOLD:
        lime_groups.reverse()

    for title, items in lime_groups:
        show_lime_group(
            title=title,
            items=items,
            feature_names=feature_columns,
            top_n=TOP_REASONS
        )

    with st.expander("View detailed LIME explanation"):
        st.caption(
            "Positive LIME weights support class 1 (payment difficulties). "
            "Negative weights support class 0 (no payment difficulties). "
            "LIME weights are local surrogate-model contributions, not "
            "probabilities or causal effects."
        )

        lime_table = pd.DataFrame(
            lime_results,
            columns=["Feature condition", "LIME contribution"]
        )

        lime_table["Direction"] = lime_table["LIME contribution"].apply(
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
            lime_figure = lime_explanation.as_pyplot_figure(label=1)
            st.pyplot(lime_figure)
            plt.close(lime_figure)
        except Exception as chart_error:
            st.warning(
                f"Could not display the LIME chart: {chart_error}"
            )

except Exception as error:
    st.error(f"LIME explanation could not be generated: {error}")


st.divider()


# ============================================================
# 11. ACTUAL TEST-DATA OUTCOME
# ============================================================

st.header("Actual Applicant Outcome")

st.caption(
    "This information is shown only for retrospective evaluation because "
    "the selected applicant comes from the held-out test dataset."
)

if "TARGET" in test_df.columns:
    actual_target = test_df.loc[applicant_index, "TARGET"]

    if pd.notna(actual_target):
        if int(actual_target) == 1:
            st.warning(
                "Actual outcome: Payment difficulties were recorded."
            )
        elif int(actual_target) == 0:
            st.success(
                "Actual outcome: No payment difficulties were recorded."
            )
        else:
            st.write(f"Unexpected TARGET value: {actual_target}")
    else:
        st.write("The actual outcome is not available for this applicant.")
else:
    st.info(
        "The TARGET column is not available. The actual outcome cannot "
        "be displayed."
    )


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
