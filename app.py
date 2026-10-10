import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide"
)

model = joblib.load("random_forest_model.pkl")
feature_names = list(model.feature_names_in_)

sample_values = [
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
    184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

st.title("Breast Cancer Classification System")
st.write("### Machine Learning Demonstration")
st.caption("Wisconsin Diagnostic Dataset • Random Forest Classifier")

st.warning(
    "This application is for educational purposes only. "
    "It is not a medical diagnostic tool."
)

st.divider()

st.subheader("Submit Feature Information")
st.write(
    "Try the sample dataset row, or enter all 30 numerical features manually."
)

mode = st.radio(
    "Select input method",
    ["Try Example Sample", "Enter Features Manually"],
    horizontal=True
)

def display_result(values):
    data = pd.DataFrame([values], columns=feature_names)
    prediction = model.predict(data)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("Model Prediction: Benign")
    else:
        st.error("Model Prediction: Malignant")

    st.caption(
        "This is a machine-learning prediction for demonstration only, "
        "not a medical diagnosis."
    )

if mode == "Try Example Sample":
    st.info("Example values from the public educational dataset.")

    if st.button("Run Example Prediction", type="primary"):
        display_result(sample_values)

else:
    st.write("Enter the feature values below.")

    with st.form("feature_form"):
        values = []

        for start in range(0, len(feature_names), 3):
            columns = st.columns(3)

            for column, feature in zip(
                columns, feature_names[start:start + 3]
            ):
                with column:
                    value = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=0.0,
                        value=0.0,
                        format="%.6f"
                    )
                    values.append(value)

        submitted = st.form_submit_button(
            "Submit Feature Information",
            type="primary"
        )

    if submitted:
        display_result(values)

st.divider()
st.caption("Academic Project | Breast Cancer Classification using Machine Learning")
