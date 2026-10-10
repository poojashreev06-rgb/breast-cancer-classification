import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="centered"
)

model = joblib.load("random_forest_model.pkl")

st.title("🎗️ Breast Cancer Classification")
st.write("An educational machine learning project using the Wisconsin Diagnostic dataset.")

st.warning(
    "Educational use only. This app is not a medical diagnostic tool. "
    "Please consult a qualified healthcare professional for medical advice."
)

# Feature names in the same order used to train the model
feature_names = list(model.feature_names_in_)

# Example row from the public educational dataset
sample_values = [
    17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
    0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
    184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
]

choice = st.radio(
    "Choose how to test the model:",
    ["Try example sample", "Enter values manually"]
)

def show_prediction(values):
    input_data = pd.DataFrame([values], columns=feature_names)
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")
    if prediction == 0:
        st.success("Model prediction: Benign")
    else:
        st.error("Model prediction: Malignant")

    st.caption("This is a model prediction, not a medical diagnosis.")

if choice == "Try example sample":
    st.info("This is an example from the public dataset, not personal medical data.")

    if st.button("Run Example Prediction"):
        show_prediction(sample_values)

else:
    st.subheader("Enter the 30 dataset features")

    with st.form("prediction_form"):
        values = []

        for feature in feature_names:
            value = st.number_input(
                feature.replace("_", " ").title(),
                min_value=0.0,
                value=0.0,
                format="%.6f"
            )
            values.append(value)

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        show_prediction(values)
