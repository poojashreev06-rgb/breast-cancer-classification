
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="centered"
)

# Load trained model
model = joblib.load("random_forest_model.pkl")

st.title("🎗️ Breast Cancer Classification")
st.write(
    "An educational machine learning demonstration "
    "using the Breast Cancer Wisconsin Diagnostic dataset."
)

st.warning(
    "For educational purposes only. This app is not a "
    "medical diagnostic tool. Please consult a qualified "
    "healthcare professional for medical advice."
)

st.subheader("Enter the 30 dataset features")

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean",
    "area_mean", "smoothness_mean", "compactness_mean",
    "concavity_mean", "concave_points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se",
    "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se",
    "symmetry_se", "fractal_dimension_se", "radius_worst",
    "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst",
    "fractal_dimension_worst"
]

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
    input_data = pd.DataFrame([values], columns=feature_names)
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 0:
        st.success("Model prediction: Benign")
    else:
        st.error("Model prediction: Malignant")

    st.caption(
        "This output is a model prediction, not a medical diagnosis."
    )