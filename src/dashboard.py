import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
import os

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("Credit Risk Probability Model Dashboard")

# Load processed data
data_path = "Data/processed/processed_data.csv"
df = pd.read_csv(data_path)

# Load model (adjust path as needed)
model_path = "artifacts/model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found. Please train and save the model first.")
    st.stop()

# Sidebar for customer selection
if "CustomerId" in df.columns:
    customer_ids = df["CustomerId"].unique()
    selected_id = st.sidebar.selectbox("Select Customer ID", customer_ids)
    customer_data = df[df["CustomerId"] == selected_id]
else:
    st.warning("CustomerId column not found in data.")
    customer_data = df.iloc[[0]]

# Display model performance metrics
st.subheader("Model Performance")
roc_path = "artifacts/model_evaluation/RandomForestClassifier_roc_curve.png"
cm_path = "artifacts/model_evaluation/RandomForestClassifier_confusion_matrix.png"
if os.path.exists(roc_path):
    st.image(roc_path, caption="ROC Curve")
if os.path.exists(cm_path):
    st.image(cm_path, caption="Confusion Matrix")

# SHAP explanations
st.subheader("Model Explainability (SHAP)")
feature_cols = [col for col in df.columns if col not in ["CustomerId", "is_high_risk"]]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df[feature_cols])

fig_summary = plt.figure()
shap.summary_plot(shap_values, df[feature_cols], show=False)
st.pyplot(fig_summary)

st.subheader("Customer-level Explanation")
if len(customer_data) > 0:
    idx = customer_data.index[0]
    fig_force = shap.force_plot(explainer.expected_value[1], shap_values[1][idx], customer_data[feature_cols], matplotlib=True)
    st.pyplot(fig_force)

# Prediction interface
st.subheader("Prediction")
prediction = model.predict(customer_data[feature_cols])
st.write(f"Predicted Risk: {'High' if prediction[0] == 1 else 'Low'}")

st.success("Dashboard loaded successfully. Project is ready for stakeholder review.")
