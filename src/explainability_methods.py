import shap
import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
import streamlit as st
from pathlib import Path

# Loads saved model
@st.cache_resource
def load_model_preprocessor():
    root_dir = Path(__file__).resolve().parent.parent
    model_path = root_dir / "models" / "xgb_pipeline.joblib"
    saved_model = joblib.load(model_path)
    model = saved_model.named_steps['model']
    preprocessor = saved_model.named_steps['preprocessor']

    return model, preprocessor

# Load SHAP and LIME explainers
def load_SHAP_explainer(model, preprocessor, X_preprocessed):
    root_dir = Path(__file__).resolve().parent.parent
    background = joblib.load(root_dir / 'data' / 'background_sample.joblib')
    explainer = shap.TreeExplainer(model, background, model_output='probability', feature_names=preprocessor.get_feature_names_out())
    return explainer

def load_LIME_explainer(model, preprocessor):
    root_dir = Path(__file__).resolve().parent.parent
    background = joblib.load(root_dir / 'data' / 'background_sample.joblib')
    explainer = LimeTabularExplainer(
    training_data=background,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=['<=50K', '>50K'],
    mode='classification'
)
    return explainer

# Get SHAP values
def get_SHAP_values(explainer, X_processed):
    shap_values = explainer(X_processed)
    return shap_values

# Summary Plot
def SHAP_summary_plot(shap_values, X_processed, feature_names):
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names)

    st.pyplot(fig)
    plt.close(fig)

# Waterfall Plot
def SHAP_waterfall_plot(shap_values, index):
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[index], max_display=10)

    st.pyplot(fig)
    plt.close(fig)

# Force Plot
def SHAP_force_plot(shap_value, index):
    shap.plots.force(shap_value[index], matplotlib=True, show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close(fig)

# LIME explanation  
def LIME_explanation(explainer, index, X_processed, model):
    lime_exp = explainer.explain_instance(
    data_row=X_processed[index],
    predict_fn=model.predict_proba,
    num_features=10
    )

    fig = lime_exp.as_pyplot_figure()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)    

    lime_data = pd.DataFrame(lime_exp.as_list(), columns=['Feature', 'Weight'])
    st.dataframe(lime_data)


