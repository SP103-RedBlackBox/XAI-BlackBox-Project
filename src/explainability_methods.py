import shap
import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer
from turtle import st

# Loads saved model
@st.cache_resource
def load_model():
    saved_model = joblib.load('xgb_pipeline.joblib')
    model = saved_model.named_steps['model']

    return model

# Load SHAP and LIME explainers
def load_SHAP_explainer(model, X_processed):
    explainer = shap.TreeExplainer(model, X_processed, model_output='probability', feature_names=model.get_feature_names_out())
    return explainer

def load_LIME_explainer(model, X_processed):
    explainer = LimeTabularExplainer(
    training_data=X_processed,
    feature_names=model.get_feature_names_out(),
    class_names=['<=50K', '>50K'],
    mode='classification'
)
    return explainer

# Get SHAP values
def get_SHAP_values(explainer, X_processed):
    shap_values = explainer.shap_values(X_processed)
    return shap_values

# Summary Plot
def SHAP_summary_plot(shap_values, X_processed, feature_names):
    shap.summary_plot(shap_values, X_processed, feature_names=feature_names)

# Waterfall Plot
def SHAP_waterfall_plot(shap_values, index):
    shap.plots.waterfall(shap_values[index], max_display=10)

# Force Plot
def SHAP_force_plot(shap_value, index):
    shap.plots.force(shap_value[index])

# LIME explanation  
def LIME_explanation(explainer, index, X_processed, model):
    lime_exp = explainer.explain_instance(
    data_row=X_processed[index],
    predict_fn=model.predict_proba,
    num_features=10
    )

    print('\nLIME Explantion: ')
    for feature, weight in lime_exp.as_list():
        print(f'{feature}: {weight: .4f}')

    fig = lime_exp.as_pyplot_figure()
    plt.tight_layout()
    plt.show()    


