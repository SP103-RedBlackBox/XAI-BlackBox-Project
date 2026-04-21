import shap
import lime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer

def load_model():
    return joblib.load('xgb_pipeline.joblib')

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


