import os
from pathlib import Path
import joblib
import shap
import mlflow.sklearn
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

# Configure MLflow tracking
mlflow.set_tracking_uri(f"sqlite:///{BASE_DIR}/mlflow.db")

print("Initializing ML artifacts...")

try:
    # Try loading the dynamic model marked as "Production" from the local MLflow registry
    model_uri = "models:/risk_profiler_model/Production"
    pipeline = mlflow.sklearn.load_model(model_uri)
    print("Successfully loaded 'Production' model from MLflow Model Registry.")
except Exception as e:
    print(f"MLflow model registry unavailable or production model not found ({e}).")
    print("Falling back to loading model locally from models/random_forest_model.joblib.")
    pipeline = joblib.load(BASE_DIR / "models" / "random_forest_model.joblib")

# Custom SHAP Explainer Ensemble for CalibratedClassifierCV
class EnsembleTreeExplainer:
    def __init__(self, calibrated_model):
        self.explainers = []
        for clf in calibrated_model.calibrated_classifiers_:
            model = clf.estimator.named_steps['model']
            self.explainers.append(shap.TreeExplainer(model))
            
    def __call__(self, X):
        # Obtain SHAP values from all individual fold explainers
        shap_values_all = [exp(X) for exp in self.explainers]
        # Standard format returns a shap.Explanation object. We average its .values
        avg_explanation = shap_values_all[0]
        avg_values = np.mean([exp.values for exp in shap_values_all], axis=0)
        avg_explanation.values = avg_values
        return avg_explanation

# Determine model structure to instantiate appropriate explainer
if hasattr(pipeline, "calibrated_classifiers_"):
    print("Instantiating EnsembleTreeExplainer for CalibratedClassifierCV.")
    explainer = EnsembleTreeExplainer(pipeline)
else:
    print("Instantiating standard TreeExplainer for single pipeline model.")
    if hasattr(pipeline, "named_steps"):
        underlying_model = pipeline.named_steps["model"]
    else:
        underlying_model = pipeline
    explainer = shap.TreeExplainer(underlying_model)

# Load feature scaler and expected features configuration
feature_scaler = joblib.load(BASE_DIR / "models" / "feature_scaler.joblib")
expected_features = joblib.load(BASE_DIR / "models" / "features_list.joblib")

print("ML artifacts initialized successfully.")
