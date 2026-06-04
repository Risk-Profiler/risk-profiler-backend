from pathlib import Path

import joblib
import shap

BASE_DIR = Path(__file__).resolve().parent.parent

pipeline = joblib.load(BASE_DIR / "models" / "random_forest_model.joblib")
rf_model = pipeline.named_steps["model"]
explainer = shap.TreeExplainer(rf_model)

feature_scaler = joblib.load(BASE_DIR / "models" / "feature_scaler.joblib")
expected_features = joblib.load(BASE_DIR / "models" / "features_list.joblib")
