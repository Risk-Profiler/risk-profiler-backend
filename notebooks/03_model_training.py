#%%
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC

from pathlib import Path

# =========================================================
# LOAD DATA
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
X_train = pd.read_csv(PROJECT_ROOT / 'data/processed/X_train.csv')
y_train = pd.read_csv(PROJECT_ROOT / 'data/processed/y_train.csv').values.ravel()
X_test  = pd.read_csv(PROJECT_ROOT / 'data/processed/X_test.csv')
y_test  = pd.read_csv(PROJECT_ROOT / 'data/processed/y_test.csv').values.ravel()

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"\nDistribusi y_train:\n{pd.Series(y_train).value_counts().sort_index()}")

# =========================================================
# SETUP MLFLOW
# =========================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Risk-Profiler")

# =========================================================
# DEFINE ESTIMATORS & PIPELINES
# =========================================================

# Identify categorical features for SMOTENC
categorical_cols = [
    'pln_delay_days_isna', 'ecommerce_rating_isna', 'pdam_bill_avg_isna', 
    'pdam_late_payments_isna', 'business_category_fnb', 'business_category_jasa', 
    'business_category_retail', 'chronic_pln_delay', 'has_both_utilities'
]
cat_indices = [X_train.columns.get_loc(col) for col in categorical_cols if col in X_train.columns]

rf_pipeline = Pipeline([
    ('smotenc', SMOTENC(categorical_features=cat_indices, random_state=42)),
    ('model', RandomForestClassifier(
        class_weight     = 'balanced',
        random_state     = 42,
        n_jobs           = -1
    ))
])

rf_param_grid = {
    'model__max_depth': [3, 5, 7, 10],
    'model__min_samples_leaf': [2, 5, 10],
    'model__min_samples_split': [5, 7, 10],
    'model__n_estimators': [100, 200, 300]
}

# XGBoost Monotonic Constraints:
# - qris_volume_monthly: -1 (higher volume, lower risk)
# - pln_delay_days: 1 (higher delay, higher risk)
# - pdam_late_payments: 1 (higher late payments, higher risk)
monotone_constraints = {
    'qris_volume_monthly': -1,
    'pln_delay_days': 1,
    'pdam_late_payments': 1
}

xgb_pipeline = Pipeline([
    ('smotenc', SMOTENC(categorical_features=cat_indices, random_state=42)),
    ('model', XGBClassifier(
        monotone_constraints=monotone_constraints,
        random_state=42,
        n_jobs=-1
    ))
])

xgb_param_grid = {
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__n_estimators': [100, 200, 300]
}

models_to_train = [
    {
        "name": "Random Forest",
        "pipeline": rf_pipeline,
        "param_distributions": rf_param_grid
    },
    {
        "name": "XGBoost",
        "pipeline": xgb_pipeline,
        "param_distributions": xgb_param_grid
    }
]

# =========================================================
# HELPER FOR MULTICLASS BRIER SCORE
# =========================================================

def multiclass_brier_score(y_true, y_prob):
    lb = LabelBinarizer()
    y_true_bin = lb.fit_transform(y_true)
    if y_true_bin.shape[1] == 1:
        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
    return np.mean(np.sum((y_prob - y_true_bin) ** 2, axis=1))

# =========================================================
# TRAIN, EVALUATE & TRACK MODELS
# =========================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
best_f1 = -1
best_model = None
best_model_name = ""
best_run_id = ""

for model_info in models_to_train:
    name = model_info["name"]
    pipeline = model_info["pipeline"]
    param_distributions = model_info["param_distributions"]
    
    print(f"\n=== TRAINING & CALIBRATING {name.upper()} ===")
    
    with mlflow.start_run(run_name=name) as run:
        # 1. Hyperparameter Tuning using RandomizedSearchCV
        print(f"Running RandomizedSearchCV for {name}...")
        search = RandomizedSearchCV(
            pipeline,
            param_distributions,
            n_iter=5, # Keep low for faster iteration
            cv=skf,
            scoring='f1_macro',
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_train, y_train)
        
        best_pipeline = search.best_estimator_
        
        mlflow.log_params(search.best_params_)
        print(f"Best params    : {search.best_params_}")
        print(f"CV F1 Mean     : {search.best_score_:.4f}")
        mlflow.log_metric("CV_F1_Mean", search.best_score_)
        
        # 2. Fit probability calibrated classifier
        print(f"Calibrating best {name} model...")
        calibrated_model = CalibratedClassifierCV(
            estimator=best_pipeline,
            method='sigmoid',
            cv=skf
        )
        calibrated_model.fit(X_train, y_train)
        
        # 4. Evaluate on test set
        y_pred = calibrated_model.predict(X_test)
        y_prob = calibrated_model.predict_proba(X_test)
        
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='macro')
        test_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        test_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        test_brier = multiclass_brier_score(y_test, y_prob)
        
        print(f"Test Accuracy  : {test_acc:.4f}")
        print(f"Test F1 (Macro): {test_f1:.4f}")
        print(f"Test Brier     : {test_brier:.4f}")
        
        mlflow.log_metric("Test_Accuracy", test_acc)
        mlflow.log_metric("Test_F1_Macro", test_f1)
        mlflow.log_metric("Test_Precision_Macro", test_prec)
        mlflow.log_metric("Test_Recall_Macro", test_rec)
        mlflow.log_metric("Test_Brier_Calibration", test_brier)
        
        # 5. Log calibrated model
        mlflow.sklearn.log_model(
            sk_model=calibrated_model,
            artifact_path="model",
            serialization_format="pickle"
        )
        
        print("Generating SHAP analysis...")
        shap_values_list = []
        for calibrated_clf in calibrated_model.calibrated_classifiers_:
            underlying_model = calibrated_clf.estimator.named_steps['model']
            explainer = shap.TreeExplainer(underlying_model)
            shap_values_list.append(explainer(X_test).values)
            
        avg_shap_values = np.mean(shap_values_list, axis=0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # SHAP Summary Plot for High Risk (class 2)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                avg_shap_values[:, :, 2],
                X_test,
                title=f'SHAP — Feature Importance for High Risk ({name})',
                show=False
            )
            plt.tight_layout()
            shap_plot_path = os.path.join(temp_dir, "shap_summary_class_2.png")
            plt.savefig(shap_plot_path)
            plt.close()
            mlflow.log_artifact(shap_plot_path)
            
            # SHAP Bar Plot for all classes
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                avg_shap_values,
                X_test,
                plot_type='bar',
                show=False
            )
            plt.tight_layout()
            bar_plot_path = os.path.join(temp_dir, "shap_feature_importance_bar.png")
            plt.savefig(bar_plot_path)
            plt.close()
            mlflow.log_artifact(bar_plot_path)
            
            # Log Scaler and Feature list for traceability
            scaler_path = PROJECT_ROOT / 'models/feature_scaler.joblib'
            if scaler_path.exists():
                mlflow.log_artifact(str(scaler_path))
                
            features_path = PROJECT_ROOT / 'models/features_list.joblib'
            if features_path.exists():
                mlflow.log_artifact(str(features_path))
                
        results[name] = {
            "f1": test_f1,
            "brier": test_brier,
            "model": calibrated_model,
            "run_id": run.info.run_id
        }
        
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model = calibrated_model
            best_model_name = name
            best_run_id = run.info.run_id

# =========================================================
# MODEL SELECTION & REGISTRY
# =========================================================

print(f"\n=== MODEL SELECTION ===")
print(f"Best Candidate Model: {best_model_name} with Test F1: {best_f1:.4f}")

client = mlflow.MlflowClient()
model_name = "risk_profiler_model"

# Fetch current production model to compare
current_prod_f1 = -1.0
try:
    prod_models = client.get_latest_versions(model_name, stages=["Production"])
    if prod_models:
        prod_run_id = prod_models[0].run_id
        prod_run = client.get_run(prod_run_id)
        if "Test_F1_Macro" in prod_run.data.metrics:
            current_prod_f1 = prod_run.data.metrics["Test_F1_Macro"]
            print(f"Current Production Model F1: {current_prod_f1:.4f}")
        else:
            print("Current Production Model doesn't have Test_F1_Macro metric logged.")
except Exception as e:
    print("No existing production model found or error fetching it.")

# Register best candidate
model_uri = f"runs:/{best_run_id}/model"
model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Candidate model '{best_model_name}' registered as '{model_name}' v{model_version.version}.")

# Champion-Challenger logic
if best_f1 >= current_prod_f1:
    print(f"Candidate model beats/equals Production! Promoting v{model_version.version} to Production.")
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )
else:
    print(f"Candidate model (F1: {best_f1:.4f}) did not beat Production model (F1: {current_prod_f1:.4f}). Not promoting.")

# Save locally to models/random_forest_model.joblib for backward compatibility/fallback loading
joblib.dump(best_model, PROJECT_ROOT / 'models/random_forest_model.joblib')
print(f"Best model saved locally to {PROJECT_ROOT / 'models/random_forest_model.joblib'}")

# Print classification report of best model
y_pred_best = best_model.predict(X_test)
print("\n" + "="*50)
print(f"  FINAL EVALUATION REPORT ({best_model_name.upper()})")
print("="*50)
print(classification_report(
    y_test, y_pred_best,
    target_names=['Low Risk (0)', 'Medium Risk (1)', 'High Risk (2)'],
    zero_division=0
))

# %%