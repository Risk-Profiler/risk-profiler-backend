#%%
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# =========================================================
# LOAD DATA
# =========================================================

X_train = pd.read_csv('../data/processed/X_train.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
X_test  = pd.read_csv('../data/processed/X_test.csv')
y_test  = pd.read_csv('../data/processed/y_test.csv').values.ravel()

print(f"X_train shape : {X_train.shape}")
print(f"X_test shape  : {X_test.shape}")
print(f"\nDistribusi y_train:\n{pd.Series(y_train).value_counts().sort_index()}")

# =========================================================
# DEFINE PIPELINE
# SMOTE di dalam pipeline — bukan di luar
# Memastikan synthetic data hanya ada di training fold
# tidak pernah menyentuh validation atau test set
# =========================================================

pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(
        max_depth        = 5,
        min_samples_leaf = 5,
        min_samples_split= 7,
        n_estimators     = 200,
        class_weight     = 'balanced',
        random_state     = 42,
        n_jobs           = -1
    ))
])

# =========================================================
# CROSS VALIDATION
# SMOTE di dalam pipeline → CV tetap valid
# Tiap fold: SMOTE fit di training fold, transform ke training fold
#            validation fold tetap pakai data original
# =========================================================

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=skf,
    scoring='f1_macro',
    n_jobs=-1
)

print("\n=== CROSS VALIDATION ===")
print(f"CV F1 per fold : {[round(s, 3) for s in cv_scores]}")
print(f"CV F1 Mean     : {cv_scores.mean():.3f}")
print(f"CV F1 Std      : {cv_scores.std():.3f}")

# =========================================================
# TRAIN FINAL MODEL
# Fit pipeline di seluruh X_train setelah CV selesai
# SMOTE akan diapply ke seluruh training data
# =========================================================

pipeline.fit(X_train, y_train)

# =========================================================
# EVALUATE
# =========================================================

y_pred = pipeline.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
recall    = recall_score(y_test, y_pred, average='macro', zero_division=0)
cm        = confusion_matrix(y_test, y_pred)

print("\n" + "="*50)
print("  FINAL EVALUATION RESULTS")
print("="*50)
print(f'accuracy :  {accuracy:.4f}')
print(f'f1_macro :  {f1:.4f}')
print(f'precision : {precision:.4f}')
print(f'recall    : {recall:.4f}\n')

print(classification_report(
    y_test, y_pred,
    target_names=['Low Risk (0)', 'Medium Risk (1)', 'High Risk (2)'],
    zero_division=0
))

print(f'conf_matrix:\n{cm}')

# Cek gap CV vs test
gap = abs(cv_scores.mean() - f1)
print(f"\nCV F1 Mean : {cv_scores.mean():.3f}")
print(f"Test F1    : {f1:.3f}")
print(f"Gap        : {gap:.3f}", end=" ")
print("✅ Konsisten" if gap <= 0.05 else "⚠️  Ada gap — perlu dicek")

# =========================================================
# SAVE MODEL
# Simpan pipeline (bukan hanya model)
# Saat inference, pipeline akan otomatis handle preprocessing
# =========================================================

joblib.dump(pipeline, '../models/random_forest_model.joblib')
print("\nPipeline disimpan ke ../models/random_forest_model.joblib")

# =========================================================
# SHAP ANALYSIS
# Akses model dari dalam pipeline
# =========================================================

print("\nGenerating SHAP analysis...")
rf_model    = pipeline.named_steps['model']
explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer(X_test)

# Summary plot — High Risk (class 2)
plt.figure()
shap.summary_plot(
    shap_values[:, :, 2],
    X_test,
    title='SHAP — Feature Importance for High Risk',
    show=False
)
plt.tight_layout()
plt.show()

# Feature importance bar — semua class
plt.figure()
shap.summary_plot(
    shap_values,
    X_test,
    plot_type='bar',
    show=False
)
plt.tight_layout()
plt.show()

# %%