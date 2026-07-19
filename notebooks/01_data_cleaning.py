#%%
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# IMPORT FEATURE ENGINEERING MODULE
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
module_path = PROJECT_ROOT / "notebooks/02_feature_engineering.py"

spec = importlib.util.spec_from_file_location(
    "feature_module",
    module_path
)

feature_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feature_module)

add_featured_engineering = feature_module.add_featured_engineering

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(PROJECT_ROOT / 'data/raw/messy_risk_profiler.csv')

print("=== RAW DATA INFO ===")
print(df.info())

#%%

# =========================================================
# BASIC CLEANING
# =========================================================

# -------------------------
# Missing Values
# -------------------------

df['pln_delay_days_isna'] = (
    df['pln_delay_days']
    .isna()
    .astype(int)
)
df['pln_delay_days'] = df['pln_delay_days'].fillna(0)

df['ecommerce_rating_isna'] = (
    df['ecommerce_rating']
    .isna()
    .astype(int)
)

ecommerce_rating_median = df['ecommerce_rating'].median()
joblib.dump({"ecommerce_rating_median": float(ecommerce_rating_median)}, PROJECT_ROOT / 'models/cleaning_constants.joblib')

df['ecommerce_rating'] = (
    df['ecommerce_rating']
    .fillna(ecommerce_rating_median)
)

df['pdam_bill_avg_isna'] = (
    df['pdam_bill_avg']
    .isna()
    .astype(int)
)
df['pdam_bill_avg'] = df['pdam_bill_avg'].fillna(0)

df['pdam_late_payments_isna'] = (
    df['pdam_late_payments']
    .isna()
    .astype(int)
)
df['pdam_late_payments'] = (
    df['pdam_late_payments']
    .fillna(0)
)

# -------------------------
# Clean Business Category
# -------------------------

df['business_category'] = (
    df['business_category']
    .astype(str)
    .str.lower()
    .str.strip()
)

correction = {
    'f & b': 'fnb',
    'f&b': 'fnb',
}

df['business_category'] = (
    df['business_category']
    .replace(correction)
)

# -------------------------
# Clean QRIS Volume
# -------------------------

df['qris_volume_monthly'] = (
    df['qris_volume_monthly']
    .astype(str)
    .str.replace(r'[^\d.]', '', regex=True)
)

df['qris_volume_monthly'] = pd.to_numeric(
    df['qris_volume_monthly'],
    errors='coerce'
)

df['qris_volume_monthly'] = (
    df['qris_volume_monthly']
    .fillna(df['qris_volume_monthly'].median())
)

# -------------------------
# Remove Invalid Data
# -------------------------

df = df[df['business_age_months'] >= 0]

df = df[df['pdam_bill_avg'] >= 0]

df = df[df['qris_active_days'] <= 31]

# =========================================================
# REMOVE DUPLICATES
# =========================================================

before_dup = len(df)

df = df.drop_duplicates(subset='merchant_id')

after_dup = len(df)

print(f'Removed duplicates: {before_dup - after_dup}')

# =========================================================
# ENCODING
# =========================================================

df = pd.get_dummies(
    df,
    columns=['business_category'],
    drop_first=True,
    dtype=int
)

df = df.drop(columns = ['business_category'], errors = 'ignore')

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df = add_featured_engineering(df)

# =========================================================
# FEATURE / TARGET SPLIT
# =========================================================

# Drop direct + derived leakage
leakage_features = [
    'risk_score'
]

X = df.drop(
    columns=[
        'merchant_id',
        'risk_level',
        *leakage_features
    ],
    errors='ignore'
)

y = df['risk_level']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# FEATURE SCALING
# AFTER SPLIT
# =========================================================

# Apply log transform to highly right-skewed cash flow columns prior to scaling
X_train['qris_volume_monthly'] = np.log1p(X_train['qris_volume_monthly'])
X_train['pdam_bill_avg'] = np.log1p(X_train['pdam_bill_avg'])

X_test['qris_volume_monthly'] = np.log1p(X_test['qris_volume_monthly'])
X_test['pdam_bill_avg'] = np.log1p(X_test['pdam_bill_avg'])

numeric_features = [
    'business_age_months',
    'qris_volume_monthly',
    'qris_active_days',
    'ecommerce_rating', 
    'pln_delay_days',   
    'pdam_bill_avg',
    'pdam_late_payments',
    'qris_active_ratio',
    'volume_to_age_ratio',
    'pln_delay_ratio'
]

feature_scaler = MinMaxScaler()

X_train[numeric_features] = feature_scaler.fit_transform(
    X_train[numeric_features]
)

X_test[numeric_features] = feature_scaler.transform(
    X_test[numeric_features]
)   

# =========================================================
# SAVE DATASETS
# =========================================================

X_train.to_csv(
    PROJECT_ROOT / 'data/processed/X_train.csv',
    index=False
)

X_test.to_csv(
    PROJECT_ROOT / 'data/processed/X_test.csv',
    index=False
)

y_train.to_csv(
    PROJECT_ROOT / 'data/processed/y_train.csv',
    index=False
)

y_test.to_csv(
    PROJECT_ROOT / 'data/processed/y_test.csv',
    index=False
)

df.to_csv(
    PROJECT_ROOT / 'data/processed/cleaned_risk_profiler.csv',
    index=False
)

# Export Scaler
joblib.dump(feature_scaler, PROJECT_ROOT / 'models/feature_scaler.joblib')

# Export daftar nama kolom secara berurutan
joblib.dump(X_train.columns.tolist(), PROJECT_ROOT / 'models/features_list.joblib')

# =========================================================
# FINAL INFO
# =========================================================

print("\n=== FINAL DATA INFO ===")
print(df.info())

print("\n=== LABEL DISTRIBUTION ===")
print(df['risk_level'].value_counts(normalize=True))

print("\n=== TRAIN TEST SHAPE ===")
print(f'X_train: {X_train.shape}')
print(f'X_test : {X_test.shape}')

print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

# %%