import joblib
import pandas as pd
import shap
from pathlib import Path
from api.schemas import RiskInput

BASE_DIR = Path(__file__).resolve().parent.parent

# Load Model, Scaler, dan list kolom 
pipeline = joblib.load(BASE_DIR / "models" / "random_forest_model.joblib")
rf_model = pipeline.named_steps['model']
explainer = shap.TreeExplainer(rf_model)

feature_scaler = joblib.load(BASE_DIR / "models" / "feature_scaler.joblib")
expected_features = joblib.load(BASE_DIR / "models" / "features_list.joblib")

RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk"
}

def generate_merchant_explanation(merchant_id, X_data, shap_values_object, pred_class):
    feature_names = X_data.columns
    merchant_shap = shap_values_object[0, :, pred_class].values
    
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': merchant_shap
    })
    
    top_positive = df_importance.sort_values(by='SHAP_Value', ascending=False).head(2)
    top_negative = df_importance.sort_values(by='SHAP_Value', ascending=True).head(1)
    
    fakta_pendukung_1 = top_positive.iloc[0]['Feature']
    fakta_pendukung_2 = top_positive.iloc[1]['Feature']
    fakta_penahan = top_negative.iloc[0]['Feature']
    
    sentence = (
        f"Sistem mengklasifikasikan merchant {merchant_id} sebagai {RISK_LABELS[pred_class]}, "
        f"terutama dipicu oleh indikasi deviasi pada metrik '{fakta_pendukung_1}' dan '{fakta_pendukung_2}'. "
        f"Namun, tingkat risiko ini sedikit ditekan oleh metrik '{fakta_penahan}'."
    )
    return sentence

def predict_risk(data: RiskInput):
    # 1. Clean Category
    category = data.business_category.lower().strip()
    correction = {"f & b": "fnb", "f&b": "fnb"}
    category = correction.get(category, category)

    # 2. Build Initial DataFrame
    df = pd.DataFrame([{
        "business_age_months": data.business_age_months,
        "qris_volume_monthly": data.qris_volume_monthly,
        "qris_active_days": data.qris_active_days,
        "pln_delay_days": data.pln_delay_days,
        "ecommerce_rating": data.ecommerce_rating,
        "pdam_bill_avg": data.pdam_bill_avg,
        "pdam_late_payments": data.pdam_late_payments,
        "business_category": category,
    }])

    # 3. Categorical Encoding (One-Hot)
    dummy_columns = ['business_category_fashion', 'business_category_fnb', 'business_category_jasa', 'business_category_retail']
    for col in dummy_columns:
        df[col] = 1 if f"business_category_{category}" == col else 0
    df = df.drop(columns=['business_category'])

    # 4. Feature Engineering (Asli dari file 02_feature_engineering.py)
    df['volume_per_active_day'] = df['qris_volume_monthly'] / (df['qris_active_days'] + 1)
    df['pln_delay_ratio'] = df['pln_delay_days'] / (df['business_age_months'] + 1)
    df['volume_to_age_ratio'] = df['qris_volume_monthly'] / (df['business_age_months'] + 1)
    df['chronic_pln_delay'] = (df['pln_delay_days'] > 14).astype(int)

    # 5. Susun urutan kolom agar sama persis dengan saat training
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_features]

    # 6. Feature Scaling (Hanya pada kolom numerik)
    numeric_features = [
        'business_age_months', 'qris_volume_monthly', 'qris_active_days',
        'ecommerce_rating', 'pln_delay_days', 'pdam_bill_avg',
        'pdam_late_payments', 'volume_per_active_day', 'volume_to_age_ratio',
        'pln_delay_ratio'
    ]
    df[numeric_features] = feature_scaler.transform(df[numeric_features])

    # 7. Prediction
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df).max()

    # 8. SHAP Explanation
    shap_values = explainer(df)
    explanation = generate_merchant_explanation(data.merchant_id, df, shap_values, prediction)

    return {
        "risk_level": RISK_LABELS.get(int(prediction), "Unknown"),
        "probability": round(float(probability), 4),
        "explanation": explanation
    }