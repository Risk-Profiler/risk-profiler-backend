import numpy as np
import pandas as pd
import joblib

from api.ml_artifacts import BASE_DIR, expected_features, feature_scaler
from api.ml_constants import NUMERIC_FEATURES
from api.ml_helpers import normalize_category
from api.schemas import RiskInput

# Load cleaning constants for ecommerce_rating median imputation
try:
    cleaning_constants = joblib.load(BASE_DIR / "models" / "cleaning_constants.joblib")
    ECOMMERCE_RATING_MEDIAN = cleaning_constants.get("ecommerce_rating_median", 4.0)
except Exception:
    ECOMMERCE_RATING_MEDIAN = 4.0


def build_feature_frame(data: RiskInput):
    category = normalize_category(data.business_category)

    # 1. Build initial dataframe with raw inputs
    raw_df = pd.DataFrame([{
        "business_age_months": data.business_age_months,
        "qris_volume_monthly": data.qris_volume_monthly,
        "qris_active_days": data.qris_active_days,
        "pln_delay_days": data.pln_delay_days,
        "ecommerce_rating": data.ecommerce_rating,
        "pdam_bill_avg": data.pdam_bill_avg,
        "pdam_late_payments": data.pdam_late_payments,
        "business_category": category,
    }])

    # 2. Extract missingness indicators
    raw_df["pln_delay_days_isna"] = raw_df["pln_delay_days"].isna().astype(int)
    raw_df["ecommerce_rating_isna"] = raw_df["ecommerce_rating"].isna().astype(int)
    raw_df["pdam_bill_avg_isna"] = raw_df["pdam_bill_avg"].isna().astype(int)
    raw_df["pdam_late_payments_isna"] = raw_df["pdam_late_payments"].isna().astype(int)

    # 3. Fill missing values as in training data cleaning
    raw_df["pln_delay_days"] = raw_df["pln_delay_days"].fillna(0)
    raw_df["ecommerce_rating"] = raw_df["ecommerce_rating"].fillna(ECOMMERCE_RATING_MEDIAN)
    raw_df["pdam_bill_avg"] = raw_df["pdam_bill_avg"].fillna(0)
    raw_df["pdam_late_payments"] = raw_df["pdam_late_payments"].fillna(0)

    # 4. Dummy columns for category
    dummy_columns = [
        "business_category_fashion",
        "business_category_fnb",
        "business_category_jasa",
        "business_category_retail",
    ]

    for col in dummy_columns:
        raw_df[col] = 1 if f"business_category_{category}" == col else 0

    raw_df = raw_df.drop(columns=["business_category"])

    # 5. Feature engineering
    raw_df["qris_active_ratio"] = raw_df["qris_active_days"] / 30
    raw_df["pln_delay_ratio"] = raw_df["pln_delay_days"] / (raw_df["business_age_months"] + 1)
    raw_df["volume_to_age_ratio"] = raw_df["qris_volume_monthly"] / (raw_df["business_age_months"] + 1)
    raw_df["chronic_pln_delay"] = (raw_df["pln_delay_days"] > 14).astype(int)
    raw_df["has_both_utilities"] = (1 - raw_df["pln_delay_days_isna"]) * (1 - raw_df["pdam_bill_avg_isna"])

    # Ensure all expected features are present (including dummy variables and isna indicators)
    for col in expected_features:
        if col not in raw_df.columns:
            raw_df[col] = 0

    # Ensure column order aligns with expected features
    raw_df = raw_df[expected_features]

    # 6. Apply log transform to highly right-skewed cash flow columns prior to scaling
    model_df = raw_df.copy()
    model_df["qris_volume_monthly"] = np.log1p(model_df["qris_volume_monthly"])
    model_df["pdam_bill_avg"] = np.log1p(model_df["pdam_bill_avg"])

    # 7. Scaling
    model_df[NUMERIC_FEATURES] = feature_scaler.transform(model_df[NUMERIC_FEATURES])

    return raw_df, model_df, category


def feature_value(data: RiskInput, category, feature):
    values = {
        "business_age_months": f"{data.business_age_months} bulan" if data.business_age_months is not None else "-",
        "qris_volume_monthly": f"Rp {data.qris_volume_monthly:,.0f}" if data.qris_volume_monthly is not None else "-",
        "qris_active_days": f"{data.qris_active_days} hari" if data.qris_active_days is not None else "-",
        "ecommerce_rating": f"{data.ecommerce_rating:.1f}" if data.ecommerce_rating is not None else "-",
        "pln_delay_days": f"{data.pln_delay_days} hari" if data.pln_delay_days is not None else "-",
        "pdam_bill_avg": f"Rp {data.pdam_bill_avg:,.0f}" if data.pdam_bill_avg is not None else "-",
        "pdam_late_payments": f"{data.pdam_late_payments} kali" if data.pdam_late_payments is not None else "-",
        "qris_active_ratio": f"{data.qris_active_days / 30:.2%}" if data.qris_active_days is not None else "-",
        "volume_to_age_ratio": f"Rp {data.qris_volume_monthly / (data.business_age_months + 1):,.0f}" if (data.qris_volume_monthly is not None and data.business_age_months is not None) else "-",
        "pln_delay_ratio": f"{data.pln_delay_days / (data.business_age_months + 1):.2f}" if (data.pln_delay_days is not None and data.business_age_months is not None) else "-",
        "chronic_pln_delay": ("Ya" if data.pln_delay_days > 14 else "Tidak") if data.pln_delay_days is not None else "-",
        "business_category_fashion": "Aktif" if category == "fashion" else "Tidak",
        "business_category_fnb": "Aktif" if category == "fnb" else "Tidak",
        "business_category_jasa": "Aktif" if category == "jasa" else "Tidak",
        "business_category_retail": "Aktif" if category == "retail" else "Tidak",
        "pln_delay_days_isna": "Ya" if data.pln_delay_days is None else "Tidak",
        "ecommerce_rating_isna": "Ya" if data.ecommerce_rating is None else "Tidak",
        "pdam_bill_avg_isna": "Ya" if data.pdam_bill_avg is None else "Tidak",
        "pdam_late_payments_isna": "Ya" if data.pdam_late_payments is None else "Tidak",
        "has_both_utilities": "Ya" if (data.pln_delay_days is not None and data.pdam_bill_avg is not None) else "Tidak",
    }
    return values.get(feature, "-")
