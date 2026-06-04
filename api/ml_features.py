import pandas as pd

from api.ml_artifacts import expected_features, feature_scaler
from api.ml_constants import NUMERIC_FEATURES
from api.ml_helpers import normalize_category
from api.schemas import RiskInput


def build_feature_frame(data: RiskInput):
    category = normalize_category(data.business_category)

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

    dummy_columns = [
        "business_category_fashion",
        "business_category_fnb",
        "business_category_jasa",
        "business_category_retail",
    ]

    for col in dummy_columns:
        raw_df[col] = 1 if f"business_category_{category}" == col else 0

    raw_df = raw_df.drop(columns=["business_category"])
    raw_df["volume_per_active_day"] = raw_df["qris_volume_monthly"] / (raw_df["qris_active_days"] + 1)
    raw_df["pln_delay_ratio"] = raw_df["pln_delay_days"] / (raw_df["business_age_months"] + 1)
    raw_df["volume_to_age_ratio"] = raw_df["qris_volume_monthly"] / (raw_df["business_age_months"] + 1)
    raw_df["chronic_pln_delay"] = (raw_df["pln_delay_days"] > 14).astype(int)

    for col in expected_features:
        if col not in raw_df.columns:
            raw_df[col] = 0

    raw_df = raw_df[expected_features]
    model_df = raw_df.copy()
    model_df[NUMERIC_FEATURES] = feature_scaler.transform(model_df[NUMERIC_FEATURES])

    return raw_df, model_df, category


def feature_value(data: RiskInput, category, feature):
    values = {
        "business_age_months": f"{data.business_age_months} bulan",
        "qris_volume_monthly": f"Rp {data.qris_volume_monthly:,.0f}",
        "qris_active_days": f"{data.qris_active_days} hari",
        "ecommerce_rating": f"{data.ecommerce_rating:.1f}",
        "pln_delay_days": f"{data.pln_delay_days} hari",
        "pdam_bill_avg": f"Rp {data.pdam_bill_avg:,.0f}",
        "pdam_late_payments": f"{data.pdam_late_payments} kali",
        "volume_per_active_day": f"Rp {data.qris_volume_monthly / (data.qris_active_days + 1):,.0f}",
        "volume_to_age_ratio": f"Rp {data.qris_volume_monthly / (data.business_age_months + 1):,.0f}",
        "pln_delay_ratio": f"{data.pln_delay_days / (data.business_age_months + 1):.2f}",
        "chronic_pln_delay": "Ya" if data.pln_delay_days > 14 else "Tidak",
        "business_category_fashion": "Aktif" if category == "fashion" else "Tidak",
        "business_category_fnb": "Aktif" if category == "fnb" else "Tidak",
        "business_category_jasa": "Aktif" if category == "jasa" else "Tidak",
        "business_category_retail": "Aktif" if category == "retail" else "Tidak",
    }
    return values.get(feature, "-")
