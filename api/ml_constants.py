RISK_LABELS = {
    0: "Low Risk",
    1: "Medium Risk",
    2: "High Risk",
}

CLASS_RISK_WEIGHTS = {
    0: 0.10,
    1: 0.55,
    2: 0.92,
}

NUMERIC_FEATURES = [
    "business_age_months",
    "qris_volume_monthly",
    "qris_active_days",
    "ecommerce_rating",
    "pln_delay_days",
    "pdam_bill_avg",
    "pdam_late_payments",
    "qris_active_ratio",
    "volume_to_age_ratio",
    "pln_delay_ratio",
]

FEATURE_LABELS = {
    "business_age_months": "Usia Usaha",
    "qris_volume_monthly": "Volume QRIS Bulanan",
    "qris_active_days": "Hari Aktif QRIS",
    "ecommerce_rating": "Rating E-Commerce",
    "pln_delay_days": "Keterlambatan PLN",
    "pdam_bill_avg": "Rata-rata Tagihan PDAM",
    "pdam_late_payments": "Keterlambatan PDAM",
    "qris_active_ratio": "Rasio Hari Aktif QRIS",
    "volume_to_age_ratio": "Rasio Volume terhadap Usia Usaha",
    "pln_delay_ratio": "Rasio Telat PLN terhadap Usia Usaha",
    "chronic_pln_delay": "Indikasi Keterlambatan PLN Kronis",
    "business_category_fashion": "Kategori Fashion",
    "business_category_fnb": "Kategori F&B",
    "business_category_jasa": "Kategori Jasa",
    "business_category_retail": "Kategori Retail",
    "pln_delay_days_isna": "Data PLN Kosong",
    "ecommerce_rating_isna": "Data Rating E-Commerce Kosong",
    "pdam_bill_avg_isna": "Data PDAM Kosong",
    "pdam_late_payments_isna": "Data Keterlambatan PDAM Kosong",
    "has_both_utilities": "Memiliki Kedua Utilitas",
}

FEATURE_GROUPS = {
    "Stabilitas Arus Kas": {
        "qris_volume_monthly",
        "qris_active_days",
        "qris_active_ratio",
        "volume_to_age_ratio",
    },
    "Kedisiplinan Utilitas": {
        "pln_delay_days",
        "pln_delay_ratio",
        "chronic_pln_delay",
        "pdam_bill_avg",
        "pdam_late_payments",
        "pln_delay_days_isna",
        "pdam_bill_avg_isna",
        "pdam_late_payments_isna",
        "has_both_utilities",
    },
    "Kredibilitas Digital": {
        "ecommerce_rating",
        "ecommerce_rating_isna",
    },
    "Ketahanan Usaha": {
        "business_age_months",
    },
    "Kategori Usaha": {
        "business_category_fashion",
        "business_category_fnb",
        "business_category_jasa",
        "business_category_retail",
        "business_category_nan",
    },
}
