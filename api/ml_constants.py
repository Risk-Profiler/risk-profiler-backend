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

# =========================================================
# SHARIAH FINANCING & DATA QUALITY CONSTANTS
# =========================================================

# Nisbah (profit share) percentage limits for the Merchant based on Risk Level
SHARIAH_NISBAH_CONFIG = {
    "Low Risk": {"min": 0.70, "max": 0.90},
    "Medium Risk": {"min": 0.50, "max": 0.70},
    "High Risk": {"min": 0.30, "max": 0.50},
}

# Murabahah markup ceiling based on Risk Level
SHARIAH_MURABAHAH_MARKUP = {
    "Low Risk": 0.05,      # 5% markup
    "Medium Risk": 0.10,   # 10% markup
    "High Risk": 0.15,     # 15% markup
}

# Weights for computing the alternative data footprint completeness (Total = 1.0)
DATA_FOOTPRINT_WEIGHTS = {
    "qris": 0.40,        # Transaction volume & days active
    "pln": 0.30,         # PLN payment reliability
    "pdam": 0.20,        # PDAM payment average & delay
    "ecommerce": 0.10,   # E-commerce platform ratings
}

