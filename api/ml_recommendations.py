from api.ml_calibration import MODEL_CALIBRATION, PEER_PROFILES
from api.ml_explainability import score_impact_factor
from api.ml_helpers import clamp, round_to
from api.ml_scoring import expected_risk_from_probabilities
from api.schemas import RiskInput


def peer_distance(row, score, data: RiskInput, category):
    category_columns = {
        "fashion": "business_category_fashion",
        "fnb": "business_category_fnb",
        "jasa": "business_category_jasa",
        "retail": "business_category_retail",
    }
    category_column = category_columns.get(category)
    category_penalty = 0 if category_column and row.get(category_column, 0) == 1 else 10
    volume_scale = max(float(PEER_PROFILES["qris_volume_monthly"].median()), 1)

    return (
        abs(float(row["model_score"]) - score)
        + abs(float(row["qris_volume_monthly"]) - data.qris_volume_monthly) / volume_scale
        + category_penalty
    )


def calculate_recommended_limit(data: RiskInput, category, score, probabilities, factors, prediction):
    expected_risk = expected_risk_from_probabilities(probabilities)
    direction = score_impact_factor(prediction)
    net_factor_impact = sum(factor["shap_value"] * direction for factor in factors)
    factor_multiplier = clamp(1 + net_factor_impact, 0.35, 1.35)
    model_capacity = data.qris_volume_monthly * (score / 100) * (1 - expected_risk)
    peer_comparison_used = False

    category_columns = {
        "fashion": "business_category_fashion",
        "fnb": "business_category_fnb",
        "jasa": "business_category_jasa",
        "retail": "business_category_retail",
    }
    category_column = category_columns.get(category)

    if (
        not PEER_PROFILES.empty
        and category_column
        and category_column in PEER_PROFILES.columns
    ):
        peer_frame = PEER_PROFILES.copy()
        peer_frame = peer_frame[peer_frame[category_column] == 1]

        if peer_frame.empty:
            raw_limit = model_capacity * factor_multiplier
        else:
            peer_comparison_used = True
            peer_frame = peer_frame.copy()
            peer_frame["distance"] = peer_frame.apply(
                lambda row: peer_distance(row, score, data, category),
                axis=1,
            )
            similar_peers = peer_frame.nsmallest(min(80, len(peer_frame)), "distance")
            peer_capacity = float(similar_peers["model_capacity"].median())
            raw_limit = (peer_capacity * 0.55 + model_capacity * 0.45) * factor_multiplier
    else:
        raw_limit = model_capacity * factor_multiplier

    bounded_limit = clamp(
        raw_limit,
        MODEL_CALIBRATION["capacity_floor"],
        MODEL_CALIBRATION["capacity_ceiling"],
    )
    return {
        "limit": int(round_to(bounded_limit, 500000)),
        "peer_comparison_used": peer_comparison_used,
    }


def generate_merchant_explanation(merchant_id, risk_level, probability, drivers):
    joined_drivers = "; ".join(
        f"{driver['label']}={driver['value']}"
        for driver in drivers[:4]
    )
    return (
        f"Model memprediksi merchant {merchant_id} sebagai {risk_level} "
        f"dengan tingkat keyakinan {probability:.4f}. "
        f"Faktor model terbesar untuk prediksi ini: {joined_drivers}."
    )


def build_data_sources(data: RiskInput):
    return [
        {
            "id": 1, 
            "label": f"QRIS - {data.qris_active_days} hari aktif" if data.qris_active_days is not None else "QRIS - Tidak ada data", 
            "available": data.qris_volume_monthly is not None and data.qris_volume_monthly > 0
        },
        {
            "id": 2, 
            "label": f"Utilitas PLN - telat {data.pln_delay_days} hari" if data.pln_delay_days is not None else "Utilitas PLN - Tidak ada data", 
            "available": data.pln_delay_days is not None
        },
        {
            "id": 3, 
            "label": f"E-Commerce - rating {data.ecommerce_rating:.1f}" if data.ecommerce_rating is not None else "E-Commerce - Tidak ada data", 
            "available": data.ecommerce_rating is not None and data.ecommerce_rating > 0
        },
        {
            "id": 4, 
            "label": f"Profil usaha - {data.business_age_months} bulan" if data.business_age_months is not None else "Profil usaha - Tidak ada data", 
            "available": data.business_age_months is not None and data.business_age_months > 0
        },
        {
            "id": 5, 
            "label": f"PDAM - {data.pdam_late_payments} kali telat" if data.pdam_late_payments is not None else "PDAM - Tidak ada data", 
            "available": data.pdam_bill_avg is not None and data.pdam_bill_avg >= 0
        },
        {"id": 6, "label": "SLIK / data bank eksternal", "available": False},
    ]


def build_recommendations(
    risk_level,
    band,
    score,
    limit,
    probability,
    drivers,
    percentile,
    peer_comparison_used,
):
    factor_summary = ", ".join(
        f"{driver['label']} ({driver['value']})"
        for driver in drivers[:3]
    )
    limit_basis = (
        "dari profil pembanding serupa"
        if peer_comparison_used
        else "berdasarkan skor, kapasitas transaksi, dan faktor risiko merchant"
    )
    return [
        f"Prediksi model: {risk_level} dengan tingkat keyakinan {probability:.4f}.",
        f"Skor model: {score}/100, Band {band}, posisi {round(percentile * 100)}% terhadap data kalibrasi.",
        f"Faktor utama model: {factor_summary}.",
        f"Plafon rekomendasi {limit_basis}: Rp {limit:,.0f}.",
    ]
