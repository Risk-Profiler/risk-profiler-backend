from api.ml_artifacts import pipeline
from api.ml_calibration import get_band, score_percentile
from api.ml_constants import RISK_LABELS
from api.ml_explainability import (
    build_breakdown,
    build_contributions,
    build_model_factors,
    build_shap_drivers,
)
from api.ml_features import build_feature_frame
from api.ml_recommendations import (
    build_data_sources,
    build_recommendations,
    calculate_recommended_limit,
    generate_merchant_explanation,
    calculate_data_quality_score,
    build_split_recommendations,
)
from api.ml_scoring import calculate_confidence, calculate_score, class_probabilities
from api.schemas import RiskInput
from api.ml_fraud import evaluate_fraud_rules


def predict_risk(data: RiskInput):
    raw_df, model_df, category = build_feature_frame(data)

    probabilities = pipeline.predict_proba(model_df)[0]
    prediction = int(probabilities.argmax())
    probability = float(probabilities[prediction])
    risk_level = RISK_LABELS.get(prediction, "Unknown")
    score = calculate_score(probabilities)
    percentile = score_percentile(score)
    band, band_range = get_band(score)
    factors = build_model_factors(data, category, model_df, prediction)
    shap_drivers = build_shap_drivers(factors)
    limit_result = calculate_recommended_limit(
        data,
        category,
        score,
        probabilities,
        factors,
        prediction,
    )
    recommended_limit = limit_result["limit"]
    peer_comparison_used = limit_result["peer_comparison_used"]
    explanation = generate_merchant_explanation(
        data.merchant_id,
        risk_level,
        probability,
        shap_drivers,
    )

    # Compute Data Quality Score (DQS) based on alternative data footprints
    dqs = calculate_data_quality_score(data)

    # Calculate Confidence, defaulting to "Perlu Review" if data footprint is poor
    confidence = calculate_confidence(probability)
    if dqs < 60:
        confidence = "Perlu Review"

    # Probability of Default (High Risk, Class 2)
    pd = float(probabilities[2])

    # Fraud evaluation
    fraud_analysis = evaluate_fraud_rules(data)
    if fraud_analysis["is_suspicious"]:
        confidence = "Rejected - Fraud Suspicion"
        risk_level = "High Risk"
        recommended_limit = 0

    # Compute split recommendations for Conventional / Shariah toggle support
    split_recs = build_split_recommendations(
        risk_level,
        band,
        score,
        recommended_limit,
        probability,
        shap_drivers,
        percentile,
        peer_comparison_used,
        dqs,
        pd,
        fraud_analysis["flags"],
    )

    return {
        "risk_level": risk_level,
        "score": score,
        "score_percentile": round(percentile, 4),
        "probability": round(probability, 4),
        "class_probabilities": class_probabilities(probabilities),
        "confidence": confidence,
        "data_quality_score": dqs,
        "band": band,
        "band_range": band_range,
        "recommended_limit": recommended_limit,
        "peer_comparison_used": peer_comparison_used,
        "explanation": explanation,
        "ai_explanation": explanation,
        "shap_drivers": shap_drivers,
        "contributions": build_contributions(factors, prediction),
        "breakdown": build_breakdown(factors, prediction),
        "data_sources": build_data_sources(data),
        "fraud_analysis": fraud_analysis,
        "recommendations": build_recommendations(
            risk_level,
            band,
            score,
            recommended_limit,
            probability,
            shap_drivers,
            percentile,
            peer_comparison_used,
            dqs,
            pd,
            fraud_analysis["flags"],
        ),
        "conventional_recommendations": split_recs["common"] + split_recs["conventional"] + split_recs["warnings"],
        "shariah_recommendations": split_recs["common"] + split_recs["shariah"] + split_recs["warnings"],
        "shariah_metrics": split_recs["shariah_metrics"],
        "model_features": raw_df.iloc[0].to_dict(),
    }
