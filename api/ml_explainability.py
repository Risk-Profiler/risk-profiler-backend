import pandas as pd

from api.ml_artifacts import explainer
from api.ml_constants import CLASS_RISK_WEIGHTS, FEATURE_GROUPS, FEATURE_LABELS
from api.ml_features import feature_value
from api.ml_helpers import clamp


def shap_for_prediction(model_df, prediction):
    shap_values = explainer(model_df)
    values = shap_values.values

    if len(values.shape) == 3:
        return values[0, :, int(prediction)]

    return values[0]


def build_model_factors(data, category, model_df, prediction):
    shap_values = shap_for_prediction(model_df, prediction)
    factors = pd.DataFrame({
        "feature": model_df.columns,
        "shap_value": shap_values,
    })
    factors["abs_value"] = factors["shap_value"].abs()
    factors = factors.sort_values("abs_value", ascending=False).reset_index(drop=True)

    max_abs = max(float(factors["abs_value"].max()), 0.000001)
    enriched = []

    for index, row in enumerate(factors.itertuples(index=False), start=1):
        shap_value = float(row.shap_value)
        feature = row.feature
        role = "supporting" if shap_value >= 0 else "balancing"
        label = FEATURE_LABELS.get(feature, feature)
        direction = "mendorong prediksi" if role == "supporting" else "menyeimbangkan prediksi"

        enriched.append({
            "id": index,
            "feature": feature,
            "label": label,
            "role": role,
            "strength": round(clamp(abs(shap_value) / max_abs * 100, 8, 100)),
            "value": feature_value(data, category, feature),
            "description": f"{label} {direction} model.",
            "shap_value": round(shap_value, 6),
            "abs_value": abs(shap_value),
        })

    return enriched


def score_impact_factor(prediction):
    return 1 - (2 * CLASS_RISK_WEIGHTS.get(int(prediction), 0.5))


def build_shap_drivers(factors):
    return [
        {key: factor[key] for key in ["id", "feature", "label", "role", "strength", "value", "description", "shap_value"]}
        for factor in factors[:5]
    ]


def build_contributions(factors, prediction):
    direction = score_impact_factor(prediction)
    direction_sign = 1 if direction >= 0 else -1
    return [
        {
            "id": index,
            "label": factor["label"],
            "value": factor["strength"],
            "impact": round(factor["strength"] * (direction_sign if factor["shap_value"] >= 0 else -direction_sign)),
            "color": "bg-green-accent" if factor["shap_value"] * direction_sign >= 0 else "bg-red-accent",
        }
        for index, factor in enumerate(factors[:5], start=1)
    ]


def build_breakdown(factors, prediction):
    direction = score_impact_factor(prediction)
    direction_sign = 1 if direction >= 0 else -1
    total_abs = sum(factor["abs_value"] for factor in factors) or 0.000001
    rows = []

    for group_index, (group_name, features) in enumerate(FEATURE_GROUPS.items(), start=1):
        group_factors = [factor for factor in factors if factor["feature"] in features]
        if not group_factors:
            continue

        abs_share = sum(factor["abs_value"] for factor in group_factors) / total_abs
        group_direction = sum(factor["shap_value"] for factor in group_factors)
        impact_points = abs_share * 100 * (direction_sign if group_direction >= 0 else -direction_sign)
        display_impact = 0.0 if abs(impact_points) < 0.05 else impact_points
        strongest = max(group_factors, key=lambda factor: factor["abs_value"])
        strength = round(clamp(abs_share * 100))
        color = (
            "bg-green-accent" if impact_points > 0.5
            else "bg-red-accent" if impact_points < -0.5
            else "bg-yellowish-accent"
        )

        rows.append({
            "id": group_index,
            "title": group_name,
            "points": (
                "0.0 dampak skor"
                if display_impact == 0
                else f"{display_impact:+.1f} dampak skor"
            ),
            "weight": f"{strength}% pengaruh model",
            "color": color,
            "description": f"Faktor terkuat: {strongest['label']} ({strongest['value']}).",
            "sort_value": abs_share,
        })

    rows = sorted(rows, key=lambda row: row["sort_value"], reverse=True)
    return [
        {key: row[key] for key in ["id", "title", "points", "weight", "color", "description"]}
        for row in rows
    ]
