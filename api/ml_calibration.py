import pandas as pd

from api.ml_artifacts import BASE_DIR, expected_features, feature_scaler, pipeline
from api.ml_constants import NUMERIC_FEATURES
from api.ml_scoring import expected_risk_from_probabilities, scores_from_probabilities


def build_training_calibration():
    calibration = {
        "score_thresholds": {"A": 85, "B": 70, "C": 55, "D": 40, "E": 0},
        "training_scores": pd.Series(dtype=float),
        "capacity_floor": 500000,
        "capacity_ceiling": 50000000,
    }
    peer_profiles = pd.DataFrame()

    x_train_path = BASE_DIR / "data" / "processed" / "X_train.csv"
    cleaned_path = BASE_DIR / "data" / "processed" / "cleaned_risk_profiler.csv"

    if x_train_path.exists():
        x_train = pd.read_csv(x_train_path)[expected_features]
        train_probabilities = pipeline.predict_proba(x_train)
        training_scores = pd.Series(scores_from_probabilities(train_probabilities))
        calibration["training_scores"] = training_scores
        calibration["score_thresholds"] = {
            "A": int(round(training_scores.quantile(0.80))),
            "B": int(round(training_scores.quantile(0.60))),
            "C": int(round(training_scores.quantile(0.40))),
            "D": int(round(training_scores.quantile(0.20))),
            "E": int(round(training_scores.min())),
        }

    if cleaned_path.exists():
        raw = pd.read_csv(cleaned_path)
        model_frame = raw[expected_features].copy()
        model_frame[NUMERIC_FEATURES] = feature_scaler.transform(model_frame[NUMERIC_FEATURES])
        peer_probabilities = pipeline.predict_proba(model_frame)
        peer_profiles = raw.copy()
        peer_profiles["model_score"] = scores_from_probabilities(peer_probabilities)
        peer_profiles["expected_risk"] = [
            expected_risk_from_probabilities(row)
            for row in peer_probabilities
        ]
        peer_profiles["model_capacity"] = (
            peer_profiles["qris_volume_monthly"]
            * (peer_profiles["model_score"] / 100)
            * (1 - peer_profiles["expected_risk"])
        )
        calibration["capacity_floor"] = int(max(500000, peer_profiles["model_capacity"].quantile(0.05)))
        calibration["capacity_ceiling"] = int(max(calibration["capacity_floor"], peer_profiles["model_capacity"].quantile(0.95)))

    return calibration, peer_profiles


MODEL_CALIBRATION, PEER_PROFILES = build_training_calibration()


def score_percentile(score):
    training_scores = MODEL_CALIBRATION["training_scores"]
    if training_scores.empty:
        return score / 100

    return float((training_scores <= score).mean())


def get_band(score):
    thresholds = MODEL_CALIBRATION["score_thresholds"]
    ordered_bands = ["A", "B", "C", "D", "E"]

    for band in ordered_bands:
        if score >= thresholds[band]:
            lower = thresholds[band]
            upper = 100 if band == "A" else max(lower, thresholds[ordered_bands[ordered_bands.index(band) - 1]] - 1)
            return band, f"{lower}-{upper}"

    return "E", f"{thresholds['E']}-{max(thresholds['D'] - 1, thresholds['E'])}"
