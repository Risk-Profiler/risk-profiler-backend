from api.ml_constants import CLASS_RISK_WEIGHTS, RISK_LABELS
from api.ml_helpers import clamp


def expected_risk_from_probabilities(probabilities):
    return sum(
        float(probability) * CLASS_RISK_WEIGHTS.get(index, 0.5)
        for index, probability in enumerate(probabilities)
    )


def class_probabilities(probabilities):
    return {
        RISK_LABELS.get(index, str(index)): round(float(probability), 4)
        for index, probability in enumerate(probabilities)
    }


def calculate_score(probabilities):
    expected_risk = expected_risk_from_probabilities(probabilities)
    return round(clamp((1 - expected_risk) * 100))


def scores_from_probabilities(probability_rows):
    return [calculate_score(row) for row in probability_rows]


def calculate_confidence(probability):
    labels = [
        (0.8, "Tinggi"),
        (0.6, "Sedang"),
        (0, "Perlu Review"),
    ]
    return next(label for threshold, label in labels if probability >= threshold)
