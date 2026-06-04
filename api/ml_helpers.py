def clamp(value, minimum=0, maximum=100):
    return min(maximum, max(minimum, value))


def round_to(value, step):
    return round(value / step) * step


def normalize_category(category):
    normalized = category.lower().strip()
    correction = {
        "f & b": "fnb",
        "f&b": "fnb",
        "food": "fnb",
        "service": "jasa",
        "services": "jasa",
    }
    return correction.get(normalized, normalized)
