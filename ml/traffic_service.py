"""
traffic_service.py — Realistic mock traffic data for Indian delivery zones.
"""

import random
from datetime import datetime, time


# ─── Zone Definitions ─────────────────────────────────────────────

ZONE_PROFILES = {
    "CBD":        {"base_density": 0.85, "accident_rate": 0.12, "risk_class": "high"},
    "Suburb":     {"base_density": 0.45, "accident_rate": 0.05, "risk_class": "medium"},
    "Industrial": {"base_density": 0.60, "accident_rate": 0.09, "risk_class": "medium"},
    "Residential":{"base_density": 0.30, "accident_rate": 0.03, "risk_class": "low"},
    "Highway":    {"base_density": 0.70, "accident_rate": 0.14, "risk_class": "high"},
    "Rural":      {"base_density": 0.15, "accident_rate": 0.06, "risk_class": "low"},
}

ZONE_RISK_SCORES = {
    "low":    20.0,
    "medium": 50.0,
    "high":   80.0,
}

# ─── Peak Hour Logic ──────────────────────────────────────────────

def _peak_factor(hour: int) -> float:
    """Return a multiplier based on time of day."""
    if   8 <= hour <= 10:  return 1.8   # morning peak
    elif 12 <= hour <= 14: return 1.4   # lunch rush
    elif 17 <= hour <= 20: return 2.0   # evening peak (worst)
    elif 22 <= hour or hour <= 5: return 0.4  # night
    return 1.0


# ─── Traffic Generator ────────────────────────────────────────────

def get_traffic_data(zone: str = "Suburb", hour: int = None, seed: int = None) -> dict:
    """
    Generate realistic traffic mock for a delivery zone.

    Parameters
    ----------
    zone : one of ZONE_PROFILES keys
    hour : 0-23, defaults to current local hour
    seed : for reproducibility in training data pipelines
    """
    if seed is not None:
        random.seed(seed)

    if hour is None:
        hour = datetime.now().hour

    profile = ZONE_PROFILES.get(zone, ZONE_PROFILES["Suburb"])
    peak    = _peak_factor(hour)

    density = min(1.0, profile["base_density"] * peak + random.uniform(-0.05, 0.05))
    accident_prob = profile["accident_rate"] * (1 + 0.5 * (peak - 1))

    # Encode density as label
    if density < 0.35:
        density_label = "low"
    elif density < 0.65:
        density_label = "medium"
    else:
        density_label = "high"

    # Traffic risk = weighted combination
    density_score = {"low": 20, "medium": 50, "high": 85}[density_label]
    accident_score = min(100, accident_prob * 700)
    is_peak = 1 if _peak_factor(hour) >= 1.5 else 0

    traffic_risk = round(
        0.50 * density_score +
        0.35 * accident_score +
        0.15 * (is_peak * 60), 2
    )

    zone_risk = ZONE_RISK_SCORES[profile["risk_class"]]

    return {
        "zone":           zone,
        "hour":           hour,
        "density":        round(density, 3),
        "density_label":  density_label,
        "accident_prone": bool(random.random() < accident_prob),
        "is_peak_hour":   bool(is_peak),
        "traffic_risk":   traffic_risk,
        "zone_risk":      zone_risk,
        "risk_class":     profile["risk_class"],
    }


def get_all_zones_snapshot(hour: int = None) -> dict:
    """Return traffic data for all zones at once."""
    return {zone: get_traffic_data(zone, hour=hour) for zone in ZONE_PROFILES}


if __name__ == "__main__":
    import json
    for z in ZONE_PROFILES:
        t = get_traffic_data(z, hour=18)
        print(f"{z:15s} | density={t['density_label']:6s} | risk={t['traffic_risk']:.1f}")
