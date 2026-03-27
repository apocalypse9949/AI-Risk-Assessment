"""
feature_engine.py — Combines weather, traffic, and worker data into an ML-ready feature vector.
"""

import numpy as np
from weather_service import get_weather
from traffic_service import get_traffic_data
from worker_service  import generate_worker


# ─── Feature Names (in model training order) ─────────────────────

FEATURE_NAMES = [
    # Weather features
    "weather_risk",
    "rain_mm_hr",
    "wind_kph",
    "visibility_m_norm",   # normalised 0-1
    "disaster_flag",

    # Traffic features
    "traffic_risk",
    "zone_risk",
    "is_peak_hour",
    "density_encoded",     # low=0, medium=1, high=2

    # Worker / exposure features
    "exposure_risk",
    "vehicle_risk",
    "weekly_hours_norm",   # normalised 0-1 (max 98hrs)
    "experience_norm",     # normalised 0-1 (max 60mo)
    "night_shift",
    "days_per_week",
]


def _encode_density(label: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(label, 1)


def extract_features(worker: dict, weather: dict, traffic: dict) -> np.ndarray:
    """
    Build a flat feature vector from the three data payloads.

    Returns
    -------
    np.ndarray of shape (15,), float32
    """
    vec = [
        # ── Weather ────────────────────────────────────────────
        weather.get("weather_risk",  0.0),
        weather.get("rain_mm_hr",    0.0),
        weather.get("wind_kph",      0.0),
        weather.get("visibility_m",  10000) / 10000.0,         # norm
        float(weather.get("disaster_flag", False)),

        # ── Traffic ────────────────────────────────────────────
        traffic.get("traffic_risk",  0.0),
        traffic.get("zone_risk",     0.0),
        float(traffic.get("is_peak_hour", False)),
        float(_encode_density(traffic.get("density_label", "medium"))),

        # ── Exposure ───────────────────────────────────────────
        worker.get("exposure_risk",  0.0),
        worker.get("vehicle_risk",   0.0),
        min(worker.get("weekly_hours", 0.0), 98) / 98.0,       # norm
        min(worker.get("experience_months", 0), 60) / 60.0,    # norm
        float(worker.get("night_shift", False)),
        float(worker.get("days_per_week", 5)),
    ]

    return np.array(vec, dtype=np.float32)


def build_feature_dict(worker: dict, weather: dict, traffic: dict) -> dict:
    """Human-readable version for explainability."""
    arr = extract_features(worker, weather, traffic)
    return dict(zip(FEATURE_NAMES, arr.tolist()))


def get_full_context(
    worker: dict = None,
    city:   str  = "Bengaluru",
    seed:   int  = None,
) -> tuple[dict, dict, dict, np.ndarray]:
    """
    One-shot convenience: generate or accept a worker, fetch weather & traffic,
    and return all three payloads plus the feature vector.
    """
    if worker is None:
        worker = generate_worker(seed=seed)

    weather = get_weather(city=city, seed=seed)
    traffic = get_traffic_data(
        zone=worker.get("zone", "Suburb"),
        seed=seed,
    )
    features = extract_features(worker, weather, traffic)
    return worker, weather, traffic, features


if __name__ == "__main__":
    from pprint import pprint
    w, wx, tx, feat = get_full_context(seed=7)
    print("Worker :", w["name"], "|", w["zone"])
    print("Weather:", wx["condition"], "risk=", wx["weather_risk"])
    print("Traffic:", tx["density_label"], "risk=", tx["traffic_risk"])
    print("\nFeature vector:")
    pprint(dict(zip(FEATURE_NAMES, feat.tolist())))
