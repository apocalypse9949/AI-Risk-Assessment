"""
risk_engine.py — DRI scoring for a gig worker.

Priority:
  1. XGBoost model (if artifacts exist)
  2. Rule-based formula (fallback, always works)
"""

import os
import json
import joblib
import numpy as np
from config import (
    DRI_MODEL_PATH, TIER_MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    DRI_WEIGHTS, DISASTER_DRI_THRESHOLD,
)
from feature_engine import extract_features, FEATURE_NAMES


# ─── Model Registry ───────────────────────────────────────────────

_dri_model  = None
_tier_model = None
_scaler     = None

def _load_models():
    global _dri_model, _tier_model, _scaler
    if _dri_model is not None:
        return True  # already loaded

    if all(os.path.exists(p) for p in [DRI_MODEL_PATH, TIER_MODEL_PATH, SCALER_PATH]):
        try:
            _dri_model  = joblib.load(DRI_MODEL_PATH)
            _tier_model = joblib.load(TIER_MODEL_PATH)
            _scaler     = joblib.load(SCALER_PATH)
            return True
        except Exception as e:
            print(f"[RiskEngine] Model load failed: {e}. Using rule-based fallback.")
    return False


# ─── Rule-Based Fallback ──────────────────────────────────────────

def _rule_based_dri(weather: dict, traffic: dict, worker: dict) -> float:
    w = DRI_WEIGHTS
    disaster_score = 90.0 if weather.get("disaster_flag", False) else 0.0

    raw = (
        w["weather"]  * weather.get("weather_risk",  0.0) +
        w["traffic"]  * traffic.get("traffic_risk",  0.0) +
        w["zone"]     * traffic.get("zone_risk",     0.0) +
        w["exposure"] * worker.get("exposure_risk",  0.0) +
        w["disaster"] * disaster_score
    )
    return float(np.clip(raw, 0.0, 100.0))


# ─── Explainability ───────────────────────────────────────────────

def explain_dri(weather: dict, traffic: dict, worker: dict) -> dict:
    """
    Returns a human-readable breakdown of what drove the DRI score.
    """
    w = DRI_WEIGHTS

    components = {
        "weather_risk":  {
            "value":       round(weather.get("weather_risk", 0.0), 2),
            "weight":      w["weather"],
            "contribution": round(w["weather"] * weather.get("weather_risk", 0), 2),
            "label":       _rain_label(weather.get("rain_mm_hr", 0)),
        },
        "traffic_risk":  {
            "value":       round(traffic.get("traffic_risk", 0.0), 2),
            "weight":      w["traffic"],
            "contribution": round(w["traffic"] * traffic.get("traffic_risk", 0), 2),
            "label":       traffic.get("density_label", "unknown"),
        },
        "zone_risk":     {
            "value":       round(traffic.get("zone_risk", 0.0), 2),
            "weight":      w["zone"],
            "contribution": round(w["zone"] * traffic.get("zone_risk", 0), 2),
            "label":       traffic.get("risk_class", "unknown"),
        },
        "exposure_risk": {
            "value":       round(worker.get("exposure_risk", 0.0), 2),
            "weight":      w["exposure"],
            "contribution": round(w["exposure"] * worker.get("exposure_risk", 0), 2),
            "label":       worker.get("exposure_label", "unknown"),
        },
        "disaster_risk": {
            "value":       90.0 if weather.get("disaster_flag") else 0.0,
            "weight":      w["disaster"],
            "contribution": round(w["disaster"] * (90.0 if weather.get("disaster_flag") else 0.0), 2),
            "label":       "ACTIVE" if weather.get("disaster_flag") else "none",
        },
    }

    top_driver = max(components.items(), key=lambda x: x[1]["contribution"])
    return {
        "components":  components,
        "top_driver":  top_driver[0],
        "summary":     _build_summary(components, worker, weather, traffic),
    }


def _rain_label(mm: float) -> str:
    if mm == 0:        return "clear"
    if mm < 2.5:       return "light drizzle"
    if mm < 7.5:       return "moderate rain"
    if mm < 15:        return "heavy rain"
    return "extreme rainfall"


def _build_summary(components: dict, worker: dict, weather: dict, traffic: dict) -> str:
    parts = []
    if weather.get("disaster_flag"):
        parts.append("⚠️ EXTREME WEATHER ALERT active")
    if components["weather_risk"]["contribution"] > 15:
        parts.append(f"High weather risk ({weather.get('condition', '')})")
    if components["traffic_risk"]["contribution"] > 12:
        parts.append(f"High traffic in {worker.get('zone', '')} zone")
    if worker.get("night_shift"):
        parts.append("Night shift → increased road risk")
    if worker.get("vehicle") == "motorcycle":
        parts.append("Motorcycle → higher injury exposure")
    if worker.get("weekly_hours", 0) > 60:
        parts.append("Extended hours → fatigue risk")
    if not parts:
        parts.append("Normal conditions — low overall risk")
    return " | ".join(parts)


# ─── Main Scoring Function ────────────────────────────────────────

def score_worker(worker: dict, weather: dict, traffic: dict) -> dict:
    """
    Compute DRI score and recommended tier for a worker.

    Returns
    -------
    dict with:
        dri_score         float 0-100
        dri_band          str   low / medium / high / extreme
        recommended_tier  int   0-4
        is_disaster       bool
        model_used        str   xgboost / rule_based
        explanation       dict
    """
    model_available = _load_models()

    # ── XGBoost path ───────────────────────────────────────────
    if model_available:
        features = extract_features(worker, weather, traffic).reshape(1, -1)
        scaled   = _scaler.transform(features)
        dri      = float(np.clip(_dri_model.predict(scaled)[0], 0.0, 100.0))
        tier_idx = int(_tier_model.predict(scaled)[0])
        model_used = "xgboost"
    else:
        # ── Rule-based fallback ────────────────────────────────
        dri = _rule_based_dri(weather, traffic, worker)

        # Derive tier from DRI
        from config import PREMIUM_TIERS
        tier_idx = 0
        for i, t in enumerate(PREMIUM_TIERS):
            if dri <= t["dri_max"]:
                tier_idx = i
                break
        else:
            tier_idx = len(PREMIUM_TIERS) - 1
        model_used = "rule_based"

    # ── DRI band ───────────────────────────────────────────────
    if dri < 25:       band = "low"
    elif dri < 55:     band = "medium"
    elif dri < 80:     band = "high"
    else:              band = "extreme"

    is_disaster = (dri >= DISASTER_DRI_THRESHOLD) or weather.get("disaster_flag", False)

    explanation = explain_dri(weather, traffic, worker)

    return {
        "dri_score":        round(dri, 2),
        "dri_band":         band,
        "recommended_tier": tier_idx,
        "is_disaster":      is_disaster,
        "model_used":       model_used,
        "explanation":      explanation,
    }


if __name__ == "__main__":
    from feature_engine  import get_full_context
    import json

    worker, weather, traffic, _ = get_full_context(seed=42)
    result = score_worker(worker, weather, traffic)

    print(f"\nWorker : {worker['name']} ({worker['zone']}, {worker['city']})")
    print(f"Weather: {weather['condition']} | rain={weather['rain_mm_hr']}mm/h")
    print(f"Traffic: {traffic['density_label']} | peak={traffic['is_peak_hour']}")
    print(f"\nDRI Score : {result['dri_score']} ({result['dri_band'].upper()})")
    print(f"Model Used: {result['model_used']}")
    print(f"Disaster  : {result['is_disaster']}")
    print(f"\nExplanation: {result['explanation']['summary']}")
