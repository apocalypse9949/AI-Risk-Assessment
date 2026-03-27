"""
synthetic_data.py — Generate 5,000 labelled training samples for the DRI + Tier models.

Labels produced
───────────────
  dri_score   : float 0-100   (regression target)
  tier_label  : int   0-4     (classification target, maps to PREMIUM_TIERS)
"""

import random
import numpy as np
import pandas as pd
from config import SYNTHETIC_SAMPLES, RANDOM_SEED, DRI_WEIGHTS, PREMIUM_TIERS
from weather_service import fetch_mock_weather
from traffic_service  import get_traffic_data, ZONE_PROFILES
from worker_service   import generate_worker
from feature_engine   import extract_features, FEATURE_NAMES


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─── Ground-Truth DRI Formula (rule-based label) ──────────────────

def compute_rule_dri(
    weather_risk: float,
    traffic_risk: float,
    zone_risk:    float,
    exposure_risk:float,
    disaster_flag:bool,
) -> float:
    """
    Deterministic DRI = weighted sum of risk components.
    This is the 'ground truth' the ML model learns to approximate,
    then generalise beyond simple linear combinations.
    """
    w = DRI_WEIGHTS
    disaster_score = 90.0 if disaster_flag else 0.0

    raw = (
        w["weather"]  * weather_risk  +
        w["traffic"]  * traffic_risk  +
        w["zone"]     * zone_risk     +
        w["exposure"] * exposure_risk +
        w["disaster"] * disaster_score
    )
    # Clip and add small noise to make the ML problem non-trivial
    noise = np.random.normal(0, 2)
    return float(np.clip(raw + noise, 0, 100))


def dri_to_tier_index(dri: float) -> int:
    """Map DRI score to the highest eligible premium tier index."""
    for i, tier in enumerate(PREMIUM_TIERS):
        if dri <= tier["dri_max"]:
            return i
    return len(PREMIUM_TIERS) - 1   # Elite


# ─── Dataset Builder ──────────────────────────────────────────────

def build_dataset(n: int = SYNTHETIC_SAMPLES) -> pd.DataFrame:
    """
    Generate n synthetic (features, DRI label, tier label) rows.

    Strategy: sample workers, weather scenarios, and traffic states
    independently to maximise coverage of the feature space.
    """
    print(f"[SyntheticData] Building {n} samples …")
    rows   = []
    zones  = list(ZONE_PROFILES.keys())
    cities = ["Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai"]

    for i in range(n):
        seed = RANDOM_SEED + i

        # ── Sample each service independently ─────────────────
        worker  = generate_worker(seed=seed)
        city    = random.choice(cities)
        zone    = worker["zone"]
        hour    = random.randint(0, 23)
        weather = fetch_mock_weather(city=city, seed=seed)
        traffic = get_traffic_data(zone=zone, hour=hour, seed=seed)

        # ── Feature vector ─────────────────────────────────────
        feat_vec = extract_features(worker, weather, traffic)

        # ── Labels ─────────────────────────────────────────────
        dri = compute_rule_dri(
            weather_risk  = weather["weather_risk"],
            traffic_risk  = traffic["traffic_risk"],
            zone_risk     = traffic["zone_risk"],
            exposure_risk = worker["exposure_risk"],
            disaster_flag = weather["disaster_flag"],
        )
        tier_idx = dri_to_tier_index(dri)

        row = dict(zip(FEATURE_NAMES, feat_vec.tolist()))
        row["dri_score"]  = round(dri, 4)
        row["tier_label"] = tier_idx
        # Keep metadata for debugging (not used in training)
        row["_city"]      = city
        row["_zone"]      = zone
        row["_condition"] = weather["condition"]
        rows.append(row)

        if (i + 1) % 1000 == 0:
            print(f"  … {i+1}/{n} done")

    df = pd.DataFrame(rows)
    print(f"[SyntheticData] Done. Shape: {df.shape}")
    print(f"  DRI  — mean={df['dri_score'].mean():.2f}, std={df['dri_score'].std():.2f}")
    print(f"  Tier — value_counts:\n{df['tier_label'].value_counts().sort_index()}")
    return df


if __name__ == "__main__":
    df = build_dataset()
    out = "artifacts/synthetic_dataset.csv"
    import os; os.makedirs("artifacts", exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved → {out}")
