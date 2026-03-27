"""
premium_engine.py — Calculate eligible premium tiers + excess options.

Applies:
  - Base tier from DRI score (or ML model output)
  - Disaster surge (+20 %)
  - Excess optimisation (higher premium → lower deductible)
"""

import math
from config import PREMIUM_TIERS, DISASTER_SURGE_FACTOR, DISASTER_DRI_THRESHOLD


# ─── Tier Eligibility ─────────────────────────────────────────────

def get_eligible_tiers(dri_score: float, recommended_tier_idx: int = None) -> list[dict]:
    """
    Return all tiers eligible for this worker.

    Eligibility rule: a tier is available if the worker's DRI ≤ tier's dri_max.
    The recommended tier is the highest eligible tier.
    """
    eligible = [t for t in PREMIUM_TIERS if dri_score <= t["dri_max"]]
    if not eligible:
        # Even at extreme DRI, always offer Elite
        eligible = [PREMIUM_TIERS[-1]]
    return eligible


def get_recommended_tier(dri_score: float) -> dict:
    """Return the highest-value tier the worker qualifies for."""
    eligible = get_eligible_tiers(dri_score)
    return eligible[-1]   # last = highest coverage


# ─── Disaster Surge ───────────────────────────────────────────────

def apply_disaster_adjustment(tier: dict, is_disaster: bool, dri_score: float) -> dict:
    """
    When a disaster is active, adjust pricing and add an alternative option.
    Returns an enriched copy of the tier dict.
    """
    tier = dict(tier)   # copy — don't mutate original
    tier["disaster_adjusted"] = False
    tier["surge_reason"]      = None

    if not is_disaster:
        return tier

    original_price = tier["weekly_inr"]
    surged_price   = math.ceil(original_price * DISASTER_SURGE_FACTOR)

    tier["weekly_inr"]        = surged_price
    tier["original_inr"]      = original_price
    tier["disaster_adjusted"] = True
    tier["surge_percent"]     = int((DISASTER_SURGE_FACTOR - 1) * 100)
    tier["surge_reason"]      = "Extreme weather / disaster alert active in your area"

    # ── Alternative: High-Excess Shield ───────────────────────
    tier["high_excess_alternative"] = {
        "weekly_inr": original_price,          # original price (no surge)
        "deductible": tier["deductible"] + 3000,
        "coverage":   tier["coverage"],
        "label":      "High-Excess Shield",
        "note":       "Keep original premium but accept higher deductible",
    }

    return tier


# ─── Excess Optimisation ─────────────────────────────────────────

def build_excess_options(recommended_tier: dict, eligible_tiers: list[dict]) -> list[dict]:
    """
    Build a sorted list of premium + deductible combinations.

    Lower premium  → higher deductible
    Higher premium → lower (or zero) deductible
    """
    options = []
    for tier in eligible_tiers:
        options.append({
            "tier":         tier["tier"],
            "weekly_inr":   tier["weekly_inr"],
            "deductible":   tier["deductible"],
            "coverage_inr": tier["coverage"],
            "is_recommended": tier["tier"] == recommended_tier["tier"],
            "savings_vs_recommended": recommended_tier["weekly_inr"] - tier["weekly_inr"],
            "extra_deductible":       tier["deductible"] - recommended_tier["deductible"],
        })
    return sorted(options, key=lambda x: x["weekly_inr"])


# ─── Full Pricing Result ──────────────────────────────────────────

def calculate_pricing(risk_result: dict) -> dict:
    """
    Main entry: given risk_result from risk_engine.score_worker(),
    return the complete pricing payload.

    Parameters
    ----------
    risk_result : dict as returned by risk_engine.score_worker()

    Returns
    -------
    dict with:
        recommended_tier     dict   (possibly disaster-adjusted)
        eligible_tiers       list   all available tiers
        excess_options       list   sorted premium+excess combos
        weekly_premium_inr   float  final recommended premium
        deductible_inr       float  deductible for recommended tier
        coverage_inr         float  max coverage
        is_disaster          bool
        pricing_notes        list[str] human-readable notes
    """
    dri           = risk_result["dri_score"]
    is_disaster   = risk_result["is_disaster"]
    rec_tier_idx  = risk_result["recommended_tier"]

    # ── Eligibility ────────────────────────────────────────────
    eligible      = get_eligible_tiers(dri)
    recommended   = eligible[-1]   # highest eligible

    # ── Disaster Adjustment ────────────────────────────────────
    recommended   = apply_disaster_adjustment(recommended, is_disaster, dri)

    # ── Excess Options ─────────────────────────────────────────
    excess_opts   = build_excess_options(recommended, eligible)

    # ── Pricing Notes ──────────────────────────────────────────
    notes = []
    if dri < 25:
        notes.append("✅ Low risk profile — best rates available")
    elif dri < 55:
        notes.append("🔵 Moderate risk — standard pricing applies")
    elif dri < 80:
        notes.append("🟠 Elevated risk — consider higher coverage")
    else:
        notes.append("🔴 High risk detected — premium rates apply")

    if is_disaster:
        notes.append(f"⚠️ Disaster surge of {recommended.get('surge_percent', 20)}% applied due to extreme conditions")
        notes.append("💡 Tip: Choose 'High-Excess Shield' to keep original premium")

    if risk_result["explanation"]["top_driver"] == "weather_risk":
        notes.append("🌧️ Weather is your primary risk driver today")
    elif risk_result["explanation"]["top_driver"] == "traffic_risk":
        notes.append("🚦 Traffic congestion is your primary risk driver")
    elif risk_result["explanation"]["top_driver"] == "exposure_risk":
        notes.append("⏰ Long working hours are your primary risk driver")

    return {
        "recommended_tier":   recommended,
        "eligible_tiers":     eligible,
        "excess_options":     excess_opts,
        "weekly_premium_inr": recommended["weekly_inr"],
        "deductible_inr":     recommended["deductible"],
        "coverage_inr":       recommended["coverage"],
        "is_disaster":        is_disaster,
        "dri_score":          dri,
        "dri_band":           risk_result["dri_band"],
        "model_used":         risk_result["model_used"],
        "pricing_notes":      notes,
        "explanation":        risk_result["explanation"],
    }


if __name__ == "__main__":
    from feature_engine import get_full_context
    from risk_engine    import score_worker
    import json

    # ── Sample: high-risk scenario ─────────────────────────────
    worker, weather, traffic, _ = get_full_context(seed=99)
    # Force disaster for demo
    weather["disaster_flag"] = True
    weather["weather_risk"]  = 95
    weather["condition"]     = "cyclone"

    risk    = score_worker(worker, weather, traffic)
    pricing = calculate_pricing(risk)

    print(f"\nWorker   : {worker['name']}")
    print(f"DRI      : {pricing['dri_score']} ({pricing['dri_band']})")
    print(f"Disaster : {pricing['is_disaster']}")
    print(f"Recommended Tier : {pricing['recommended_tier']['tier']}")
    print(f"Weekly Premium   : ₹{pricing['weekly_premium_inr']}")
    print(f"Deductible       : ₹{pricing['deductible_inr']}")
    print(f"Coverage         : ₹{pricing['coverage_inr']:,}")
    print("\nEligible Options:")
    for opt in pricing["excess_options"]:
        tag = " ← RECOMMENDED" if opt["is_recommended"] else ""
        print(f"  {opt['tier']:10s} | ₹{opt['weekly_premium_inr']:3d}/wk | deductible ₹{opt['deductible']:,}{tag}")
    print("\nNotes:")
    for n in pricing["pricing_notes"]:
        print(" ", n)
