"""
predict.py — Unified prediction interface for a gig worker.

Usage (CLI):
    python predict.py                          # random worker
    python predict.py --seed 42               # reproducible
    python predict.py --city Mumbai           # custom city
    python predict.py --disaster              # force disaster scenario
    python predict.py --workers 5 --seed 1   # batch of 5 workers

Outputs a full JSON result with DRI, pricing, and explainability.
"""

import argparse
import json
import sys
from datetime import datetime

from worker_service  import generate_worker, generate_worker_fleet
from feature_engine  import get_full_context
from weather_service import get_weather, fetch_mock_weather
from traffic_service import get_traffic_data
from risk_engine     import score_worker
from premium_engine  import calculate_pricing


# ─── Single Worker ────────────────────────────────────────────────

def predict_single(
    seed:     int  = None,
    city:     str  = "Bengaluru",
    disaster: bool = False,
) -> dict:
    worker, weather, traffic, _ = get_full_context(seed=seed, city=city)

    if disaster:
        weather["disaster_flag"] = True
        weather["weather_risk"]  = 92.0
        weather["rain_mm_hr"]    = 45.0
        weather["wind_kph"]      = 120.0
        weather["condition"]     = "cyclone"

    risk    = score_worker(worker, weather, traffic)
    pricing = calculate_pricing(risk)

    return {
        "worker":        worker,
        "weather":       weather,
        "traffic":       traffic,
        "risk":          risk,
        "pricing":       pricing,
        "generated_at":  datetime.utcnow().isoformat(),
    }


# ─── Batch Workers ────────────────────────────────────────────────

def predict_batch(n: int = 5, seed: int = None, city: str = "Bengaluru") -> list[dict]:
    import random
    if seed is not None:
        random.seed(seed)
    results = []
    for i in range(n):
        s = (seed or 0) + i
        results.append(predict_single(seed=s, city=city))
    return results


# ─── Pretty Printer ───────────────────────────────────────────────

def pretty_print(result: dict):
    w  = result["worker"]
    wx = result["weather"]
    tx = result["traffic"]
    r  = result["risk"]
    p  = result["pricing"]

    sep = "─" * 60

    print(f"\n{sep}")
    print(f"  WORKER: {w['name']}  |  {w['platform']}  |  {w['city']}")
    print(f"  Zone: {w['zone']}  |  Vehicle: {w['vehicle']}  |  Age: {w['age']}")
    print(f"  Hours/day: {w['hours_per_day']}  |  Days/wk: {w['days_per_week']}  "
          f"|  Night shift: {w['night_shift']}")
    print(sep)

    print(f"\n  WEATHER ({wx['source'].upper()}): {wx['condition'].upper()}")
    print(f"    Rain: {wx['rain_mm_hr']} mm/hr  |  Wind: {wx['wind_kph']} km/h  "
          f"|  Risk: {wx['weather_risk']}")
    if wx["disaster_flag"]:
        print("    🚨 DISASTER FLAG: active")

    print(f"\n  TRAFFIC: {tx['density_label'].upper()} density  |  Zone: {tx['risk_class']}")
    print(f"    Peak hour: {tx['is_peak_hour']}  |  Traffic risk: {tx['traffic_risk']}")

    print(f"\n{sep}")
    print(f"  DRI SCORE : {r['dri_score']} / 100  ({r['dri_band'].upper()})")
    print(f"  Model     : {r['model_used']}")
    print(f"  Disaster  : {'YES 🚨' if r['is_disaster'] else 'No'}")

    print(f"\n  Top driver : {r['explanation']['top_driver']}")
    print(f"  Summary    : {r['explanation']['summary']}")

    print(f"\n  RECOMMENDED TIER : {p['recommended_tier']['tier']}")
    print(f"  Weekly Premium   : ₹{p['weekly_premium_inr']}")
    print(f"  Deductible       : ₹{p['deductible_inr']:,}")
    print(f"  Coverage         : ₹{p['coverage_inr']:,}")

    if p["is_disaster"] and p["recommended_tier"].get("high_excess_alternative"):
        alt = p["recommended_tier"]["high_excess_alternative"]
        print(f"\n  💡 ALTERNATIVE (High-Excess Shield):")
        print(f"     ₹{alt['weekly_inr']}/wk | Deductible ₹{alt['deductible']:,}")

    print(f"\n  ALL ELIGIBLE OPTIONS:")
    for opt in p["excess_options"]:
        tag = " ← RECOMMENDED" if opt["is_recommended"] else ""
        print(f"    {opt['tier']:10s} ₹{opt['weekly_premium_inr']:3d}/wk"
              f"  deductible ₹{opt['deductible']:,}"
              f"  coverage ₹{opt['coverage_inr']:,}{tag}")

    print(f"\n  NOTES:")
    for note in p["pricing_notes"]:
        print(f"    {note}")

    print(sep)


# ─── CLI ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Insurance Pricing Predictor")
    parser.add_argument("--seed",     type=int,   default=None)
    parser.add_argument("--city",     type=str,   default="Bengaluru")
    parser.add_argument("--disaster", action="store_true")
    parser.add_argument("--workers",  type=int,   default=1)
    parser.add_argument("--json",     action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.workers == 1:
        result = predict_single(seed=args.seed, city=args.city, disaster=args.disaster)
        if args.json:
            print(json.dumps(result, default=str, indent=2))
        else:
            pretty_print(result)
    else:
        results = predict_batch(n=args.workers, seed=args.seed, city=args.city)
        if args.json:
            print(json.dumps(results, default=str, indent=2))
        else:
            for r in results:
                pretty_print(r)


if __name__ == "__main__":
    main()
