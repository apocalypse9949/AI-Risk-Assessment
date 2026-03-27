"""
worker_service.py — Simulated gig-worker profiles for grocery delivery platform.
"""

import random
import uuid
from datetime import datetime
from config import INDIA_CITIES
from traffic_service import ZONE_PROFILES


# ─── Realistic Name Pool ──────────────────────────────────────────
FIRST_NAMES = [
    "Arjun", "Ravi", "Suresh", "Priya", "Anita", "Kiran",
    "Vijay", "Meena", "Rahul", "Sunita", "Deepak", "Pooja",
    "Amit", "Kavya", "Naveen", "Divya", "Manoj", "Sneha",
    "Ganesh", "Lakshmi", "Rajesh", "Usha", "Venkat", "Asha",
]
LAST_NAMES = [
    "Kumar", "Sharma", "Reddy", "Nair", "Patel", "Singh",
    "Rao", "Iyer", "Verma", "Pillai", "Mehta", "Gupta",
    "Das", "Joshi", "Naidu", "Mishra", "Shetty", "Bhat",
]

PLATFORMS   = ["Blinkit", "Zepto", "Swiggy Instamart", "Dunzo", "BigBasket Now"]
VEHICLE_TYPES = ["bicycle", "motorcycle", "e-scooter"]

# Hours per day → risk buckets
EXPOSURE_RISK = {
    (0, 4):  ("part_time", 15),
    (4, 7):  ("standard",  40),
    (7, 10): ("full_time", 65),
    (10, 14):("over_time", 90),
}


def _exposure_risk(hours_per_day: float) -> tuple[str, float]:
    for (lo, hi), (label, score) in EXPOSURE_RISK.items():
        if lo <= hours_per_day < hi:
            return label, score
    return "over_time", 90.0


def generate_worker(worker_id: str = None, seed: int = None) -> dict:
    """
    Generate a realistic gig-worker profile.

    Returns
    -------
    dict with worker metadata + pre-computed exposure metrics
    """
    if seed is not None:
        random.seed(seed)

    worker_id  = worker_id or f"GW-{uuid.uuid4().hex[:8].upper()}"
    name       = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    city       = random.choice(list(INDIA_CITIES.keys()))
    zone       = random.choice(list(ZONE_PROFILES.keys()))
    platform   = random.choice(PLATFORMS)
    vehicle    = random.choice(VEHICLE_TYPES)
    age        = random.randint(19, 52)
    experience_months = random.randint(1, 60)

    # Working pattern
    hours_per_day   = round(random.triangular(2, 12, 7), 1)
    days_per_week   = random.randint(3, 7)
    weekly_hours    = round(hours_per_day * days_per_week, 1)

    # Shift preference
    slots = random.sample(["morning", "afternoon", "evening", "night"], k=random.randint(1, 3))
    night_shift = "night" in slots

    # Deliveries
    avg_deliveries_per_hr = round(random.uniform(1.5, 4.5), 2)
    weekly_deliveries     = int(avg_deliveries_per_hr * weekly_hours)

    exp_label, exposure_risk = _exposure_risk(hours_per_day)

    # Night and rain bonus risk
    night_risk_bonus = 15 if night_shift else 0
    vehicle_risk     = {"bicycle": 20, "e-scooter": 10, "motorcycle": 35}[vehicle]

    return {
        "worker_id":          worker_id,
        "name":               name,
        "age":                age,
        "city":               city,
        "zone":               zone,
        "platform":           platform,
        "vehicle":            vehicle,
        "experience_months":  experience_months,
        "hours_per_day":      hours_per_day,
        "days_per_week":      days_per_week,
        "weekly_hours":       weekly_hours,
        "active_slots":       slots,
        "night_shift":        night_shift,
        "avg_deliveries_hr":  avg_deliveries_per_hr,
        "weekly_deliveries":  weekly_deliveries,
        "exposure_label":     exp_label,
        "exposure_risk":      round(exposure_risk + night_risk_bonus, 2),
        "vehicle_risk":       vehicle_risk,
        "registered_at":      datetime.utcnow().isoformat(),
    }


def generate_worker_fleet(n: int = 10, seed: int = None) -> list[dict]:
    """Generate n worker profiles."""
    if seed is not None:
        random.seed(seed)
    return [generate_worker(seed=random.randint(0, 9999)) for _ in range(n)]


if __name__ == "__main__":
    import json
    fleet = generate_worker_fleet(3, seed=42)
    for w in fleet:
        print(json.dumps(w, indent=2))
        print("─" * 60)
