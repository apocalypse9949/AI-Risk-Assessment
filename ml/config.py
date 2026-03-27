"""
config.py — Central configuration for the Insurance Pricing ML System.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── OpenWeatherMap ───────────────────────────────────────────────
OWM_API_KEY = os.getenv("OWM_API_KEY", "")          # leave blank → auto-mock
OWM_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Default city (Bengaluru)
DEFAULT_CITY = "Bengaluru"
DEFAULT_LAT  = 12.9716
DEFAULT_LON  = 77.5946

INDIA_CITIES = {
    "Bengaluru": (12.9716, 77.5946),
    "Mumbai":    (19.0760, 72.8777),
    "Delhi":     (28.6139, 77.2090),
    "Hyderabad": (17.3850, 78.4867),
    "Chennai":   (13.0827, 80.2707),
    "Kolkata":   (22.5726, 88.3639),
    "Pune":      (18.5204, 73.8567),
}

# ─── DRI Weights ─────────────────────────────────────────────────
DRI_WEIGHTS = {
    "weather":  0.30,
    "traffic":  0.25,
    "zone":     0.20,
    "exposure": 0.15,
    "disaster": 0.10,
}

# ─── Premium Tiers (INR / week) ───────────────────────────────────
PREMIUM_TIERS = [
    {
        "tier":       "Basic",
        "weekly_inr": 49,
        "deductible": 5000,
        "coverage":   100000,
        "dri_max":    30,          # available when DRI ≤ 30
        "color":      "#4ADE80",
    },
    {
        "tier":       "Standard",
        "weekly_inr": 99,
        "deductible": 3000,
        "coverage":   250000,
        "dri_max":    50,
        "color":      "#60A5FA",
    },
    {
        "tier":       "Enhanced",
        "weekly_inr": 149,
        "deductible": 1500,
        "coverage":   500000,
        "dri_max":    70,
        "color":      "#FACC15",
    },
    {
        "tier":       "Premium",
        "weekly_inr": 199,
        "deductible": 500,
        "coverage":   750000,
        "dri_max":    85,
        "color":      "#F97316",
    },
    {
        "tier":       "Elite",
        "weekly_inr": 249,
        "deductible": 0,
        "coverage":   1000000,
        "dri_max":    100,
        "color":      "#A855F7",
    },
]

# ─── Disaster Thresholds ──────────────────────────────────────────
DISASTER_DRI_THRESHOLD = 80        # above this → surge pricing
DISASTER_SURGE_FACTOR  = 1.20      # +20 % premium

# ─── ML Model Paths ───────────────────────────────────────────────
MODEL_DIR          = os.path.join(os.path.dirname(__file__), "artifacts")
DRI_MODEL_PATH     = os.path.join(MODEL_DIR, "dri_model.pkl")
TIER_MODEL_PATH    = os.path.join(MODEL_DIR, "tier_model.pkl")
SCALER_PATH        = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

# ─── Synthetic Data ───────────────────────────────────────────────
SYNTHETIC_SAMPLES = 5000
RANDOM_SEED       = 42
