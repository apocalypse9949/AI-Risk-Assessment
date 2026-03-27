"""
weather_service.py — Fetch live weather or fall back to realistic mock data.
"""

import random
import requests
from datetime import datetime
from config import OWM_API_KEY, OWM_BASE_URL, DEFAULT_LAT, DEFAULT_LON, INDIA_CITIES


# ─── Helpers ──────────────────────────────────────────────────────

def _rain_risk(rain_mm: float) -> float:
    """Convert mm/hr rainfall to a 0-100 risk score."""
    if rain_mm == 0:    return 0.0
    if rain_mm < 2.5:   return 15.0   # light drizzle
    if rain_mm < 7.5:   return 35.0   # moderate
    if rain_mm < 15.0:  return 60.0   # heavy
    if rain_mm < 30.0:  return 80.0   # very heavy
    return 100.0                       # extreme / cyclone-level


def _wind_risk(wind_kph: float) -> float:
    if wind_kph < 20:   return 0.0
    if wind_kph < 40:   return 20.0
    if wind_kph < 65:   return 50.0
    if wind_kph < 90:   return 75.0
    return 100.0


def _visibility_risk(vis_m: float) -> float:
    if vis_m >= 10000:  return 0.0
    if vis_m >= 5000:   return 20.0
    if vis_m >= 1000:   return 50.0
    return 80.0


# ─── Live API ──────────────────────────────────────────────────────

def fetch_live_weather(city: str = "Bengaluru") -> dict:
    """Call OpenWeatherMap current-weather endpoint."""
    lat, lon = INDIA_CITIES.get(city, (DEFAULT_LAT, DEFAULT_LON))
    url = f"{OWM_BASE_URL}/weather"
    params = {
        "lat": lat, "lon": lon,
        "appid": OWM_API_KEY,
        "units": "metric",
    }
    resp = requests.get(url, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    rain_mm   = data.get("rain", {}).get("1h", 0.0)
    wind_kph  = data.get("wind", {}).get("speed", 0) * 3.6
    vis_m     = data.get("visibility", 10000)
    condition = data["weather"][0]["main"].lower()
    temp_c    = data["main"]["temp"]

    disaster_flag = condition in ("thunderstorm", "tornado", "squall")

    weather_risk = (
        0.50 * _rain_risk(rain_mm) +
        0.30 * _wind_risk(wind_kph) +
        0.20 * _visibility_risk(vis_m)
    )

    return {
        "source":        "live",
        "city":          city,
        "condition":     condition,
        "temp_c":        round(temp_c, 1),
        "rain_mm_hr":    round(rain_mm, 2),
        "wind_kph":      round(wind_kph, 1),
        "visibility_m":  vis_m,
        "disaster_flag": disaster_flag,
        "weather_risk":  round(weather_risk, 2),
        "timestamp":     datetime.utcnow().isoformat(),
    }


# ─── Realistic Mock ───────────────────────────────────────────────

_SCENARIOS = [
    # (weight, label, rain, wind_kph, vis_m, disaster)
    (35, "clear",        0.0,  10, 10000, False),
    (20, "light_rain",   2.0,  20,  7000, False),
    (15, "moderate_rain",7.0,  30,  5000, False),
    (10, "heavy_rain",  15.0,  45,  2000, False),
    ( 8, "fog",          0.0,  10,   500, False),
    ( 5, "strong_wind",  0.0,  70,  8000, False),
    ( 4, "thunderstorm",20.0,  80,  1500, True),
    ( 2, "cyclone",     40.0, 110,   500, True),
    ( 1, "extreme",     50.0, 130,   200, True),
]

def fetch_mock_weather(city: str = "Bengaluru", seed: int = None) -> dict:
    """Generate a statistically realistic mock weather reading."""
    if seed is not None:
        random.seed(seed)

    weights  = [s[0] for s in _SCENARIOS]
    scenario = random.choices(_SCENARIOS, weights=weights, k=1)[0]
    _, label, base_rain, base_wind, base_vis, disaster = scenario

    rain_mm  = max(0, base_rain  + random.gauss(0, base_rain * 0.15 + 0.1))
    wind_kph = max(0, base_wind  + random.gauss(0, 5))
    vis_m    = max(100, base_vis + random.randint(-500, 500))
    temp_c   = round(random.uniform(18, 38), 1)

    weather_risk = (
        0.50 * _rain_risk(rain_mm) +
        0.30 * _wind_risk(wind_kph) +
        0.20 * _visibility_risk(vis_m)
    )

    return {
        "source":        "mock",
        "city":          city,
        "condition":     label,
        "temp_c":        temp_c,
        "rain_mm_hr":    round(rain_mm, 2),
        "wind_kph":      round(wind_kph, 1),
        "visibility_m":  int(vis_m),
        "disaster_flag": disaster,
        "weather_risk":  round(weather_risk, 2),
        "timestamp":     datetime.utcnow().isoformat(),
    }


# ─── Auto-routing ─────────────────────────────────────────────────

def get_weather(city: str = "Bengaluru", seed: int = None) -> dict:
    """Returns live weather if API key present, else mock."""
    if OWM_API_KEY:
        try:
            return fetch_live_weather(city)
        except Exception as e:
            print(f"[WeatherService] Live API failed ({e}), falling back to mock.")
    return fetch_mock_weather(city, seed=seed)


if __name__ == "__main__":
    import json
    print(json.dumps(get_weather("Mumbai"), indent=2))
