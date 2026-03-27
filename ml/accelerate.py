"""
accelerate.py — Vectorized batch DRI + Premium calculation engine.

Why this exists
───────────────
The standard predict.py scores workers one-by-one (Python loops).
This module processes thousands of workers in a single NumPy pass,
making it suitable for real-time fleet re-pricing and bulk actuarial runs.

Key techniques
──────────────
  • Fully vectorized NumPy arithmetic (no Python loops for scoring)
  • Pre-computed tier boundary lookup via np.searchsorted
  • LRU-cached weather / traffic calls (avoids redundant API hits)
  • Parallel synthetic batch generation with ThreadPoolExecutor
  • Built-in benchmarker comparing vectorized vs. loop speed

Usage (CLI)
───────────
  python accelerate.py                          # benchmark with 1,000 workers
  python accelerate.py --n 5000                # larger batch
  python accelerate.py --n 100 --print         # show individual results
  python accelerate.py --city Mumbai --n 200   # city-specific batch
"""

import os
import sys
import time
import json
import argparse
import random
import warnings

import numpy as np
import pandas as pd
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

# ── local imports ──────────────────────────────────────────────────
from config import (
    DRI_WEIGHTS, PREMIUM_TIERS, DISASTER_SURGE_FACTOR,
    DISASTER_DRI_THRESHOLD, RANDOM_SEED, DRI_MODEL_PATH,
    SCALER_PATH, TIER_MODEL_PATH,
)
from weather_service import fetch_mock_weather
from traffic_service import get_traffic_data, ZONE_PROFILES
from worker_service  import generate_worker
from feature_engine  import extract_features, FEATURE_NAMES


# ══════════════════════════════════════════════════════════════════
# 0.  CONSTANTS & PRE-COMPUTED LOOKUP TABLES
# ══════════════════════════════════════════════════════════════════

# Weight vector (shape 5,) in component order: weather, traffic, zone, exposure, disaster
W = np.array([
    DRI_WEIGHTS["weather"],
    DRI_WEIGHTS["traffic"],
    DRI_WEIGHTS["zone"],
    DRI_WEIGHTS["exposure"],
    DRI_WEIGHTS["disaster"],
], dtype=np.float32)

# Tier boundary array: sorted dri_max values for searchsorted
TIER_BOUNDARIES = np.array([t["dri_max"] for t in PREMIUM_TIERS], dtype=np.float32)
TIER_PREMIUMS   = np.array([t["weekly_inr"] for t in PREMIUM_TIERS], dtype=np.float32)
TIER_DEDUCTS    = np.array([t["deductible"] for t in PREMIUM_TIERS], dtype=np.float32)
TIER_COVERAGE   = np.array([t["coverage"] for t in PREMIUM_TIERS], dtype=np.float32)
TIER_NAMES_ARR  = np.array([t["tier"] for t in PREMIUM_TIERS])

# Feature indices inside the 15-dim vector
_FI = {name: i for i, name in enumerate(FEATURE_NAMES)}
IDX_WEATHER_RISK = _FI["weather_risk"]
IDX_TRAFFIC_RISK = _FI["traffic_risk"]
IDX_ZONE_RISK    = _FI["zone_risk"]
IDX_EXPOSURE     = _FI["exposure_risk"]
IDX_DISASTER     = _FI["disaster_flag"]


# ══════════════════════════════════════════════════════════════════
# 1.  CACHED DATA LOADERS
# ══════════════════════════════════════════════════════════════════

@lru_cache(maxsize=256)
def _cached_weather(city: str, seed: int) -> tuple:
    """Cache weather payloads to avoid redundant calls in large batches."""
    w = fetch_mock_weather(city=city, seed=seed)
    return (
        w["weather_risk"],
        w["rain_mm_hr"],
        w["wind_kph"],
        w["visibility_m"] / 10000.0,
        float(w["disaster_flag"]),
    )


@lru_cache(maxsize=512)
def _cached_traffic(zone: str, hour: int, seed: int) -> tuple:
    """Cache traffic payloads."""
    t = get_traffic_data(zone=zone, hour=hour, seed=seed)
    density_enc = {"low": 0, "medium": 1, "high": 2}.get(t["density_label"], 1)
    return (
        t["traffic_risk"],
        t["zone_risk"],
        float(t["is_peak_hour"]),
        float(density_enc),
    )


# ══════════════════════════════════════════════════════════════════
# 2.  BATCH FEATURE MATRIX BUILDER
# ══════════════════════════════════════════════════════════════════

def build_feature_matrix(workers: list[dict], city: str = "Bengaluru") -> np.ndarray:
    """
    Build an (N × 15) float32 feature matrix for N workers in one pass.

    Caching ensures duplicate city/zone/hour combinations
    hit memory instead of re-running mock logic.
    """
    n   = len(workers)
    mat = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    for i, w in enumerate(workers):
        seed = RANDOM_SEED + i
        hour = (i * 3 + 8) % 24          # spread workers across day

        # ── Weather (cached per city × seed bucket) ───────────
        wx_seed = seed % 200              # bucket → more cache hits
        wx = _cached_weather(city, wx_seed)

        # ── Traffic (cached per zone × hour × seed bucket) ────
        zone     = w.get("zone", "Suburb")
        tx_seed  = seed % 100
        tx = _cached_traffic(zone, hour, tx_seed)

        # ── Worker-level features ─────────────────────────────
        mat[i, _FI["weather_risk"]]       = wx[0]
        mat[i, _FI["rain_mm_hr"]]         = wx[1]
        mat[i, _FI["wind_kph"]]           = wx[2]
        mat[i, _FI["visibility_m_norm"]]  = wx[3]
        mat[i, _FI["disaster_flag"]]      = wx[4]
        mat[i, _FI["traffic_risk"]]       = tx[0]
        mat[i, _FI["zone_risk"]]          = tx[1]
        mat[i, _FI["is_peak_hour"]]       = tx[2]
        mat[i, _FI["density_encoded"]]    = tx[3]
        mat[i, _FI["exposure_risk"]]      = w.get("exposure_risk",  0.0)
        mat[i, _FI["vehicle_risk"]]       = w.get("vehicle_risk",   0.0)
        mat[i, _FI["weekly_hours_norm"]]  = min(w.get("weekly_hours", 0.0), 98) / 98.0
        mat[i, _FI["experience_norm"]]    = min(w.get("experience_months", 0), 60) / 60.0
        mat[i, _FI["night_shift"]]        = float(w.get("night_shift", False))
        mat[i, _FI["days_per_week"]]      = float(w.get("days_per_week", 5))

    return mat


# ══════════════════════════════════════════════════════════════════
# 3.  VECTORIZED DRI SCORER
# ══════════════════════════════════════════════════════════════════

def vectorized_dri(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Compute DRI for N workers in a single matrix multiply.

    DRI = W[0]*weather + W[1]*traffic + W[2]*zone + W[3]*exposure + W[4]*disaster*90

    Parameters
    ----------
    feature_matrix : (N, 15) float32

    Returns
    -------
    dri : (N,) float32, clipped to [0, 100]
    """
    # Extract the 5 risk components (slice columns)
    weather_risk  = feature_matrix[:, IDX_WEATHER_RISK]
    traffic_risk  = feature_matrix[:, IDX_TRAFFIC_RISK]
    zone_risk     = feature_matrix[:, IDX_ZONE_RISK]
    exposure_risk = feature_matrix[:, IDX_EXPOSURE]
    disaster      = feature_matrix[:, IDX_DISASTER] * 90.0   # flag → score

    dri = (
        W[0] * weather_risk  +
        W[1] * traffic_risk  +
        W[2] * zone_risk     +
        W[3] * exposure_risk +
        W[4] * disaster
    )
    return np.clip(dri, 0.0, 100.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# 4.  ML-ACCELERATED DRI (XGBoost batch inference)
# ══════════════════════════════════════════════════════════════════

_xgb_dri_model  = None
_xgb_tier_model = None
_scaler         = None

def _try_load_models() -> bool:
    global _xgb_dri_model, _xgb_tier_model, _scaler
    if _xgb_dri_model is not None:
        return True
    try:
        import joblib
        if all(os.path.exists(p) for p in [DRI_MODEL_PATH, TIER_MODEL_PATH, SCALER_PATH]):
            _xgb_dri_model  = joblib.load(DRI_MODEL_PATH)
            _xgb_tier_model = joblib.load(TIER_MODEL_PATH)
            _scaler         = joblib.load(SCALER_PATH)
            return True
    except Exception as e:
        print(f"[Accelerate] Model load failed: {e}")
    return False


def ml_batch_dri(feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Use XGBoost to score entire batch in one call (most efficient path).

    Returns
    -------
    dri_scores : (N,) float32
    tier_indices: (N,) int32
    """
    scaled = _scaler.transform(feature_matrix)
    dri    = np.clip(_xgb_dri_model.predict(scaled), 0.0, 100.0).astype(np.float32)
    tiers  = _xgb_tier_model.predict(scaled).astype(np.int32)
    return dri, tiers


# ══════════════════════════════════════════════════════════════════
# 5.  VECTORIZED TIER + PREMIUM CALCULATOR
# ══════════════════════════════════════════════════════════════════

def vectorized_tiers(dri_scores: np.ndarray) -> np.ndarray:
    """
    Map N DRI scores → tier indices using binary search (O(N log K)).

    np.searchsorted is ~200× faster than a Python loop over PREMIUM_TIERS.
    """
    # searchsorted gives the insertion index = first boundary that dri <= boundary
    indices = np.searchsorted(TIER_BOUNDARIES, dri_scores, side="left")
    return np.clip(indices, 0, len(PREMIUM_TIERS) - 1).astype(np.int32)


def vectorized_premiums(
    tier_indices:  np.ndarray,
    disaster_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Look up premiums, deductibles and coverage for N workers.
    Apply disaster surge where flagged.

    Returns
    -------
    premiums    : (N,) float32  weekly INR
    deductibles : (N,) float32
    coverages   : (N,) float32
    surged      : (N,) bool     True if disaster surge applied
    """
    premiums    = TIER_PREMIUMS[tier_indices].copy()
    deductibles = TIER_DEDUCTS[tier_indices].copy()
    coverages   = TIER_COVERAGE[tier_indices].copy()

    # Disaster surge: ceil( premium * 1.20 )
    surged   = disaster_mask.astype(bool)
    premiums[surged] = np.ceil(premiums[surged] * DISASTER_SURGE_FACTOR).astype(np.float32)

    return premiums, deductibles, coverages, surged


# ══════════════════════════════════════════════════════════════════
# 6.  MAIN BATCH SCORER
# ══════════════════════════════════════════════════════════════════

def score_batch(
    workers: list[dict],
    city:    str = "Bengaluru",
    use_ml:  bool = True,
) -> pd.DataFrame:
    """
    Score a batch of N workers.

    Returns a DataFrame with one row per worker containing:
      worker_id, name, zone, dri_score, dri_band,
      tier, weekly_premium_inr, deductible_inr, coverage_inr,
      is_disaster, model_used
    """
    # ── 1. Feature matrix ─────────────────────────────────────
    mat = build_feature_matrix(workers, city=city)

    # ── 2. DRI scoring ────────────────────────────────────────
    model_loaded = use_ml and _try_load_models()

    if model_loaded:
        dri_scores, tier_indices = ml_batch_dri(mat)
        model_used = "xgboost"
    else:
        dri_scores   = vectorized_dri(mat)
        tier_indices = vectorized_tiers(dri_scores)
        model_used   = "vectorized_rule"

    # ── 3. Disaster mask ──────────────────────────────────────
    disaster_from_score = dri_scores >= DISASTER_DRI_THRESHOLD
    disaster_from_flag  = mat[:, IDX_DISASTER].astype(bool)
    disaster_mask       = disaster_from_score | disaster_from_flag

    # ── 4. Premium lookup ─────────────────────────────────────
    premiums, deductibles, coverages, surged = vectorized_premiums(
        tier_indices, disaster_mask
    )

    # ── 5. DRI band (vectorized string mapping) ───────────────
    bands = np.select(
        [dri_scores < 25, dri_scores < 55, dri_scores < 80],
        ["low",           "medium",        "high"],
        default="extreme",
    )

    # ── 6. Assemble DataFrame ─────────────────────────────────
    records = []
    for i, w in enumerate(workers):
        records.append({
            "worker_id":        w.get("worker_id",  f"W{i:05d}"),
            "name":             w.get("name",        "Unknown"),
            "city":             w.get("city",        city),
            "zone":             w.get("zone",        "Suburb"),
            "vehicle":          w.get("vehicle",     "motorcycle"),
            "weekly_hours":     w.get("weekly_hours", 0),
            "night_shift":      w.get("night_shift",  False),
            "dri_score":        round(float(dri_scores[i]), 2),
            "dri_band":         str(bands[i]),
            "tier":             str(TIER_NAMES_ARR[tier_indices[i]]),
            "weekly_premium_inr": int(premiums[i]),
            "deductible_inr":   int(deductibles[i]),
            "coverage_inr":     int(coverages[i]),
            "is_disaster":      bool(disaster_mask[i]),
            "surge_applied":    bool(surged[i]),
            "model_used":       model_used,
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════
# 7.  PARALLEL WORKER GENERATION (for large benchmarks)
# ══════════════════════════════════════════════════════════════════

def generate_workers_parallel(n: int, seed: int = RANDOM_SEED) -> list[dict]:
    """Generate N workers using a thread pool (I/O-bound friendly)."""
    random.seed(seed)
    seeds = [seed + i for i in range(n)]

    def _gen(s):
        return generate_worker(seed=s)

    workers = []
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
        futures = {ex.submit(_gen, s): s for s in seeds}
        for fut in as_completed(futures):
            workers.append(fut.result())

    # Sort by seed order for reproducibility
    workers.sort(key=lambda w: w.get("worker_id", ""))
    return workers


# ══════════════════════════════════════════════════════════════════
# 8.  BENCHMARKER
# ══════════════════════════════════════════════════════════════════

def _loop_score(workers: list[dict], city: str) -> pd.DataFrame:
    """Reference implementation: score one by one (slow baseline)."""
    from risk_engine    import score_worker
    from premium_engine import calculate_pricing
    from weather_service import fetch_mock_weather
    from traffic_service import get_traffic_data

    rows = []
    for i, w in enumerate(workers):
        seed    = RANDOM_SEED + i
        weather = fetch_mock_weather(city=city, seed=seed % 200)
        traffic = get_traffic_data(zone=w.get("zone","Suburb"), seed=seed % 100)
        risk    = score_worker(w, weather, traffic)
        pricing = calculate_pricing(risk)
        rows.append({
            "worker_id":          w["worker_id"],
            "dri_score":          risk["dri_score"],
            "tier":               pricing["recommended_tier"]["tier"],
            "weekly_premium_inr": pricing["weekly_premium_inr"],
        })
    return pd.DataFrame(rows)


def benchmark(n: int = 1000, city: str = "Bengaluru"):
    """Compare vectorized vs. loop scoring and print a timing table."""
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  Accelerated Calculation Benchmark  |  N={n}  |  {city}")
    print(sep)

    # ── Generate workers ──────────────────────────────────────
    print(f"\n  Generating {n} worker profiles …", end=" ", flush=True)
    t0 = time.perf_counter()
    workers = generate_workers_parallel(n)
    gen_time = time.perf_counter() - t0
    print(f"done in {gen_time:.2f}s")

    # ── Vectorized batch ──────────────────────────────────────
    print(f"  Vectorized batch scoring …",  end=" ", flush=True)
    t0 = time.perf_counter()
    df_vec = score_batch(workers, city=city, use_ml=True)
    vec_time = time.perf_counter() - t0
    print(f"done in {vec_time:.4f}s")

    # ── Loop baseline (only for small n to avoid long wait) ───
    loop_time = None
    if n <= 500:
        print(f"  Loop baseline scoring …",    end=" ", flush=True)
        t0 = time.perf_counter()
        df_loop = _loop_score(workers, city=city)
        loop_time = time.perf_counter() - t0
        print(f"done in {loop_time:.4f}s")

    # ── Results ───────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  {'Metric':<30} {'Value':>15}")
    print(f"{'─'*60}")
    print(f"  {'Workers scored':<30} {n:>15,}")
    print(f"  {'Vectorized time (s)':<30} {vec_time:>15.4f}")
    print(f"  {'Throughput (workers/sec)':<30} {n/vec_time:>15,.0f}")
    if loop_time:
        speedup = loop_time / vec_time
        print(f"  {'Loop time (s)':<30} {loop_time:>15.4f}")
        print(f"  {'Speedup':<30} {speedup:>14.1f}×")
    print(f"{'─'*60}")

    # ── DRI Distribution ──────────────────────────────────────
    print(f"\n  DRI Score Distribution:")
    print(f"    Mean  : {df_vec['dri_score'].mean():.2f}")
    print(f"    Std   : {df_vec['dri_score'].std():.2f}")
    print(f"    Min   : {df_vec['dri_score'].min():.2f}")
    print(f"    Max   : {df_vec['dri_score'].max():.2f}")

    # ── Tier Distribution ─────────────────────────────────────
    print(f"\n  Tier Distribution:")
    tier_counts = df_vec["tier"].value_counts().sort_index()
    for tier, count in tier_counts.items():
        pct = count / n * 100
        bar = "█" * int(pct / 2)
        print(f"    {tier:<12} {count:>5} ({pct:5.1f}%)  {bar}")

    # ── Disaster Stats ────────────────────────────────────────
    dis_count = df_vec["is_disaster"].sum()
    print(f"\n  Disaster / Surge cases : {dis_count} ({dis_count/n*100:.1f}%)")
    print(f"  Model used             : {df_vec['model_used'].iloc[0]}")

    print(f"\n{sep}\n")
    return df_vec


# ══════════════════════════════════════════════════════════════════
# 9.  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Accelerated Insurance Pricing Engine")
    parser.add_argument("--n",      type=int, default=1000,        help="Number of workers to score")
    parser.add_argument("--city",   type=str, default="Bengaluru", help="City for weather context")
    parser.add_argument("--seed",   type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--print",  action="store_true",           help="Print individual results")
    parser.add_argument("--csv",    type=str, default="",          help="Save results to CSV path")
    parser.add_argument("--no-ml",  action="store_true",           help="Force rule-based (skip ML)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Run benchmark
    df = benchmark(n=args.n, city=args.city)

    # Optional: print individual rows
    if args.print:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 160)
        print(df[["name", "zone", "dri_score", "dri_band",
                   "tier", "weekly_premium_inr", "deductible_inr",
                   "is_disaster", "surge_applied"]].to_string(index=False))

    # Optional: save CSV
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\n  Results saved → {args.csv}")


if __name__ == "__main__":
    main()
