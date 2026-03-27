"""run_benchmark.py — Simple benchmark runner (ASCII output only)."""
import sys, time, random
import numpy as np

random.seed(42)
np.random.seed(42)

import accelerate as ac
from risk_engine import score_worker
from premium_engine import calculate_pricing
from weather_service import fetch_mock_weather
from traffic_service import get_traffic_data
from config import RANDOM_SEED

N = 500

# ── Generate workers ──────────────────────────────────────────────
print("Generating %d workers..." % N)
t0 = time.perf_counter()
workers = ac.generate_workers_parallel(N, seed=42)
gen_t = time.perf_counter() - t0
print("  Done in %.2fs" % gen_t)

# ── Vectorized / ML batch ─────────────────────────────────────────
print("Running vectorized batch score...")
t0 = time.perf_counter()
df = ac.score_batch(workers, city="Bengaluru", use_ml=True)
vec_t = time.perf_counter() - t0
print("  Done in %.4fs" % vec_t)

# ── Loop baseline ─────────────────────────────────────────────────
print("Running loop baseline (reference)...")
t0 = time.perf_counter()
for i, w in enumerate(workers):
    seed = RANDOM_SEED + i
    wx = fetch_mock_weather(city="Bengaluru", seed=seed % 200)
    tx = get_traffic_data(zone=w.get("zone","Suburb"), seed=seed % 100)
    risk = score_worker(w, wx, tx)
    _ = calculate_pricing(risk)
loop_t = time.perf_counter() - t0
print("  Done in %.4fs" % loop_t)

speedup = loop_t / vec_t

# ── Print results ─────────────────────────────────────────────────
print()
print("=" * 55)
print("  ACCELERATED CALCULATION BENCHMARK RESULTS")
print("=" * 55)
print("  Workers scored      : %d" % N)
print("  --")
print("  Vectorized time     : %.4fs" % vec_t)
print("  Throughput          : %d workers/sec" % int(N / vec_t))
print("  --")
print("  Loop time           : %.4fs" % loop_t)
print("  SPEEDUP             : %.1fx" % speedup)
print("=" * 55)
print()
print("DRI Score Statistics:")
print("  Mean  : %.2f" % df["dri_score"].mean())
print("  Std   : %.2f" % df["dri_score"].std())
print("  Min   : %.2f" % df["dri_score"].min())
print("  Max   : %.2f" % df["dri_score"].max())
print()
print("Tier Distribution:")
tc = df["tier"].value_counts().sort_index()
for tier, cnt in tc.items():
    pct = cnt / N * 100
    bar = "#" * int(pct / 2)
    print("  %-12s %4d  (%5.1f%%)  %s" % (tier, cnt, pct, bar))
print()
dis = int(df["is_disaster"].sum())
print("Disaster/Surge cases: %d (%.1f%%)" % (dis, dis/N*100))
print("Model used          : %s" % df["model_used"].iloc[0])
print()
print("Sample output (first 5 workers):")
print("  %-20s %6s  %-10s  %8s  %10s" % ("Name","DRI","Tier","Premium","Deductible"))
print("  " + "-"*63)
for _, row in df.head(5).iterrows():
    print("  %-20s %6.1f  %-10s  Rs%6d  Rs%9d" % (
        str(row["name"])[:20], row["dri_score"], row["tier"],
        row["weekly_premium_inr"], row["deductible_inr"]
    ))
