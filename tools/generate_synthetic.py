"""
tools/generate_synthetic.py — Generate a small synthetic pNEUMA-format CSV
for unit-testing and pipeline smoke-tests without the real 15.8 GB dataset.

Creates a realistic Athens-area grid of 9 intersections, routes ~2000 vehicles
through the network, and writes a single pNEUMA-compatible CSV.

Usage
-----
    python tools/generate_synthetic.py --out data/synthetic.csv --vehicles 2000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("synth")

# ── Athens downtown bounding box ──────────────────────────────────────────
LAT_MIN, LAT_MAX = 37.975, 37.990
LON_MIN, LON_MAX = 23.726, 23.742

VEHICLE_TYPES = [
    "Motorcycle", "Automobile", "Automobile", "Automobile",
    "Taxi", "Taxi", "Medium Vehicle", "Heavy Vehicle", "Bus",
]

# ── 3×3 grid of signals ───────────────────────────────────────────────────
SIGNALS = [
    (0, 37.9770, 23.7270),
    (1, 37.9770, 23.7330),
    (2, 37.9770, 23.7390),
    (3, 37.9820, 23.7270),
    (4, 37.9820, 23.7330),
    (5, 37.9820, 23.7390),
    (6, 37.9870, 23.7270),
    (7, 37.9870, 23.7330),
    (8, 37.9870, 23.7390),
]

# Adjacency: each signal → possible next signals
ADJACENCY = {
    0: [1, 3], 1: [0, 2, 4], 2: [1, 5],
    3: [0, 4, 6], 4: [1, 3, 5, 7], 5: [2, 4, 8],
    6: [3, 7], 7: [4, 6, 8], 8: [5, 7],
}

RNG = np.random.default_rng(42)


def _travel_speed_kmh(vtype: str) -> float:
    """Mean travel speed by vehicle type."""
    base = {
        "Motorcycle": 35, "Automobile": 28, "Taxi": 25,
        "Medium Vehicle": 22, "Heavy Vehicle": 18, "Bus": 20,
    }
    mu = base.get(vtype, 25)
    return float(RNG.normal(mu, mu * 0.15))


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a  = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _interpolate_path(
    lat1, lon1, lat2, lon2,
    speed_kmh: float,
    t_start: float,
    freq_s: float = 0.5,
) -> list[tuple]:
    """Return list of (lat, lon, speed, lon_acc, lat_acc, time) tuples."""
    dist_m    = _haversine_m(lat1, lon1, lat2, lon2)
    speed_ms  = speed_kmh / 3.6
    duration  = max(1.0, dist_m / max(speed_ms, 0.1))
    n_steps   = max(3, int(duration / freq_s))

    lats    = np.linspace(lat1, lat2, n_steps)
    lons    = np.linspace(lon1, lon2, n_steps)
    times   = np.linspace(t_start, t_start + duration, n_steps)
    speeds  = np.clip(RNG.normal(speed_kmh, 3, n_steps), 0.5, 80.0)

    # Decelerate near destination (last 20% of steps)
    decel_start = int(n_steps * 0.8)
    speeds[decel_start:] = np.linspace(speeds[decel_start], 1.5, n_steps - decel_start)

    # Brief stop at destination
    stop_dur = float(RNG.uniform(5, 30))
    stop_steps = max(1, int(stop_dur / freq_s))
    stop_lats   = np.full(stop_steps, lat2)
    stop_lons   = np.full(stop_steps, lon2)
    stop_times  = np.linspace(times[-1], times[-1] + stop_dur, stop_steps)
    stop_speeds = np.zeros(stop_steps)

    lats   = np.concatenate([lats, stop_lats])
    lons   = np.concatenate([lons, stop_lons])
    times  = np.concatenate([times, stop_times])
    speeds = np.concatenate([speeds, stop_speeds])

    lon_accs = np.gradient(speeds, times) * 0.1 if len(speeds) > 2 else np.zeros(len(speeds))
    lat_accs = RNG.normal(0, 0.02, len(speeds))

    return list(zip(lats, lons, speeds, lon_accs, lat_accs, times))


def _generate_vehicle(
    track_id: int,
    t_start: float,
    n_hops: int = None,
) -> list:
    """Generate one vehicle's full trajectory as a wide pNEUMA row."""
    vtype    = RNG.choice(VEHICLE_TYPES)
    start_sig = int(RNG.integers(0, len(SIGNALS)))
    if n_hops is None:
        n_hops = int(RNG.integers(2, 6))

    # Route through the network
    route = [start_sig]
    for _ in range(n_hops - 1):
        nxt_candidates = ADJACENCY[route[-1]]
        # Avoid immediate backtracking when possible
        candidates = [s for s in nxt_candidates if s != route[-2]] if len(route) > 1 else nxt_candidates
        route.append(int(RNG.choice(candidates or nxt_candidates)))

    all_points: list[tuple] = []
    t_now = t_start

    for i in range(len(route) - 1):
        _, lat1, lon1 = SIGNALS[route[i]]
        _, lat2, lon2 = SIGNALS[route[i + 1]]
        speed = max(5.0, _travel_speed_kmh(vtype))
        pts = _interpolate_path(lat1, lon1, lat2, lon2, speed, t_now)
        all_points.extend(pts)
        if pts:
            t_now = pts[-1][-1]   # last timestamp

    if not all_points:
        return []

    lats   = [p[0] for p in all_points]
    lons   = [p[1] for p in all_points]
    speeds = [p[2] for p in all_points]

    total_dist  = sum(
        _haversine_m(lats[i], lons[i], lats[i+1], lons[i+1])
        for i in range(len(lats) - 1)
    )
    avg_speed_kmh = float(np.mean([s for s in speeds if s > 0])) if speeds else 0.0

    row_fixed = [track_id, vtype, round(total_dist, 1), round(avg_speed_kmh, 2)]

    row_traj = []
    for p in all_points:
        lat_, lon_, spd_, lacc_, lacc2_, t_ = p
        row_traj.extend([
            round(lat_, 6), round(lon_, 6),
            round(spd_, 2), round(lacc_, 4), round(lacc2_, 4),
            round(t_, 2),
        ])

    return row_fixed + row_traj


def generate_csv(n_vehicles: int, out_path: Path, session_hours: float = 1.0) -> None:
    log.info("Generating %d synthetic vehicles …", n_vehicles)

    rows = []
    max_cols = 0

    for vid in range(n_vehicles):
        t_start = float(RNG.uniform(0, session_hours * 3600 * 0.5))
        row = _generate_vehicle(vid, t_start)
        if row:
            rows.append(row)
            max_cols = max(max_cols, len(row))

    # Pad shorter rows to equal length (pNEUMA format requires rectangular CSV)
    fixed_cols = 4
    padded = []
    for row in rows:
        if len(row) < max_cols:
            row = row + [None] * (max_cols - len(row))
        padded.append(row)

    # Build header
    repeat_groups = (max_cols - fixed_cols) // 6
    header = ["track_id", "type", "traveled_d(m)", "avg_speed(km/h)"]
    for g in range(repeat_groups):
        suffix = f"_{g+1}"
        header += [
            f"lat{suffix}", f"lon{suffix}", f"speed{suffix}",
            f"lon_acc{suffix}", f"lat_acc{suffix}", f"time{suffix}",
        ]

    # Trim or pad header to match columns
    while len(header) < max_cols:
        header.append(f"col_{len(header)}")
    header = header[:max_cols]

    df = pd.DataFrame(padded, columns=header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, sep=";")

    log.info("Written %d vehicles to %s  (%.1f KB)", len(df), out_path,
             out_path.stat().st_size / 1024)


def main():
    p = argparse.ArgumentParser(description="Generate synthetic pNEUMA CSV")
    p.add_argument("--out",      type=Path, default=Path("data/synthetic.csv"))
    p.add_argument("--vehicles", type=int,  default=2000)
    p.add_argument("--hours",    type=float, default=1.0)
    args = p.parse_args()
    generate_csv(args.vehicles, args.out, args.hours)
    log.info("Done. Run: python train.py --data-dir data/")


if __name__ == "__main__":
    main()
