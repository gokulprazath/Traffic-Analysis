"""
pNEUMA Traffic Analysis Script
================================
Processes pNEUMA drone-trajectory CSV files to extract vehicle traffic
passing through a geographic point (e.g. a signalised intersection) at
any given time window.

Dataset source: https://zenodo.org/records/10491409
README: https://open-traffic.epfl.ch/

CSV Structure (per the README):
  - Each row = one vehicle's full trajectory
  - Columns 0-3  : trackID, type, distance_m, avg_speed_kmh
  - Columns 4-9  : lat, lon, speed, lon_acc, lat_acc, time   (t=0)
  - Columns 10-15: lat, lon, speed, lon_acc, lat_acc, time   (t=1)
  - ... repeated every 6 columns

Usage
-----
    python pneuma_traffic_analysis.py \
        --file 20181024_d1_0830_0900.csv \
        --lat 37.9837 \
        --lon 23.7281 \
        --radius 15 \
        --t_start 600 \
        --t_end   660

Or import and call the functions directly in a notebook.
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from typing import Iterator

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two WGS-84 coordinates."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_pneuma_csv(filepath: str) -> Iterator[dict]:
    """
    Lazily parse a pNEUMA CSV file, yielding one dict per vehicle.

    Each dict has:
        track_id  : int
        veh_type  : str
        distance  : float  (metres)
        avg_speed : float  (km/h)
        trajectory: list of dicts with keys lat, lon, speed, lon_acc, lat_acc, time
    """
    with open(filepath, "r") as fh:
        for row_idx, raw in enumerate(fh):
            raw = raw.strip()
            if not raw:
                continue

            parts = raw.split(";") if ";" in raw else raw.split(",")

            # Skip header rows (contain non-numeric trackID)
            try:
                track_id = int(float(parts[0]))
            except (ValueError, IndexError):
                continue

            veh_type  = parts[1].strip() if len(parts) > 1 else "unknown"
            try:
                distance  = float(parts[2])
                avg_speed = float(parts[3])
            except (ValueError, IndexError):
                distance = avg_speed = float("nan")

            # Trajectory columns start at index 4, repeated every 6
            trajectory = []
            idx = 4
            while idx + 5 < len(parts):
                try:
                    lat     = float(parts[idx])
                    lon     = float(parts[idx + 1])
                    speed   = float(parts[idx + 2])
                    lon_acc = float(parts[idx + 3])
                    lat_acc = float(parts[idx + 4])
                    t       = float(parts[idx + 5])
                    trajectory.append(
                        dict(lat=lat, lon=lon, speed=speed,
                             lon_acc=lon_acc, lat_acc=lat_acc, time=t)
                    )
                except (ValueError, IndexError):
                    pass
                idx += 6

            yield {
                "track_id":  track_id,
                "veh_type":  veh_type,
                "distance":  distance,
                "avg_speed": avg_speed,
                "trajectory": trajectory,
            }


# ---------------------------------------------------------------------------
# Core query: vehicles near a point in a time window
# ---------------------------------------------------------------------------

def query_vehicles_at_signal(
    filepath: str,
    signal_lat: float,
    signal_lon: float,
    radius_m: float = 20.0,
    t_start: float = 0.0,
    t_end: float = float("inf"),
) -> pd.DataFrame:
    """
    Return a DataFrame of every vehicle that came within *radius_m* metres of
    (signal_lat, signal_lon) during the interval [t_start, t_end].

    Columns returned
    ----------------
    track_id, veh_type, distance_m (total trip), avg_speed_kmh,
    first_seen_s, last_seen_s, closest_dist_m,
    speed_at_closest_kmh, n_obs_in_zone
    """
    records = []

    for veh in parse_pneuma_csv(filepath):
        hits = []
        for obs in veh["trajectory"]:
            t = obs["time"]
            if t < t_start or t > t_end:
                continue
            d = haversine_m(signal_lat, signal_lon, obs["lat"], obs["lon"])
            if d <= radius_m:
                hits.append((t, d, obs["speed"]))

        if hits:
            times, dists, speeds = zip(*hits)
            min_dist_idx = dists.index(min(dists))
            records.append(
                {
                    "track_id":           veh["track_id"],
                    "veh_type":           veh["veh_type"],
                    "distance_m":         veh["distance"],
                    "avg_speed_kmh":      veh["avg_speed"],
                    "first_seen_s":       min(times),
                    "last_seen_s":        max(times),
                    "closest_dist_m":     round(min(dists), 2),
                    "speed_at_closest_kmh": round(speeds[min_dist_idx], 2),
                    "n_obs_in_zone":      len(hits),
                }
            )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values("first_seen_s").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Aggregate / summary helpers
# ---------------------------------------------------------------------------

def summarise_by_type(df: pd.DataFrame) -> pd.DataFrame:
    """Count vehicles and compute mean speed grouped by vehicle type."""
    if df.empty:
        return df
    return (
        df.groupby("veh_type")
        .agg(
            count=("track_id", "count"),
            mean_speed_kmh=("speed_at_closest_kmh", "mean"),
            mean_closest_dist_m=("closest_dist_m", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("count", ascending=False)
    )


def flow_over_time(
    filepath: str,
    signal_lat: float,
    signal_lon: float,
    radius_m: float = 20.0,
    bin_seconds: float = 60.0,
) -> pd.DataFrame:
    """
    Compute vehicle flow rate (vehicles / bin) over the full recording.

    Returns a DataFrame with columns: bin_start_s, bin_end_s, count, veh_types_counts.
    """
    bin_counts: dict = defaultdict(lambda: defaultdict(int))

    for veh in parse_pneuma_csv(filepath):
        seen_bins: set = set()
        for obs in veh["trajectory"]:
            d = haversine_m(signal_lat, signal_lon, obs["lat"], obs["lon"])
            if d <= radius_m:
                b = int(obs["time"] // bin_seconds)
                if b not in seen_bins:
                    bin_counts[b][veh["veh_type"]] += 1
                    seen_bins.add(b)

    rows = []
    for b, type_counts in sorted(bin_counts.items()):
        rows.append(
            {
                "bin_start_s":    b * bin_seconds,
                "bin_end_s":      (b + 1) * bin_seconds,
                "total":          sum(type_counts.values()),
                **{f"n_{k}": v for k, v in type_counts.items()},
            }
        )
    return pd.DataFrame(rows).fillna(0)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query pNEUMA vehicle traffic at a geographic signal point."
    )
    class args:
        file    = "D:/Innovation/Traffic Analysis/pNEUMA_dataset/20181024_d1_0830_0900.csv"
        lat     = 37.9837
        lon     = 23.7281
        radius  = 20.0
        t_start = 0.0
        t_end   = 300.0
        bin     = 60.0
        out     = "results.csv"

    if not os.path.exists(args.file):
        sys.exit(f"[ERROR] File not found: {args.file}")

    print(f"\n🚦 pNEUMA Traffic Query")
    print(f"   File    : {args.file}")
    print(f"   Signal  : ({args.lat}, {args.lon})")
    print(f"   Radius  : {args.radius} m")
    print(f"   Window  : [{args.t_start}s, {args.t_end}s]\n")

    df = query_vehicles_at_signal(
        filepath=args.file,
        signal_lat=args.lat,
        signal_lon=args.lon,
        radius_m=args.radius,
        t_start=args.t_start,
        t_end=args.t_end,
    )

    if df.empty:
        print("No vehicles found near this point in the specified time window.")
        return

    print(f"✅ {len(df)} vehicles detected\n")
    print("── Per-vehicle detail ──────────────────────────────────────────")
    print(df.to_string(index=False))

    print("\n── Summary by vehicle type ─────────────────────────────────────")
    print(summarise_by_type(df).to_string(index=False))

    print(f"\n── Flow over time (bin = {args.bin}s) ──────────────────────────")
    flow_df = flow_over_time(
        filepath=args.file,
        signal_lat=args.lat,
        signal_lon=args.lon,
        radius_m=args.radius,
        bin_seconds=args.bin,
    )
    print(flow_df.to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n💾 Results saved to {args.out}")


if __name__ == "__main__":
    main()