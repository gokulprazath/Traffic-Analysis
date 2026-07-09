"""
pipeline/arm_detector.py — Detect unique road arms per traffic signal.

For each signal, we look at every vehicle that stopped there and compute the
compass bearing of its approach direction (bearing from the previous GPS point
to the stop point).  We then cluster those bearings using agglomerative
clustering (angular distance, wrapping at 360°).  Each resulting cluster is
one road arm (leg) entering the intersection.

Output
------
arm_table.parquet   — (signal_id, arm_id, mean_bearing, n_vehicles)
stop_arm_map.parquet— (stop_event_index, signal_id, arm_id)  — used in feature extraction
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fclusterdata

from config import (
    ARM_BEARING_EPS_DEG,
    ARM_MIN_VEHICLES,
    DETECTION_RADIUS_M,
    EARTH_RADIUS_M,
    MODEL_DIR,
)

log = logging.getLogger(__name__)
ARM_TABLE_PATH    = MODEL_DIR / "arms.parquet"
STOP_ARM_MAP_PATH = MODEL_DIR / "stop_arm_map.parquet"


# ── Geometry helpers ────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised haversine distance in metres."""
    R = EARTH_RADIUS_M
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dφ = np.radians(lat2 - lat1)
    dλ = np.radians(lon2 - lon1)
    a  = np.sin(dφ / 2) ** 2 + np.cos(φ1) * np.cos(φ2) * np.sin(dλ / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Bearing from point 1 → point 2, in degrees [0, 360)."""
    φ1, φ2 = np.radians(lat1), np.radians(lat2)
    dλ = np.radians(lon2 - lon1)
    x  = np.sin(dλ) * np.cos(φ2)
    y  = np.cos(φ1) * np.sin(φ2) - np.sin(φ1) * np.cos(φ2) * np.cos(dλ)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def _angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest angle between two bearing vectors (0–180°)."""
    diff = np.abs(a - b) % 360
    return np.minimum(diff, 360 - diff)


def _cluster_bearings(bearings: np.ndarray) -> np.ndarray:
    """
    Cluster an array of approach bearings using average-linkage agglomerative
    clustering with angular distance.  Returns integer cluster labels.
    """
    if len(bearings) < 2:
        return np.zeros(len(bearings), dtype=int)

    # fclusterdata expects (n, 1) matrix
    b = bearings.reshape(-1, 1)
    labels = fclusterdata(
        b,
        t=ARM_BEARING_EPS_DEG,
        criterion="distance",
        metric=lambda u, v: _angular_distance(u, v)[0],
        method="average",
    )
    return labels - 1   # make 0-indexed


# ── Main entry point ────────────────────────────────────────────────────────

def detect_arms(stops_annotated: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """
    Detect road arms for every signal and persist tables.

    Parameters
    ----------
    stops_annotated : pd.DataFrame
        Stop events with (lat, lon, track_id, vehicle_type, time_s, cluster)
        as returned by signal_detector.detect_signals.
    signals : pd.DataFrame
        Signal table with (signal_id, lat, lon).

    Returns
    -------
    pd.DataFrame
        Arm table: (signal_id, arm_id, mean_bearing_deg, n_vehicles)
    """
    # For each stop event, find the previous GPS point of the same vehicle
    # so we can compute an approach bearing.
    #
    # stops_annotated may not contain consecutive points — it only has points
    # where speed < threshold.  We need to pair each stop with the immediately
    # *preceding* non-stop point for the same vehicle.  That information lives
    # in the full trajectory; the caller must provide it via the
    # stops_annotated DataFrame enriched with prev_lat / prev_lon columns.
    #
    # If those columns are absent (e.g. caller did not enrich), we fall back
    # to using the stop's own position for consecutive stop pairs within the
    # same cluster — computing the bearing between the first and last stop
    # point of each vehicle visit.

    stops = stops_annotated.copy()
    stops["signal_id"] = stops["cluster"].astype(int)

    arm_records = []
    stop_arm_rows = []

    for sid, sig_stops in stops[stops["signal_id"] >= 0].groupby("signal_id"):
        if sid not in signals["signal_id"].values:
            continue
        sig_row = signals.set_index("signal_id").loc[sid]
        sig_lat, sig_lon = sig_row["lat"], sig_row["lon"]

        # Per-vehicle: take the first stop observation as the "arrival" point
        arrivals = (
            sig_stops.sort_values("time_s")
            .groupby("track_id")
            .first()
            .reset_index()
        )

        # We need a "previous point" to compute bearing.
        # Use the signal centre as target and each arrival point as origin;
        # bearing = direction from arrival → signal centre (approach direction).
        if "prev_lat" in arrivals.columns and "prev_lon" in arrivals.columns:
            b = bearing_deg(
                arrivals["prev_lat"].values, arrivals["prev_lon"].values,
                sig_lat, sig_lon,
            )
        else:
            # Fallback: bearing from vehicle arrival position → signal centre
            b = bearing_deg(
                arrivals["lat"].values, arrivals["lon"].values,
                sig_lat, sig_lon,
            )

        arrivals["bearing"] = b

        if len(arrivals) < 2:
            continue

        cluster_labels = _cluster_bearings(arrivals["bearing"].values)
        arrivals["arm_cluster"] = cluster_labels

        for arm_cluster, arm_group in arrivals.groupby("arm_cluster"):
            n_veh = len(arm_group)
            if n_veh < ARM_MIN_VEHICLES:
                continue

            # Compute circular mean bearing
            angles_rad = np.radians(arm_group["bearing"].values)
            mean_bearing = float(
                np.degrees(np.arctan2(np.sin(angles_rad).mean(), np.cos(angles_rad).mean())) % 360
            )

            arm_id = len(arm_records)
            arm_records.append({
                "arm_id":          arm_id,
                "signal_id":       int(sid),
                "mean_bearing_deg": mean_bearing,
                "n_vehicles":      n_veh,
            })

            for tid in arm_group["track_id"]:
                stop_arm_rows.append({
                    "track_id":  tid,
                    "signal_id": int(sid),
                    "arm_id":    arm_id,
                })

    arms_df = pd.DataFrame(arm_records)
    stop_arm_df = pd.DataFrame(stop_arm_rows)

    arms_df.to_parquet(ARM_TABLE_PATH, index=False)
    stop_arm_df.to_parquet(STOP_ARM_MAP_PATH, index=False)

    n_arms = len(arms_df)
    log.info(
        "Detected %d arms across %d signals (avg %.1f arms/signal)",
        n_arms,
        arms_df["signal_id"].nunique() if not arms_df.empty else 0,
        n_arms / max(1, arms_df["signal_id"].nunique()) if not arms_df.empty else 0,
    )
    return arms_df


def load_arms() -> pd.DataFrame:
    if not ARM_TABLE_PATH.exists():
        raise FileNotFoundError("Run detect_arms first.")
    return pd.read_parquet(ARM_TABLE_PATH)


def load_stop_arm_map() -> pd.DataFrame:
    if not STOP_ARM_MAP_PATH.exists():
        raise FileNotFoundError("Run detect_arms first.")
    return pd.read_parquet(STOP_ARM_MAP_PATH)


def assign_arm_for_vehicle(
    veh_lat: float, veh_lon: float,
    signal_id: int,
    arms: pd.DataFrame,
) -> int:
    """
    Given a vehicle's current position and a target signal, return the arm_id
    that best matches the vehicle's approach bearing.

    Used during inference to slot an in-transit vehicle into a forecast bin.
    """
    sig_arms = arms[arms["signal_id"] == signal_id]
    if sig_arms.empty:
        return -1

    sig_lat = sig_arms.iloc[0]["lat"] if "lat" in sig_arms.columns else None
    # If signal lat/lon is not embedded in arms table, we skip bearing check
    # and return the arm with the smallest id (placeholder)
    if sig_lat is None:
        return int(sig_arms.iloc[0]["arm_id"])

    approach_bearing = bearing_deg(veh_lat, veh_lon, sig_lat, sig_arms.iloc[0]["lon"])
    diffs = _angular_distance(
        np.array([approach_bearing]),
        sig_arms["mean_bearing_deg"].values,
    )
    best_idx = int(np.argmin(diffs))
    return int(sig_arms.iloc[best_idx]["arm_id"])
