"""
pipeline/feature_extractor.py — Build the signal_events and vehicle_sequences
tables used to train all three models.

For each vehicle × signal visit the extractor records:
  - arrival time, departure time, approach speed
  - vehicle type, signal_id, arm_id
  - next_signal_id (the next signal the vehicle visits after this one)
  - travel_time_s  (seconds between departure from this signal and arrival at next)

These two tables are the single source of truth for model training.
"""

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import (
    DETECTION_RADIUS_M,
    EARTH_RADIUS_M,
    MODEL_DIR,
)
from pipeline.arm_detector import bearing_deg, haversine_m

log = logging.getLogger(__name__)

EVENTS_PATH    = MODEL_DIR / "signal_events.parquet"
SEQUENCES_PATH = MODEL_DIR / "vehicle_sequences.parquet"


def _time_of_day_features(time_s: pd.Series) -> pd.DataFrame:
    """
    Convert raw seconds-since-midnight into cyclic sin/cos features.
    pNEUMA timestamps are seconds from the start of the recording session,
    not wall-clock time — we treat them as relative time-of-day.
    """
    day_s = 24 * 3600
    angle = 2 * np.pi * (time_s % day_s) / day_s
    return pd.DataFrame({
        "tod_sin": np.sin(angle),
        "tod_cos": np.cos(angle),
    })


def _nearest_signal(lat: float, lon: float, signals: pd.DataFrame) -> tuple[int, float]:
    """Return (signal_id, distance_m) of the closest signal to (lat, lon)."""
    dists = haversine_m(
        lat, lon,
        signals["lat"].values,
        signals["lon"].values,
    )
    idx = int(np.argmin(dists))
    return int(signals.iloc[idx]["signal_id"]), float(dists[idx])


def extract_features(
    chunks: Iterator[pd.DataFrame],
    signals: pd.DataFrame,
    stop_arm_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Iterate over trajectory chunks and produce:
      1. signal_events  — one row per (vehicle, signal) visit
      2. vehicle_sequences — vehicle-ordered list of (signal, arm, time) for
         building transition matrix and travel-time labels.

    Parameters
    ----------
    chunks : Iterator[pd.DataFrame]
        Long-format trajectory chunks.
    signals : pd.DataFrame
        Signal table (signal_id, lat, lon).
    stop_arm_map : pd.DataFrame
        Maps (track_id, signal_id) → arm_id.

    Returns
    -------
    (signal_events, vehicle_sequences)
    """
    # Build fast lookup: track_id → list of (signal_id, arm_id)
    arm_lookup = (
        stop_arm_map
        .set_index(["track_id", "signal_id"])["arm_id"]
        .to_dict()
    )

    all_events: list[dict] = []
    all_seqs:   list[dict] = []

    for chunk in chunks:
        for track_id, traj in chunk.groupby("track_id"):
            traj = traj.sort_values("time_s").reset_index(drop=True)
            vtype = traj["vehicle_type"].iloc[0]

            # Detect signal visits: contiguous blocks where the vehicle is
            # within DETECTION_RADIUS_M of the same signal.
            visits = _extract_visits(traj, signals)
            if not visits:
                continue

            prev_sig_id = None
            prev_dep_time = None

            for visit in visits:
                sig_id = visit["signal_id"]
                arm_id = arm_lookup.get((track_id, sig_id), -1)

                tod = _time_of_day_features(pd.Series([visit["arrival_time_s"]]))

                event = {
                    "track_id":      track_id,
                    "vehicle_type":  vtype,
                    "signal_id":     sig_id,
                    "arm_id":        arm_id,
                    "arrival_time_s": visit["arrival_time_s"],
                    "departure_time_s": visit["departure_time_s"],
                    "dwell_time_s":  visit["departure_time_s"] - visit["arrival_time_s"],
                    "approach_speed_kmh": visit["approach_speed_kmh"],
                    "tod_sin":       float(tod["tod_sin"].iloc[0]),
                    "tod_cos":       float(tod["tod_cos"].iloc[0]),
                }

                # Travel time label (filled in when we see the NEXT visit)
                if prev_sig_id is not None and prev_dep_time is not None:
                    travel_time = visit["arrival_time_s"] - prev_dep_time
                    if 0 < travel_time < 1800:       # sanity: < 30 min
                        # Back-patch the previous event
                        if all_events:
                            last = all_events[-1]
                            if last["track_id"] == track_id and last["signal_id"] == prev_sig_id:
                                last["next_signal_id"] = sig_id
                                last["travel_time_s"] = travel_time

                        all_seqs.append({
                            "track_id":       track_id,
                            "vehicle_type":   vtype,
                            "from_signal_id": prev_sig_id,
                            "to_signal_id":   sig_id,
                            "travel_time_s":  travel_time,
                            "departure_time_s": prev_dep_time,
                        })

                event["next_signal_id"] = -1     # will be filled by back-patch
                event["travel_time_s"]  = np.nan
                all_events.append(event)

                prev_sig_id   = sig_id
                prev_dep_time = visit["departure_time_s"]

    events_df = pd.DataFrame(all_events)
    seqs_df   = pd.DataFrame(all_seqs)

    if not events_df.empty:
        events_df.to_parquet(EVENTS_PATH, index=False)
        log.info("Saved %d signal events → %s", len(events_df), EVENTS_PATH)

    if not seqs_df.empty:
        seqs_df.to_parquet(SEQUENCES_PATH, index=False)
        log.info("Saved %d vehicle sequences → %s", len(seqs_df), SEQUENCES_PATH)

    return events_df, seqs_df


def _extract_visits(traj: pd.DataFrame, signals: pd.DataFrame) -> list[dict]:
    """
    Find contiguous windows where a vehicle is within DETECTION_RADIUS_M of
    the same signal.  Returns a list of visit dicts.
    """
    visits = []
    in_signal: int | None = None
    entry_idx: int = 0

    for i, row in traj.iterrows():
        sig_id, dist = _nearest_signal(row["lat"], row["lon"], signals)
        inside = dist <= DETECTION_RADIUS_M

        if inside and in_signal is None:
            in_signal = sig_id
            entry_idx = i
        elif inside and sig_id != in_signal:
            # Switched directly to a different signal — close old, open new
            _close_visit(traj, entry_idx, i - 1, in_signal, visits)
            in_signal = sig_id
            entry_idx = i
        elif not inside and in_signal is not None:
            _close_visit(traj, entry_idx, i - 1, in_signal, visits)
            in_signal = None

    # Close any open visit at trajectory end
    if in_signal is not None:
        _close_visit(traj, entry_idx, len(traj) - 1, in_signal, visits)

    return visits


def _close_visit(
    traj: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    signal_id: int,
    visits: list,
) -> None:
    """Append a completed visit dict to visits list."""
    window = traj.loc[entry_idx:exit_idx]
    if window.empty:
        return

    # Approach speed = average speed of the 3 observations before entry
    pre = traj.loc[max(0, entry_idx - 3): entry_idx - 1]
    approach_speed = float(pre["speed_kmh"].mean()) if not pre.empty else 0.0

    visits.append({
        "signal_id":          signal_id,
        "arrival_time_s":     float(window["time_s"].iloc[0]),
        "departure_time_s":   float(window["time_s"].iloc[-1]),
        "approach_speed_kmh": approach_speed,
    })


def load_events() -> pd.DataFrame:
    return pd.read_parquet(EVENTS_PATH)


def load_sequences() -> pd.DataFrame:
    return pd.read_parquet(SEQUENCES_PATH)
