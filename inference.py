"""
inference.py — Live inference pipeline / historical replay simulator.

Loads trained artefacts, then replays one or more pNEUMA CSV files as a
time-ordered stream of vehicle events.  For every vehicle that enters a
signal's detection radius the controller:

  1. Looks up the most probable next signal via the transition matrix
  2. Predicts ETA at that next signal via the travel-time model
  3. Registers the in-transit vehicle with the priority controller
  4. Emits a green-arm decision every BIN_SECONDS

Output is a CSV of decisions: (time_s, signal_id, green_arm_id, rationale).

Usage
-----
    python inference.py --csv data/20181024_d1_0900_0930.csv
    python inference.py --csv data/ --out-csv results/decisions.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("inference")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pNEUMA signal-control inference / replay")
    p.add_argument(
        "--csv", type=Path, required=True,
        help="Path to one pNEUMA CSV file OR a directory of CSV files",
    )
    p.add_argument(
        "--out-csv", type=Path, default=Path("results/decisions.csv"),
        help="Where to write the green-signal decision log",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    # ── Load artefacts ─────────────────────────────────────────────────────
    log.info("Loading trained artefacts …")
    from pipeline.signal_detector import load_signals
    from pipeline.arm_detector import load_arms, assign_arm_for_vehicle
    from pipeline.transition_matrix import TransitionMatrix
    from pipeline.travel_time_model import TravelTimeModel
    from pipeline.volume_model import VolumeModel
    from pipeline.priority_controller import PriorityController
    from pipeline.arm_detector import haversine_m

    signals   = load_signals()
    arms      = load_arms()
    tm        = TransitionMatrix.load()
    tt_model  = TravelTimeModel.load()
    vm        = VolumeModel.load()

    controller = PriorityController(signals, arms, vm)

    log.info(
        "Ready: %d signals, %d arms, %d TT models, %d volume models",
        len(signals), len(arms), len(tt_model.models), len(vm.models),
    )

    # ── Stream trajectory data ─────────────────────────────────────────────
    from config import DETECTION_RADIUS_M, BIN_SECONDS
    from pipeline.parser import parse_csv, load_all_csvs

    csv_path = args.csv
    if csv_path.is_dir():
        chunk_gen = load_all_csvs(csv_path)
    else:
        chunk_gen = parse_csv(csv_path)

    # Collect all trajectory data into memory (for time-ordered replay)
    # For 15 GB datasets use only ONE CSV at a time or a streaming approach.
    log.info("Loading trajectory data from %s …", csv_path)
    all_frames = []
    for chunk in chunk_gen:
        all_frames.append(chunk)
    if not all_frames:
        log.error("No data loaded from %s", csv_path)
        sys.exit(1)

    traj_df = pd.concat(all_frames, ignore_index=True).sort_values("time_s")
    t_start = float(traj_df["time_s"].min())
    t_end   = float(traj_df["time_s"].max())
    log.info(
        "Trajectory span: %.0f s → %.0f s (%.1f min)",
        t_start, t_end, (t_end - t_start) / 60,
    )

    # Build fast signal lookup (KD-tree would be faster for large N)
    sig_lats = signals["lat"].values
    sig_lons = signals["lon"].values
    sig_ids  = signals["signal_id"].values

    decision_log: list[dict] = []

    # ── Replay loop ────────────────────────────────────────────────────────
    t = t_start
    n_events_processed = 0

    while t <= t_end:
        # All observations in [t, t + BIN_SECONDS)
        window = traj_df[(traj_df["time_s"] >= t) & (traj_df["time_s"] < t + BIN_SECONDS)]

        # Count observed arrivals per (signal, arm) for bin buffer update
        arrival_counts: dict[tuple, float] = {}

        for _, obs in window.iterrows():
            lat, lon = obs["lat"], obs["lon"]
            vtype    = str(obs["vehicle_type"]).strip()
            track_id = obs["track_id"]

            # Find nearest signal
            dists = haversine_m(lat, lon, sig_lats, sig_lons)
            nearest_idx  = int(np.argmin(dists))
            nearest_dist = float(dists[nearest_idx])

            if nearest_dist > DETECTION_RADIUS_M:
                continue   # not at a signal yet

            current_sig_id = int(sig_ids[nearest_idx])

            # Predict next signal and ETA
            next_sig_id = tm.most_probable_next(current_sig_id, vtype)
            if next_sig_id < 0:
                continue

            eta_s = tt_model.predict_eta(
                from_signal_id=current_sig_id,
                to_signal_id=next_sig_id,
                vehicle_type=vtype,
                time_of_day_s=float(obs["time_s"]),
                approach_speed_kmh=float(obs["speed_kmh"]),
            ) + float(obs["time_s"])   # convert relative → absolute

            # Determine arm at next signal
            arm_id = assign_arm_for_vehicle(lat, lon, next_sig_id, arms)

            controller.register_arrival(track_id, vtype, next_sig_id, arm_id, eta_s)

            # Track observed arrivals for bin buffer
            key = (current_sig_id, arm_id)
            arrival_counts[key] = arrival_counts.get(key, 0.0) + 1.0
            n_events_processed += 1

        # Push observed counts into rolling buffers
        for (sig_id, arm_id_), count in arrival_counts.items():
            controller.update_bin_buffer(sig_id, arm_id_, count)

        # Controller tick → green decisions
        decisions = controller.tick(t)

        for sig_id, green_arm in decisions.items():
            decision_log.append({
                "time_s":       t,
                "signal_id":    sig_id,
                "green_arm_id": green_arm,
            })

        t += BIN_SECONDS

    log.info(
        "Replay complete. Processed %d signal events over %.0f s",
        n_events_processed, t_end - t_start,
    )

    # ── Save results ───────────────────────────────────────────────────────
    results_df = pd.DataFrame(decision_log)
    results_df.to_csv(args.out_csv, index=False)
    log.info("Decision log saved → %s  (%d rows)", args.out_csv, len(results_df))

    # Quick summary statistics
    if not results_df.empty:
        green_dist = (
            results_df.groupby(["signal_id", "green_arm_id"])
            .size()
            .rename("green_phases")
            .reset_index()
        )
        log.info("\nGreen phase distribution (top 20):\n%s", green_dist.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
