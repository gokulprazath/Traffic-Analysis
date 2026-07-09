"""
train.py — Full training pipeline for pNEUMA signal control.

Run once per batch of historical CSV data.  Stages:
  1. Parse all pNEUMA CSVs (streaming, memory-bounded)
  2. Detect signal locations (DBSCAN on stop events)
  3. Detect road arms per signal (bearing clustering)
  4. Extract signal events and vehicle sequences
  5. Train transition matrix
  6. Train travel-time models (GBR per signal pair + global fallback)
  7. Train volume models (autoregressive GBR per signal-arm)

All artefacts are saved to MODEL_DIR and can be loaded by inference.py.

Usage
-----
    python train.py [--data-dir PATH] [--chunksize N] [--log-level LEVEL]
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pNEUMA signal-control models")
    p.add_argument(
        "--data-dir", type=Path, default=Path("data"),
        help="Directory containing pNEUMA CSV files (default: ./data)",
    )
    p.add_argument(
        "--chunksize", type=int, default=3_000,
        help="Number of vehicle rows per parse chunk (trade memory vs speed)",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    from config import DATA_DIR, MODEL_DIR
    data_dir = args.data_dir or DATA_DIR
    log.info("══ pNEUMA Signal Control — Training Pipeline ══")
    log.info("Data dir : %s", data_dir)
    log.info("Model dir: %s", MODEL_DIR)

    # ── Stage 1 + 2: Parse CSVs & detect signals ─────────────────────────
    log.info("\n─── Stage 1 & 2: Parse + Signal Detection ───")
    from pipeline.parser import load_all_csvs
    from pipeline.signal_detector import detect_signals

    def iter_chunks():
        yield from load_all_csvs(data_dir, chunksize=args.chunksize)

    signals, stops_annotated = detect_signals(iter_chunks())
    log.info("Signals detected: %d", len(signals))

    # ── Stage 3: Detect road arms ─────────────────────────────────────────
    log.info("\n─── Stage 3: Arm Detection ───")
    from pipeline.arm_detector import detect_arms
    arms = detect_arms(stops_annotated, signals)
    log.info(
        "Arms detected: %d across %d signals",
        len(arms), arms["signal_id"].nunique(),
    )

    # ── Stage 4: Feature extraction ───────────────────────────────────────
    log.info("\n─── Stage 4: Feature Extraction ───")
    from pipeline.arm_detector import load_stop_arm_map
    from pipeline.feature_extractor import extract_features

    stop_arm_map = load_stop_arm_map()

    def iter_chunks2():
        yield from load_all_csvs(data_dir, chunksize=args.chunksize)

    events, sequences = extract_features(iter_chunks2(), signals, stop_arm_map)
    log.info("Signal events: %d", len(events))
    log.info("Vehicle sequences: %d", len(sequences))

    if events.empty or sequences.empty:
        log.error("Insufficient data for model training.  Check data paths and config.")
        sys.exit(1)

    # ── Stage 5: Transition matrix ────────────────────────────────────────
    log.info("\n─── Stage 5: Transition Matrix ───")
    from pipeline.transition_matrix import TransitionMatrix
    tm = TransitionMatrix()
    tm.fit(sequences)
    tm.save()

    # ── Stage 6: Travel-time model ────────────────────────────────────────
    log.info("\n─── Stage 6: Travel-Time Models ───")
    from pipeline.travel_time_model import TravelTimeModel
    tt = TravelTimeModel()
    tt.fit(events)
    tt.save()

    # ── Stage 7: Volume model ─────────────────────────────────────────────
    log.info("\n─── Stage 7: Volume Models ───")
    from pipeline.volume_model import VolumeModel
    vm = VolumeModel()
    vm.fit(events)
    vm.save()

    log.info("\n══ Training complete. All artefacts saved to %s ══", MODEL_DIR)
    log.info(
        "  signals.parquet          %6d rows",  len(signals),
    )
    log.info(
        "  arms.parquet             %6d rows",  len(arms),
    )
    log.info(
        "  signal_events.parquet    %6d rows",  len(events),
    )
    log.info(
        "  transition_matrix.pkl    %6d pairs", len(tm.matrix),
    )
    log.info(
        "  travel_time_models.pkl   %6d pairs + 1 global",  len(tt.models),
    )
    log.info(
        "  volume_models.pkl        %6d (signal,arm) models", len(vm.models),
    )


if __name__ == "__main__":
    main()
