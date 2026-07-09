"""
pipeline/signal_detector.py — Automatically detect traffic signal locations
from vehicle trajectory data using DBSCAN clustering of stop events.

GPU-accelerated with cuML when RAPIDS is available; falls back to sklearn CPU.
"""

import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# ── GPU / CPU backend selection ───────────────────────────────────────────
try:
    from cuml.cluster import DBSCAN
    import cudf
    _GPU = True
except ImportError:
    from sklearn.cluster import DBSCAN
    _GPU = False

from config import (
    STOP_SPEED_KMH,
    DBSCAN_EPS_DEG,
    DBSCAN_MIN_SAMPLES,
    MODEL_DIR,
)

log = logging.getLogger(__name__)
SIGNAL_TABLE_PATH = MODEL_DIR / "signals.parquet"


def _collect_stop_points(chunks: Iterator[pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for chunk in chunks:
        stopped = chunk[chunk["speed_kmh"] < STOP_SPEED_KMH][
            ["lat", "lon", "track_id", "vehicle_type", "time_s"]
        ].copy()
        if not stopped.empty:
            frames.append(stopped)
    if not frames:
        raise RuntimeError("No stop events found. Check STOP_SPEED_KMH.")
    return pd.concat(frames, ignore_index=True)


def detect_signals(chunks: Iterator[pd.DataFrame]) -> tuple:
    log.info("Collecting stop events …")
    stops = _collect_stop_points(chunks)
    log.info("  %d stop observations collected", len(stops))

    coords = stops[["lat", "lon"]].values

    if _GPU:
        log.info("Running GPU DBSCAN via cuML (eps=%.5f, min_samples=%d) …",
                 DBSCAN_EPS_DEG, DBSCAN_MIN_SAMPLES)
        # cuML DBSCAN uses euclidean on degree coords — eps in degrees
        import cudf as cd
        coords_gpu = cd.DataFrame({"lat": coords[:, 0], "lon": coords[:, 1]})
        db = DBSCAN(eps=DBSCAN_EPS_DEG, min_samples=DBSCAN_MIN_SAMPLES)
        labels = db.fit_predict(coords_gpu).to_numpy()
    else:
        log.info("Running CPU DBSCAN via sklearn (eps=%.5f, min_samples=%d) …",
                 DBSCAN_EPS_DEG, DBSCAN_MIN_SAMPLES)
        coords_rad = np.radians(coords)
        db = DBSCAN(
            eps=DBSCAN_EPS_DEG,
            min_samples=DBSCAN_MIN_SAMPLES,
            algorithm="ball_tree",
            metric="haversine",
        )
        labels = db.fit_predict(coords_rad)

    stops = stops.copy()
    stops["cluster"] = labels

    signal_rows = []
    for cid in sorted(set(labels) - {-1}):
        mask = labels == cid
        cluster_pts = coords[mask]
        signal_rows.append({
            "signal_id":     int(cid),
            "lat":           float(cluster_pts[:, 0].mean()),
            "lon":           float(cluster_pts[:, 1].mean()),
            "n_stop_events": int(mask.sum()),
        })

    signals = pd.DataFrame(signal_rows)
    log.info("Detected %d signal locations (backend: %s)",
             len(signals), "GPU cuML" if _GPU else "CPU sklearn")
    signals.to_parquet(SIGNAL_TABLE_PATH, index=False)
    return signals, stops


def load_signals() -> pd.DataFrame:
    if not SIGNAL_TABLE_PATH.exists():
        raise FileNotFoundError(f"Run detect_signals first.")
    return pd.read_parquet(SIGNAL_TABLE_PATH)
