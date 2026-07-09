"""
pipeline/volume_model.py — Arrival-volume forecaster per (signal, arm).

Uses cuML RandomForestRegressor on GPU when RAPIDS is available.
Falls back to sklearn GradientBoostingRegressor on CPU.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ── GPU / CPU backend selection ───────────────────────────────────────────
try:
    from cuml.ensemble import RandomForestRegressor as Regressor
    _GPU = True
    _REG_NAME = "cuML RandomForest (GPU)"
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as Regressor
    _GPU = False
    _REG_NAME = "sklearn GBR (CPU)"

from config import (
    BIN_SECONDS,
    FORECAST_HORIZON,
    GBR_LEARNING_RATE,
    GBR_MAX_DEPTH,
    GBR_N_ESTIMATORS,
    MODEL_DIR,
    ROLLING_WINDOW_BINS,
)

log = logging.getLogger(__name__)
VOLUME_MODEL_PATH = MODEL_DIR / "volume_models.pkl"
MIN_BINS_TO_TRAIN = ROLLING_WINDOW_BINS + FORECAST_HORIZON + 10


def _make_regressor():
    if _GPU:
        return Regressor(
            n_estimators=GBR_N_ESTIMATORS,
            max_depth=GBR_MAX_DEPTH,
            n_streams=4,
        )
    return Regressor(
        n_estimators=GBR_N_ESTIMATORS,
        max_depth=GBR_MAX_DEPTH,
        learning_rate=GBR_LEARNING_RATE,
        random_state=42,
    )


def _make_windows(series, window, horizon):
    X, y = [], []
    for i in range(window, len(series) - horizon + 1):
        X.append(series[i - window: i])
        y.append(series[i: i + horizon].sum())
    return np.array(X), np.array(y)


class VolumeModel:
    def __init__(self):
        self.models:     dict = {}
        self.bin_series: dict = {}

    def fit(self, events: pd.DataFrame) -> "VolumeModel":
        log.info("Volume model backend: %s", _REG_NAME)
        required = {"signal_id", "arm_id", "arrival_time_s"}
        if not required.issubset(events.columns):
            raise ValueError(f"events missing: {required - set(events.columns)}")

        groups = events[events["arm_id"] >= 0].groupby(["signal_id", "arm_id"])
        log.info("Training volume models for %d (signal, arm) pairs …", len(groups))

        for (sig_id, arm_id), grp in groups:
            t = grp["arrival_time_s"].values
            if len(t) == 0:
                continue
            t_min, t_max = t.min(), t.max()
            n_bins = max(1, int((t_max - t_min) / BIN_SECONDS) + 1)
            bin_indices = ((t - t_min) / BIN_SECONDS).astype(int)
            bin_counts  = np.bincount(bin_indices, minlength=n_bins).astype(np.float32)
            self.bin_series[(sig_id, arm_id)] = bin_counts

            if len(bin_counts) < MIN_BINS_TO_TRAIN:
                continue
            X, y = _make_windows(bin_counts, ROLLING_WINDOW_BINS, FORECAST_HORIZON)
            if len(X) < 5:
                continue
            X = X.astype(np.float32)
            y = y.astype(np.float32)
            m = _make_regressor()
            m.fit(X, y)
            self.models[(sig_id, arm_id)] = m

        log.info("Trained %d volume models (%s)", len(self.models), _REG_NAME)
        return self

    def forecast(self, signal_id, arm_id, recent_bin_counts=None) -> float:
        key = (signal_id, arm_id)
        if recent_bin_counts is None:
            history = self.bin_series.get(key)
            if history is None or len(history) < ROLLING_WINDOW_BINS:
                return 0.0
            recent_bin_counts = history[-ROLLING_WINDOW_BINS:]

        if len(recent_bin_counts) < ROLLING_WINDOW_BINS:
            pad = np.zeros(ROLLING_WINDOW_BINS - len(recent_bin_counts))
            recent_bin_counts = np.concatenate([pad, recent_bin_counts])

        model = self.models.get(key)
        if model is None:
            return float(recent_bin_counts.mean() * FORECAST_HORIZON)

        X = recent_bin_counts[-ROLLING_WINDOW_BINS:].reshape(1, -1).astype(np.float32)
        return float(max(0.0, model.predict(X)[0]))

    def save(self, path: Path = VOLUME_MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Volume models saved → %s", path)

    @classmethod
    def load(cls, path: Path = VOLUME_MODEL_PATH) -> "VolumeModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def aggregate_etas_to_bins(etas, current_time_s, n_bins=FORECAST_HORIZON):
    bins = np.zeros(n_bins, dtype=float)
    for eta in etas:
        dt = eta - current_time_s
        bin_idx = 0 if dt < 0 else int(dt / BIN_SECONDS)
        if bin_idx < n_bins:
            bins[bin_idx] += 1
    return bins
