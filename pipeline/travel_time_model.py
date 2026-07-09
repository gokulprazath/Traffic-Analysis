"""
pipeline/travel_time_model.py — Travel-time regressor.

Uses cuML RandomForestRegressor on GPU when RAPIDS is available.
Falls back to sklearn GradientBoostingRegressor on CPU.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    GBR_LEARNING_RATE,
    GBR_MAX_DEPTH,
    GBR_N_ESTIMATORS,
    MODEL_DIR,
    VEHICLE_TYPES,
)

log = logging.getLogger(__name__)
TT_MODEL_PATH = MODEL_DIR / "travel_time_models.pkl"
MIN_SAMPLES_PER_PAIR = 30


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


def _encode_vtype(series: pd.Series, encoder: LabelEncoder) -> np.ndarray:
    def safe_transform(v):
        try:
            return encoder.transform([v])[0]
        except ValueError:
            return 0
    return np.array([safe_transform(v) for v in series])


def _build_features(df: pd.DataFrame, encoder: LabelEncoder) -> np.ndarray:
    vtype_enc = _encode_vtype(df["vehicle_type"], encoder)
    return np.column_stack([
        vtype_enc,
        df["tod_sin"].fillna(0).values,
        df["tod_cos"].fillna(0).values,
        df["approach_speed_kmh"].fillna(0).values,
    ])


class TravelTimeModel:
    def __init__(self):
        self.models:   dict = {}
        self.fallback = None
        self.encoder   = LabelEncoder()
        self.encoder.fit(VEHICLE_TYPES)

    def fit(self, events: pd.DataFrame) -> "TravelTimeModel":
        log.info("Travel-time backend: %s", _REG_NAME)

        df = events.copy()
        if "from_signal_id" not in df.columns:
            df = df.rename(columns={"signal_id": "from_signal_id"})
        if "to_signal_id" not in df.columns:
            df = df.rename(columns={"next_signal_id": "to_signal_id"})

        df = df.dropna(subset=["travel_time_s"])
        df = df[df["to_signal_id"] >= 0]
        df = df[df["travel_time_s"] > 0]

        if df.empty:
            raise ValueError("No valid travel-time observations after filtering.")

        if "tod_sin" not in df.columns:
            from pipeline.feature_extractor import _time_of_day_features
            tod = _time_of_day_features(df["arrival_time_s"])
            df = pd.concat([df, tod], axis=1)

        # Global fallback
        log.info("Training global fallback on %d samples …", len(df))
        X_all = _build_features(df, self.encoder).astype(np.float32)
        y_all = df["travel_time_s"].values.astype(np.float32)
        self.fallback = _make_regressor()
        self.fallback.fit(X_all, y_all)

        # Per-pair
        pairs = df.groupby(["from_signal_id", "to_signal_id"])
        log.info("Training per-pair models for %d signal pairs …", len(pairs))
        for (frm, to), grp in pairs:
            if len(grp) < MIN_SAMPLES_PER_PAIR:
                continue
            X = _build_features(grp, self.encoder).astype(np.float32)
            y = grp["travel_time_s"].values.astype(np.float32)
            m = _make_regressor()
            m.fit(X, y)
            self.models[(int(frm), int(to))] = m

        log.info("Trained %d per-pair models + 1 global (%s)",
                 len(self.models), _REG_NAME)
        return self

    def predict_eta(self, from_signal_id, to_signal_id, vehicle_type,
                    time_of_day_s, approach_speed_kmh) -> float:
        from pipeline.feature_extractor import _time_of_day_features
        tod = _time_of_day_features(pd.Series([time_of_day_s]))
        row = pd.DataFrame([{
            "vehicle_type":       vehicle_type,
            "tod_sin":            float(tod["tod_sin"].iloc[0]),
            "tod_cos":            float(tod["tod_cos"].iloc[0]),
            "approach_speed_kmh": approach_speed_kmh,
        }])
        X = _build_features(row, self.encoder).astype(np.float32)
        key = (int(from_signal_id), int(to_signal_id))
        model = self.models.get(key, self.fallback)
        if model is None:
            return 60.0
        return float(max(1.0, model.predict(X)[0]))

    def save(self, path: Path = TT_MODEL_PATH):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Travel-time models saved → %s", path)

    @classmethod
    def load(cls, path: Path = TT_MODEL_PATH) -> "TravelTimeModel":
        with open(path, "rb") as f:
            return pickle.load(f)
