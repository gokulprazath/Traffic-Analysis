"""
pipeline/transition_matrix.py — Build and query the probabilistic transition
matrix  P(next_signal | current_signal, vehicle_type).

The matrix is simply a normalised count table built from observed vehicle
sequences.  Laplace smoothing avoids zero-probability entries for unseen
(signal, vehicle_type) pairs.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import MODEL_DIR, VEHICLE_TYPES

log = logging.getLogger(__name__)
MATRIX_PATH = MODEL_DIR / "transition_matrix.pkl"

# Smoothing constant α for Laplace smoothing
ALPHA = 0.5


class TransitionMatrix:
    """
    Probabilistic transition matrix  P(next | current, vtype).

    Attributes
    ----------
    matrix : dict[(from_signal_id, vehicle_type), dict[to_signal_id, float]]
        Nested dict of conditional probabilities.
    signal_ids : list[int]
        Sorted list of all signal IDs observed during training.
    """

    def __init__(self) -> None:
        self.counts: dict = {}       # raw counts before normalisation
        self.matrix: dict = {}       # normalised probabilities
        self.signal_ids: list[int] = []

    # ── Training ────────────────────────────────────────────────────────────

    def fit(self, sequences: pd.DataFrame) -> "TransitionMatrix":
        """
        Build the matrix from a vehicle_sequences DataFrame.

        Parameters
        ----------
        sequences : pd.DataFrame
            Must have columns: from_signal_id, to_signal_id, vehicle_type
        """
        required = {"from_signal_id", "to_signal_id", "vehicle_type"}
        missing  = required - set(sequences.columns)
        if missing:
            raise ValueError(f"sequences missing columns: {missing}")

        # Collect all known signal IDs
        all_ids = pd.concat([
            sequences["from_signal_id"],
            sequences["to_signal_id"],
        ]).unique()
        self.signal_ids = sorted(int(x) for x in all_ids if x >= 0)

        # Count transitions per (from, vtype) → to
        raw: dict = {}
        for _, row in sequences.iterrows():
            frm   = int(row["from_signal_id"])
            to    = int(row["to_signal_id"])
            vtype = str(row["vehicle_type"]).strip()
            if frm < 0 or to < 0:
                continue
            key = (frm, vtype)
            raw.setdefault(key, {})
            raw[key][to] = raw[key].get(to, 0) + 1

        self.counts = raw

        # Normalise with Laplace smoothing
        n_targets = len(self.signal_ids)
        matrix: dict = {}
        for (frm, vtype), counts in raw.items():
            total = sum(counts.values()) + ALPHA * n_targets
            matrix[(frm, vtype)] = {
                sig: (counts.get(sig, 0) + ALPHA) / total
                for sig in self.signal_ids
            }
        self.matrix = matrix

        log.info(
            "Transition matrix built: %d (signal, type) pairs, %d unique signals",
            len(matrix), n_targets,
        )
        return self

    # ── Inference ──────────────────────────────────────────────────────────

    def predict_next(
        self,
        current_signal_id: int,
        vehicle_type: str,
        top_k: int = 3,
    ) -> list[tuple[int, float]]:
        """
        Return the top-k most probable next signals with their probabilities.

        Parameters
        ----------
        current_signal_id : int
        vehicle_type : str
        top_k : int

        Returns
        -------
        list of (signal_id, probability) sorted descending by probability
        """
        key = (current_signal_id, vehicle_type)
        if key not in self.matrix:
            # Fall back to uniform over all signals
            log.debug(
                "No transition data for signal %d / type '%s'; using uniform prior",
                current_signal_id, vehicle_type,
            )
            probs = {s: 1.0 / len(self.signal_ids) for s in self.signal_ids}
        else:
            probs = self.matrix[key]

        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_probs[:top_k]

    def most_probable_next(self, current_signal_id: int, vehicle_type: str) -> int:
        """Return the single most probable next signal ID."""
        top = self.predict_next(current_signal_id, vehicle_type, top_k=1)
        return top[0][0] if top else -1

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: Path = MATRIX_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info("Transition matrix saved → %s", path)

    @classmethod
    def load(cls, path: Path = MATRIX_PATH) -> "TransitionMatrix":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info("Transition matrix loaded ← %s", path)
        return obj
