"""
pipeline/priority_controller.py — Real-time traffic signal priority controller.

The controller maintains a state machine for every signal:
  - current green arm
  - time since each arm last had green (starvation counter)
  - in-transit vehicle registry (vehicles currently en route to each signal)

On every tick (e.g. every 1 s of simulated time) it:
  1. Updates in-transit registry (remove vehicles whose ETA has passed)
  2. Requests a volume forecast per arm from VolumeModel
  3. Boosts score by starvation penalty to avoid indefinite red arms
  4. Gives green to the arm with the highest weighted score
  5. Enforces MIN_GREEN_SECONDS / MAX_GREEN_SECONDS phase constraints
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import (
    BIN_SECONDS,
    FORECAST_HORIZON,
    MAX_GREEN_SECONDS,
    MIN_GREEN_SECONDS,
    STARVATION_PENALTY,
)
from pipeline.volume_model import VolumeModel, aggregate_etas_to_bins

log = logging.getLogger(__name__)


@dataclass
class InTransitVehicle:
    """A vehicle predicted to arrive at a (signal, arm) at a given time."""
    track_id:    str | int
    vehicle_type: str
    eta_s:       float     # absolute simulated seconds
    arm_id:      int


@dataclass
class SignalState:
    """Runtime state for one intersection."""
    signal_id:   int
    arm_ids:     list[int]

    # Phase tracking
    green_arm_id:          int   = -1      # -1 = no arm yet
    green_start_s:         float = 0.0
    seconds_since_green:   dict  = field(default_factory=dict)   # arm_id → float

    # In-transit registry: arm_id → list[InTransitVehicle]
    in_transit: dict = field(default_factory=dict)

    def __post_init__(self):
        for arm_id in self.arm_ids:
            self.seconds_since_green[arm_id] = 0.0
            self.in_transit[arm_id] = []


class PriorityController:
    """
    Central controller managing all signals.

    Usage
    -----
    >>> ctrl = PriorityController(signals_df, arms_df, volume_model)
    >>> ctrl.register_arrival(track_id, vehicle_type, to_signal_id, arm_id, eta_s)
    >>> decisions = ctrl.tick(current_time_s)
    >>> # decisions = {signal_id: green_arm_id}
    """

    def __init__(
        self,
        signals: "pd.DataFrame",
        arms: "pd.DataFrame",
        volume_model: VolumeModel,
    ) -> None:
        import pandas as pd
        self.volume_model = volume_model

        # Build SignalState objects
        self.states: dict[int, SignalState] = {}
        for sig_id, sig_arms in arms.groupby("signal_id"):
            arm_ids = sorted(sig_arms["arm_id"].tolist())
            self.states[int(sig_id)] = SignalState(
                signal_id=int(sig_id),
                arm_ids=arm_ids,
            )

        # Rolling bin buffers: (signal_id, arm_id) → deque of recent bin counts
        from collections import deque
        self._bin_buffers: dict[tuple, deque] = {
            (int(sig_id), int(arm_id)): deque(maxlen=6)
            for sig_id, arm_id in zip(arms["signal_id"], arms["arm_id"])
        }
        self._last_bin_time: dict[tuple, float] = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def register_arrival(
        self,
        track_id: str | int,
        vehicle_type: str,
        to_signal_id: int,
        arm_id: int,
        eta_s: float,
    ) -> None:
        """
        Register an in-transit vehicle expected at (to_signal_id, arm_id) at eta_s.
        """
        state = self.states.get(int(to_signal_id))
        if state is None or arm_id not in state.in_transit:
            return
        state.in_transit[arm_id].append(
            InTransitVehicle(track_id, vehicle_type, eta_s, arm_id)
        )

    def tick(self, current_time_s: float) -> dict[int, int]:
        """
        Advance the controller by one tick.

        Parameters
        ----------
        current_time_s : float
            Current simulated / wall-clock time.

        Returns
        -------
        dict[signal_id, green_arm_id]
            The recommended green arm for every managed signal.
        """
        decisions: dict[int, int] = {}

        for sig_id, state in self.states.items():
            # Prune vehicles that have already arrived
            for arm_id in state.arm_ids:
                state.in_transit[arm_id] = [
                    v for v in state.in_transit[arm_id]
                    if v.eta_s > current_time_s
                ]

            # Update starvation counters
            for arm_id in state.arm_ids:
                if arm_id != state.green_arm_id:
                    state.seconds_since_green[arm_id] = (
                        state.seconds_since_green.get(arm_id, 0.0) + 1.0
                    )
                else:
                    state.seconds_since_green[arm_id] = 0.0

            # Enforce minimum green time
            current_green_dur = current_time_s - state.green_start_s
            if state.green_arm_id >= 0 and current_green_dur < MIN_GREEN_SECONDS:
                decisions[sig_id] = state.green_arm_id
                continue

            # Enforce maximum green time (force switch)
            force_switch = (
                state.green_arm_id >= 0
                and current_green_dur >= MAX_GREEN_SECONDS
            )

            # Score each arm
            scores = self._score_arms(state, current_time_s, force_switch)

            best_arm = max(scores, key=scores.__getitem__)
            if best_arm != state.green_arm_id:
                log.debug(
                    "Signal %d: green arm %d → %d (scores: %s)",
                    sig_id, state.green_arm_id, best_arm,
                    {k: f"{v:.2f}" for k, v in scores.items()},
                )
                state.green_arm_id  = best_arm
                state.green_start_s = current_time_s

            decisions[sig_id] = state.green_arm_id

        return decisions

    # ── Internal scoring ────────────────────────────────────────────────────

    def _score_arms(
        self,
        state: SignalState,
        current_time_s: float,
        force_switch: bool,
    ) -> dict[int, float]:
        """
        Compute a priority score for each arm of one signal.

        Score = forecasted_volume + starvation_bonus
        If force_switch is True, the current green arm gets score = 0.
        """
        scores: dict[int, float] = {}

        for arm_id in state.arm_ids:
            # Get ETA list for this arm
            etas = [v.eta_s for v in state.in_transit[arm_id]]

            # Bin the in-transit ETAs
            eta_bins = aggregate_etas_to_bins(etas, current_time_s)

            # Model forecast (historical pattern)
            key = (state.signal_id, arm_id)
            buf = self._bin_buffers.get(key)
            if buf and len(buf) >= 1:
                recent = np.array(list(buf))
            else:
                recent = None

            model_forecast = self.volume_model.forecast(
                state.signal_id, arm_id, recent_bin_counts=recent
            )

            # Combine: immediate ETA pressure + model forecast
            # Weight near-term bins more heavily
            bin_weights = np.array(
                [1.0 / (i + 1) for i in range(FORECAST_HORIZON)]
            )
            bin_weights /= bin_weights.sum()
            immediate_score = float(np.dot(eta_bins, bin_weights))

            volume_score = 0.6 * immediate_score + 0.4 * model_forecast

            # Starvation bonus
            starvation_bonus = (
                state.seconds_since_green.get(arm_id, 0.0) * STARVATION_PENALTY
            )

            score = volume_score + starvation_bonus

            if force_switch and arm_id == state.green_arm_id:
                score = 0.0

            scores[arm_id] = score

        return scores

    def update_bin_buffer(
        self,
        signal_id: int,
        arm_id: int,
        observed_count: float,
    ) -> None:
        """
        Push the most recently observed bin count into the rolling buffer.
        Called once per BIN_SECONDS by the simulation/live loop.
        """
        key = (signal_id, arm_id)
        if key in self._bin_buffers:
            self._bin_buffers[key].append(observed_count)

    def summary(self, current_time_s: float) -> list[dict]:
        """Return a human-readable list of current signal states."""
        rows = []
        for sig_id, state in self.states.items():
            for arm_id in state.arm_ids:
                n_in_transit = len(state.in_transit.get(arm_id, []))
                rows.append({
                    "signal_id":     sig_id,
                    "arm_id":        arm_id,
                    "is_green":      arm_id == state.green_arm_id,
                    "green_age_s":   current_time_s - state.green_start_s
                                     if arm_id == state.green_arm_id else 0.0,
                    "starvation_s":  state.seconds_since_green.get(arm_id, 0.0),
                    "in_transit":    n_in_transit,
                })
        return rows
