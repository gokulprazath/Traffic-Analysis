"""
evaluate.py — Measure accuracy of all three models and controller quality.

Metrics reported
----------------
Travel-time model  : MAE, RMSE, R² per signal pair + global
Volume model       : MAE, RMSE per (signal, arm)
Transition matrix  : top-1 accuracy, top-3 accuracy, cross-entropy
Controller         : green-phase fairness (Gini), avg starvation time,
                     throughput (vehicles processed per green second)

Usage
-----
    python evaluate.py --decisions results/decisions.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("evaluate")


# ── Helpers ────────────────────────────────────────────────────────────────

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def gini(values: np.ndarray) -> float:
    """Gini impurity as a fairness measure: 0 = perfectly fair, 1 = totally unfair."""
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)


# ── Travel-time model evaluation ───────────────────────────────────────────

def eval_travel_time(events: pd.DataFrame, tt_model) -> pd.DataFrame:
    df = events.copy()
    if "from_signal_id" not in df.columns:
        df = df.rename(columns={"signal_id": "from_signal_id"})
    if "to_signal_id" not in df.columns:
        df = df.rename(columns={"next_signal_id": "to_signal_id"})

    df = df.dropna(subset=["travel_time_s"])
    df = df[(df["to_signal_id"] >= 0) & (df["travel_time_s"] > 0)]

    if df.empty:
        log.warning("No labelled travel-time rows — skipping TT evaluation")
        return pd.DataFrame()

    y_true, y_pred = [], []
    for _, row in df.iterrows():
        pred = tt_model.predict_eta(
            from_signal_id=int(row["from_signal_id"]),
            to_signal_id=int(row["to_signal_id"]),
            vehicle_type=str(row["vehicle_type"]),
            time_of_day_s=float(row.get("arrival_time_s", 0)),
            approach_speed_kmh=float(row.get("approach_speed_kmh", 20)),
        )
        y_true.append(row["travel_time_s"])
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    global_row = {
        "pair": "GLOBAL",
        "n_samples": len(y_true),
        "MAE_s":  round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE_s": round(rmse(y_true, y_pred), 2),
        "R2":     round(r2_score(y_true, y_pred), 4),
    }
    rows = [global_row]

    # Per-pair breakdown (top 20 by sample count)
    for (frm, to), grp in df.groupby(["from_signal_id", "to_signal_id"]):
        if len(grp) < 10:
            continue
        yt = grp["travel_time_s"].values
        yp = np.array([
            tt_model.predict_eta(
                int(frm), int(to),
                str(r["vehicle_type"]),
                float(r.get("arrival_time_s", 0)),
                float(r.get("approach_speed_kmh", 20)),
            )
            for _, r in grp.iterrows()
        ])
        rows.append({
            "pair":      f"{int(frm)}→{int(to)}",
            "n_samples": len(yt),
            "MAE_s":     round(mean_absolute_error(yt, yp), 2),
            "RMSE_s":    round(rmse(yt, yp), 2),
            "R2":        round(r2_score(yt, yp), 4) if len(yt) > 1 else None,
        })

    result = pd.DataFrame(rows).sort_values("n_samples", ascending=False)
    log.info("\n── Travel-Time Model ──\n%s", result.head(15).to_string(index=False))
    return result


# ── Transition matrix evaluation ───────────────────────────────────────────

def eval_transition_matrix(sequences: pd.DataFrame, tm) -> dict:
    df = sequences[
        (sequences["from_signal_id"] >= 0) &
        (sequences["to_signal_id"] >= 0)
    ].copy()

    if df.empty:
        log.warning("No sequences — skipping transition matrix evaluation")
        return {}

    top1_hits, top3_hits, log_losses = 0, 0, []

    for _, row in df.iterrows():
        frm   = int(row["from_signal_id"])
        to    = int(row["to_signal_id"])
        vtype = str(row["vehicle_type"])

        preds = tm.predict_next(frm, vtype, top_k=len(tm.signal_ids))
        if not preds:
            continue

        pred_ids = [p[0] for p in preds]
        pred_probs = {p[0]: p[1] for p in preds}

        top1_hits += int(pred_ids[0] == to)
        top3_hits += int(to in pred_ids[:3])

        p_true = pred_probs.get(to, 1e-9)
        log_losses.append(-np.log(p_true))

    n = len(df)
    metrics = {
        "n_transitions": n,
        "top1_accuracy": round(top1_hits / n, 4) if n else 0,
        "top3_accuracy": round(top3_hits / n, 4) if n else 0,
        "mean_log_loss": round(float(np.mean(log_losses)), 4) if log_losses else None,
    }
    log.info(
        "\n── Transition Matrix ──\n"
        "  Top-1 accuracy : %.1f%%\n"
        "  Top-3 accuracy : %.1f%%\n"
        "  Mean log-loss  : %.4f\n"
        "  N transitions  : %d",
        metrics["top1_accuracy"] * 100,
        metrics["top3_accuracy"] * 100,
        metrics["mean_log_loss"] or 0,
        n,
    )
    return metrics


# ── Volume model evaluation ────────────────────────────────────────────────

def eval_volume_model(events: pd.DataFrame, vm) -> pd.DataFrame:
    from config import BIN_SECONDS, ROLLING_WINDOW_BINS, FORECAST_HORIZON

    rows = []
    for (sig_id, arm_id), grp in events[events["arm_id"] >= 0].groupby(["signal_id", "arm_id"]):
        t  = grp["arrival_time_s"].values
        if len(t) < (ROLLING_WINDOW_BINS + FORECAST_HORIZON + 5) * 2:
            continue

        t_min = t.min()
        n_bins = int((t.max() - t_min) / BIN_SECONDS) + 1
        bin_counts = np.bincount(
            ((t - t_min) / BIN_SECONDS).astype(int),
            minlength=n_bins
        ).astype(float)

        y_true, y_pred = [], []
        for i in range(ROLLING_WINDOW_BINS, len(bin_counts) - FORECAST_HORIZON):
            window   = bin_counts[i - ROLLING_WINDOW_BINS: i]
            actual   = bin_counts[i: i + FORECAST_HORIZON].sum()
            forecast = vm.forecast(int(sig_id), int(arm_id), recent_bin_counts=window)
            y_true.append(actual)
            y_pred.append(forecast)

        if len(y_true) < 3:
            continue

        yt = np.array(y_true)
        yp = np.array(y_pred)
        rows.append({
            "signal_id": int(sig_id),
            "arm_id":    int(arm_id),
            "n_windows": len(yt),
            "MAE":       round(mean_absolute_error(yt, yp), 3),
            "RMSE":      round(rmse(yt, yp), 3),
            "R2":        round(r2_score(yt, yp), 4) if len(yt) > 1 else None,
        })

    result = pd.DataFrame(rows).sort_values("n_windows", ascending=False)
    if not result.empty:
        log.info(
            "\n── Volume Model ──\n"
            "  Mean MAE  : %.3f vehicles/bin\n"
            "  Mean RMSE : %.3f\n"
            "  Mean R²   : %.4f\n"
            "  Models evaluated: %d",
            result["MAE"].mean(), result["RMSE"].mean(),
            result["R2"].dropna().mean(), len(result),
        )
    return result


# ── Controller evaluation ──────────────────────────────────────────────────

def eval_controller(decisions_csv: Path) -> dict:
    if not decisions_csv.exists():
        log.warning("decisions.csv not found at %s — skipping controller eval", decisions_csv)
        return {}

    df = pd.read_csv(decisions_csv)

    # Green time per (signal, arm)
    green_counts = (
        df.groupby(["signal_id", "green_arm_id"])
        .size()
        .rename("green_bins")
        .reset_index()
    )

    # Fairness per signal: how evenly is green distributed across arms?
    fairness_rows = []
    for sig_id, grp in green_counts.groupby("signal_id"):
        counts = grp["green_bins"].values
        total  = counts.sum()
        shares = counts / total
        gini_score = gini(shares)
        fairness_rows.append({
            "signal_id":    sig_id,
            "n_arms":       len(counts),
            "total_phases": int(total),
            "gini":         round(gini_score, 4),
            "min_share%":   round(shares.min() * 100, 1),
            "max_share%":   round(shares.max() * 100, 1),
        })

    fairness_df = pd.DataFrame(fairness_rows)
    metrics = {
        "total_decisions":   len(df),
        "signals_controlled": df["signal_id"].nunique(),
        "mean_gini":         round(fairness_df["gini"].mean(), 4) if not fairness_df.empty else None,
        "mean_min_share%":   round(fairness_df["min_share%"].mean(), 1) if not fairness_df.empty else None,
    }

    log.info(
        "\n── Controller Fairness ──\n"
        "  Signals controlled : %d\n"
        "  Total decisions    : %d\n"
        "  Mean Gini (fairness, 0=fair) : %.4f\n"
        "  Mean min-arm share : %.1f%%\n\n%s",
        metrics["signals_controlled"],
        metrics["total_decisions"],
        metrics["mean_gini"] or 0,
        metrics["mean_min_share%"] or 0,
        fairness_df.to_string(index=False),
    )
    return metrics, fairness_df


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Evaluate pNEUMA pipeline models")
    p.add_argument("--decisions", type=Path, default=Path("results/decisions.csv"))
    p.add_argument("--out-dir",   type=Path, default=Path("results"))
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load artefacts
    from pipeline.signal_detector import load_signals
    from pipeline.arm_detector import load_arms
    from pipeline.transition_matrix import TransitionMatrix
    from pipeline.travel_time_model import TravelTimeModel
    from pipeline.volume_model import VolumeModel
    from pipeline.feature_extractor import load_events, load_sequences

    events    = load_events()
    sequences = load_sequences()
    tm        = TransitionMatrix.load()
    tt        = TravelTimeModel.load()
    vm        = VolumeModel.load()

    # Run evaluations
    tt_results  = eval_travel_time(events, tt)
    tm_metrics  = eval_transition_matrix(sequences, tm)
    vol_results = eval_volume_model(events, vm)
    ctrl_result = eval_controller(args.decisions)

    # Save CSVs
    if not tt_results.empty:
        tt_results.to_csv(args.out_dir / "eval_travel_time.csv", index=False)
    if not vol_results.empty:
        vol_results.to_csv(args.out_dir / "eval_volume.csv", index=False)

    pd.DataFrame([tm_metrics]).to_csv(args.out_dir / "eval_transition.csv", index=False)

    if isinstance(ctrl_result, tuple):
        _, fairness_df = ctrl_result
        fairness_df.to_csv(args.out_dir / "eval_controller.csv", index=False)

    log.info("\nAll evaluation CSVs saved to %s/", args.out_dir)


if __name__ == "__main__":
    main()
