"""
pipeline/parser.py — Parse pNEUMA CSV files into long-format trajectory DataFrames.

pNEUMA format (per row):
  trackID | type | distance_m | avg_speed_kmh | [lat, lon, speed, lon_acc, lat_acc, time] * N

Every vehicle row has a DIFFERENT number of columns (different trajectory lengths).
pandas read_csv with a fixed schema fails with "Expected N fields, saw M".
We therefore read the file line-by-line so each row can be any width.
"""

import csv
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from config import (
    FIXED_COLS, REPEAT_COLS,
    COL_LAT, COL_LON, COL_SPEED, COL_LON_ACC, COL_LAT_ACC, COL_TIME,
)

log = logging.getLogger(__name__)


def _detect_sep(path: Path) -> str:
    """Sniff the delimiter from the first line of a pNEUMA CSV."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        first = fh.readline()
    # pNEUMA files use '; ' (semicolon+space) or plain ','
    return ";" if first.count(";") > first.count(",") else ","


_BAD_VALS = {"", "none", "nan", "null", "na", "-"}


def _safe_float(v: str) -> float:
    """Convert a CSV field to float, returning NaN for blank/non-numeric values."""
    s = v.strip().lower()
    if s in _BAD_VALS:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_fields(fields: list[str]) -> pd.DataFrame | None:
    """
    Convert one list of raw string fields (one vehicle row) into a
    long-format DataFrame.  Returns None if the row is malformed.

    Handles variable-width rows (pNEUMA's defining feature) and safely
    converts empty / None / NaN strings to np.nan rather than raising.
    """
    if len(fields) < FIXED_COLS + REPEAT_COLS:
        return None

    try:
        track_id   = fields[0].strip()
        vtype      = fields[1].strip()
        distance_m = _safe_float(fields[2])
        avg_speed  = _safe_float(fields[3])
    except IndexError:
        return None

    if not track_id or not vtype:
        return None

    raw_traj = list(fields[FIXED_COLS:])

    # Strip trailing empty / None / nan fields (padding artefacts)
    while raw_traj and raw_traj[-1].strip().lower() in _BAD_VALS:
        raw_traj.pop()

    n_obs = len(raw_traj) // REPEAT_COLS
    if n_obs == 0:
        return None

    traj_flat = [_safe_float(v) for v in raw_traj[: n_obs * REPEAT_COLS]]
    traj = np.array(traj_flat, dtype=float).reshape(n_obs, REPEAT_COLS)

    # Drop observation rows that have NaN in lat, lon, or time
    valid = (
        ~np.isnan(traj[:, COL_LAT]) &
        ~np.isnan(traj[:, COL_LON]) &
        ~np.isnan(traj[:, COL_TIME])
    )
    traj = traj[valid]
    if len(traj) == 0:
        return None

    return pd.DataFrame({
        "track_id":      track_id,
        "vehicle_type":  vtype,
        "distance_m":    distance_m,
        "avg_speed_kmh": avg_speed,
        "lat":           traj[:, COL_LAT],
        "lon":           traj[:, COL_LON],
        "speed_kmh":     traj[:, COL_SPEED],
        "lon_acc":       traj[:, COL_LON_ACC],
        "lat_acc":       traj[:, COL_LAT_ACC],
        "time_s":        traj[:, COL_TIME],
    })


def parse_csv(path: Path, chunksize: int = 5_000) -> Iterator[pd.DataFrame]:
    """
    Yield long-format trajectory DataFrames from a single pNEUMA CSV.

    Reads the file line-by-line with Python's csv module so that rows with
    different numbers of columns are handled correctly.  Yields one DataFrame
    per `chunksize` vehicles to keep memory usage bounded.

    Parameters
    ----------
    path : Path
        Path to one pNEUMA CSV file.
    chunksize : int
        Number of vehicle rows to accumulate before yielding.

    Yields
    ------
    pd.DataFrame
        Long-format observations with columns:
        track_id, vehicle_type, distance_m, avg_speed_kmh,
        lat, lon, speed_kmh, lon_acc, lat_acc, time_s
    """
    log.info("Parsing %s", path.name)
    sep = _detect_sep(path)

    frames: list[pd.DataFrame] = []
    vehicles_in_chunk = 0
    skipped = 0

    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=sep, skipinitialspace=True)

        for line_no, fields in enumerate(reader):
            # Skip the header row (contains column names, not numbers)
            if line_no == 0:
                try:
                    float(fields[2])   # distance_m should be numeric
                except (ValueError, IndexError):
                    continue           # it's a header — skip it

            df = _parse_fields(fields)
            if df is None:
                skipped += 1
                continue

            frames.append(df)
            vehicles_in_chunk += 1

            if vehicles_in_chunk >= chunksize:
                yield pd.concat(frames, ignore_index=True)
                frames = []
                vehicles_in_chunk = 0

    # Yield any remaining vehicles
    if frames:
        yield pd.concat(frames, ignore_index=True)

    if skipped:
        log.debug("  Skipped %d malformed rows in %s", skipped, path.name)


def load_all_csvs(data_dir: Path, chunksize: int = 5_000) -> Iterator[pd.DataFrame]:
    """
    Walk *data_dir* and yield long-format DataFrames from every .csv file found.

    Parameters
    ----------
    data_dir : Path
        Root directory containing pNEUMA CSVs (searched recursively).
    chunksize : int
        Forwarded to parse_csv.

    Yields
    ------
    pd.DataFrame
        Same schema as parse_csv output, with an extra `source_file` column.
    """
    csv_files = sorted(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    log.info("Found %d CSV files in %s", len(csv_files), data_dir)
    for csv_path in csv_files:
        for chunk in parse_csv(csv_path, chunksize=chunksize):
            chunk["source_file"] = csv_path.name
            yield chunk
