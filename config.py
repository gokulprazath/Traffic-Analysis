"""
config.py — All tunable constants for the pNEUMA signal control pipeline.
Edit this file to adapt the system to different datasets or cities.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")          # folder containing pNEUMA CSV files
MODEL_DIR       = Path("models")        # persisted model artefacts
MODEL_DIR.mkdir(exist_ok=True)

# ── pNEUMA CSV format ──────────────────────────────────────────────────────
# Columns 0-3: trackID, type, distance(m), avg_speed(km/h)
# Then repeating groups of 6: lat, lon, speed(km/h), lon_acc, lat_acc, time(s)
FIXED_COLS      = 4
REPEAT_COLS     = 6                     # size of each trajectory group
COL_LAT         = 0                     # offset within group
COL_LON         = 1
COL_SPEED       = 2
COL_LON_ACC     = 3
COL_LAT_ACC     = 4
COL_TIME        = 5

# ── Signal detection (DBSCAN on stop events) ───────────────────────────────
STOP_SPEED_KMH      = 3.0       # vehicle is "stopped" when speed < this
DBSCAN_EPS_DEG      = 0.0003    # ~33 m at Athens latitude; tune to intersection density
DBSCAN_MIN_SAMPLES  = 30        # minimum stop-events to form a signal cluster
DETECTION_RADIUS_M  = 40.0      # how close (metres) a vehicle must be to "enter" a signal

# ── Arm / road-leg detection ───────────────────────────────────────────────
ARM_BEARING_EPS_DEG = 25.0      # max angular spread to merge bearings into one arm
ARM_MIN_VEHICLES    = 10        # minimum vehicles approaching from that bearing to count

# ── Modelling ─────────────────────────────────────────────────────────────
BIN_SECONDS         = 30        # width of each volume-forecast bin (seconds)
FORECAST_HORIZON    = 4         # how many bins ahead to forecast
ROLLING_WINDOW_BINS = 6         # historical bins fed into volume model (= 3 min)
GBR_N_ESTIMATORS    = 200
GBR_MAX_DEPTH       = 5
GBR_LEARNING_RATE   = 0.08

# ── Priority controller ────────────────────────────────────────────────────
MIN_GREEN_SECONDS   = 15        # no arm gets < this green time
MAX_GREEN_SECONDS   = 90        # safety cap on any single green phase
STARVATION_PENALTY  = 0.15      # volume boost per extra second since last green
                                # (prevents arms from being permanently starved)

# ── Misc ───────────────────────────────────────────────────────────────────
EARTH_RADIUS_M      = 6_371_000  # for haversine distance
VEHICLE_TYPES = [
    "Motorcycle", "Automobile", "Taxi",
    "Medium Vehicle", "Heavy Vehicle", "Bus",
]
