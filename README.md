# pNEUMA Signal Control Pipeline
### Probabilistic traffic signal control from drone trajectory data

---

## What this system does

Given the raw pNEUMA CSV files (500k vehicle trajectories, Athens), the pipeline:

1. **Detects traffic signals automatically** by clustering GPS stop-events (no pre-labelled map needed)
2. **Detects road arms** per intersection by clustering vehicle approach bearings
3. **Trains three probabilistic models**:
   - *Transition matrix* — P(next signal | current signal, vehicle type)
   - *Travel-time model* — gradient-boosted ETA per signal pair
   - *Volume model* — autoregressive arrival-count forecast per arm
4. **Controls green phases in real time** — the priority controller scores every arm and gives green to the highest-forecast-volume arm, with starvation protection

---

## File layout

```
pneuma_signal/
│
├── config.py                    ← ALL tunable knobs — edit this first
├── requirements.txt
├── train.py                     ← STEP 3: run once on historical CSVs
├── inference.py                 ← STEP 4: replay any CSV to get decisions
├── evaluate.py                  ← STEP 5: measure model accuracy
│
├── pipeline/
│   ├── parser.py                  pNEUMA wide→long streaming CSV parser
│   ├── signal_detector.py         DBSCAN on stop events → signal locations
│   ├── arm_detector.py            Bearing clustering → road arms per signal
│   ├── feature_extractor.py       Signal events + vehicle sequences tables
│   ├── transition_matrix.py       P(next | current, type)  [Laplace-smoothed]
│   ├── travel_time_model.py       GBR travel-time regressor per signal pair
│   ├── volume_model.py            Autoregressive arrival-volume forecaster
│   └── priority_controller.py     Real-time green-phase decision engine
│
├── tools/
│   ├── generate_synthetic.py      Generate a small test CSV (no real data needed)
│   └── dashboard.py               Build a self-contained HTML results dashboard
│
├── data/                        ← PUT YOUR pNEUMA CSVs HERE
├── models/                      ← auto-created; stores trained artefacts
└── results/                     ← auto-created; stores decisions + eval CSVs
```

---

## Step-by-step instructions

### Prerequisites

- Python 3.10 or newer
- ~4 GB free RAM (streaming parser keeps memory bounded)
- The pNEUMA dataset CSVs in a folder you can point to

---

### STEP 1 — Install dependencies

```bash
cd pneuma_signal
pip install -r requirements.txt
```

Expected output: clean install of numpy, pandas, scikit-learn, scipy, pyarrow.

---

### STEP 2 — Place your data

Copy (or symlink) your pNEUMA CSVs into the `data/` folder:

```
pneuma_signal/
└── data/
    ├── 20181024_d1_0900_0930.csv
    ├── 20181024_d1_0930_1000.csv
    └── ...   (all files you want to train on)
```

The parser walks the folder recursively, so subfolders are fine.

**Optional smoke-test (no real data needed):**
```bash
python tools/generate_synthetic.py --out data/synthetic.csv --vehicles 3000
```
This generates a 3000-vehicle synthetic Athens grid CSV you can use to verify
the full pipeline runs end-to-end before touching the 15.8 GB dataset.

---

### STEP 3 — Train

```bash
python train.py --data-dir data/
```

This runs all training stages in sequence.  Expected terminal output:

```
08:01:02  INFO  train — ══ pNEUMA Signal Control — Training Pipeline ══
08:03:45  INFO  signal_detector — Detected 187 signal locations
08:03:51  INFO  arm_detector — Detected 634 arms across 187 signals (avg 3.4 arms/signal)
08:07:22  INFO  feature_extractor — Saved 4812340 signal events
08:09:14  INFO  travel_time_model — Trained 412 per-pair models (plus 1 global)
08:10:58  INFO  volume_model — Trained 601 volume models
08:10:58  INFO  train — ══ Training complete ══
```

Artefacts written to `models/`:

| File | Contents |
|---|---|
| `signals.parquet` | Detected intersection locations (lat, lon) |
| `arms.parquet` | Road arms per signal (bearing, vehicle count) |
| `signal_events.parquet` | Per-visit feature rows used in training |
| `vehicle_sequences.parquet` | Ordered signal-transition records |
| `transition_matrix.pkl` | Serialised TransitionMatrix object |
| `travel_time_models.pkl` | Serialised TravelTimeModel object |
| `volume_models.pkl` | Serialised VolumeModel object |

**Memory note:** The default `--chunksize 3000` keeps peak RAM under 4 GB.
If you have more RAM, raise it to speed up parsing:
```bash
python train.py --data-dir data/ --chunksize 8000
```

---

### STEP 4 — Run inference (signal control)

Point at any single pNEUMA CSV (typically a different day or time window from training):

```bash
python inference.py \
    --csv  data/20181024_d1_0900_0930.csv \
    --out-csv results/decisions.csv
```

Output file `results/decisions.csv` — one row per 30-second tick per signal:

```
time_s,signal_id,green_arm_id
0.0,3,12
0.0,7,28
30.0,3,12
60.0,3,14        ← arm switched at t=60s
...
```

**Run on an entire directory of CSVs:**
```bash
python inference.py --csv data/ --out-csv results/decisions.csv
```

---

### STEP 5 — Evaluate model accuracy

```bash
python evaluate.py --decisions results/decisions.csv --out-dir results/
```

Prints three metric blocks to the terminal and writes four CSVs:

| Output file | What it measures |
|---|---|
| `results/eval_travel_time.csv` | MAE, RMSE, R² per signal pair |
| `results/eval_volume.csv` | MAE, RMSE, R² per (signal, arm) |
| `results/eval_transition.csv` | Top-1 accuracy, top-3 accuracy, log-loss |
| `results/eval_controller.csv` | Green-phase fairness (Gini) per signal |

Example terminal output:
```
── Travel-Time Model ──
       pair   n_samples  MAE_s  RMSE_s     R2
     GLOBAL     3961004   18.4    27.1  0.712
      3→ 7        12841   12.1    18.3  0.801

── Transition Matrix ──
  Top-1 accuracy : 74.3%
  Top-3 accuracy : 95.1%
  Mean log-loss  : 0.4821

── Controller Fairness ──
  Mean Gini (0=perfectly fair) : 0.071
  Mean min-arm share           : 18.4%
```

---

### STEP 6 — View the dashboard

```bash
python tools/dashboard.py --results-dir results/ --out results/dashboard.html
```

Then open `results/dashboard.html` in any browser. No server needed.
Shows KPI cards, green-phase bar chart, and all evaluation tables with
colour-coded R² / Gini badges.

---

## Tuning — `config.py` knobs

| Parameter | Default | What to change it for |
|---|---|---|
| `STOP_SPEED_KMH` | `3.0` | Raise if too few signals detected (noisy stops) |
| `DBSCAN_EPS_DEG` | `0.0003` | ~33 m radius; lower for denser city grids |
| `DBSCAN_MIN_SAMPLES` | `30` | Lower to detect minor intersections |
| `DETECTION_RADIUS_M` | `40` | Raise for larger intersection footprints |
| `ARM_BEARING_EPS_DEG` | `25` | Lower to split arms with similar approach angles |
| `ARM_MIN_VEHICLES` | `10` | Raise to suppress rarely-used arms |
| `BIN_SECONDS` | `30` | Forecast resolution (lower = finer, needs more data) |
| `FORECAST_HORIZON` | `4` | Bins ahead to forecast (4 × 30 s = 2 min lookahead) |
| `ROLLING_WINDOW_BINS` | `6` | History fed to volume model (= 3 min) |
| `MIN_GREEN_SECONDS` | `15` | Minimum green phase before a switch is allowed |
| `MAX_GREEN_SECONDS` | `90` | Maximum before forced arm switch |
| `STARVATION_PENALTY` | `0.15` | Score boost per extra second an arm waits for green |

---

## Quick command reference

```bash
# ── Smoke-test with synthetic data (no real CSVs needed) ──────────────────
python tools/generate_synthetic.py --out data/synthetic.csv
python train.py     --data-dir data/
python inference.py --csv data/synthetic.csv --out-csv results/decisions.csv
python evaluate.py  --decisions results/decisions.csv
python tools/dashboard.py

# ── Full run on real pNEUMA data ──────────────────────────────────────────
python train.py     --data-dir data/ --chunksize 6000
python inference.py --csv data/20181024_d1_0900_0930.csv --out-csv results/decisions.csv
python evaluate.py
python tools/dashboard.py
```
