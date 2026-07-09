"""
tools/dashboard.py — Generate a self-contained HTML dashboard from
results/decisions.csv and the evaluation CSVs.

Opens in any browser; no server needed.

Usage
-----
    python tools/dashboard.py --out results/dashboard.html
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pNEUMA Signal Control Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #f5f4f0; --card: #fff; --border: #e2e0d8;
    --text: #1a1a18; --muted: #6b6a65; --accent: #1d9e75;
    --accent2: #534ab7; --warn: #ba7517; --danger: #a32d2d;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); font-size: 14px; }}
  header {{ background: var(--text); color: #fff; padding: 18px 32px; display: flex; align-items: center; gap: 16px; }}
  header h1 {{ font-size: 18px; font-weight: 500; }}
  header span {{ font-size: 12px; color: #aaa; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; padding: 24px 32px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 20px; }}
  .card h2 {{ font-size: 13px; font-weight: 500; color: var(--muted); margin-bottom: 14px; text-transform: uppercase; letter-spacing: .05em; }}
  .kpi {{ font-size: 32px; font-weight: 500; color: var(--accent); }}
  .kpi-sub {{ font-size: 12px; color: var(--muted); margin-top: 4px; }}
  .wide {{ grid-column: 1 / -1; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ text-align: left; padding: 6px 10px; border-bottom: 2px solid var(--border); font-weight: 500; color: var(--muted); font-size: 11px; text-transform: uppercase; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: #f9f8f4; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; }}
  .good  {{ background: #e1f5ee; color: #0f6e56; }}
  .ok    {{ background: #faeeda; color: #854f0b; }}
  .bad   {{ background: #fcebeb; color: #a32d2d; }}
  canvas {{ max-height: 260px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>pNEUMA Signal Control — Dashboard</h1>
    <span>Generated from decisions.csv + evaluation outputs</span>
  </div>
</header>

<div class="grid">

  <!-- KPI cards -->
  <div class="card">
    <h2>Signals controlled</h2>
    <div class="kpi" id="kpi-signals">—</div>
    <div class="kpi-sub">unique intersections</div>
  </div>
  <div class="card">
    <h2>Total green decisions</h2>
    <div class="kpi" id="kpi-decisions">—</div>
    <div class="kpi-sub">30-second ticks</div>
  </div>
  <div class="card">
    <h2>TT Model MAE</h2>
    <div class="kpi" id="kpi-mae">—</div>
    <div class="kpi-sub">seconds (global)</div>
  </div>
  <div class="card">
    <h2>Transition Top-1</h2>
    <div class="kpi" id="kpi-top1">—</div>
    <div class="kpi-sub">next-signal accuracy</div>
  </div>

  <!-- Green share distribution chart -->
  <div class="card wide">
    <h2>Green time share per arm (all signals)</h2>
    <canvas id="chart-green"></canvas>
  </div>

  <!-- Travel-time model table -->
  <div class="card wide">
    <h2>Travel-time model — per signal pair</h2>
    <div id="tt-table-wrap" style="overflow-x:auto"></div>
  </div>

  <!-- Volume model table -->
  <div class="card wide">
    <h2>Volume model — per (signal, arm)</h2>
    <div id="vol-table-wrap" style="overflow-x:auto"></div>
  </div>

  <!-- Controller fairness table -->
  <div class="card wide">
    <h2>Controller fairness per signal (Gini: 0 = perfectly fair)</h2>
    <div id="ctrl-table-wrap" style="overflow-x:auto"></div>
  </div>

</div>

<script>
const DATA = __DATA_PLACEHOLDER__;

// KPIs
document.getElementById("kpi-signals").textContent   = DATA.kpi.signals ?? "—";
document.getElementById("kpi-decisions").textContent = DATA.kpi.decisions?.toLocaleString() ?? "—";
document.getElementById("kpi-mae").textContent       = DATA.kpi.tt_mae != null ? DATA.kpi.tt_mae + "s" : "—";
document.getElementById("kpi-top1").textContent      = DATA.kpi.top1 != null ? (DATA.kpi.top1 * 100).toFixed(1) + "%" : "—";

// Green share bar chart
if (DATA.green_shares && DATA.green_shares.length) {
  const labels = DATA.green_shares.map(r => `Sig ${r.signal_id} Arm ${r.green_arm_id}`);
  const values = DATA.green_shares.map(r => r.share_pct);
  new Chart(document.getElementById("chart-green"), {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "Green share %", data: values,
        backgroundColor: "#1d9e75", borderRadius: 4 }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: "%" } } },
      responsive: true, maintainAspectRatio: true,
    }
  });
}

// Generic table builder
function makeTable(rows, cols) {
  if (!rows || !rows.length) return "<p style='color:#999;font-size:12px'>No data</p>";
  let html = "<table><thead><tr>" + cols.map(c => `<th>${c}</th>`).join("") + "</tr></thead><tbody>";
  for (const r of rows) {
    html += "<tr>" + cols.map(c => {
      const v = r[c];
      if (c === "R2" || c === "r2") {
        const cls = v == null ? "" : v > 0.7 ? "good" : v > 0.4 ? "ok" : "bad";
        return `<td><span class="badge ${cls}">${v ?? "—"}</span></td>`;
      }
      if (c === "gini") {
        const cls = v < 0.1 ? "good" : v < 0.3 ? "ok" : "bad";
        return `<td><span class="badge ${cls}">${v ?? "—"}</span></td>`;
      }
      return `<td>${v ?? "—"}</td>`;
    }).join("") + "</tr>";
  }
  return html + "</tbody></table>";
}

document.getElementById("tt-table-wrap").innerHTML  = makeTable(DATA.tt,  ["pair","n_samples","MAE_s","RMSE_s","R2"]);
document.getElementById("vol-table-wrap").innerHTML = makeTable(DATA.vol, ["signal_id","arm_id","n_windows","MAE","RMSE","R2"]);
document.getElementById("ctrl-table-wrap").innerHTML = makeTable(DATA.ctrl, ["signal_id","n_arms","total_phases","gini","min_share%","max_share%"]);
</script>
</body>
</html>
"""


def _safe_load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def build_dashboard(results_dir: Path, out_path: Path) -> None:
    decisions = _safe_load(results_dir / "decisions.csv")
    tt_eval   = _safe_load(results_dir / "eval_travel_time.csv")
    vol_eval  = _safe_load(results_dir / "eval_volume.csv")
    ctrl_eval = _safe_load(results_dir / "eval_controller.csv")
    tm_eval   = _safe_load(results_dir / "eval_transition.csv")

    # KPIs
    kpi = {
        "signals":   int(decisions["signal_id"].nunique()) if not decisions.empty else None,
        "decisions": int(len(decisions)) if not decisions.empty else None,
        "tt_mae":    float(round(tt_eval[tt_eval["pair"] == "GLOBAL"]["MAE_s"].iloc[0], 1))
                     if not tt_eval.empty and "MAE_s" in tt_eval.columns else None,
        "top1":      float(tm_eval["top1_accuracy"].iloc[0])
                     if not tm_eval.empty and "top1_accuracy" in tm_eval.columns else None,
    }

    # Green share per (signal, arm)
    green_shares = []
    if not decisions.empty:
        counts = decisions.groupby(["signal_id", "green_arm_id"]).size().reset_index(name="bins")
        totals = counts.groupby("signal_id")["bins"].transform("sum")
        counts["share_pct"] = (counts["bins"] / totals * 100).round(1)
        green_shares = counts.to_dict(orient="records")

    data = {
        "kpi":          kpi,
        "green_shares": green_shares,
        "tt":           tt_eval.head(50).to_dict(orient="records") if not tt_eval.empty else [],
        "vol":          vol_eval.head(50).to_dict(orient="records") if not vol_eval.empty else [],
        "ctrl":         ctrl_eval.to_dict(orient="records") if not ctrl_eval.empty else [],
    }

    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", json.dumps(data))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    log.info("Dashboard written → %s", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--out",         type=Path, default=Path("results/dashboard.html"))
    args = p.parse_args()
    build_dashboard(args.results_dir, args.out)


if __name__ == "__main__":
    main()
