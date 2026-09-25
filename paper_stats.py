"""Compute every derived statistic quoted in the AES paper from raw results.

Single source of truth for paper numbers: reads per-condition outputs
(results/<cond>/all_trials.csv, summary.json) and comparison outputs
(results/comparisons/<name>/comparison.json), recomputes the paper's derived
quantities, and writes results/paper_stats.json. The analysis notebook loads
that JSON for display; nothing here writes into any condition directory.

Run:  python3 paper_stats.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
R = EXP_DIR / "results"
SPEAKERS_CSV = EXP_DIR.parent / "AvLocalization3D" / "Assets" / "CaveSpeakerPositions.csv"
OPT_NOTEBOOK = EXP_DIR.parent / "SpeakerOptimization" / "speaker_optimization_viz.ipynb"

NOISE_LIKE = ["pink noise", "applause"]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def room_effect_stats() -> dict:
    comp = _load(R / "comparisons" / "room_effect" / "comparison.json")
    per_stim = {}
    for stim, d in comp["per_stimulus"].items():
        cave, anech = d["gt_cave"], d["gt_anechoic"]
        per_stim[stim] = {
            "delta_median": d["delta"]["delta"],
            "delta_ci95": d["delta"]["ci95"],
            "cave_median": cave["median"], "cave_p90": cave["p90"],
            "anechoic_median": anech["median"], "anechoic_p90": anech["p90"],
            "room_multiplier": (round(cave["median"] / anech["median"], 2)
                                 if anech["median"] else None),
        }
    by_el = {el: {"delta_median": row["delta"]["delta"], "delta_ci95": row["delta"]["ci95"]}
             for el, row in comp["noise_like_by_elevation"].items()}
    return {"per_stimulus": per_stim, "noise_like_by_elevation": by_el,
            "n_shared_cells": comp["n_shared_cells"]}


def vbap_structure_stats() -> dict:
    comp = _load(R / "comparisons" / "vbap_gtmount_vs_gt" / "comparison.json")
    df = pd.read_csv(R / "vbap_gtmount" / "all_trials.csv")
    df["cell_az"] = df["truth_az"].round().astype(int) % 360
    nl = df[df["stim_label"].isin(NOISE_LIKE) & df["reliable"]].copy()
    nl["elerr"] = nl["doa_el_cal"] - nl["truth_el"]

    ear = nl[nl["truth_el"].round() == 0]
    per_az = ear.groupby("cell_az")["miss_corrected"].median().sort_values(ascending=False)
    up_by_az = ear.groupby("cell_az")["elerr"].median()
    low = nl[nl["truth_el"].round() == -25]

    return {
        "penalty_by_elevation": {el: {"delta_median": row["delta"]["delta"],
                                       "delta_ci95": row["delta"]["ci95"]}
                                 for el, row in comp["noise_like_by_elevation"].items()},
        "ear_level": {
            "worst_cells": {int(k): round(v, 1) for k, v in per_az.head(5).items()},
            "frontal_cells": {a: round(float(per_az.get(a, np.nan)), 1) for a in (0, 15, 345)},
            "band_median": round(float(per_az.median()), 1),
            "elevation_error_median": round(float(ear["elerr"].median()), 1),
            "azimuths_biased_upward": int((up_by_az > 0).sum()),
            "azimuths_total": int(up_by_az.notna().sum()),
        },
        "el_minus25_measured_elevation_median": round(float(low["doa_el_cal"].median()), 1),
        "n_shared_cells": comp["n_shared_cells"],
    }


def instrument_stats() -> dict:
    anech = _load(R / "gt_anechoic" / "summary.json")
    cave = _load(R / "gt_cave" / "summary.json")
    prec = anech["within_position_precision_by_stimulus"]
    days = anech["mounting_by_day"]["by_group"]
    az_offsets = [d["az_offset"] for d in days.values()]
    counts = {}
    for cond in ("gt_cave", "gt_anechoic", "vbap"):
        with open(R / cond / "all_trials.csv") as fh:
            counts[cond] = sum(1 for _ in fh) - 1
    return {
        "anechoic_precision_by_stimulus": {k: round(v, 3) for k, v in prec.items()},
        "anechoic_precision_max": round(max(prec.values()), 2),
        "anechoic_day_az_offset_spread": round(max(az_offsets) - min(az_offsets), 2),
        "mount_elevation_offset": {"cave": round(cave["mounting"]["el_offset"], 1),
                                    "anechoic": round(anech["mounting"]["el_offset"], 1)},
        "trials": {**counts, "total": sum(counts.values())},
    }


def speaker_geometry() -> dict:
    els = sorted(float(r["EL_Degrees"]) for r in csv.DictReader(open(SPEAKERS_CSV)))
    return {
        "n_speakers": len(els),
        "n_above_20deg": sum(1 for e in els if e > 20),
        "n_within_6deg_ear_level": sum(1 for e in els if abs(e) <= 6),
        "mean_elevation": round(sum(els) / len(els), 1),
        "min_elevation": round(els[0], 1), "max_elevation": round(els[-1], 1),
        "elevations": [round(e, 1) for e in els],
    }


def layout_comparison() -> dict:
    """Reference-layout metrics from the executed optimization notebook (cell table)."""
    nb = json.loads(OPT_NOTEBOOK.read_text(encoding="utf-8"))
    rows = {}
    for c in nb["cells"]:
        out = ""
        for o in c.get("outputs", []):
            if "text" in o:
                out += "".join(o["text"])
        if "Cov(1)" in out and "Optimized" in out:
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 9 and parts[-1].replace(".", "", 1).isdigit() \
                        and parts[-7].endswith("%"):
                    name = " ".join(parts[:-8])
                    if not name or name.startswith("="):
                        continue
                    rows[name] = {"coverage_pct": float(parts[-7].rstrip("%")),
                                   "energy_err": float(parts[-3]),
                                   "up_max_gap_rad": float(parts[-2]),
                                   "cost": float(parts[-1])}
            break
    return rows


def _spectral_predictor() -> dict:
    """Effective-bandwidth predictor of the room penalty (see spectral_predictor.py)."""
    import spectral_predictor
    return spectral_predictor.compute()


def main() -> None:
    stats = {
        "_provenance": "Derived statistics quoted in the AES paper; regenerate with "
                       "python3 paper_stats.py. Sources: results/<cond>/, "
                       "results/comparisons/<name>/, CaveSpeakerPositions.csv, "
                       "speaker_optimization_viz.ipynb.",
        "room_effect": room_effect_stats(),
        "vbap_structure": vbap_structure_stats(),
        "instrument": instrument_stats(),
        "speaker_geometry": speaker_geometry(),
        "layout_comparison": layout_comparison(),
        "spectral_predictor": _spectral_predictor(),
    }
    out = R / "paper_stats.json"
    out.write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({k: (v if k == "_provenance" else "...") for k, v in stats.items()}))


if __name__ == "__main__":
    main()
