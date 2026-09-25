"""Generic cross-condition comparison over the condition registry (conditions.json).

Any two analysed conditions are compared on their MATCHED (azimuth, elevation) cells only
(nominal cells: az/el rounded to the nearest degree), per stimulus, with bootstrap 95% CIs on
the medians and on the A-B delta. Deltas are oriented `a - b` (the registry orders each pair
so a positive delta reads as a penalty: room_effect = cave - anechoic, vbap_vs_gt = render -
physical). The VBAP comparison also overlays the optimizer's per-direction energy-vector (rE)
prediction (optim_predict.py, predictions_full.json).

The June comparison layer (compare.py) is frozen; this module supersedes it for the July
dataset, sharing its statistics via zyldoa.stats.

Usage:
  python compare_conditions.py                      # every runnable comparison in the registry
  python compare_conditions.py --comparison room_effect
  python compare_conditions.py --a vbap --b gt_cave
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from zyldoa import stats

EXP_DIR = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, GRAY = "#5B7FA5", "#E8873C", "#4C9A6B", "#9aa0a6"


def _load_registry(conditions_file: Path | str = "conditions.json") -> dict:
    cf = Path(conditions_file)
    if not cf.is_absolute():
        cf = EXP_DIR / cf
    return json.loads(cf.read_text(encoding="utf-8"))


def _results_dir(registry: dict, name: str) -> Path:
    p = Path(registry["conditions"][name]["results_dir"])
    return p if p.is_absolute() else EXP_DIR / p


def load_condition_results(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(results_dir / "all_trials.csv")
    for c in ("reliable", "in_calibration"):
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: str(x).strip().lower() == "true")
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    return df, summary


def _cells(df: pd.DataFrame) -> pd.DataFrame:
    """Attach nominal direction cells: az/el rounded to the nearest degree (az mod 360).

    Packed-session truth comes from Unity positions printed at 2 decimals, so measured truth
    like 14.98 deg lands on the nominal 15 deg cell."""
    out = df.copy()
    out["cell_az"] = np.round(out["truth_az"].astype(float)).astype(int) % 360
    out["cell_el"] = np.round(out["truth_el"].astype(float)).astype(int)
    return out


def _usable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["reliable"] & df["in_calibration"]]


def compare_pair(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> dict:
    """Matched-cell comparison: per-stimulus pooled stats + per-cell paired deltas."""
    a = _cells(_usable(df_a))
    b = _cells(_usable(df_b))
    cells_a = {(int(az), int(el)) for az, el in a[["cell_az", "cell_el"]].drop_duplicates().values}
    cells_b = {(int(az), int(el)) for az, el in b[["cell_az", "cell_el"]].drop_duplicates().values}
    shared = sorted(cells_a & cells_b)
    if not shared:
        return {"error": f"no shared (az, el) cells between {label_a} and {label_b}",
                "cells_a": len(cells_a), "cells_b": len(cells_b)}
    sh = pd.DataFrame(shared, columns=["cell_az", "cell_el"])
    a = a.merge(sh, on=["cell_az", "cell_el"])
    b = b.merge(sh, on=["cell_az", "cell_el"])

    labels = stats.stim_order(sorted(set(a["stim_label"]) & set(b["stim_label"])))
    per_stim = {}
    for lab in labels:
        xa = a.loc[a["stim_label"] == lab, "miss_corrected"].to_numpy(float)
        xb = b.loc[b["stim_label"] == lab, "miss_corrected"].to_numpy(float)
        # per-cell paired deltas (each matched direction contributes one median-vs-median delta)
        cell_deltas = []
        for az, el in shared:
            va = a.loc[(a["stim_label"] == lab) & (a["cell_az"] == az) & (a["cell_el"] == el),
                       "miss_corrected"]
            vb = b.loc[(b["stim_label"] == lab) & (b["cell_az"] == az) & (b["cell_el"] == el),
                       "miss_corrected"]
            if len(va) and len(vb):
                cell_deltas.append(float(np.median(va) - np.median(vb)))
        per_stim[lab] = {
            f"{label_a}": stats.stat(xa), f"{label_b}": stats.stat(xb),
            "delta": stats.delta_stat(xa, xb),
            "per_cell_delta_median": round(float(np.median(cell_deltas)), 2) if cell_deltas else None,
            "n_matched_cells": len(cell_deltas),
        }

    # noise-like pooled headline (pink noise + applause)
    nl = stats.NOISE_LIKE
    xa = a.loc[a["stim_label"].isin(nl), "miss_corrected"].to_numpy(float)
    xb = b.loc[b["stim_label"].isin(nl), "miss_corrected"].to_numpy(float)
    headline = {f"{label_a}": stats.stat(xa), f"{label_b}": stats.stat(xb),
                "delta": stats.delta_stat(xa, xb)}

    # per-elevation view of the noise-like delta
    by_el = {}
    for el in sorted({e for _, e in shared}):
        va = a.loc[a["stim_label"].isin(nl) & (a["cell_el"] == el), "miss_corrected"].to_numpy(float)
        vb = b.loc[b["stim_label"].isin(nl) & (b["cell_el"] == el), "miss_corrected"].to_numpy(float)
        by_el[str(el)] = {f"{label_a}": stats.stat(va), f"{label_b}": stats.stat(vb),
                          "delta": stats.delta_stat(va, vb)}

    return {"a": label_a, "b": label_b, "delta_orientation": f"{label_a} - {label_b}",
            "n_shared_cells": len(shared),
            "shared_elevations": sorted({e for _, e in shared}),
            "shared_azimuths": sorted({z for z, _ in shared}),
            "noise_like_headline": headline, "noise_like_by_elevation": by_el,
            "per_stimulus": per_stim}


def _load_predictions(pred_path: Path) -> pd.DataFrame | None:
    if not pred_path.exists():
        return None
    pred = json.loads(pred_path.read_text(encoding="utf-8"))
    rows = [{"cell_az": int(round(e["az"])) % 360, "cell_el": int(round(e["el"])),
             "rE_err_deg": e["energy_err_deg"], "covered": e["covered"]}
            for e in pred["deployed"]["experiment_directions"]]
    return pd.DataFrame(rows)


def _fig_by_stimulus(comp: dict, out_png: Path) -> None:
    labs = list(comp["per_stimulus"].keys())
    la, lb = comp["a"], comp["b"]
    ma = [comp["per_stimulus"][s][la]["median"] for s in labs]
    mb = [comp["per_stimulus"][s][lb]["median"] for s in labs]

    def _err(stat_d):
        ci = stat_d["ci95"]
        if ci[0] is None or stat_d["median"] is None:
            return [0, 0]
        return [max(0.0, stat_d["median"] - ci[0]), max(0.0, ci[1] - stat_d["median"])]

    ea = np.array([_err(comp["per_stimulus"][s][la]) for s in labs]).T
    eb = np.array([_err(comp["per_stimulus"][s][lb]) for s in labs]).T
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x - 0.2, ma, 0.38, yerr=ea, color=ORANGE, capsize=3, label=la)
    ax.bar(x + 0.2, mb, 0.38, yerr=eb, color=BLUE, capsize=3, label=lb)
    ax.set_xticks(x); ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("median corrected miss (deg)")
    ax.set_title(f"{la} vs {lb} — matched directions (n={comp['n_shared_cells']} cells), "
                 "bars = bootstrap 95% CI")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def _fig_azimuth_profile(df_a, df_b, comp: dict, out_png: Path,
                         pred: pd.DataFrame | None = None) -> None:
    a = _cells(_usable(df_a)); b = _cells(_usable(df_b))
    nl = stats.NOISE_LIKE
    els = comp["shared_elevations"]
    fig, axes = plt.subplots(1, len(els), figsize=(4.2 * len(els), 3.8),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    for ax, el in zip(axes[0], els):
        for df, col, lab in ((a, ORANGE, comp["a"]), (b, BLUE, comp["b"])):
            sub = df[df["stim_label"].isin(nl) & (df["cell_el"] == el)]
            med = sub.groupby("cell_az")["miss_corrected"].median().sort_index()
            if len(med):
                th = np.radians(np.append(med.index.values, med.index.values[0]))
                rr = np.append(med.values, med.values[0])
                ax.plot(th, rr, color=col, lw=1.6, label=lab)
        if pred is not None:
            p = pred[(pred["cell_el"] == el) & pred["covered"]].sort_values("cell_az")
            if len(p):
                th = np.radians(np.append(p["cell_az"].values, p["cell_az"].values[0]))
                rr = np.append(p["rE_err_deg"].values, p["rE_err_deg"].values[0])
                ax.plot(th, rr, color=GRAY, lw=1.2, ls="--", label="rE prediction")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_title(f"el {el:+d}°", fontsize=10)
    axes[0][0].legend(loc="upper left", bbox_to_anchor=(-0.35, 1.15), fontsize=8, frameon=False)
    fig.suptitle(f"noise-like median miss by azimuth — {comp['a']} vs {comp['b']}", y=1.02)
    fig.tight_layout(); fig.savefig(out_png, dpi=200, bbox_inches="tight"); plt.close(fig)


def run_comparison(registry: dict, spec: dict, force: bool = False) -> dict | None:
    name = spec["name"]
    la, lb = spec["a"], spec["b"]
    out_dir = EXP_DIR / "results" / "comparisons" / name
    fig_dir = EXP_DIR / "figures" / "comparisons"
    out_json = out_dir / "comparison.json"
    for cond in (la, lb):
        rd = _results_dir(registry, cond)
        if not (rd / "all_trials.csv").exists():
            print(f"  [skip] comparison {name!r}: condition {cond!r} has no results yet")
            return None
    if out_json.exists() and not force:
        comp = json.loads(out_json.read_text(encoding="utf-8"))
        comp["figures"] = [str(fig_dir / f"{name}_by_stimulus.png"),
                           str(fig_dir / f"{name}_azimuth_profile.png")]
        return comp

    df_a, _ = load_condition_results(_results_dir(registry, la))
    df_b, _ = load_condition_results(_results_dir(registry, lb))
    comp = compare_pair(df_a, df_b, la, lb)
    if "error" in comp:
        print(f"  [skip] comparison {name!r}: {comp['error']}")
        return None

    pred = None
    if spec.get("predictions"):
        pp = Path(spec["predictions"])
        pred = _load_predictions(pp if pp.is_absolute() else EXP_DIR / pp)
        if pred is None:
            print(f"  [note] {name}: predictions file missing "
                  f"({spec['predictions']}); run optim_predict.py")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = []
    f1 = fig_dir / f"{name}_by_stimulus.png"
    _fig_by_stimulus(comp, f1); figs.append(str(f1))
    f2 = fig_dir / f"{name}_azimuth_profile.png"
    _fig_azimuth_profile(df_a, df_b, comp, f2, pred=pred); figs.append(str(f2))
    comp["figures"] = figs
    out_json.write_text(json.dumps(comp, indent=2), encoding="utf-8")
    print(f"  [ok] {name}: {comp['n_shared_cells']} matched cells; wrote {out_json}")
    return comp


def run_all(registry: dict | None = None, force: bool = False,
            conditions_file: Path | str = "conditions.json") -> dict:
    if registry is None:
        registry = _load_registry(conditions_file)
    out = {}
    for spec in registry["comparisons"]:
        comp = run_comparison(registry, spec, force=force)
        if comp is not None:
            out[spec["name"]] = comp
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--comparison", default=None, help="run one named comparison from the registry")
    ap.add_argument("--a", default=None, help="ad-hoc: condition A (delta = A - B)")
    ap.add_argument("--b", default=None, help="ad-hoc: condition B")
    ap.add_argument("--conditions-file", default="conditions.json")
    ap.add_argument("--force", action="store_true", help="recompute even if comparison.json exists")
    args = ap.parse_args()
    registry = _load_registry(args.conditions_file)
    if args.a and args.b:
        spec = {"name": f"{args.a}_vs_{args.b}", "a": args.a, "b": args.b}
        run_comparison(registry, spec, force=True)
    elif args.comparison:
        spec = next((s for s in registry["comparisons"] if s["name"] == args.comparison), None)
        if spec is None:
            raise SystemExit(f"No comparison named {args.comparison!r} in the registry")
        run_comparison(registry, spec, force=args.force)
    else:
        run_all(registry, force=args.force)


if __name__ == "__main__":
    main()
