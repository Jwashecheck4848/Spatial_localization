"""Test the "one pooled mount can't fit two physical rigs" hypothesis for the elevated
mono miss in pn_s12_mono_cc / pn_s12_vbap_cc.

Both conditions pool Setup1 + Setup2 sessions and self-calibrate with ONE global mount
(`per_session` is not set in conditions.json for these two conditions). If the two physical
rig mountings actually differ, a single pooled offset/rotation cannot register both: one rig
ends up well-corrected, the other carries a residual bias in every direction -- which would
show up as elevated miss broadly (not just at specific elevations), and/or a rig-skewed
calibration inlier/outlier split.

This reads all_trials.csv directly (not summary.json) so it can split by rig using the raw
`folder` column, which is not otherwise broken out: `position` is `folder|azXX`, and
`in_calibration` / mounting are only reported POOLED across both rigs in summary.json.

Checks, for pink-noise trials (the calibration-reference stimulus), split by rig:
  1. RAW (pre-calibration) azimuth/elevation residual median per rig -- a persistent
     difference here directly means the two rigs are mounted differently.
  2. CORRECTED (post pooled-mount) miss per rig -- if rig B's corrected miss is much worse
     than rig A's, the pooled mount is systematically fitting rig A and penalizing rig B.
  3. in_calibration inlier rate per rig -- a pooled mount can't literally include incompatible
     positions as inliers, so a skewed split (e.g. rig A mostly inliers, rig B mostly
     outliers) is the other signature of this failure mode.

Run:  python3 diagnose_rig_pooling.py [condition_name ...]
      (default: pn_s12_mono_cc pn_s12_vbap_cc)
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

from compare_conditions import _load_registry, _results_dir, load_condition_results

DEFAULT_CONDITIONS = ["pn_s12_mono_cc", "pn_s12_vbap_cc"]


def _rig_of(folder: str) -> str:
    m = re.search(r"setup([12])", folder, re.I)
    return f"Setup{m.group(1)}" if m else "unknown"


def _circ_resid(meas_deg, truth_deg) -> np.ndarray:
    """Signed shortest-path residual (meas - truth) in degrees, wrapped to [-180, 180]."""
    return (np.asarray(meas_deg, float) - np.asarray(truth_deg, float) + 180) % 360 - 180


def diagnose(name: str, registry: dict) -> None:
    rd = _results_dir(registry, name)
    if not (rd / "all_trials.csv").exists():
        print(f"[skip] {name}: no results yet")
        return
    df, _ = load_condition_results(rd)
    df["rig"] = df["folder"].astype(str).map(_rig_of)
    pn = df[(df["stim_label"] == "pink noise") & df["reliable"]].copy()
    if pn.empty:
        print(f"\n=== {name}: no reliable pink-noise trials ===")
        return

    print(f"\n=== {name}: pooling check (rig vs one global mount) ===")
    print(f"{'rig':>8} | {'n_trials':>8} | {'n_pos':>6} | {'raw az resid':>12} | "
          f"{'raw el resid':>12} | {'corrected miss':>14} | {'in_cal rate':>11}")
    print("-" * 92)
    for rig, g in sorted(pn.groupby("rig")):
        n_trials = len(g)
        n_pos = g["position"].nunique()
        az_resid = _circ_resid(g["doa_az"], g["truth_az"])
        el_resid = g["doa_el"].astype(float) - g["truth_el"].astype(float)
        corrected = g["miss_corrected"].astype(float)
        in_cal_rate = g["in_calibration"].mean()
        print(f"{rig:>8} | {n_trials:>8} | {n_pos:>6} | {np.median(az_resid):>+11.1f}\N{DEGREE SIGN} | "
              f"{np.median(el_resid):>+11.1f}\N{DEGREE SIGN} | {corrected.median():>13.1f}\N{DEGREE SIGN} | "
              f"{in_cal_rate:>10.0%}")

    rigs = sorted(pn["rig"].unique())
    if len(rigs) == 2:
        a = pn[pn["rig"] == rigs[0]]
        b = pn[pn["rig"] == rigs[1]]
        az_a = np.median(_circ_resid(a["doa_az"], a["truth_az"]))
        az_b = np.median(_circ_resid(b["doa_az"], b["truth_az"]))
        el_a = np.median(a["doa_el"].astype(float) - a["truth_el"].astype(float))
        el_b = np.median(b["doa_el"].astype(float) - b["truth_el"].astype(float))
        print(f"\n  rig-to-rig raw offset difference: az {abs(az_a - az_b):.1f}\N{DEGREE SIGN}, "
              f"el {abs(el_a - el_b):.1f}\N{DEGREE SIGN}")
        print("  (a few degrees is normal mounting slop; tens of degrees means one global "
              "mount cannot fit both rigs -- re-fit with calibration.per_session: true)")


def main() -> None:
    registry = _load_registry()
    names = sys.argv[1:] or DEFAULT_CONDITIONS
    for name in names:
        diagnose(name, registry)


if __name__ == "__main__":
    main()
