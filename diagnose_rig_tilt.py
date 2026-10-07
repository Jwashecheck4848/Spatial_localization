"""THROWAWAY DIAGNOSTIC: for one condition's all_trials.csv, print the raw (uncorrected)
measured-vs-truth azimuth relationship for pink-noise calibration trials, split by rig
(_rig_of on the folder), sorted by truth azimuth.

Context: after fixing (1) per-rig mount propagation, (2) a KeyError on the rotation matrix,
(3) an over-constrained auto model-selection gate, and (4) a missing elevation-handedness
parameter, pn_s12_mono_cc's Setup2 rig STILL shows ~25-48 deg corrected miss across ALL of
its own el-20 azimuths even under ITS OWN in-sample offset fit (vs Setup1's el+20 getting to
2-9 deg). An in-sample constant-offset fit should drive its own dominant cluster's median
residual close to zero if a constant offset is actually an adequate model -- it is not here,
which means either (a) there's a real azimuth-dependent error (tilt) that only a rotation fit
could capture, but Setup2 has only one well-populated elevation so rotation cannot be
constrained from this condition's own data, or (b) the raw DOA measurement itself is bad for
this rig/elevation (not a calibration/model problem at all).

This script shows the raw numbers needed to tell those apart: if (truth_az - meas_az) drifts
smoothly/sinusoidally with azimuth, that is the signature of a real tilt (a). If it is noisy/
inconsistent with no clear pattern, that points to (b).

Run (needs real data / Teba access):
    python diagnose_rig_tilt.py pn_s12_mono_cc
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from run_analysis import _rig_of
from compare_conditions import _load_registry, _results_dir


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "pn_s12_mono_cc"
    registry = _load_registry()
    rd = _results_dir(registry, name)
    csv_path = rd / "all_trials.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} missing -- run `python run_analysis.py --condition {name}` first")

    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["stim_label"] != "pink noise" or r["reliable"] != "True":
                continue
            rows.append(r)

    by_rig_pos: dict[str, dict[str, list]] = {}
    for r in rows:
        rig = _rig_of(r["folder"])
        by_rig_pos.setdefault(rig, {}).setdefault(r["position"], []).append(r)

    for rig, positions in sorted(by_rig_pos.items()):
        print(f"\n=== rig: {rig} ({len(positions)} positions) ===")
        print(f"{'truth_az':>9} {'truth_el':>9} | {'meas_az_med':>11} {'meas_el_med':>11} | "
              f"{'az_diff':>8} {'el_diff':>8} | {'az_std':>7} {'el_std':>7}  n")
        entries = []
        for pid, prows in positions.items():
            ta = float(prows[0]["truth_az"]); te = float(prows[0]["truth_el"])
            mas = sorted(float(r["doa_az"]) for r in prows)
            mes = sorted(float(r["doa_el"]) for r in prows)
            ma = mas[len(mas) // 2]; me = mes[len(mes) // 2]
            az_diff = ((ta - ma) + 180) % 360 - 180
            # Trial-level jitter (circular std for az, linear for el) -- distinguishes a NOISY
            # raw estimate (large std: onset/signal-quality problem at this position) from a
            # TIGHT-BUT-WRONG one (small std: systematic/physical error, not a signal problem).
            import numpy as np
            az_arr = np.array(mas)
            az_std = float(np.degrees(np.sqrt(-2 * np.log(np.hypot(
                np.mean(np.cos(np.radians(az_arr))), np.mean(np.sin(np.radians(az_arr))))))))
            el_std = float(np.std(mes))
            entries.append((ta, te, ma, me, az_diff, te - me, az_std, el_std, len(prows)))
        for ta, te, ma, me, az_diff, el_diff, az_std, el_std, n in sorted(entries):
            print(f"{ta:9.1f} {te:9.1f} | {ma:11.2f} {me:11.2f} | "
                  f"{az_diff:8.2f} {el_diff:8.2f} | {az_std:7.2f} {el_std:7.2f} {n:3d}")


if __name__ == "__main__":
    main()
