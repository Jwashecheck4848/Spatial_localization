"""Why did a stimulus disappear from compare_all_sounds.py's matched-cell comparison?

compare_conditions.compare_pair() only includes a stim_label if, within the shared
(azimuth, elevation) cell grid, BOTH conditions being compared have at least one trial that
is both `reliable` (SNR gate + the MAX_RELIABLE_HZ cap in run_analysis.py) and
`in_calibration` (its position was an inlier of the fitted mount). If every trial of a
stimulus fails either gate on either side, that stimulus silently vanishes from every bar
chart / slopegraph / miss-map built on top of it, with no error -- this script shows exactly
which gate did it.

Run:  python3 diagnose_stim_reliability.py pn_s12_mono_cc pn_s12_vbap_cc ...
      python3 diagnose_stim_reliability.py                      # all 6 new-data conditions
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from compare_conditions import _load_registry, _results_dir, load_condition_results

DEFAULT_CONDITIONS = ["pn_s12_mono_cc", "pn_s12_vbap_cc", "fphon_s12_mono_cc",
                      "fphon_s12_vbap_cc", "newstim_s12_mono_cc", "newstim_s12_vbap_cc"]


def diagnose(name: str, registry: dict) -> None:
    rd = _results_dir(registry, name)
    if not (rd / "all_trials.csv").exists():
        print(f"[skip] {name}: no results yet")
        return
    df, _ = load_condition_results(rd)
    print(f"\n=== {name} ({len(df)} trials) ===")
    print(f"{'stimulus':>16} | {'n':>5} | {'reliable':>8} | {'in_cal':>8} | {'usable':>8} | "
          f"{'snr median':>10} | {'snr<6dB':>8} | {'hz':>6}")
    print("-" * 92)
    for lab, g in df.groupby("stim_label"):
        n = len(g)
        rel = int(g["reliable"].sum())
        cal = int(g["in_calibration"].sum())
        usable = int((g["reliable"] & g["in_calibration"]).sum())
        snr = g["inband_snr_db"].astype(float)
        hz = g["stim_hz"].iloc[0] if "stim_hz" in g and pd.notna(g["stim_hz"]).any() else ""
        below_gate = int((snr < 6.0).sum())
        print(f"{lab:>16} | {n:>5} | {rel:>8} | {cal:>8} | {usable:>8} | "
              f"{snr.median():>10.1f} | {below_gate:>8} | {hz!s:>6}")


def main() -> None:
    registry = _load_registry()
    names = sys.argv[1:] or DEFAULT_CONDITIONS
    for name in names:
        diagnose(name, registry)


if __name__ == "__main__":
    main()
