"""Mono vs VBAP, broken down by every recorded sound type, pooled Setup1 + Setup2 (CC).

Each stimulus family lives in its own condition (pn_s12_*, fphon_s12_*, newstim_s12_*) because
the three batches were recorded at different times with different folder-naming conventions
(see conditions.json). This script concatenates their per-trial results into one "mono" and one
"vbap" dataframe and reuses compare_conditions.py's matched-cell statistics, so the output is
ONE comparison covering all seven stimuli (pink noise, phonemes, 250/1000/5000 Hz tone, phone
ringing, applause) instead of three separate per-family comparisons.

Prereq: run_analysis.py must already have produced results for all six conditions below.

Run:  python3 compare_all_sounds.py
Writes results/comparisons/all_sounds_mono_vs_vbap/comparison.json and
figures/comparisons/all_sounds_mono_vs_vbap_by_stimulus.png (+ azimuth_profile.png).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from compare_conditions import (
    EXP_DIR, _load_registry, _results_dir, load_condition_results,
    compare_pair, _fig_by_stimulus, _fig_azimuth_profile,
)

MONO_CONDITIONS = ["pn_s12_mono_cc", "fphon_s12_mono_cc", "newstim_s12_mono_cc"]
VBAP_CONDITIONS = ["pn_s12_vbap_cc", "fphon_s12_vbap_cc", "newstim_s12_vbap_cc"]
NAME = "all_sounds_mono_vs_vbap"


def _load_and_concat(registry: dict, names: list[str]) -> pd.DataFrame:
    frames = []
    for name in names:
        rd = _results_dir(registry, name)
        if not (rd / "all_trials.csv").exists():
            print(f"  [skip] {name}: no results yet (run `python run_analysis.py "
                  f"--condition {name}` first)")
            continue
        df, _ = load_condition_results(rd)
        frames.append(df)
    if not frames:
        raise SystemExit(f"None of {names} have results yet.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    registry = _load_registry()
    df_mono = _load_and_concat(registry, MONO_CONDITIONS)
    df_vbap = _load_and_concat(registry, VBAP_CONDITIONS)

    comp = compare_pair(df_mono, df_vbap, "mono", "vbap")
    if "error" in comp:
        raise SystemExit(f"{NAME}: {comp['error']}")

    out_dir = EXP_DIR / "results" / "comparisons" / NAME
    fig_dir = EXP_DIR / "figures" / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    f1 = fig_dir / f"{NAME}_by_stimulus.png"
    _fig_by_stimulus(comp, f1)
    f2 = fig_dir / f"{NAME}_azimuth_profile.png"
    _fig_azimuth_profile(df_mono, df_vbap, comp, f2)
    comp["figures"] = [str(f1), str(f2)]
    comp["_provenance"] = {"mono_conditions": MONO_CONDITIONS, "vbap_conditions": VBAP_CONDITIONS}

    out_json = out_dir / "comparison.json"
    out_json.write_text(json.dumps(comp, indent=2), encoding="utf-8")
    print(f"[ok] {NAME}: {comp['n_shared_cells']} matched cells; "
          f"stimuli = {list(comp['per_stimulus'].keys())}")
    print(f"wrote {out_json}")
    print(f"wrote {f1}")
    print(f"wrote {f2}")


if __name__ == "__main__":
    main()
