"""Diagnostic for the VBAP-beats-mono anomaly: mono shows a HIGHER median corrected miss than
VBAP for some stimuli (observed: applause / phone ringing / phonemes), which should be physically
impossible -- a VBAP phantom image should never out-localize a real physical speaker. Pink noise
does NOT show the inversion, so the shared mount isn't obviously broken; this narrows down where
the mono numbers go wrong, stimulus by stimulus:

  1. Matched-cell mono-vs-vbap delta per stimulus (same statistic as compare_all_sounds.py),
     flagging every stimulus where mono is significantly worse than vbap (95% CI of the delta
     excludes zero and favors vbap).
  2. For each flagged stimulus, split mono's miss_raw/miss_corrected by physical rig (Setup1 vs
     Setup2, parsed from the session folder name) -- if one rig drives the anomaly, it should be
     concentrated there.
  3. miss_raw vs miss_corrected for mono on the flagged stimuli -- if the corrected number is
     WORSE than raw, the borrowed mount (not the raw measurement) is adding the error.
  4. Cross-reference capsule-quality flags (summary.json signal_quality.flagged_sessions) for the
     sessions feeding each flagged stimulus.

Prereq: run_analysis.py must already have produced results for the six conditions below (and,
for a clean read, should be re-run after the run_analysis.py in_calibration fix so tone trials
are no longer dropped from the mono/vbap pools).

Run:  python3 diagnose_mono_vbap.py
"""
from __future__ import annotations

import pandas as pd

from compare_conditions import _load_registry, _results_dir, load_condition_results, _cells, _usable, compare_pair

MONO_CONDITIONS = ["pn_s12_mono_cc", "fphon_s12_mono_cc", "newstim_s12_mono_cc"]
VBAP_CONDITIONS = ["pn_s12_vbap_cc", "fphon_s12_vbap_cc", "newstim_s12_vbap_cc"]


def _setup_of(folder: str) -> str:
    low = folder.lower()
    if "setup1" in low:
        return "Setup1"
    if "setup2" in low:
        return "Setup2"
    return "?"


def _load_all(registry: dict, names: list[str]) -> tuple[pd.DataFrame, dict]:
    frames, summaries = [], {}
    for name in names:
        rd = _results_dir(registry, name)
        if not (rd / "all_trials.csv").exists():
            print(f"  [skip] {name}: no results yet (run `python run_analysis.py "
                  f"--condition {name}` first)")
            continue
        df, summary = load_condition_results(rd)
        df["_condition"] = name
        frames.append(df)
        summaries[name] = summary
    if not frames:
        raise SystemExit(f"None of {names} have results yet.")
    return pd.concat(frames, ignore_index=True), summaries


def _flagged_stimuli(df_mono: pd.DataFrame, df_vbap: pd.DataFrame) -> list[str]:
    comp = compare_pair(df_mono, df_vbap, "mono", "vbap")
    if "error" in comp:
        raise SystemExit(comp["error"])
    flagged = []
    for lab, row in comp["per_stimulus"].items():
        d = row["delta"]
        lo, _hi = d["ci95"]
        if d["delta"] is not None and d["delta"] > 0 and lo is not None and lo > 0:
            flagged.append(lab)
    for lab, row in comp["per_stimulus"].items():
        d = row["delta"]
        tag = "  <-- MONO WORSE (suspicious)" if lab in flagged else ""
        print(f"  {lab:>16s}: mono={row['mono']['median']:>5}  vbap={row['vbap']['median']:>5}  "
              f"delta(mono-vbap)={d['delta']:>5} 95%CI={d['ci95']}{tag}")
    return flagged


def _setup_breakdown(mono_usable: pd.DataFrame, stim_label: str) -> None:
    sub = mono_usable[mono_usable["stim_label"] == stim_label].copy()
    sub["setup"] = sub["folder"].map(_setup_of)
    print(f"    Setup breakdown for '{stim_label}' (mono):")
    for setup, g in sub.groupby("setup"):
        print(f"      {setup}: n={len(g)}  miss_raw median={g['miss_raw'].median():.2f}  "
              f"miss_corrected median={g['miss_corrected'].median():.2f}")


def _raw_vs_corrected(mono_usable: pd.DataFrame, stim_label: str) -> None:
    sub = mono_usable[mono_usable["stim_label"] == stim_label]
    raw, cor = sub["miss_raw"].median(), sub["miss_corrected"].median()
    verdict = ("mount correction INCREASES error -- suspect the borrowed mount"
               if cor > raw else "mount correction reduces error (normal)")
    print(f"    raw vs corrected for '{stim_label}' (mono): raw median={raw:.2f}  "
          f"corrected median={cor:.2f}  -- {verdict}")


def _quality_for_stim(summaries: dict, df_mono: pd.DataFrame, stim_label: str) -> None:
    folders = sorted(df_mono.loc[df_mono["stim_label"] == stim_label, "folder"].unique())
    any_flag = False
    for name, summary in summaries.items():
        flagged = summary.get("signal_quality", {}).get("flagged_sessions", {})
        for f in folders:
            if f in flagged:
                any_flag = True
                print(f"      [{name}] {f}: {flagged[f]}")
    print(f"    capsule/stimulus quality flags for '{stim_label}' sessions: "
          f"{'none' if not any_flag else 'see above'} ({len(folders)} sessions checked)")


def main() -> None:
    registry = _load_registry()
    df_mono, summ_mono = _load_all(registry, MONO_CONDITIONS)
    df_vbap, _summ_vbap = _load_all(registry, VBAP_CONDITIONS)

    print("=== matched-cell mono vs vbap delta per stimulus ===")
    flagged = _flagged_stimuli(df_mono, df_vbap)

    if not flagged:
        print("\nNo stimulus shows mono significantly worse than vbap -- anomaly not reproduced "
              "with the current data (may be resolved by the in_calibration fix, or was a "
              "transient/pre-fix artifact). Re-run after refreshing results if this looks stale.")
        return

    mono_usable = _cells(_usable(df_mono))
    print(f"\n=== drill-down on flagged stimuli: {flagged} ===")
    for lab in flagged:
        print(f"\n--- {lab} ---")
        _setup_breakdown(mono_usable, lab)
        _raw_vs_corrected(mono_usable, lab)
        _quality_for_stim(summ_mono, df_mono, lab)


if __name__ == "__main__":
    main()
