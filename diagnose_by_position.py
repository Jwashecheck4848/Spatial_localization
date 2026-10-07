"""THROWAWAY DIAGNOSTIC: per-position breakdown of mono vs vbap corrected miss, for one
stimulus, to tell apart two remaining hypotheses for the mono-worse-than-vbap anomaly (window-
length and onset-alignment bugs have both been ruled out by prior checks):

  A) a uniform pipeline/DOA bug -- every mono position for the affected stimulus is elevated
     by roughly the same amount, regardless of which physical speaker channel played it.
  B) a per-speaker TRUTH error -- only a subset of positions (SpeakerChannelIndex / physical
     mono speaker channels) are bad, while the rest look fine. This would point to those
     specific channels' surveyed position in Unity being wrong (plausible if the five new
     stimulus families route through physical speaker channels that weren't exercised/
     validated by the original pink-noise survey) -- VBAP's truth is always a mathematically
     synthesized phantom direction (never subject to a physical survey error), which is
     exactly why VBAP would stay clean even if specific real speakers are mis-surveyed.

Run (needs real data / Teba access):
    python diagnose_by_position.py --stim "applause"
    python diagnose_by_position.py --stim "phone ringing"
    python diagnose_by_position.py --stim "phonemes"          # uses fphon_s12_* instead
    python diagnose_by_position.py --stim "1000 Hz tone"
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from compare_conditions import _load_registry, _results_dir, load_condition_results

NEWSTIM_MONO, NEWSTIM_VBAP = "newstim_s12_mono_cc", "newstim_s12_vbap_cc"
FPHON_MONO, FPHON_VBAP = "fphon_s12_mono_cc", "fphon_s12_vbap_cc"


def _load(registry: dict, name: str) -> pd.DataFrame:
    rd = _results_dir(registry, name)
    df, _ = load_condition_results(rd)
    return df


def _usable(df: pd.DataFrame, stim: str) -> pd.DataFrame:
    sub = df[(df["stim_label"] == stim) & df["reliable"] & df["in_calibration"]].copy()
    sub["cell_az"] = np.round(sub["truth_az"].astype(float)).astype(int) % 360
    sub["cell_el"] = np.round(sub["truth_el"].astype(float)).astype(int)
    return sub


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stim", required=True, help="stim_label, e.g. 'applause', 'phone ringing', "
                    "'phonemes', '1000 Hz tone'")
    args = ap.parse_args()

    registry = _load_registry()
    mono_name, vbap_name = ((FPHON_MONO, FPHON_VBAP) if args.stim == "phonemes"
                            else (NEWSTIM_MONO, NEWSTIM_VBAP))
    mono = _usable(_load(registry, mono_name), args.stim)
    vbap = _usable(_load(registry, vbap_name), args.stim)
    if mono.empty or vbap.empty:
        raise SystemExit(f"No usable rows for stim={args.stim!r} in {mono_name}/{vbap_name}")

    print(f"=== per-position corrected miss: '{args.stim}' (mono vs vbap) ===")
    print(f"{'folder':<55} {'az':>6} {'el':>5} {'mono n':>7} {'mono med':>9} "
          f"{'vbap n':>7} {'vbap med':>9}  flag")
    mono_pos = mono.groupby(["folder", "cell_az", "cell_el"])["miss_corrected"]
    vbap_by_cell = vbap.groupby(["cell_az", "cell_el"])["miss_corrected"]
    vbap_stats = {k: (len(g), float(g.median())) for k, g in vbap_by_cell}
    rows = []
    for (folder, az, el), g in mono_pos:
        m_n, m_med = len(g), float(g.median())
        v_n, v_med = vbap_stats.get((az, el), (0, None))
        flag = ""
        if v_med is not None and m_med - v_med > 15:
            flag = "<-- mono much worse at this position"
        rows.append((folder, az, el, m_n, m_med, v_n, v_med, flag))
    for folder, az, el, m_n, m_med, v_n, v_med, flag in sorted(rows, key=lambda r: r[2]):
        v_str = f"{v_med:9.2f}" if v_med is not None else "     n/a "
        print(f"{folder:<55} {az:>6.0f} {el:>5.0f} {m_n:>7d} {m_med:>9.2f} "
              f"{v_n:>7d} {v_str}  {flag}")

    meds = [r[4] for r in rows]
    spread = max(meds) - min(meds) if meds else 0
    print(f"\nspread across mono positions (max-min median): {spread:.1f} deg")
    if spread > 30:
        print("-> large spread: error is concentrated in specific positions/speaker channels "
              "(points to a per-channel truth/survey problem, not a uniform pipeline bug)")
    else:
        print("-> small spread: error is roughly uniform across positions "
              "(points to a uniform pipeline/DOA bug affecting mono broadly, not a few speakers)")


if __name__ == "__main__":
    main()
