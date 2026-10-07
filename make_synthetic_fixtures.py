"""THROWAWAY DEV TOOL -- generates fake results/<condition>/{all_trials.csv,summary.json} for
the six mono/vbap conditions used by compare_all_sounds.py / diagnose_mono_vbap.py, so those
scripts can be dry-run on a machine with no Teba access (no real recordings needed).

Schema matches run_analysis.py's _write_master_csv() columns exactly. Three shared (truth_az,
truth_el) positions are used across every condition so compare_conditions.py's matched-cell
logic has something to match.

Deliberately plants the reported anomaly (mono scoring WORSE than vbap) in 'phonemes',
'applause', and 'phone ringing' -- concentrated in 'Setup1' sessions, with miss_corrected worse
than miss_raw there (mimicking a bad borrowed mount) -- while 'pink noise' and '250 Hz tone'
stay normal (mono better than vbap), matching what was actually observed. Also plants a capsule
quality flag on the Setup1 sessions so diagnose_mono_vbap.py's quality cross-reference has
something to show.

Run once after cloning (no venv/data prerequisites beyond numpy/pandas):
    python make_synthetic_fixtures.py
Then:
    python diagnose_mono_vbap.py
    python compare_all_sounds.py

Delete this file and the generated results/ once real Teba data is available again -- it only
exists to validate the diagnostic scripts' logic, not to represent real measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
POSITIONS = [(0.0, 0.0), (90.0, 0.0), (180.0, 0.0)]
N_PER_CELL = 15
COLS = ["folder", "position", "trial", "block", "stim_index", "stim_type", "stim_label",
        "stim_hz", "truth_az", "truth_el", "radius_m", "onset_s", "inband_snr_db",
        "reliable", "in_calibration", "doa_az", "doa_el", "doa_az_cal", "doa_el_cal",
        "miss_raw", "miss_corrected"]

rng = np.random.default_rng(20261005)


def _rows(folder: str, stim_label: str, stim_type: str, stim_hz, raw_loc: float,
          raw_scale: float, cal_loc: float, cal_scale: float) -> list[dict]:
    rows = []
    for az, el in POSITIONS:
        for i in range(N_PER_CELL):
            raw = float(abs(rng.normal(raw_loc, raw_scale)))
            cor = float(abs(rng.normal(cal_loc, cal_scale)))
            rows.append({
                "folder": folder, "position": f"{folder}|az{az:g}", "trial": i, "block": 0,
                "stim_index": 0, "stim_type": stim_type, "stim_label": stim_label,
                "stim_hz": stim_hz, "truth_az": az, "truth_el": el, "radius_m": 2.5,
                "onset_s": round(0.1 + 0.01 * i, 2), "inband_snr_db": 20.0,
                "reliable": True, "in_calibration": True,
                "doa_az": round(az + rng.normal(0, 3), 2), "doa_el": round(el + rng.normal(0, 3), 2),
                "doa_az_cal": round(az + rng.normal(0, 3), 2), "doa_el_cal": round(el + rng.normal(0, 3), 2),
                "miss_raw": round(raw, 2), "miss_corrected": round(cor, 2),
            })
    return rows


def _write(name: str, rows: list[dict], flagged_sessions: dict | None = None) -> None:
    results_dir = EXP_DIR / "results" / name
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(results_dir / "all_trials.csv", index=False)
    summary = {
        "mounting": {"az_sign": 1.0, "az_offset": 0.0, "el_offset": 0.0},
        "signal_quality": {"n_sessions": len({r['folder'] for r in rows}),
                            "n_flagged": len(flagged_sessions or {}),
                            "persistent_capsule_suspects": [],
                            "flagged_sessions": flagged_sessions or {}},
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {results_dir / 'all_trials.csv'} ({len(rows)} rows)")


def main() -> None:
    # pn_s12: pink noise, NORMAL (mono better than vbap) -- matches the real, non-anomalous case
    _write("pn_s12_mono_cc",
           _rows("pn_synth_Setup1_CC___synth", "pink noise", "broadband_noise", None,
                 raw_loc=4.0, raw_scale=1.0, cal_loc=4.0, cal_scale=1.0) +
           _rows("pn_synth_Setup2_CC___synth", "pink noise", "broadband_noise", None,
                 raw_loc=4.0, raw_scale=1.0, cal_loc=4.0, cal_scale=1.0))
    _write("pn_s12_vbap_cc",
           _rows("pn_synth_vbap_Setup1_CC___synth", "pink noise", "broadband_noise", None,
                 raw_loc=7.6, raw_scale=1.5, cal_loc=7.6, cal_scale=1.5) +
           _rows("pn_synth_vbap_Setup2_CC___synth", "pink noise", "broadband_noise", None,
                 raw_loc=7.6, raw_scale=1.5, cal_loc=7.6, cal_scale=1.5))

    # fphon_s12: phonemes, ANOMALY planted -- Setup1 mono badly corrected (raw ok, corrected bad)
    _write("fphon_s12_mono_cc",
           _rows("fphon_synth_Setup1_CC___synth", "phonemes", "broadband_noise", None,
                 raw_loc=5.0, raw_scale=1.0, cal_loc=14.0, cal_scale=2.0) +
           _rows("fphon_synth_Setup2_CC___synth", "phonemes", "broadband_noise", None,
                 raw_loc=5.0, raw_scale=1.0, cal_loc=5.0, cal_scale=1.0),
           flagged_sessions={"fphon_synth_Setup1_CC___synth":
                              ["capsule 3: incoherent"]})
    _write("fphon_s12_vbap_cc",
           _rows("fphon_synth_vbap_Setup1_CC___synth", "phonemes", "broadband_noise", None,
                 raw_loc=4.0, raw_scale=1.0, cal_loc=4.0, cal_scale=1.0) +
           _rows("fphon_synth_vbap_Setup2_CC___synth", "phonemes", "broadband_noise", None,
                 raw_loc=4.0, raw_scale=1.0, cal_loc=4.0, cal_scale=1.0))

    # newstim_s12: applause + phone ringing ANOMALOUS (same Setup1 pattern); 250 Hz tone NORMAL
    newstim_mono_rows = (
        _rows("newstim_synth_applause_Setup1_200ms_CC___synth", "applause", "broadband_noise", None,
              raw_loc=5.0, raw_scale=1.0, cal_loc=13.0, cal_scale=2.0) +
        _rows("newstim_synth_applause_Setup2_200ms_CC___synth", "applause", "broadband_noise", None,
              raw_loc=5.0, raw_scale=1.0, cal_loc=5.0, cal_scale=1.0) +
        _rows("newstim_synth_phonering_Setup1_200ms_CC___synth", "phone ringing", "broadband_noise", None,
              raw_loc=5.5, raw_scale=1.0, cal_loc=12.0, cal_scale=2.0) +
        _rows("newstim_synth_phonering_Setup2_200ms_CC___synth", "phone ringing", "broadband_noise", None,
              raw_loc=5.5, raw_scale=1.0, cal_loc=5.5, cal_scale=1.0) +
        _rows("newstim_synth_250hz_Setup1_200ms_CC___synth", "250 Hz tone", "tone", 250.0,
              raw_loc=5.0, raw_scale=1.0, cal_loc=5.0, cal_scale=1.0) +
        _rows("newstim_synth_250hz_Setup2_200ms_CC___synth", "250 Hz tone", "tone", 250.0,
              raw_loc=5.0, raw_scale=1.0, cal_loc=5.0, cal_scale=1.0)
    )
    _write("newstim_s12_mono_cc", newstim_mono_rows,
           flagged_sessions={"newstim_synth_applause_Setup1_200ms_CC___synth":
                              ["capsule 5: incoherent"],
                             "newstim_synth_phonering_Setup1_200ms_CC___synth":
                              ["capsule 5: incoherent"]})
    _write("newstim_s12_vbap_cc",
           _rows("newstim_synth_vbap_applause_Setup1_200ms_CC___synth", "applause", "broadband_noise", None,
                 raw_loc=4.0, raw_scale=1.0, cal_loc=4.0, cal_scale=1.0) +
           _rows("newstim_synth_vbap_applause_Setup2_200ms_CC___synth", "applause", "broadband_noise", None,
                 raw_loc=4.0, raw_scale=1.0, cal_loc=4.0, cal_scale=1.0) +
           _rows("newstim_synth_vbap_phonering_Setup1_200ms_CC___synth", "phone ringing", "broadband_noise", None,
                 raw_loc=4.5, raw_scale=1.0, cal_loc=4.5, cal_scale=1.0) +
           _rows("newstim_synth_vbap_phonering_Setup2_200ms_CC___synth", "phone ringing", "broadband_noise", None,
                 raw_loc=4.5, raw_scale=1.0, cal_loc=4.5, cal_scale=1.0) +
           _rows("newstim_synth_vbap_250hz_Setup1_200ms_CC___synth", "250 Hz tone", "tone", 250.0,
                 raw_loc=7.0, raw_scale=1.5, cal_loc=7.0, cal_scale=1.5) +
           _rows("newstim_synth_vbap_250hz_Setup2_200ms_CC___synth", "250 Hz tone", "tone", 250.0,
                 raw_loc=7.0, raw_scale=1.5, cal_loc=7.0, cal_scale=1.5))

    print("\n[ok] synthetic fixtures written under results/. These are FAKE data for dry-running "
          "diagnose_mono_vbap.py / compare_all_sounds.py only -- delete results/ once real Teba "
          "data is available.")


if __name__ == "__main__":
    main()
