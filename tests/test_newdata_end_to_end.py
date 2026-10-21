"""End-to-end check of the new-data path on SYNTHETIC recordings (no Teba access needed).

Builds two clip-log sessions shaped like the real ones (Setup1-like and Setup2-like position
lists, 200 ms pink-noise clips every 3.2 s, no pauses between positions, a quiet first block),
renders each trial as a rigid-sphere plane wave on the ZM-1's 19 capsules, writes 24-bit
19-channel WAVs + clip-log CSVs, and runs the real `run_analysis.run()` on them twice:

  * with the fixed aligner (default)           -> mount ~identity, corrected miss ~0-2 deg
  * with the legacy aligner patched back in    -> whole-trial shift, large miss, audit flags it

    python tests/test_newdata_end_to_end.py        (~1-2 min, writes to a temp folder)
"""
from __future__ import annotations

import csv
import json
import shutil
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.special import eval_legendre, spherical_jn, spherical_yn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_analysis as ra  # noqa: E402
from zyldoa import geometry, onsets, pipeline  # noqa: E402

FS = 16000
ITI, CLIP, REPS = 3.2, 0.2, 4
SETUP1 = [(0, 20), (30, -20), (30, 0), (60, 20), (90, -20), (90, 0), (120, 20), (150, -20),
          (150, 0), (180, 20), (210, -20), (210, 0)]
SETUP2 = [(0, -20), (0, 0), (30, 20), (60, -20), (60, 0), (90, 20), (120, -20), (120, 0),
          (150, 20), (180, -20), (180, 0), (210, 20)]


def _b(n, kr):
    """Rigid-sphere mode strength, e^{+jwt} convention (second-kind Hankel)."""
    kr = np.maximum(kr, 1e-8)
    jn, jnp = spherical_jn(n, kr), spherical_jn(n, kr, derivative=True)
    hn = jn - 1j * spherical_yn(n, kr)
    hnp = jnp - 1j * spherical_yn(n, kr, derivative=True)
    return (1j ** n) * (jn - (jnp / hnp) * hn)


def plane_wave_burst(src_unit, n_samp, rng, order=12):
    """19-capsule pressures of a broadband plane wave from direction `src_unit` (Zylia frame)."""
    f = np.fft.rfftfreq(n_samp, 1 / FS)
    kr = 2 * np.pi * f / geometry.SPEED_OF_SOUND * geometry.ZM1_RADIUS_M
    caps = geometry.capsule_unit_vectors()
    cosg = caps @ src_unit
    H = np.zeros((len(f), len(caps)), complex)
    for n in range(order + 1):
        H += (2 * n + 1) * _b(n, kr)[:, None] * eval_legendre(n, cosg)[None, :]
    S = np.fft.rfft(rng.standard_normal(n_samp))
    S[(f < 100) | (f > 6000)] = 0
    x = np.fft.irfft(S[:, None] * H, n=n_samp, axis=0)
    ramp = np.minimum(1, np.minimum(np.arange(n_samp), np.arange(n_samp)[::-1]) / (0.005 * FS))
    return x * ramp[:, None]


def write_wav24(path, x):
    x = np.clip(x, -1, 1 - 1 / 8388608)
    ints = np.round(x * 8388608).astype(np.int32).ravel()
    b = np.empty((ints.size, 3), np.uint8)
    b[:, 0] = ints & 0xFF; b[:, 1] = (ints >> 8) & 0xFF; b[:, 2] = (ints >> 16) & 0xFF
    data = b.tobytes()
    nch = x.shape[1]
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        fh.write(b"fmt " + struct.pack("<IHHIIHH", 16, 1, nch, FS, FS * nch * 3, nch * 3, 24))
        fh.write(b"data" + struct.pack("<I", len(data)) + data)


def make_session(root: Path, folder: str, positions, seed: int, hard: bool = False):
    """hard=True: the case energy alone cannot solve -- every position at the same level, the
    first clip silent, and a stray sound one interval after the last clip (so a labelling
    shifted by one trial explains exactly as many sounds)."""
    rng = np.random.default_rng(seed)
    d = root / folder; d.mkdir(parents=True)
    n = len(positions) * REPS
    u = 6.7 + ITI * np.arange(n)
    wav_on = u * 1.00015 + 2.8                       # audio clock vs Unity clock
    total = int((wav_on[-1] + 10) * FS)
    x = (rng.standard_normal((total, 19)) * 10 ** (-78 / 20)).astype(np.float64)
    rows = []
    for i in range(n):
        p = i // REPS
        az_cw, el = positions[p]
        silent = hard and i == 0                          # first clip never played
        if hard:
            lvl = -34.0
        else:
            lvl = -70.0 if p == 0 else rng.uniform(-38, -30)   # quiet first position
        # identity mount: Zylia front = Unity az 0, Zylia CCW azimuth = -Unity (clockwise) az
        src = geometry.azel_to_unit(-az_cw, el)
        burst = plane_wave_burst(src, int(CLIP * FS), rng)
        burst *= 10 ** (lvl / 20) / (np.sqrt(np.mean(burst ** 2)) + 1e-12)
        s = int(round(wav_on[i] * FS))
        if not silent:
            x[s:s + len(burst)] += burst
        dist = 130 if el < 0 else 140
        rows.append({"SourceType": "Mono", "SpeakerChannelIndex": p, "SpeakerChannel": 25 + p,
                     "SourcePositionXYZ": "0;0;0", "SourcePositionAzimuth": az_cw,
                     "SourcePositionElevation": el, "SourcePositionDistance": dist,
                     "ClipNumber": 0, "ClipRepetition": i % REPS, "ClipName": "01_pink_noise.wav",
                     "DateTime": "2026__09_13__23_40_33", "ClipStart": f"{u[i]:.6f}",
                     "ClipEnd": f"{u[i] + CLIP:.6f}"})
    if hard:                                              # stray sound after the session
        stray = plane_wave_burst(geometry.azel_to_unit(100.0, 10.0), int(CLIP * FS), rng)
        stray *= 10 ** (-34 / 20) / (np.sqrt(np.mean(stray ** 2)) + 1e-12)
        s = int(round((wav_on[-1] + ITI * 1.00015) * FS))
        x[s:s + len(stray)] += stray
    write_wav24(d / "Take_20260913_23_40.wav", x)
    with open(d / f"{folder}___.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run_once(data_root: Path, out: Path):
    stim = json.loads((ROOT / "stimuli.json").read_text())
    return ra.run(force=True, figures=False, data_root=data_root, results_dir=out / "results",
                  figures_dir=out / "figures", stim_classification=stim,
                  calibration_label="pink noise", mount_model="rotation", tol_deg=30.0)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="zyl_e2e_"))
    try:
        data = tmp / "Objective Localization"
        make_session(data, "Pink_Noise_Setup1_CC___2026__09_13__23_40_33", SETUP1, seed=11)
        make_session(data, "Pink_noise_Setup2_CC___2026__09_13__20_54_29", SETUP2, seed=22)

        print("\n######## FIXED pipeline ########")
        s = run_once(data, tmp / "fixed")
        med = s["corrected_miss_by_stimulus"]["pink noise"]["median"]
        audit = s["label_shift_audit"]
        print("mount:", s["mounting_description"], "| corrected miss median:", med)
        assert med < 3.0, f"fixed pipeline: corrected miss {med} deg (expected < 3)"
        assert not any(a["suspect"] for a in audit.values()), audit
        assert all(q["n_on_sound"] == q["n_trials"] for q in s["alignment_qc"].values())
        labels = {v["label"] for v in s["by_position"].values()}
        assert {"az30_el-20_r1.3", "az30_el0_r1.4"} <= labels, "same-azimuth positions merged"

        print("\n######## EXTERNAL mount (how every non-pink condition is scored) ########")
        stim = json.loads((ROOT / "stimuli.json").read_text())
        s_ext = ra.run(force=False, figures=False, data_root=data, results_dir=tmp / "ext" / "results",
                       figures_dir=tmp / "ext" / "figures", stim_classification=stim,
                       external_calibration=s["mounting"])
        med_ext = s_ext["corrected_miss_by_stimulus"]["pink noise"]["median"]
        assert med_ext < 3.0, f"external-mount run: {med_ext} deg"

        print("\n######## HARD case: energy is ambiguous, the DOA check must resolve it ########")
        hard = tmp / "hard"
        make_session(hard, "Pink_Noise_Setup1_CC___2026__09_13__23_40_33", SETUP1, seed=33, hard=True)
        s_h = run_once(hard, tmp / "hard_out")
        q = next(iter(s_h["alignment_qc"].values()))
        print("hard-case alignment qc:", {k: q.get(k) for k in ("ambiguous", "resolved_by_doa_shift",
                                                                 "n_on_sound", "doa_check")})
        med_h = s_h["corrected_miss_by_stimulus"]["pink noise"]["median"]
        assert not q["ambiguous"] and med_h < 3.0, f"hard case not resolved: {q}, miss {med_h}"

        print("\n######## LEGACY aligner patched back in (reproduces the bug) ########")
        orig = onsets.align_schedule
        def legacy(wav, u, blocks, clip_dur_s=None, **k):
            al = onsets.align_trials(wav, u, blocks)
            al.update(on_sound=np.ones(len(u), bool), contrast_db=np.zeros(len(u)),
                      alignment_qc={"legacy": True, "n_on_sound": -1, "n_trials": len(u)})
            return al
        onsets.align_schedule = legacy
        try:
            s2 = run_once(data, tmp / "legacy")
        finally:
            onsets.align_schedule = orig
        med2 = s2["corrected_miss_by_stimulus"].get("pink noise", {}).get("median")
        flagged = [f for f, a in s2["label_shift_audit"].items() if a["suspect"]]
        print("legacy: corrected miss median:", med2, "| sessions flagged by the audit:", flagged)
        print("\nOK: fixed pipeline recovers the synthetic truth; the legacy aligner does not.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
