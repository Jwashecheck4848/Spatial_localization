"""Spectral-complexity predictor of the room localization penalty.

Documented, reproducible replacement for the paper's earlier (unscripted, and
male-speech-exception-dependent) rho = -0.83 claim. With the recovered speech data the
relationship is cleaner: a stimulus's effective spectral bandwidth, measured from its own
anechoic spectrum, orders the CAVE-vs-anechoic room penalty monotonically.

HONEST STATUS (after adversarial verification, 2026-07-19): this is a DESCRIPTIVE,
hypothesis-generating trend, NOT a validated independent predictor. Three caveats the
compute() output makes explicit so nobody over-reads it:
  - BAND-DEPENDENT: the negative correlation only holds when the metric band includes low
    frequencies overlapping the DOA analysis range (rho -0.96 at 200-5000 Hz, but +0.25 at
    1500-8000 Hz and ~0 at 2000-12000 Hz). Physically sensible (room modes are a low-frequency
    phenomenon, so low-frequency spectral spread averages over more modes), but it means the
    metric is NOT independent of the estimator's operating band; do not claim it is.
  - METRIC-SPECIFIC: the resolution-invariant analog (spectral flatness / Wiener entropy)
    reaches only -0.82 (all 7) and -0.60 on the five naturalistic stimuli, so the perfect
    ordering is specific to the resolution-dependent N_eff, not to spectral shape in general.
  - SMALL N: n=7 (5 without tones). Report the all-7 rho and its p; the -1.0 on the 5
    broadband stimuli is descriptive (a single adjacent-rank swap moves rho ~0.036) -- do NOT
    attach an inferential p-value to it. The one attack that FAILED: in-band level does not
    explain the penalty (+0.43 all 7, -0.10 no-tones), so the effect is spectral shape.

Method (all reproducible from committed data):
  1. For each stimulus, average the measured anechoic power spectrum across all
     gt_anechoic sessions (per-block Welch PSD cached in align.npz, block_ids == stimulus
     playlist index for single-position ground-truth sessions).
  2. Restrict to BAND (default 200-5000 Hz). N_eff = exp(H), H = -sum q ln q of the
     normalized power spectrum q (effective number of independent spectral components).
  3. Spearman correlation of log N_eff against the room penalty. Reported under BOTH penalty
     definitions (they differ): headline pooled delta and matched-cell per_cell_delta_median.
     6 kHz tone excluded a priori (above the estimator's first-order band).
  4. Diagnostics reported alongside: band-sensitivity, spectral-flatness comparison, and the
     level-confound check, so the fragility is visible in the artifact itself.

Run:  python3 spectral_predictor.py  ->  results/spectral_predictor.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

EXP_DIR = Path(__file__).resolve().parent
R = EXP_DIR / "results"
BAND_HZ = (200.0, 5000.0)
# stimulus playlist index -> label (6 kHz tone, index 3, excluded a priori)
STIM = {0: "pink noise", 1: "250 Hz tone", 2: "1000 Hz tone", 4: "phone ringing",
        5: "applause", 6: "female speech", 7: "male speech"}
PURE_TONES = {1, 2}


def _anechoic_spectra() -> tuple[dict, np.ndarray]:
    """Mean measured anechoic PSD per stimulus, from the cached per-block Welch spectra."""
    acc = {k: [] for k in STIM}
    freqs = None
    for f in sorted(glob.glob(str(R / "gt_anechoic" / "*_align.npz"))):
        d = np.load(f, allow_pickle=True)
        freqs = d["block_freqs"]
        ids = list(d["block_ids"])
        for k in STIM:
            if k in ids:
                acc[k].append(d["block_psd"][ids.index(k)])
    return {k: np.mean(v, axis=0) for k, v in acc.items() if v}, freqs


def _log_neff(psd: np.ndarray, freqs: np.ndarray, band=BAND_HZ) -> float:
    m = (freqs >= band[0]) & (freqs <= band[1])
    p = np.clip(psd[m].astype(float), 1e-20, None)
    q = p / p.sum()
    return float(np.log(np.exp(-np.sum(q * np.log(q)))))  # = H (nats); log N_eff


def _flatness(psd, freqs, band):
    m = (freqs >= band[0]) & (freqs <= band[1])
    p = np.clip(psd[m].astype(float), 1e-20, None)
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def compute(band=BAND_HZ) -> dict:
    spectra, freqs = _anechoic_spectra()
    room = json.loads((R / "comparisons" / "room_effect" / "comparison.json").read_text())["per_stimulus"]
    ks = [k for k in STIM if k in spectra and STIM[k] in room]
    x = {k: _log_neff(spectra[k], freqs, band) for k in ks}
    y = {k: room[STIM[k]]["delta"]["delta"] for k in ks}                 # headline pooled delta
    y_mc = {k: room[STIM[k]]["per_cell_delta_median"] for k in ks}       # matched-cell delta
    lvl = {k: float(np.log(np.mean(spectra[k][(freqs >= band[0]) & (freqs <= band[1])]))) for k in ks}
    nt = [k for k in ks if k not in PURE_TONES]
    rho = lambda a, sub: round(float(spearmanr([x[k] for k in sub], [a[k] for k in sub]).statistic), 3)
    sp = spearmanr([x[k] for k in ks], [y[k] for k in ks])
    loo = {STIM[d]: round(float(spearmanr([x[k] for k in ks if k != d],
                                           [y[k] for k in ks if k != d]).statistic), 3) for d in ks}
    band_sens = {f"{lo}-{hi}": round(float(spearmanr(
                    [_log_neff(spectra[k], freqs, (lo, hi)) for k in ks], [y[k] for k in ks]).statistic), 3)
                 for lo, hi in [(200, 5000), (400, 1200), (1500, 8000), (2000, 12000)]}
    return {
        "_status": "DESCRIPTIVE TREND, not a validated independent predictor; see module docstring",
        "band_hz": list(band),
        "n_stimuli": len(ks),
        "spearman_rho_headline_delta": round(float(sp.statistic), 3),
        "spearman_p_all7": round(float(sp.pvalue), 4),
        "spearman_rho_matched_cell_delta": rho(y_mc, ks),
        "spearman_rho_no_pure_tones": rho(y, nt),   # descriptive; do not attach an inferential p
        "pearson_logNeff_headline": round(float(pearsonr([x[k] for k in ks], [y[k] for k in ks]).statistic), 3),
        "leave_one_out_rho": loo,
        "diagnostics": {
            "band_sensitivity_rho": band_sens,
            "flatness_rho_all7": round(float(spearmanr([_flatness(spectra[k], freqs, band) for k in ks],
                                                       [y[k] for k in ks]).statistic), 3),
            "flatness_rho_no_tones": round(float(spearmanr([_flatness(spectra[k], freqs, band) for k in nt],
                                                           [y[k] for k in nt]).statistic), 3),
            "level_confound_rho_all7": round(float(spearmanr([lvl[k] for k in ks], [y[k] for k in ks]).statistic), 3),
            "level_confound_rho_no_tones": round(float(spearmanr([lvl[k] for k in nt], [y[k] for k in nt]).statistic), 3),
        },
        "per_stimulus": {STIM[k]: {"log_Neff": round(x[k], 3), "room_penalty_deg": round(y[k], 2)}
                         for k in ks},
    }


def main() -> None:
    out = compute()
    (R / "spectral_predictor.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {R / 'spectral_predictor.json'}: "
          f"Spearman rho={out['spearman_rho_headline_delta']} (p={out['spearman_p_all7']}), "
          f"no-tones rho={out['spearman_rho_no_pure_tones']}, n={out['n_stimuli']} "
          f"[DESCRIPTIVE TREND; band-dependent {out['diagnostics']['band_sensitivity_rho']}]")


if __name__ == "__main__":
    main()
