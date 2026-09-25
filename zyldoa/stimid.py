"""Identify each block's stimulus from the audio itself (no reliance on Unity labels).

For every block we average the spectrum of a few trials' steady portion and classify it as a
pure tone (narrow dominant peak) or broadband noise (flat / 1-over-f). The dominant frequency
labels the tone. These spectra are also what the spectrogram figures visualize.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .wavio import ZyliaWav

ANALYSIS_WIN = (0.15, 0.55)  # seconds after onset: steady portion of the ~1 s stimulus


def trial_window(wav: ZyliaWav, onset_sample: int, win=ANALYSIS_WIN) -> np.ndarray:
    start = onset_sample + int(win[0] * wav.sample_rate)
    n = int((win[1] - win[0]) * wav.sample_rate)
    return wav.read(start, n)


def _psd(x: np.ndarray, sr: int, nperseg: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    mono = x.mean(axis=1)
    f, p = welch(mono, fs=sr, nperseg=min(nperseg, len(mono)))
    return f, p


def block_spectrum(wav: ZyliaWav, onset_samples: np.ndarray, sr: int) -> tuple:
    """Average PSD over the given trials of one block.

    Windows clipped by the end of the recording produce shorter Welch grids; only the
    dominant (full-length) grid is averaged so a truncated straggler cannot poison the
    stack (callers should already exclude truncated trials)."""
    out = []
    for s in onset_samples:
        w = trial_window(wav, int(s))
        if len(w) >= 64:
            out.append(_psd(w, sr))
    if not out:
        raise ValueError("no analyzable trial windows for block spectrum")
    n = max(len(p) for _, p in out)
    keep = [(f, p) for f, p in out if len(p) == n]
    return keep[0][0], np.mean([p for _, p in keep], axis=0)


def classify_spectrum(f: np.ndarray, psd: np.ndarray, fmin: float = 80.0,
                      tonal_consistency: float | None = None) -> dict:
    """Classify one stimulus into 'tone', 'complex_tone', or 'broadband_noise'.

    A tone must be peaky in frequency (high peak_fraction, low flatness) AND hold that frequency
    across time (high tonal_consistency) -- both signals must agree, so a transient spectral peak
    or a steady but spread spectrum is not mistaken for a tone. A tone with strong harmonics at
    integer multiples is 'complex_tone'. Everything else is 'broadband_noise'. peak_fraction,
    spectral_flatness and spectral_slope are returned as metadata only (slope is unreliable here
    because the loudspeaker + sphere + room roll-off dominate it, so pink/white are not split)."""
    band = f >= fmin
    fb, pb = f[band], psd[band]
    peak_i = int(np.argmax(pb))
    peak_f = float(fb[peak_i])
    near = np.abs(fb - peak_f) <= peak_f * 0.12  # +/- ~1/6 octave
    peak_frac = float(pb[near].sum() / pb.sum())
    flatness = float(np.exp(np.mean(np.log(pb + 1e-20))) / (pb.mean() + 1e-20))
    slope = float(np.polyfit(np.log(fb + 1e-9), np.log(pb + 1e-20), 1)[0])
    info = {"peak_fraction": round(peak_frac, 3), "spectral_flatness": round(flatness, 4),
            "spectral_slope": round(slope, 2), "dominant_hz": None,
            "tonal_consistency": round(tonal_consistency, 3) if tonal_consistency is not None else None}
    # A tone must be peaky in frequency AND steady in time (both signals must agree).
    is_tone = (peak_frac > 0.4 and flatness < 0.05
               and (tonal_consistency is None or tonal_consistency > 0.8))
    if is_tone:
        n_harm = sum(1 for k in (2, 3, 4)
                     if np.any(pb[np.abs(fb - k * peak_f) <= peak_f * 0.1] > 0.1 * pb[near].max()))
        info.update(type="complex_tone" if n_harm >= 2 else "tone",
                    dominant_hz=round(peak_f, 1), n_harmonics=int(n_harm))
    else:
        # Sub-typing noise by spectral slope is unreliable here: the loudspeaker + sphere + room
        # roll-off dominates the measured slope, not the source's intrinsic colour. Report one
        # honest broadband class (the slope is kept as metadata).
        info["type"] = "broadband_noise"
    return info


TONE_TYPES = ("tone", "complex_tone")


def label_text(info: dict) -> str:
    if info["type"] in TONE_TYPES:
        kind = "complex tone" if info["type"] == "complex_tone" else "tone"
        return f"{int(round(info['dominant_hz']))} Hz {kind}"
    return "broadband noise"


def in_band_snr(wav: ZyliaWav, onset_sample: int, info: dict,
                post=(0.15, 0.55), pre=(-0.45, -0.05), capsule_mask=None) -> float:
    """In-band post/pre power ratio (dB). In-band = tone +/-1/6 oct, or 0.2-5 kHz for noise."""
    sr = wav.sample_rate
    import numpy as _np
    _sel = (lambda seg: seg[:, _np.asarray(capsule_mask)]) if capsule_mask is not None else (lambda seg: seg)
    def band_power(seg):
        if len(seg) < 64:
            return 1e-12
        f, p = _psd(seg, sr, nperseg=min(8192, len(seg)))
        if info["type"] in TONE_TYPES and info.get("dominant_hz"):
            fc = info["dominant_hz"]
            m = np.abs(f - fc) <= fc * 0.12
        else:
            m = (f >= 200) & (f <= 5000)
        return float(p[m].mean())
    post_seg = _sel(wav.read(onset_sample + int(post[0] * sr), int((post[1] - post[0]) * sr)))
    pre_seg = _sel(wav.read(onset_sample + int(pre[0] * sr), int((pre[1] - pre[0]) * sr)))
    return 10 * np.log10((band_power(post_seg) + 1e-12) / (band_power(pre_seg) + 1e-12))


def tonal_consistency(wav: ZyliaWav, onset_samples: np.ndarray, win=ANALYSIS_WIN,
                      fmin: float = 120.0) -> float:
    """Fraction of spectrogram frames (within the safe analysis window) whose dominant frequency
    sits within 5 % of the median — ~1 for a steady tone, low for speech/noise."""
    from scipy.signal import spectrogram
    sr = wav.sample_rate; consist = []
    for s in onset_samples:
        seg = wav.read(int(s) + int(win[0] * sr), int((win[1] - win[0]) * sr)).mean(axis=1)
        if len(seg) < 1024:      # truncated window (recording ended): skip
            continue
        f, _, S = spectrogram(seg, fs=sr, nperseg=1024, noverlap=768)
        m = f >= fmin
        domf = f[m][np.argmax(S[m], axis=0)]
        med = np.median(domf)
        consist.append(float(np.mean(np.abs(domf - med) <= 0.05 * med)))
    return float(np.median(consist)) if consist else 0.0


def identify_blocks(wav: ZyliaWav, onset_samples: np.ndarray, blocks: np.ndarray,
                    trials_per_block: int = 8) -> dict:
    """Per-block stimulus identity + averaged spectrum (for plotting/classification)."""
    out = {}
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        sub = idx[np.linspace(0, len(idx) - 1, min(trials_per_block, len(idx))).astype(int)]
        f, psd = block_spectrum(wav, onset_samples[sub], wav.sample_rate)
        tc = tonal_consistency(wav, onset_samples[sub])
        info = classify_spectrum(f, psd, tonal_consistency=tc)
        info["block"] = int(b)
        info["n_trials"] = int(len(idx))
        out[int(b)] = {"info": info, "freqs": f, "psd": psd}
    return out
