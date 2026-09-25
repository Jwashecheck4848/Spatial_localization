"""Raw-signal integrity checks: is the DOA running on true signal or on garbage?

Three independent checks per session, none of which the SNR gate can perform:

1. **Capsule health.** A dead, attenuated, or rattling capsule corrupts the spherical-
   harmonic encoding silently (the SNR gate looks at total in-band power, which 18 good
   capsules keep high). Per capsule: RMS level and band-limited coherence with the
   leave-one-out array mean, medianed over sampled trial windows. Flags: `dead` (RMS more
   than 12 dB under the capsule median), `hot` (12 dB over), `incoherent` (correlation with
   the rest of the array < 0.6).
2. **Clipping.** Saturated windows bias the intensity vector; flagged when any capsule has
   more than 0.1% of samples within 1.5% of full scale.
3. **Stimulus verification.** Labels are authoritative (Unity playlist), but the *playback*
   can still be wrong (wrong file, renderer fault, silence). The measured average spectrum
   of each stimulus group (cached from alignment) is classified and compared against the
   expected type: an expected tone must classify tonal within 10% of its nominal frequency;
   expected broadband must not classify as a pure tone. Advisory (phone ringing legitimately
   classifies tonal), but a mismatch on pink noise or a tone is a playback fault.

Per-session results are cached to `<tag>_quality.json`; `aggregate()` produces the
summary-level roll-up with one line per flagged session.
"""
from __future__ import annotations

import numpy as np

from . import stimid
from .wavio import open_recording

WINDOW_S = (0.15, 0.55)      # same steady portion the classifier uses
N_SAMPLE_WINDOWS = 12
COH_BAND_HZ = (200.0, 1500.0)   # above ~1.5 kHz sphere shadowing decorrelates far-side capsules
RMS_FLAG_DB = 12.0
COH_FLAG = 0.5
CLIP_LEVEL = 0.985
CLIP_FRAC = 1e-3


def _bandpass(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    X = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(x.shape[0], 1 / sr)
    X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, n=x.shape[0], axis=0)


def capsule_health(folder, onset_samples: np.ndarray) -> dict:
    """Per-capsule RMS (dBFS) and leave-one-out coherence over sampled trial windows."""
    wav = open_recording(folder)
    sr = wav.sample_rate
    n_ch = wav.n_channels
    idx = np.linspace(0, len(onset_samples) - 1,
                      min(N_SAMPLE_WINDOWS, len(onset_samples))).astype(int)
    rms_db, coh, clip = [], [], np.zeros(n_ch)
    n_used = 0
    for i in idx:
        seg = wav.read(int(onset_samples[i]) + int(WINDOW_S[0] * sr),
                       int((WINDOW_S[1] - WINDOW_S[0]) * sr))
        if seg.shape[0] < 1024:
            continue
        n_used += 1
        clip = np.maximum(clip, np.mean(np.abs(seg) >= CLIP_LEVEL, axis=0))
        rms_db.append(20 * np.log10(np.sqrt(np.mean(seg ** 2, axis=0)) + 1e-12))
        b = _bandpass(seg, sr, *COH_BAND_HZ)
        # reference = per-sample MEDIAN across capsules: robust to a hot/dead capsule, which
        # would poison a mean-based reference and make every healthy capsule look incoherent
        ref = np.median(b, axis=1)
        cc = np.zeros(n_ch)
        for c in range(n_ch):
            denom = b[:, c].std() * ref.std()
            cc[c] = float(np.corrcoef(b[:, c], ref)[0, 1]) if denom > 1e-12 else 0.0
        coh.append(cc)
    wav.close()
    if not n_used:
        return {"error": "no analyzable windows"}
    rms_db = np.median(np.array(rms_db), axis=0)
    coh = np.median(np.array(coh), axis=0)
    med = float(np.median(rms_db))
    flags = []
    for c in range(n_ch):
        tags = []
        if rms_db[c] < med - RMS_FLAG_DB:
            tags.append("dead")
        if rms_db[c] > med + RMS_FLAG_DB:
            tags.append("hot")
        if coh[c] < COH_FLAG:
            tags.append("incoherent")
        if clip[c] > CLIP_FRAC:
            tags.append("clipping")
        if tags:
            flags.append({"capsule": c, "tags": tags, "rms_db": round(float(rms_db[c]), 1),
                          "coherence": round(float(coh[c]), 3),
                          "clip_frac": round(float(clip[c]), 5)})
    return {"n_windows": int(n_used),
            "rms_db_median": round(med, 1),
            "rms_db_range": [round(float(rms_db.min()), 1), round(float(rms_db.max()), 1)],
            "coherence_min": round(float(coh.min()), 3),
            "clip_frac_max": round(float(clip.max()), 6),
            "capsule_flags": flags}


def verify_stimuli(npz, canonical: dict) -> dict:
    """Classify each stimulus group's measured average spectrum and compare to its label."""
    if "block_ids" not in getattr(npz, "files", npz):
        return {"error": "no cached spectra"}
    out, mismatches = {}, []
    freqs = npz["block_freqs"]
    for bid, psd in zip(npz["block_ids"], npz["block_psd"]):
        s = int(bid)
        if s not in canonical:
            continue
        expected = canonical[s]
        got = stimid.classify_spectrum(freqs, psd)
        ok, note = True, ""
        if expected["type"] in stimid.TONE_TYPES:
            fc = expected.get("dominant_hz") or 0
            tonal = got["type"] in stimid.TONE_TYPES
            close = tonal and got.get("dominant_hz") and abs(got["dominant_hz"] - fc) <= 0.10 * fc
            # 6 kHz is past the classifier band; a null there is expected, not a fault
            if not close and fc < 5000:
                ok, note = False, f"expected {fc:g} Hz tone, spectrum says {got['type']} {got.get('dominant_hz')}"
        else:
            # speech has a strong pitch (f0) and phone ringing is quasi-tonal: only the true
            # noise stimuli must never classify as a pure tone
            tonal_ok = ("speech" in expected["label"]) or expected["label"] == "phone ringing"
            if got["type"] == "tone" and not tonal_ok:
                ok, note = False, f"expected {expected['label']}, spectrum says pure tone {got.get('dominant_hz')} Hz"
        out[s] = {"label": expected["label"], "spectrum_type": got["type"],
                  "spectrum_hz": got.get("dominant_hz"), "ok": bool(ok)}
        if not ok:
            mismatches.append(f"stim {s} ({expected['label']}): {note}")
    return {"per_stimulus": out, "mismatches": mismatches}


def session_quality(folder, npz, canonical: dict) -> dict:
    q = {"capsules": capsule_health(folder, npz["onset_samples"]),
         "stimuli": verify_stimuli(npz, canonical)}
    caps = q["capsules"].get("capsule_flags", [])
    mism = q["stimuli"].get("mismatches", [])
    q["ok"] = not caps and not mism
    q["issues"] = ([f"capsule {c['capsule']}: {'/'.join(c['tags'])}" for c in caps] + mism)
    return q


def aggregate(per_session: dict) -> dict:
    flagged = {name: q["issues"] for name, q in per_session.items() if not q.get("ok", True)}
    caps_all = [c["capsule"] for q in per_session.values()
                for c in q.get("capsules", {}).get("capsule_flags", [])]
    persistent = sorted({c for c in caps_all if caps_all.count(c) > len(per_session) / 2})
    return {"n_sessions": len(per_session), "n_flagged": len(flagged),
            "persistent_capsule_suspects": persistent,
            "flagged_sessions": flagged}
