"""Per-folder pipeline stages: (1) separate trials + classify stimuli, (2) estimate DOA.

One folder = one recording session; the current clip-log format always sweeps every speaker
position for one stimulus family in a single recording (`unity.parse_clip_log`). Stages are
split so the stimulus classification can be reconciled across all positions (same speaker,
same sounds) before DOA picks an analysis band. Results are cached (NPZ + CSV + JSON); pass
force=True to recompute. WAVs are streamed.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .wavio import ZyliaWav, open_recording
from . import unity, onsets, stimid, doa, geometry

ANALYSIS_WIN = (0.10, 0.40)  # seconds after onset: inside the gated stimuli
CACHE_SCHEMA = 3  # v3: single clip-log CSV per session (unity.parse_clip_log); v1/v2 caches are stale

# Speech and the telephone ring have variable / intermittent envelopes, so the fixed early
# window frequently lands on a quiet portion and the trial fails the SNR gate even though the
# stimulus is present. For these stimuli ONLY, search from onset for the EARLIEST 0.40 s window
# that clears the SNR gate (keeping direct sound; the max-SNR window grabs reverberation and
# localizes poorly). Tones and continuous broadband (pink noise, applause) are left untouched,
# so the headline room/tone results are unchanged. Changing this requires recomputing the DOA
# stage (delete the *_trials.csv caches or run with force).
ADAPTIVE_WIN_LABELS = {"male speech", "female speech", "phone ringing", "phonemes"}
ADAPTIVE_WIN_LEN, ADAPTIVE_WIN_MAX, ADAPTIVE_WIN_STEP = 0.40, 1.90, 0.10
ADAPTIVE_SNR_DB = 6.0  # match run_analysis.RELIABLE_SNR_DB

# Widest windows any stage reads around a trial onset: the SNR pre-window reaches 0.45 s
# before it, classification/SNR post-windows 0.55 s after (stimid.py). A trial whose windows
# leave the recording (e.g. the Zylia was stopped before the session finished) cannot be
# analysed and is marked truncated instead of crashing the spectra/DOA stages.
TRIAL_PRE_S, TRIAL_POST_S = 0.45, 0.55


def complete_trials_mask(onset_samples, sample_rate: float, n_frames: int) -> np.ndarray:
    o = np.asarray(onset_samples, float)
    return (o - TRIAL_PRE_S * sample_rate >= 0) & (o + TRIAL_POST_S * sample_rate <= n_frames)


def folder_tag(folder: Path) -> str:
    """Collision-free slug of the FULL folder name (keeps the session timestamp)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(folder).name).strip("_")


def align_and_classify(folder: Path, results_dir: Path, force: bool = False) -> dict:
    """Stage 1: separate trials (per-block clock) and identify each stimulus.

    Trial timing, stimulus identity, and truth position all come from the session's single
    clip-log CSV (`unity.parse_clip_log`): `SpeakerChannelIndex` is used directly as the
    clock-alignment block (trials for one speaker position are recorded contiguously), and
    `stim_index` is constant for the whole session (the file's stimulus family, from its
    filename) -- QC spectra and classification are grouped by stim_index as before."""
    folder = Path(folder)
    tag = folder_tag(folder)
    npz_path = results_dir / f"{tag}_align.npz"
    meta_path = results_dir / f"{tag}_align.json"
    if npz_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("schema") == CACHE_SCHEMA:
            meta["npz_path"] = str(npz_path)
            return meta

    log = unity.parse_clip_log(folder)
    blocks = log["block"]
    stim_index = log["stim_index"]
    truth_az, truth_el, truth_r = log["truth_az"], log["truth_el"], log["truth_r"]

    wav = open_recording(folder)
    al = onsets.align_trials(wav, log["onset_unity_s"], blocks)
    osamp = al["onset_samples"]
    complete = complete_trials_mask(osamp, wav.sample_rate, wav.n_frames)
    if not complete.all():
        print(f"  [warn] {folder.name}: {int((~complete).sum())} trial(s) truncated "
              f"(recording ends at {wav.duration_s:.1f} s, last onset at "
              f"{osamp.max() / wav.sample_rate:.1f} s)")
    if not complete.any():
        raise ValueError(f"{folder.name}: no complete trials inside the recording")
    ident = stimid.identify_blocks(wav, osamp[complete], stim_index[complete])
    wav.close()

    meta = {
        "schema": CACHE_SCHEMA,
        "folder": folder.name, "tag": tag,
        "ground_truth": {"packed": True, "azimuth_deg": None, "azimuth_range_deg": None,
                         "elevation_deg": None, "radius_m": None, "family": log["family"]},
        "n_trials": int(len(blocks)), "n_truncated": int((~complete).sum()),
        "n_bursts_detected": al["n_bursts_detected"],
        "block_offsets": al["block_offsets"], "block_offset_confidence": al["block_offset_confidence"],
        "block_repaired": al["block_repaired"],
        "block_info": {int(b): ident[b]["info"] for b in ident},
        "has_trial_truth": bool(np.isfinite(truth_az).any()),
        "npz_path": str(npz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    np.savez_compressed(
        npz_path, env=al["env"], env_rate=al["env_rate"], onset_samples=osamp,
        onset_s=al["onset_s"], bursts_s=al["bursts_s"], blocks=blocks, trial=log["trial"],
        stim_index=stim_index, truth_az=truth_az, truth_el=truth_el, truth_r=truth_r,
        block_freqs=ident[next(iter(ident))]["freqs"],
        block_psd=np.array([ident[b]["psd"] for b in sorted(ident)]),
        block_ids=np.array(sorted(ident)),
    )
    return meta


def reconcile_classification(metas: list[dict]) -> dict:
    """One canonical stimulus per block across all positions (identical speaker/sounds).

    A block is a tone if a sizable, frequency-consistent *cluster* of positions heard a tone -- not
    a strict majority. Pure-tone detectability varies with position: a side/back speaker or a room
    null can push a clear tone's peak_fraction just under threshold (so only a minority of positions
    flag it), and a few positions can lock onto a harmonic/octave (so the spread of all tone
    frequencies is large even when most agree). Both defeat a "majority + small global std" rule and
    wrongly collapse a real tone to broadband. Instead we take the positions that heard a tone, find
    the frequency they cluster on (robust median, keep those within 10 %), and accept the tone when
    that agreeing cluster is non-trivial (>=25 % of positions, or >=3). A single spurious detection
    (e.g. one room mode) never reaches the cluster size, so noise still stays noise."""
    by_block: dict[int, list[dict]] = {}
    for meta in metas:
        for b, info in meta["block_info"].items():
            by_block.setdefault(int(b), []).append(info)
    canonical = {}
    for b, infos in by_block.items():
        tonal = [i for i in infos if i["type"] in stimid.TONE_TYPES]
        hz = np.array([i["dominant_hz"] for i in tonal], float) if tonal else np.array([])
        cluster = hz[np.abs(hz - np.median(hz)) <= 0.10 * np.median(hz)] if len(hz) else hz
        min_cluster = max(3, len(infos) // 4)
        if len(cluster) >= min_cluster:
            fc = float(np.median(cluster))
            agree = [i for i in tonal if abs(i["dominant_hz"] - fc) <= 0.10 * fc]
            typ = "complex_tone" if sum(i["type"] == "complex_tone" for i in agree) > len(agree) / 2 else "tone"
            canonical[b] = {"type": typ, "dominant_hz": round(fc, 1)}
        else:
            canonical[b] = {"type": "broadband_noise", "dominant_hz": None}
        canonical[b]["label"] = stimid.label_text(canonical[b])
    return canonical


def estimate_doa(folder: Path, canonical_stim: dict, results_dir: Path,
                 force: bool = False, exclude_capsules=()) -> list[dict]:
    """Stage 2: per-trial DOA + in-band SNR, using the stimulus identity per trial.

    Rows carry `stim_index` (the session's stimulus family) alongside `block` (the speaker
    position group used for clock alignment); truth columns are always populated since every
    session is now a full multi-position sweep. `exclude_capsules` (from the signal-quality
    layer: hot/dead sensors) removes those capsules from the SH encoding — the remaining
    18+ capsules still resolve first order."""
    folder = Path(folder)
    tag = folder_tag(folder)
    excl = tuple(sorted(int(c) for c in exclude_capsules))
    excl_str = ",".join(str(c) for c in excl)
    csv_path = results_dir / f"{tag}_trials.csv"
    if csv_path.exists() and not force:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and "stim_index" in rows[0] \
                and rows[0].get("capsules_excluded", "") == excl_str:
            return rows
        # v1 cache (pre stim_index) or exclusion set changed: recompute below

    npz = np.load(results_dir / f"{tag}_align.npz")
    osamp = npz["onset_samples"]; blocks = npz["blocks"]; onset_s = npz["onset_s"]; trial = npz["trial"]
    stim_index = npz["stim_index"] if "stim_index" in npz.files else blocks
    t_az = npz["truth_az"] if "truth_az" in npz.files else np.full(len(blocks), np.nan)
    t_el = npz["truth_el"] if "truth_el" in npz.files else np.full(len(blocks), np.nan)
    t_r = npz["truth_r"] if "truth_r" in npz.files else np.full(len(blocks), np.nan)
    wav = open_recording(folder)
    complete = complete_trials_mask(osamp, wav.sample_rate, wav.n_frames)
    mask = None
    if excl:
        mask = np.ones(wav.n_channels, dtype=bool)
        mask[list(excl)] = False
    rows = []
    for i in range(len(blocks)):
        s = int(stim_index[i]); info = canonical_stim[s]; band = doa.analysis_band(info)
        if not complete[i]:
            # windows leave the recording (stopped early): keep the trial for accounting,
            # unreliable by construction (SNR gate) so it never enters any statistic
            az, el, snr = 0.0, 0.0, -99.0
        else:
            found = (_adaptive_window(wav, int(osamp[i]), info, mask)
                     if info.get("label") in ADAPTIVE_WIN_LABELS else None)
            if found is not None:
                win, snr = found                    # speech/phone: early voiced-segment window
            else:
                win = ANALYSIS_WIN                   # tones, pink noise, applause: unchanged
                snr = stimid.in_band_snr(wav, int(osamp[i]), info, capsule_mask=mask)
            az, el, mag = _trial_doa(wav, int(osamp[i]), band, win=win, mask=mask)
        rows.append({
            "trial": int(trial[i]), "block": int(blocks[i]), "stim_index": s,
            "capsules_excluded": excl_str,
            "stim_type": info["type"],
            "stim_label": info["label"], "stim_hz": info["dominant_hz"] or "",
            "trial_truth_az": round(float(t_az[i]), 2) if np.isfinite(t_az[i]) else "",
            "trial_truth_el": round(float(t_el[i]), 2) if np.isfinite(t_el[i]) else "",
            "trial_truth_r": round(float(t_r[i]), 3) if np.isfinite(t_r[i]) else "",
            "onset_s": round(float(onset_s[i]), 4),
            "doa_az": round(az, 2), "doa_el": round(el, 2),
            "inband_snr_db": round(float(snr), 1),
        })
    wav.close()
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    return rows


def _adaptive_window(wav: ZyliaWav, onset_sample: int, info: dict, mask):
    """Earliest ADAPTIVE_WIN_LEN window (from onset) whose in-band SNR clears the gate, for
    variable-envelope stimuli (speech / telephone ring). Prefers the earliest passing window
    (direct sound) over the loudest one (which is often reverberant). Returns ((t0, t1), snr)
    or None if no window clears -- in which case the caller falls back to the fixed window and
    the trial is excluded exactly as before."""
    sr = wav.sample_rate
    t = 0.05
    while t + ADAPTIVE_WIN_LEN <= ADAPTIVE_WIN_MAX:
        if (onset_sample / sr) + t + ADAPTIVE_WIN_LEN > wav.duration_s - 0.05:
            break
        snr = stimid.in_band_snr(wav, onset_sample, info,
                                 post=(t, t + ADAPTIVE_WIN_LEN), capsule_mask=mask)
        if snr >= ADAPTIVE_SNR_DB:
            return (t, t + ADAPTIVE_WIN_LEN), float(snr)
        t += ADAPTIVE_WIN_STEP
    return None


def _trial_doa(wav: ZyliaWav, onset_sample: int, band, win=ANALYSIS_WIN, mask=None):
    seg = wav.read(onset_sample + int(win[0] * wav.sample_rate),
                   int((win[1] - win[0]) * wav.sample_rate))
    I = doa.intensity_vector(seg, wav.sample_rate, band, capsule_mask=mask)
    az, el = geometry.unit_to_azel(I)
    return az, el, float(np.linalg.norm(I))
