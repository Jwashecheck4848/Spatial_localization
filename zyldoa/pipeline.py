"""Per-folder pipeline stages: (1) separate trials + classify stimuli, (2) estimate DOA.

One folder = one recording session; the current clip-log format always sweeps every speaker
position for one stimulus family in a single recording (`unity.parse_clip_log`). Stages are
split so the stimulus classification can be reconciled across all positions (same speaker,
same sounds) before DOA picks an analysis band. Results are cached (NPZ + CSV + JSON); pass
force=True to recompute. WAVs are streamed.
"""
from __future__ import annotations

import csv
import itertools
import json
import re
from pathlib import Path

import numpy as np

from .wavio import ZyliaWav, open_recording
from . import unity, onsets, stimid, doa, geometry

ANALYSIS_WIN = (0.10, 0.40)  # seconds after onset: inside the gated stimuli
CACHE_SCHEMA = 4  # v4: global schedule alignment (onsets.align_schedule) -- v1-v3 caches carry
                  # whole-trial-shifted onsets for clip-log sessions and MUST NOT be reused

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


# Analysis window for clip-log trials, from each trial's OWN logged duration (ClipEnd -
# ClipStart): skip the first 20 ms (onset transient) and stop 20 ms before the clip ends, so
# neither the DOA nor the SNR gate ever integrates post-stimulus silence/reverberation. Every
# new-format stimulus is short (200 ms bursts incl. PINK NOISE; phonemes 0.32-0.53 s), so the
# legacy fixed 0.10-0.40 s window (tuned for ~1 s July clips) must not be used for them.
CLIP_WIN_LEAD_S, CLIP_WIN_TAIL_S, CLIP_WIN_MIN_S = 0.02, 0.02, 0.05


def clip_window(dur_s: float) -> tuple[float, float]:
    return (CLIP_WIN_LEAD_S, max(CLIP_WIN_LEAD_S + CLIP_WIN_MIN_S, float(dur_s) - CLIP_WIN_TAIL_S))


def complete_trials_mask(onset_samples, sample_rate: float, n_frames: int) -> np.ndarray:
    o = np.asarray(onset_samples, float)
    return (o - TRIAL_PRE_S * sample_rate >= 0) & (o + TRIAL_POST_S * sample_rate <= n_frames)


def folder_tag(folder: Path) -> str:
    """Collision-free slug of the FULL folder name (keeps the session timestamp)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(folder).name).strip("_")


def _family_band(family: str) -> tuple[float, float]:
    m = re.match(r"(\d+) hz tone", family or "")
    if m:
        fc = float(m.group(1))
        return (fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6))
    return doa.BROADBAND_HZ


OUTLIER_DEG = 15.0   # a trial this far from its position's medoid DOA is "in the wrong group"


def _doa_shift_check(wav, blocks: np.ndarray, band, win, max_shift: int = 3):
    """Build the DOA-consistency callback for `onsets.align_schedule`.

    For every trial that landed on sound, estimate a quick DOA (same intensity estimator and
    band as the main analysis) at the aligned onset, then score each candidate relabelling
    k = -max_shift..max_shift by the median within-position angular scatter when the audio at
    trial i is attributed to trial i+k. Positions differ by >= 30 deg, so the true labelling
    has (almost) no trial far from its position's medoid DOA, while k = +-1 puts ~1 trial per
    position into the wrong group. Decisive when every non-equivalent k has at least
    max(3, 3% of trials) more such outliers than the chosen labelling."""
    sr = wav.sample_rate

    # shifts must stay below half a position run, or k and k -/+ run_length alias
    runs = [len(list(g)) for _, g in itertools.groupby(np.asarray(blocks).tolist())]
    run_len = int(np.median(runs)) if runs else 10
    max_shift = int(max(1, min(max_shift, (run_len - 1) // 2)))

    def check(onset_s, on_sound):
        n = len(onset_s)
        U = np.full((n, 3), np.nan)
        for i in np.flatnonzero(on_sound):
            seg = wav.read(int(round((onset_s[i] + win[0]) * sr)), int((win[1] - win[0]) * sr))
            if len(seg) < 256:
                continue
            I = doa.intensity_vector(seg, sr, band)
            nrm = np.linalg.norm(I)
            if nrm > 0:
                U[i] = I / nrm
        has = np.isfinite(U[:, 0])
        n_has = int(has.sum())
        sc, outl, equiv = {}, {}, {}
        for k in range(-max_shift, max_shift + 1):
            groups: dict = {}
            same = []
            for i in range(n):
                j = i + k
                if 0 <= j < n and has[i]:
                    groups.setdefault(int(blocks[j]), []).append(U[i])
                    same.append(blocks[j] == blocks[i])
            # A one-trial shift puts ~1 trial per position into the WRONG position group, i.e. a
            # >= 30 deg outlier. Count trials > OUTLIER_DEG from their group's medoid (robust
            # centre) -- the true labelling has (almost) none; also keep the mean deviation.
            dev, n_out = [], 0
            for g in groups.values():
                if len(g) < 3:
                    continue
                g = np.array(g)
                A = np.degrees(np.arccos(np.clip(g @ g.T, -1, 1)))
                d = A[int(np.argmin(np.median(A, axis=1)))]
                dev += list(d); n_out += int((d > OUTLIER_DEG).sum())
            sc[k] = float(np.mean(dev)) if dev else None
            outl[k] = n_out if dev else None
            # labelling k assigns every on-sound trial to the SAME position as k=0 (happens when
            # e.g. the first clip of every position is silent): harmless, not a real ambiguity
            equiv[k] = bool(same) and float(np.mean(same)) >= 0.99
        valid = [k for k in sc if sc[k] is not None]
        if 0 not in valid:
            return {"best_shift": None, "decisive": False, "scatter_deg": {}, "outliers": {}}
        best = min(valid, key=lambda k: (outl[k], sc[k]))
        if equiv.get(best) and best != 0:
            best = 0                                   # equivalent to the energy solution
        others = [k for k in valid if k != best and not (equiv.get(k) and equiv.get(best))]
        need = outl[best] + max(3, int(0.03 * n_has))
        decisive = (not others) or all(outl[k] >= need for k in others)
        return {"best_shift": int(best), "decisive": bool(decisive), "n_trials_used": n_has,
                "outliers": {str(k): outl[k] for k in valid},
                "scatter_deg": {str(k): round(sc[k], 2) for k in valid},
                "equivalent_shifts": [int(k) for k, e in equiv.items() if e and k != 0]}
    return check


def align_and_classify(folder: Path, results_dir: Path, force: bool = False,
                       user_shift: int | None = None) -> dict:
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
        if meta.get("schema") == CACHE_SCHEMA and meta.get("user_shift") == user_shift:
            meta["npz_path"] = str(npz_path)
            return meta

    log = unity.parse_clip_log(folder)
    blocks = log["block"]
    stim_index = log["stim_index"]
    truth_az, truth_el, truth_r = log["truth_az"], log["truth_el"], log["truth_r"]

    wav = open_recording(folder)
    # Clip-log sessions run on a strictly periodic schedule with no inter-block pauses, which
    # the legacy per-block aligner (`onsets.align_trials`) cannot place: it anchored whole
    # sessions 1-11+ trials late. Use the global schedule aligner instead.
    st = log["source_type"].lower(); nm = folder.name.lower()
    if ("vbap" in nm and "vbp" not in st) or ("mono" in nm and "mono" not in st):
        print(f"  [warn] {folder.name}: folder name and clip-log SourceType "
              f"({log['source_type']}) disagree -- check which render this session really is")
    dur_ok = log["clip_dur_s"][np.isfinite(log["clip_dur_s"])]
    min_dur = float(np.min(dur_ok)) if len(dur_ok) else 0.2
    check = _doa_shift_check(wav, blocks, _family_band(log["family"]), clip_window(min_dur))
    al = onsets.align_schedule(wav, log["onset_unity_s"], blocks, clip_dur_s=log["clip_dur_s"],
                               doa_check=check, user_shift=user_shift)
    osamp = al["onset_samples"]
    complete = complete_trials_mask(osamp, wav.sample_rate, wav.n_frames)
    if not complete.all():
        print(f"  [warn] {folder.name}: {int((~complete).sum())} trial(s) truncated "
              f"(recording ends at {wav.duration_s:.1f} s, last onset at "
              f"{osamp.max() / wav.sample_rate:.1f} s)")
    if not complete.any():
        raise ValueError(f"{folder.name}: no complete trials inside the recording")
    # QC spectra from INSIDE the clip (the legacy 0.15-0.55 s window sits after a 200 ms clip)
    spec_win = clip_window(min_dur)
    ident = stimid.identify_blocks(wav, osamp[complete], stim_index[complete], win=spec_win)
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
        "source_type": log["source_type"],
        "alignment_qc": al["alignment_qc"],
        "user_shift": user_shift,
        "npz_path": str(npz_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    np.savez_compressed(
        npz_path, env=al["env"], env_rate=al["env_rate"], onset_samples=osamp,
        onset_s=al["onset_s"], bursts_s=al["bursts_s"], blocks=blocks, trial=log["trial"],
        stim_index=stim_index, truth_az=truth_az, truth_el=truth_el, truth_r=truth_r,
        clip_dur_s=log["clip_dur_s"], on_sound=al["on_sound"], contrast_db=al["contrast_db"],
        speaker_channel=log["speaker_channel"],
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
        ok = (rows and "stim_index" in rows[0]
              and rows[0].get("capsules_excluded", "") == excl_str
              and str(rows[0].get("cache_schema", "")) == str(CACHE_SCHEMA))
        if ok:
            # the DOAs must have been computed at THESE onsets (stage 1 may have been re-run,
            # e.g. after an alignment override)
            on_npz = np.load(results_dir / f"{tag}_align.npz")["onset_s"]
            on_csv = np.array([float(r["onset_s"]) for r in rows])
            ok = len(on_csv) == len(on_npz) and np.allclose(on_csv, np.round(on_npz, 4), atol=2e-4)
        if ok:
            return rows
        # stale cache (older schema = possibly shifted alignment) or exclusion set changed

    npz = np.load(results_dir / f"{tag}_align.npz")
    osamp = npz["onset_samples"]; blocks = npz["blocks"]; onset_s = npz["onset_s"]; trial = npz["trial"]
    stim_index = npz["stim_index"] if "stim_index" in npz.files else blocks
    t_az = npz["truth_az"] if "truth_az" in npz.files else np.full(len(blocks), np.nan)
    t_el = npz["truth_el"] if "truth_el" in npz.files else np.full(len(blocks), np.nan)
    t_r = npz["truth_r"] if "truth_r" in npz.files else np.full(len(blocks), np.nan)
    clip_dur = npz["clip_dur_s"] if "clip_dur_s" in npz.files else None
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
            dur = (float(clip_dur[i]) if clip_dur is not None and np.isfinite(clip_dur[i])
                   else None) or info.get("clip_duration_s")
            if dur:
                # Short-burst stimulus (declared clip_duration_s): the legacy fixed/adaptive
                # windows below assume a ~1 s clip and would run past this one's end into
                # post-stimulus silence/reverberation, corrupting both the SNR gate and the DOA
                # estimate itself. Use (almost) the whole burst instead: skip a brief onset ramp,
                # stop briefly before the clip ends.
                win = clip_window(dur)
                snr = stimid.in_band_snr(wav, int(osamp[i]), info, post=win, capsule_mask=mask)
            else:
                found = (_adaptive_window(wav, int(osamp[i]), info, mask)
                         if info.get("label") in ADAPTIVE_WIN_LABELS else None)
                if found is not None:
                    win, snr = found                # speech/phone: early voiced-segment window
                else:
                    win = ANALYSIS_WIN               # tones, pink noise, applause: unchanged
                    snr = stimid.in_band_snr(wav, int(osamp[i]), info, capsule_mask=mask)
            az, el, mag = _trial_doa(wav, int(osamp[i]), band, win=win, mask=mask)
        rows.append({
            "trial": int(trial[i]), "block": int(blocks[i]), "stim_index": s,
            "capsules_excluded": excl_str, "cache_schema": CACHE_SCHEMA,
            "win_start_s": round(float(win[0]), 3) if complete[i] else "",
            "win_end_s": round(float(win[1]), 3) if complete[i] else "",
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
