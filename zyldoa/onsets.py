"""Detect stimulus onsets in the continuous recording and align them to the Unity trials.

Strategy: stream a broadband RMS envelope over the whole file, detect loud bursts, then fit
an affine clock map (Unity time -> WAV time) so every one of the 310 Unity trials gets a WAV
onset even if a few bursts are missed. Onsets are refined to the local rising edge.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .wavio import ZyliaWav


def compute_envelope(wav: ZyliaWav, hop: int = 256, ref_channels=None,
                     mode: str = "mean") -> tuple[np.ndarray, float]:
    """Broadband RMS envelope. Returns (env, env_rate_hz).

    mode="mean"  (legacy): RMS of the channel-MEAN signal. Above ~1.5 kHz the 19 capsules sit
                 at very different phases on the 98 mm sphere, so averaging the channels
                 cancels most of the signal (a 5 kHz tone almost vanishes from this envelope).
    mode="power": sqrt(mean over channels of x^2) -- no inter-capsule cancellation; used by the
                 clip-log schedule aligner (`align_schedule`)."""
    chunk = (1 << 20) // hop * hop  # multiple of hop
    env_parts, carry = [], np.zeros(0, dtype=np.float32)
    for _, block in wav.iter_chunks(chunk):
        blk = block if ref_channels is None else block[:, ref_channels]
        if mode == "power":
            # mean square across channels; sqrt after the frame average below gives frame RMS
            mono = np.sqrt((blk.astype(np.float64) ** 2).mean(axis=1))
        else:
            mono = blk.mean(axis=1)
        mono = np.concatenate([carry, mono])
        n = (len(mono) // hop) * hop
        carry = mono[n:]
        frames = mono[:n].reshape(-1, hop)
        env_parts.append(np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)))
    if len(carry):
        env_parts.append(np.array([np.sqrt((carry.astype(np.float64) ** 2).mean())]))
    return np.concatenate(env_parts), wav.sample_rate / hop


def detect_bursts(env: np.ndarray, env_rate: float, min_gap_s: float = 0.6,
                  min_dur_s: float = 0.15) -> np.ndarray:
    """Return onset times (s) of loud bursts using hysteresis on the envelope.

    Diagnostic only — alignment does not depend on getting this count exactly right. A high
    threshold opens a burst, a lower one closes it, so amplitude ripple inside a stimulus does
    not split it into several onsets.
    """
    edb = 20 * np.log10(env + 1e-9)
    floor = np.percentile(edb, 20)
    peak = np.percentile(edb, 99)
    hi = floor + 0.50 * (peak - floor)
    lo = floor + 0.30 * (peak - floor)
    onsets, active, start = [], False, 0
    min_gap = int(min_gap_s * env_rate)
    min_dur = int(min_dur_s * env_rate)
    for i, v in enumerate(edb):
        if not active and v > hi:
            active, start = True, i
            if not onsets or i - onsets[-1] >= min_gap:
                onsets.append(i)
        elif active and v < lo:
            if i - start < min_dur and onsets:
                onsets.pop()  # too short, drop it
            active = False
    return np.array(onsets) / env_rate


def _block_starts(times: np.ndarray, gap_s: float) -> np.ndarray:
    """Times that begin a run, i.e. the first plus any preceded by a gap > gap_s."""
    if len(times) == 0:
        return times
    idx = np.where(np.diff(times) > gap_s)[0] + 1
    return np.concatenate([[times[0]], times[idx]])


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Robust (slope, intercept) = median pairwise slope, median intercept."""
    n = len(x)
    sl = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(n) for j in range(i + 1, n)
          if x[j] != x[i]]
    slope = float(np.median(sl))
    return slope, float(np.median(y - slope * x))


def align_clock(unity_onsets: np.ndarray, bursts: np.ndarray,
                gap_s: float = 4.0, match_tol_s: float = 2.5) -> tuple[float, float, int]:
    """Robust Unity->WAV clock (scale, offset) from inter-block-gap landmarks.

    Block-start times (first stimulus after each long break) form a non-periodic fingerprint,
    so matching them avoids the ~1.1 s sidelobe that defeats a global cross-correlation. A
    Theil-Sen fit on the matched pairs recovers a real clock scale (the Unity frame clock drifts
    ~0.2% from the audio hardware clock). Returns (scale, offset, n_matched)."""
    u_bs = _block_starts(unity_onsets, gap_s)
    b_bs = _block_starts(bursts, gap_s)
    if len(b_bs) < 2:
        return 1.0, (float(bursts[0] - unity_onsets[0]) if len(bursts) else 0.0), 0
    best = (-1, float(b_bs[0] - u_bs[0]))
    for seed in (b_bs - u_bs[0]):   # try aligning trial 0 to every detected block-start
        d = np.abs((u_bs + seed)[:, None] - b_bs[None, :]).min(axis=1)
        n = int((d < match_tol_s).sum())
        if n > best[0]:
            best = (n, float(seed))
    pred = u_bs + best[1]
    j = np.abs(pred[:, None] - b_bs[None, :]).argmin(axis=1)
    matched = np.abs(b_bs[j] - pred) < match_tol_s
    if matched.sum() >= 3:
        scale, offset = _theil_sen(u_bs[matched], b_bs[j][matched])
    else:
        scale, offset = 1.0, best[1]
    return float(scale), float(offset), int(matched.sum())


def _env_at(env: np.ndarray, env_rate: float, t: np.ndarray) -> np.ndarray:
    idx = np.clip(np.round(t * env_rate).astype(int), 0, len(env) - 1)
    return env[idx]


def _block_offset(env, env_rate, u_block, center_off, bursts,
                  span_s: float = 3.0, step_s: float = 0.02) -> tuple[float, float]:
    """Best Unity->WAV offset for one block by maximizing the mean onset-energy step
    (energy just after each onset minus just before), which locks onto the rising edge while
    averaging over all trials in the block. Sharpened on detected bursts when enough are present.
    Returns (offset, confidence); confidence is the energy contrast of the peak (>~1.5 = clear
    lock, ~1 = block too weak to trust)."""
    offs = np.arange(center_off - span_s, center_off + span_s, step_s)
    post = np.array([_env_at(env, env_rate, u_block + o + 0.15).mean() for o in offs])
    pre = np.array([_env_at(env, env_rate, u_block + o - 0.30).mean() for o in offs])
    step = post - pre
    off = float(offs[np.argmax(step)])
    confidence = float(post.max() / (np.median(post) + 1e-12))
    if len(bursts):  # sharpen with bursts that fall within the trial spacing
        j = np.abs((u_block + off)[:, None] - bursts[None, :]).argmin(axis=1)
        r = bursts[j] - u_block
        m = np.abs(r - off) < 0.20
        if m.sum() >= max(5, len(u_block) // 4):
            off = float(np.median(r[m]))
    return off, confidence


def align_trials(wav: ZyliaWav, unity_onsets: np.ndarray, blocks: np.ndarray,
                 env_hop: int = 256, min_confidence: float = 1.5) -> dict:
    """Map every Unity trial to a WAV onset with a per-block offset model.

    The Unity frame clock and the audio hardware clock drift slowly across the session and jump
    by a couple of seconds across the long inter-block breaks, so a single global linear clock
    does not fit. Instead each block gets its own offset (Unity within-block timing is accurate),
    found by maximizing onset energy and sharpened on detected bursts. Blocks too weak to lock
    (e.g. a faint high-frequency tone) inherit an offset interpolated from confident neighbours.
    """
    env, env_rate = compute_envelope(wav, hop=env_hop)
    bursts = detect_bursts(env, env_rate)
    _, center, _ = align_clock(unity_onsets, bursts)

    # Align blocks in order, each searching a narrow window around the previous confident
    # block's offset, since the clock drifts smoothly block-to-block (a wide search on a short
    # 5-trial block can otherwise lock onto a spurious peak).
    uniq = np.unique(blocks)
    raw_off, conf, t_block = {}, {}, {}
    anchor = center
    for i, b in enumerate(uniq):
        ub = unity_onsets[blocks == b]
        raw_off[b], conf[b] = _block_offset(env, env_rate, ub, anchor, bursts,
                                            span_s=(5.0 if i == 0 else 3.0))
        t_block[b] = float(ub.mean())
        if conf[b] >= min_confidence:
            anchor = raw_off[b]

    # repair low-confidence blocks from confident ones (interpolate vs block time; nearest if
    # extrapolating) so a weak block never throws its trials into silence.
    good = [b for b in uniq if conf[b] >= min_confidence]
    off = dict(raw_off)
    if good:
        gt = np.array([t_block[b] for b in good]); go = np.array([raw_off[b] for b in good])
        for b in uniq:
            if conf[b] < min_confidence:
                if gt.min() <= t_block[b] <= gt.max():
                    off[b] = float(np.interp(t_block[b], gt, go))
                else:
                    off[b] = float(go[np.argmin(np.abs(gt - t_block[b]))])

    onset_s = unity_onsets + np.array([off[b] for b in blocks])
    onset_samples = np.round(onset_s * wav.sample_rate).astype(np.int64)
    nn = (np.abs(onset_s[:, None] - bursts[None, :]).min(axis=1)
          if len(bursts) else np.full(len(onset_s), np.nan))
    return {
        "n_bursts_detected": int(len(bursts)),
        "block_offsets": {int(b): off[b] for b in uniq},
        "block_offset_confidence": {int(b): conf[b] for b in uniq},
        "block_repaired": {int(b): conf[b] < min_confidence for b in uniq},
        "nearest_burst_s": nn,
        "onset_samples": onset_samples,
        "onset_s": onset_s,
        "env": env,
        "env_rate": env_rate,
        "bursts_s": bursts,
    }


# ---------------------------------------------------------------------------------------------
# Clip-log (2026-09+) sessions: global schedule alignment.
#
# The new Unity clip log plays every trial on a strictly periodic schedule (one clip every
# 3.2 s for the 200 ms stimuli, ~3.46 s for phonemes) with NO pause between speaker positions.
# `align_trials` above anchors the absolute offset on inter-block silences (> 4 s) and on the
# first detected burst; with no silences it falls back to "first detected burst == trial 0",
# and its burst detector (tuned for ~1 s stimuli) misses many 200 ms bursts. The per-block
# energy search is then ambiguous modulo one inter-trial interval, so whole sessions were
# mapped N trials late (N = 0..11 observed, and one session entirely past the end of the WAV):
# every trial inherited the truth position of a different trial. The aligner below instead
# fits ONE clock map (scale, offset) for the whole schedule by counting how many of the logged
# trials land on real sound, resolves the residual whole-trial ambiguity explicitly, refines
# each onset to its rising edge, and reports how decisive the solution was.
# ---------------------------------------------------------------------------------------------

def _cum_power(env: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.asarray(env, np.float64) ** 2)])


def _win_db(cum: np.ndarray, i0, i1) -> np.ndarray:
    """dB of the mean envelope power over envelope-sample windows [i0, i1)."""
    n = len(cum) - 1
    i0 = np.clip(np.asarray(i0), 0, n)
    i1 = np.clip(np.asarray(i1), 0, n)
    i1 = np.maximum(i1, i0 + 1)
    i1 = np.minimum(i1, n)
    i0 = np.minimum(i0, i1 - 1)
    return 10.0 * np.log10((cum[i1] - cum[i0]) / (i1 - i0) + 1e-20)


def onset_contrast_curve(env: np.ndarray, env_rate: float, post_s=(0.01, 0.15),
                         pre_s=(-0.60, -0.10)) -> np.ndarray:
    """C[j] = dB(mean power just after j) - dB(mean power before j), for every envelope
    sample j: large where a stimulus starts at j. Pre-window ends 0.1 s before the onset so
    the previous trial's reverberant tail (>= 3 s earlier) never matters."""
    cum = _cum_power(env)
    j = np.arange(len(env))
    post = _win_db(cum, j + int(round(post_s[0] * env_rate)), j + int(round(post_s[1] * env_rate)))
    pre = _win_db(cum, j + int(round(pre_s[0] * env_rate)), j + int(round(pre_s[1] * env_rate)))
    # near the start of the file the pre-window is simply shortened (>= 2 envelope samples),
    # so a first trial that begins < 0.6 s into the WAV is still detectable
    valid = (j + pre_s[1] * env_rate >= 2) & (j + post_s[1] * env_rate < len(env))
    return np.where(valid, post - pre, -np.inf)


def _count_on_sound(C: np.ndarray, env_rate: float, t: np.ndarray, thr_db: float):
    idx = np.round(np.asarray(t) * env_rate).astype(np.int64)
    inside = (idx >= 0) & (idx < len(C))
    c = np.full(len(idx), -np.inf)
    c[inside] = C[idx[inside]]
    return c, c > thr_db


DISP_RATIO, DISP_MIN_DB2 = 3.0, 0.01  # level-signature separation needed to call a shift decisive


def align_schedule_env(env: np.ndarray, env_rate: float, unity_onsets: np.ndarray,
                       blocks: np.ndarray | None = None, clip_dur_s: np.ndarray | None = None,
                       max_drift: float = 0.003, thr_db: float = 6.0,
                       max_shift_trials: int = 20, shift_override: int | None = None) -> dict:
    """Align a periodic Unity clip schedule to a recording envelope.

    1. Coarse global search over clock scale (+/- max_drift) and offset (whole recording):
       maximize the number of logged trials whose onset lands on a >= thr_db energy rise
       (ties broken by mean contrast). Trials mapped outside the recording count as misses,
       so a solution shifted by k trials loses k trials at one end.
    2. Explicit whole-trial ambiguity check: re-score offset + k*ITI for |k| <= max_shift_trials.
       Among the best counts, prefer the shift whose per-position-block levels are most uniform
       (each block is one speaker at one distance, so its trials share a level; a shifted
       labelling straddles block boundaries).
    3. Refine every trial onset to the steepest rising edge within +/- 60 ms of the prediction.

    Returns onset times (s) plus a `qc` dict: n_on_sound, the level dispersion of the chosen vs
    the best rival whole-trial shift, and `ambiguous` (neither the trial count nor the level
    signature separates the chosen labelling from a shifted one). `shift_override` forces a
    whole-trial shift relative to the count optimum (manual fix after inspecting a session)."""
    u = np.asarray(unity_onsets, float)
    n = len(u)
    if blocks is None:
        blocks = np.zeros(n, int)
    min_dur = float(np.min(clip_dur_s)) if clip_dur_s is not None and len(clip_dur_s) else 0.2
    post_len = float(np.clip(min_dur - 0.02, 0.04, 0.15))
    C = onset_contrast_curve(env, env_rate, post_s=(0.01, post_len))
    T = len(env) / env_rate
    span = u[-1] - u[0]

    # --- 1. coarse grid (offset step = 4 envelope samples ~ 21 ms at 187.5 Hz) -------------
    step = 4
    best = (-np.inf, 1.0, 0.0)
    Cs = np.where(np.isfinite(C), np.clip(C, -20, 40), -20.0)
    hit = np.isfinite(C) & (C > thr_db)
    for scale in np.arange(1.0 - max_drift, 1.0 + max_drift + 1e-12, 2e-4):
        # offsets such that trial 0 maps anywhere from "schedule ends at t=0" to "starts at T"
        o_lo, o_hi = -span * scale - u[0] * scale, T - u[0] * scale
        k0 = int(np.floor(o_lo * env_rate)); k1 = int(np.ceil(o_hi * env_rate))
        ks_all = np.arange(k0, k1 + 1, step)
        base = np.round(u * scale * env_rate).astype(np.int64)
        for c0 in range(0, len(ks_all), 4096):          # chunked: bounded memory
            ks = ks_all[c0:c0 + 4096]
            idx = ks[:, None] + base[None, :]
            inside = (idx >= 0) & (idx < len(C))
            idc = np.clip(idx, 0, len(C) - 1)
            cnt = (hit[idc] & inside).sum(axis=1)
            mean_c = np.where(inside, Cs[idc], -20.0).mean(axis=1)
            score = cnt + mean_c / 1000.0
            i = int(np.argmax(score))
            if score[i] > best[0]:
                best = (float(score[i]), float(scale), float(ks[i] / env_rate))
    _, scale, off = best

    # --- fine refinement around the coarse optimum ------------------------------------------
    best_f = (-np.inf, scale, off)
    for sc in np.arange(scale - 2.4e-4, scale + 2.4e-4 + 1e-12, 2e-5):
        for o in off + np.arange(-6, 7) / env_rate:
            c, h = _count_on_sound(C, env_rate, u * sc + o, thr_db)
            s = h.sum() + np.where(np.isfinite(c), np.clip(c, -20, 40), -20).mean() / 1000.0
            if s > best_f[0]:
                best_f = (float(s), float(sc), float(o))
    _, scale, off = best_f

    # --- 2. whole-trial ambiguity --------------------------------------------------------------
    # A labelling shifted by k trials means trial i's audio sits where trial i+k was scheduled.
    # Shifts are applied as INDEX shifts on the schedule (extrapolated past both ends with the
    # median interval), so they stay exact for non-periodic schedules (phonemes).
    iti = float(np.median(np.diff(u))) * scale if n > 1 else 0.0
    cum = _cum_power(env)
    lvl_post = int(round(post_len * env_rate))
    K = max_shift_trials if n > 1 else 0
    du = float(np.median(np.diff(u))) if n > 1 else 0.0
    u_ext = np.concatenate([u[0] - du * np.arange(K, 0, -1), u, u[-1] + du * np.arange(1, K + 1)])

    def _times(k):
        return u_ext[K + k: K + k + n] * scale + off

    def _shift_stats(k):
        t = _times(k)
        c, h = _count_on_sound(C, env_rate, t, thr_db)
        idx = np.round(t * env_rate).astype(np.int64)
        lv = _win_db(cum, idx + 1, idx + 1 + lvl_post)
        disp = 0.0
        for b in np.unique(blocks):
            m = (blocks == b) & h
            if m.sum() >= 2:
                disp += float(np.var(lv[m]) * m.sum())
        disp /= max(1, int(h.sum()))
        return int(h.sum()), disp

    shifts = list(range(-K, K + 1))
    stats = {k: _shift_stats(k) for k in shifts}
    top = max(v[0] for v in stats.values())

    def _better(a, b):
        """labelling a has a decisively more uniform per-position level signature than b"""
        da, db = stats[a][1], stats[b][1]
        return db >= DISP_RATIO * da and db - da >= DISP_MIN_DB2

    # Count optimum first (ties -> most uniform levels -> smallest |k|). A shift of +-1 trial
    # only loses ONE edge trial, so the count alone is fragile (a missed first trial or a stray
    # sound after the session flips it): a rival within 1 trial of the count replaces it only
    # if its level signature is decisively more uniform.
    k_cnt = min(stats, key=lambda k: (-stats[k][0], stats[k][1], abs(k)))
    if shift_override is not None:
        k_best = int(shift_override)
    else:
        challengers = [k for k in stats if k != k_cnt and stats[k][0] >= stats[k_cnt][0] - 1
                       and _better(k, k_cnt)]
        k_best = min(challengers, key=lambda k: stats[k][1]) if challengers else k_cnt
    others = [k for k in stats if k != k_best]
    margin = stats[k_best][0] - (max(stats[k][0] for k in others) if others else 0)
    disp_best = stats[k_best][1]
    rivals = [k for k in others if stats[k][0] >= stats[k_best][0] - 1]
    disp_next = min((stats[k][1] for k in rivals), default=None)
    # decisive = no rival explains (almost) as many trials with a comparably uniform signature.
    # Non-decisive sessions are resolved downstream from the DOAs (pipeline._doa_shift_check).
    decisive = all(_better(k_best, k) for k in rivals)
    off_shift_k = k_best
    sched = _times(k_best)

    # --- 3. per-trial rising-edge refinement ----------------------------------------------------
    edge = onset_contrast_curve(env, env_rate, post_s=(0.0, 0.03), pre_s=(-0.06, -0.005))
    pred = sched
    onset = pred.copy()
    w = int(round(0.06 * env_rate))
    for i, t in enumerate(pred):
        j0 = int(round(t * env_rate))
        lo, hi = max(0, j0 - w), min(len(edge), j0 + w + 1)
        if hi - lo < 3:
            continue
        seg = edge[lo:hi]
        j = int(np.argmax(seg))
        if np.isfinite(seg[j]) and seg[j] > 3.0:
            onset[i] = (lo + j) / env_rate
    c_final, h_final = _count_on_sound(C, env_rate, onset, thr_db)

    n_on = int(h_final.sum())
    ambiguous = bool(not decisive) and shift_override is None
    qc = {"scale": round(scale, 6), "offset_s": round(float(sched[0] - u[0] * scale), 4),
          "iti_s": round(iti, 4), "n_trials": int(n), "n_on_sound": n_on,
          "chosen_shift_vs_count_optimum": int(off_shift_k - k_cnt),
          "shift_margin_trials": int(margin),
          "block_level_dispersion_db2": round(disp_best, 3),
          "rival_dispersion_db2": (round(disp_next, 3) if disp_next is not None else None),
          "shift_override": (int(shift_override) if shift_override is not None else None),
          "index_shift_abs": int(k_best),
          "ambiguous": ambiguous,
          "onset_refine_median_ms": round(float(np.median(np.abs(onset - pred))) * 1000, 1)}
    return {"onset_s": onset, "contrast_db": c_final, "on_sound": h_final, "qc": qc}


def align_schedule(wav: ZyliaWav, unity_onsets: np.ndarray, blocks: np.ndarray,
                   clip_dur_s: np.ndarray | None = None, env_hop: int | None = None,
                   min_on_sound_frac: float = 0.9, doa_check=None,
                   user_shift: int | None = None) -> dict:
    """`align_trials`-compatible wrapper of `align_schedule_env` for clip-log sessions.

    Uses the channel-POWER envelope (no inter-capsule cancellation, so the 5 kHz tone and weak
    VBAP renders stay visible).

    `doa_check(onset_s, on_sound) -> dict` (optional, see pipeline._doa_shift_check) re-checks
    the trial->position labelling from the DOAs themselves: within a session every position is
    a run of consecutive trials >= 30 deg from its neighbours, so a labelling shifted by k
    trials inflates the within-position DOA scatter. It returns {"best_shift", "decisive",
    "scatter_deg"}. When the energy alignment is ambiguous and the DOA check is decisive, its
    shift is applied; when the two DISAGREE decisively the session is flagged ambiguous.

    `user_shift` (manual fix, from conditions.json `alignment_shift_override`): relabel so
    that the audio found at trial i is attributed to trial i + user_shift -- the same sign
    convention as `label_shift_audit`'s / the DOA check's `best_shift`.

    Sessions left ambiguous, or with < `min_on_sound_frac` of trials on sound, get a loud
    warning; run_analysis marks their trials unreliable unless accepted in conditions.json."""
    if env_hop is None:                      # ~5.3 ms envelope resolution at any sample rate
        env_hop = max(16, int(round(wav.sample_rate / 187.5)))
    env, env_rate = compute_envelope(wav, hop=env_hop, mode="power")
    res = align_schedule_env(env, env_rate, unity_onsets, blocks=blocks, clip_dur_s=clip_dur_s)
    qc = res["qc"]
    k_abs = qc["index_shift_abs"]
    doa_info, conflict = None, False
    if user_shift is not None:
        res = align_schedule_env(env, env_rate, unity_onsets, blocks=blocks, clip_dur_s=clip_dur_s,
                                 shift_override=k_abs - int(user_shift))
        res["qc"]["user_shift"] = int(user_shift)
    elif doa_check is not None:
        doa_info = doa_check(res["onset_s"], res["on_sound"])
        k_doa = doa_info.get("best_shift")
        if doa_info.get("decisive") and k_doa not in (None, 0):
            if qc["ambiguous"]:
                res = align_schedule_env(env, env_rate, unity_onsets, blocks=blocks,
                                         clip_dur_s=clip_dur_s, shift_override=k_abs - int(k_doa))
                res["qc"]["resolved_by_doa_shift"] = int(k_doa)
            else:
                conflict = True               # energy says decisive, DOAs say shifted
        elif qc["ambiguous"] and doa_info.get("decisive") and k_doa == 0:
            res["qc"]["ambiguous"] = False
            res["qc"]["resolved_by_doa_shift"] = 0
    qc = res["qc"]
    if doa_info is not None:
        qc["doa_check"] = doa_info
    if conflict:
        qc["ambiguous"] = True
        qc["energy_doa_conflict"] = True
    name = getattr(wav, "path", "")
    if qc["ambiguous"] or qc["n_on_sound"] < min_on_sound_frac * qc["n_trials"]:
        print(f"  [ALIGNMENT WARNING] {Path(name).parent.name if name else ''}: "
              f"{qc['n_on_sound']}/{qc['n_trials']} trials on sound, ambiguous={qc['ambiguous']}"
              f"{' (energy and DOA disagree)' if conflict else ''} -> inspect its onsets figure; "
              "ambiguous sessions are excluded from statistics until accepted or given an "
              "'alignment_shift_override' in conditions.json")
    onset_s = res["onset_s"]
    uniq = np.unique(blocks)
    u = np.asarray(unity_onsets, float)
    block_off = {int(b): float(np.median(onset_s[blocks == b] - u[blocks == b])) for b in uniq}
    block_conf = {int(b): float(np.mean(res["on_sound"][blocks == b])) for b in uniq}
    bursts = onset_s[res["on_sound"]]
    return {
        "n_bursts_detected": int(res["on_sound"].sum()),
        "block_offsets": block_off,
        "block_offset_confidence": block_conf,
        "block_repaired": {int(b): False for b in uniq},
        "nearest_burst_s": np.where(res["on_sound"], 0.0, np.nan),
        "onset_samples": np.round(onset_s * wav.sample_rate).astype(np.int64),
        "onset_s": onset_s,
        "env": env,
        "env_rate": env_rate,
        "bursts_s": bursts,
        "on_sound": res["on_sound"],
        "contrast_db": res["contrast_db"],
        "alignment_qc": qc,
    }
