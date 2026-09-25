"""Detect stimulus onsets in the continuous recording and align them to the Unity trials.

Strategy: stream a broadband RMS envelope over the whole file, detect loud bursts, then fit
an affine clock map (Unity time -> WAV time) so every one of the 310 Unity trials gets a WAV
onset even if a few bursts are missed. Onsets are refined to the local rising edge.
"""
from __future__ import annotations

import numpy as np

from .wavio import ZyliaWav


def compute_envelope(wav: ZyliaWav, hop: int = 256, ref_channels=None) -> tuple[np.ndarray, float]:
    """Broadband RMS envelope of the channel-mean signal. Returns (env, env_rate_hz)."""
    chunk = (1 << 20) // hop * hop  # multiple of hop
    env_parts, carry = [], np.zeros(0, dtype=np.float32)
    for _, block in wav.iter_chunks(chunk):
        mono = block.mean(axis=1) if ref_channels is None else block[:, ref_channels].mean(axis=1)
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
