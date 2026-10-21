"""Regression test for the clip-log trial aligner (onsets.align_schedule).

Reproduces the failure that broke the first/second-pass new-data analysis: a strictly
periodic schedule (one 200 ms clip every 3.2 s, NO pause between speaker positions) whose
first position block is quiet. The legacy `align_trials` anchors on the first burst its
detector finds, so it maps the whole session whole trials late; `align_schedule` must recover
every trial exactly.

Runs on a streamed synthetic 19-channel recording (no data needed):
    python tests/test_align_schedule.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zyldoa import onsets  # noqa: E402

FS = 12000           # low rate keeps the synthetic stream light; hop 64 -> 187.5 Hz envelope
HOP = 64


class SyntheticWav:
    """Minimal ZyliaWav stand-in: 19-ch noise floor + 200 ms noise bursts, generated per chunk."""

    def __init__(self, onsets_s, levels_db, dur_s, burst_s=0.2, floor_db=-78.0, seed=0):
        self.sample_rate = FS
        self.n_channels = 19
        self.n_frames = int(dur_s * FS)
        self.duration_s = self.n_frames / FS
        self.on = np.asarray(onsets_s); self.lv = np.asarray(levels_db)
        self.burst = int(burst_s * FS); self.floor = 10 ** (floor_db / 20); self.seed = seed
        self.path = Path("synthetic/Take.wav")

    def read(self, start, n):
        start = max(0, int(start)); n = int(min(n, self.n_frames - start))
        rng = np.random.default_rng(self.seed + start)
        x = rng.standard_normal((n, self.n_channels)).astype(np.float32) * self.floor
        t0 = start
        for o, l in zip(self.on, self.lv):
            s = int(round(o * FS))
            a, b = max(s, t0), min(s + self.burst, t0 + n)
            if a < b:
                x[a - t0:b - t0] += (rng.standard_normal((b - a, self.n_channels)).astype(np.float32)
                                     * 10 ** (l / 20))
        return x

    def iter_chunks(self, chunk_frames=1 << 20):
        pos = 0
        while pos < self.n_frames:
            k = min(chunk_frames, self.n_frames - pos)
            yield pos, self.read(pos, k)
            pos += k


def make_session(first_block_db=-70.0, seed=1):
    rng = np.random.default_rng(seed)
    n_blocks, reps, iti = 18, 10, 3.2
    u = 6.7 + iti * np.arange(n_blocks * reps) + rng.normal(0, 0.003, n_blocks * reps)  # Unity ClipStart
    blocks = np.repeat(np.arange(n_blocks), reps)
    block_db = rng.uniform(-60, -52, n_blocks)
    block_db[0] = first_block_db                  # a quiet first position (far/occluded speaker)
    scale, offset = 1.00015, 2.85                 # audio clock vs Unity clock
    true_on = u * scale + offset
    wav = SyntheticWav(true_on, block_db[blocks], dur_s=true_on[-1] + 15.0, seed=seed)
    return wav, u, blocks, true_on


def test_new_aligner_recovers_every_trial():
    wav, u, blocks, true_on = make_session()
    al = onsets.align_schedule(wav, u, blocks, clip_dur_s=np.full(len(u), 0.2), env_hop=HOP)
    err = al["onset_s"] - true_on
    qc = al["alignment_qc"]
    print("new aligner:", qc, " max |onset error| = %.1f ms" % (1000 * np.abs(err).max()))
    assert np.abs(err).max() < 0.015, "an onset is off by more than 15 ms"
    assert qc["n_on_sound"] >= 175 and not qc["ambiguous"]


def test_legacy_aligner_shifts_whole_trials():
    """Documents the bug: with a quiet first block and no inter-block pauses the legacy aligner
    lands an integer number of trials late. (If this ever starts passing as 'correct', the
    legacy path changed -- the new data must still use align_schedule.)"""
    wav, u, blocks, true_on = make_session()
    al = onsets.align_trials(wav, u, blocks, env_hop=HOP)
    shift = np.round((al["onset_s"] - true_on) / 3.2).astype(int)
    vals, cnt = np.unique(shift, return_counts=True)
    print("legacy aligner: whole-trial shift ->", dict(zip(vals.tolist(), cnt.tolist())))
    assert np.any(shift != 0), "legacy aligner unexpectedly got every trial right"


if __name__ == "__main__":
    test_new_aligner_recovers_every_trial()
    test_legacy_aligner_shifts_whole_trials()
    print("OK")
