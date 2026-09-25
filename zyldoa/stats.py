"""Shared statistics for the localization analyses: circular medians, bootstrap CIs, and
summary stats. Lifted from the June comparison layer (compare.py) so the frozen June scripts
and the new cross-condition runner share one implementation. Seeds are fixed: every CI is
deterministic and reproducible.
"""
from __future__ import annotations

import numpy as np

STIM_ORDER = ["pink noise", "applause", "phone ringing", "250 Hz tone", "1000 Hz tone",
              "6000 Hz tone", "male speech", "female speech"]
NOISE_LIKE = ["pink noise", "applause"]
CAL_REF = "pink noise"


def circmedian(deg) -> float:
    """Median of angles (deg), computed about the circular mean to dodge the wrap."""
    a = np.radians(np.asarray(deg, float))
    c = np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    rel = (np.asarray(deg, float) - c + 180) % 360 - 180
    return float((c + np.median(rel) + 180) % 360 - 180)


def stim_order(labels) -> list:
    return [s for s in STIM_ORDER if s in labels] + [s for s in labels if s not in STIM_ORDER]


def boot_ci_median(x, n_boot: int = 2000, seed: int = 12345):
    """Percentile bootstrap 95% CI for the median (deterministic seed). None if n<3."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    return [round(float(np.percentile(meds, 2.5)), 1), round(float(np.percentile(meds, 97.5)), 1)]


def boot_ci_delta(a, b, n_boot: int = 2000, seed: int = 777):
    """Bootstrap 95% CI for (median(a) - median(b)) by independently resampling each group."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    da = np.median(a[rng.integers(0, len(a), size=(n_boot, len(a)))], axis=1)
    db = np.median(b[rng.integers(0, len(b), size=(n_boot, len(b)))], axis=1)
    d = da - db
    return [round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]


def stat(x) -> dict:
    x = np.asarray(x, float)
    return {"n": int(len(x)),
            "median": round(float(np.median(x)), 2) if len(x) else None,
            "p90": round(float(np.percentile(x, 90)), 2) if len(x) else None,
            "ci95": boot_ci_median(x)}


def delta_stat(a, b) -> dict:
    """median(a) - median(b) with its bootstrap CI and a CI-excludes-zero significance flag."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if not len(a) or not len(b):
        return {"delta": None, "ci95": [None, None], "significant": None,
                "n_a": int(len(a)), "n_b": int(len(b))}
    d = float(np.median(a) - np.median(b))
    ci = boot_ci_delta(a, b)
    sig = None if ci[0] is None else bool(ci[0] > 0 or ci[1] < 0)
    return {"delta": round(d, 2), "ci95": ci, "significant": sig,
            "n_a": int(len(a)), "n_b": int(len(b))}
