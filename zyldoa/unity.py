"""Parse the Unity experiment CSVs for per-trial block index and stimulus-onset times.

Position ground truth is the folder name; sound type comes from the block index. We use the
FrameData stream (one row per rendered frame) to read, for each trial, its block and the
clock time of its first StimOn frame.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


# Unity places the listener origin at this height (m); AudioTargetPosition is relative to the
# room origin on the floor, so target elevation is measured about (0, LISTENER_HEIGHT_M, 0).
LISTENER_HEIGHT_M = 1.20


def parse_ground_truth(folder_name: str) -> dict:
    """Parse the session position from the folder name.

    'Subject_15,0,1.63_AVLoc_Data_...'      -> single position {azimuth, elevation, radius}.
    'Subject_0-345,25,1.63,V_AVLoc_Data...' -> packed session (one recording sweeps an azimuth
    range at a fixed elevation): azimuth_deg is None and per-trial truth must come from the
    TrialData CSV (`parse_trial_data`)."""
    m = re.search(r"Subject_(\d+\.?\d*)-(\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*)", folder_name)
    if m:
        a0, a1, el, r = (float(x) for x in m.groups())
        return {"packed": True, "azimuth_deg": None, "azimuth_range_deg": [a0, a1],
                "elevation_deg": el, "radius_m": r}
    m = re.search(r"Subject_(-?\d+\.?\d*),(-?\d+\.?\d*),(-?\d+\.?\d*)", folder_name)
    if not m:
        raise ValueError(f"Cannot parse position from folder: {folder_name}")
    az, el, r = (float(x) for x in m.groups())
    return {"packed": False, "azimuth_deg": az, "elevation_deg": el, "radius_m": r}


def _find(folder: Path, key: str) -> Path:
    hits = list(folder.glob(f"*{key}*.csv"))
    if not hits:
        raise FileNotFoundError(f"No *{key}*.csv in {folder}")
    return hits[0]


_VEC3 = re.compile(r"\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)")


def parse_trial_data(folder: Path) -> dict | None:
    """Per-trial stimulus identity and (when present) per-trial target truth, keyed by trial.

    The TrialData CSV is authoritative for what played on each trial: `AudioTargetIndex` is the
    stimulus playlist index in every session layout (in packed sessions `BlockCount` is the
    azimuth block instead of the stimulus, so FrameData's block cannot identify the sound).
    `AudioTargetPosition` (Unity metres, listener at (0, 1.20, 0), az = atan2(x, z) clockwise
    from +z to match the folder-name convention) carries per-trial truth for packed sessions;
    ground-truth sessions leave it at (0,0,0) and keep the folder-name truth.

    Returns None when no TrialData CSV exists (early pilots), so callers can fall back to the
    FrameData block index."""
    try:
        path = _find(folder, "TrialData")
    except FileNotFoundError:
        return None
    per_trial: dict[int, dict] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            try:
                t = int(row["TrialCountInExpt"])
            except (KeyError, ValueError):
                continue
            entry = {"block": int(row["BlockCount"])}
            ai = (row.get("AudioTargetIndex") or "").strip()
            entry["stim_index"] = int(ai) if ai.lstrip("-").isdigit() else None
            entry["stim_type_str"] = (row.get("AudioStimType") or "").strip()
            m = _VEC3.search(row.get("AudioTargetPosition") or "")
            if m:
                x, y, z = (float(g) for g in m.groups())
                if x == 0.0 and y == 0.0 and z == 0.0:
                    entry["truth"] = None
                else:
                    dy = y - LISTENER_HEIGHT_M
                    r = float(np.sqrt(x * x + dy * dy + z * z))
                    entry["truth"] = {
                        "azimuth_deg": float(np.degrees(np.arctan2(x, z)) % 360.0),
                        "elevation_deg": float(np.degrees(np.arcsin(np.clip(dy / max(r, 1e-9), -1, 1)))),
                        "radius_m": r,
                    }
            else:
                entry["truth"] = None
            per_trial[t] = entry
    return per_trial


def parse_frame_data(folder: Path) -> dict:
    """Return per-trial arrays: trial index, block index, and Unity StimOn onset time (s)."""
    path = _find(folder, "FrameData")
    onset, block = {}, {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["TrialState"] != "StimOn":
                continue
            t = int(row["TrialCountInExpt"])
            if t not in onset:
                onset[t] = float(row["FrameStart"])
                block[t] = int(row["BlockCount"])
    trials = sorted(onset)
    return {
        "trial": np.array(trials, dtype=int),
        "block": np.array([block[t] for t in trials], dtype=int),
        "onset_unity_s": np.array([onset[t] for t in trials], dtype=float),
    }
