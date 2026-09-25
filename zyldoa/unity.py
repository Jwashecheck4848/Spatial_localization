"""Parse the Unity experiment logs for per-trial timing, stimulus identity, and truth.

The current (2026-09) format is a single comma-delimited clip-log CSV per session folder, one
row per trial (columns: SourceType, SpeakerChannelIndex, SpeakerChannel, SourcePositionXYZ,
SourcePositionAzimuth, SourcePositionElevation, SourcePositionDistance, ClipNumber,
ClipRepetition, ClipName, DateTime, ClipStart, ClipEnd). One session always sweeps every
speaker position for a single stimulus family (pink noise OR phonemes, never mixed in one
file); per-trial truth position and clip timing both live directly on each row, so no
separate frame-by-frame log or per-trial join is needed (`parse_clip_log`). Legacy
ground-truth sessions (single fixed physical position, folder name `Subject_<az>,<el>,<r>`)
are still identified by `parse_ground_truth`, used only for those folders' dedup key.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


def parse_ground_truth(folder_name: str) -> dict:
    """Parse the session position from a LEGACY folder name (single fixed physical speaker).

    'Subject_15,0,1.63_AVLoc_Data_...'      -> single position {azimuth, elevation, radius}.
    'Subject_0-345,25,1.63,V_AVLoc_Data...' -> packed session (one recording sweeps an azimuth
    range at a fixed elevation): azimuth_deg is None (per-trial truth came from the retired
    TrialData CSV in that era). New clip-log sessions carry no position in the folder name at
    all -- see `parse_clip_log`, which supplies fully per-trial truth instead."""
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


# Trailing session timestamp on every new-format folder/file name, e.g.
# '..._2026__09_14__18_22_17'. Stripping it yields the retake-identity used for dedup: two
# takes of the same stimulus/speaker-setup/VBAP/curtain combination share this key, and
# 'duplicates: latest' (run_analysis.discover_sessions) keeps the newer one.
_TIMESTAMP_SUFFIX_RE = re.compile(r"_*\d{4}__\d{2}_\d{2}__\d{2}_\d{2}_\d{2}_*$")


def session_key(folder_name: str) -> str:
    """Retake-identity key for a new-format clip-log session: the folder name with its
    trailing timestamp (and any surrounding underscores) stripped."""
    return _TIMESTAMP_SUFFIX_RE.sub("", folder_name)


# Stimulus family is constant for an entire clip-log session and is read from the CSV's
# FILENAME, not any column -- ClipNumber is reused (0-9) with different meaning in each
# family (always 0 for the single pink-noise burst; 0-9 selects one of ten phonemes), so it
# cannot identify which family a file belongs to on its own.
FAMILY_STIM_INDEX = {"pink noise": 0, "phonemes": 1}


def clip_family(csv_name: str) -> str:
    """Stimulus family ('pink noise' or 'phonemes') from a clip-log CSV's filename."""
    n = csv_name.lower()
    if "phoneme" in n:
        return "phonemes"
    if "pink" in n or "noise" in n:
        return "pink noise"
    raise ValueError(f"Cannot determine stimulus family from CSV filename: {csv_name}")


def find_clip_log(folder: Path) -> Path:
    """The session's single clip-log CSV (exactly one per folder in the new format)."""
    hits = sorted(Path(folder).glob("*.csv"))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"Expected exactly one clip-log CSV in {folder}, found {len(hits)}: "
            f"{[h.name for h in hits]}")
    return hits[0]


def parse_clip_log(folder: Path) -> dict:
    """Parse the session's single clip-log CSV into the arrays `align_and_classify` needs.

    One row = one trial. `SpeakerChannelIndex` groups trials by speaker position and is used
    directly as the clock-alignment block id (trials for one position are recorded back to
    back, exactly like the legacy per-block clock assumption). `stim_index` is constant for
    the whole session (the file's stimulus family, from its filename). Truth position comes
    straight from `SourcePositionAzimuth/Elevation/Distance`: azimuth is normalized to 0-360
    deg clockwise (source files use either that convention directly, or a signed -180..+180
    convention -- `% 360.0` reconciles both since they share the same rotational sense);
    elevation is already referenced to listener ear height; distance is in CENTIMETRES and is
    converted to metres."""
    path = find_clip_log(folder)
    family = clip_family(path.name)
    stim_index_val = FAMILY_STIM_INDEX[family]
    trial, block, onset_s, truth_az, truth_el, truth_r = [], [], [], [], [], []
    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            trial.append(i)
            block.append(int(row["SpeakerChannelIndex"]))
            onset_s.append(float(row["ClipStart"]))
            truth_az.append(float(row["SourcePositionAzimuth"]) % 360.0)
            truth_el.append(float(row["SourcePositionElevation"]))
            truth_r.append(float(row["SourcePositionDistance"]) / 100.0)
    if not trial:
        raise ValueError(f"No rows in clip log: {path}")
    return {
        "trial": np.array(trial, dtype=int),
        "block": np.array(block, dtype=int),
        "onset_unity_s": np.array(onset_s, dtype=float),
        "stim_index": np.full(len(trial), stim_index_val, dtype=int),
        "truth_az": np.array(truth_az, dtype=float),
        "truth_el": np.array(truth_el, dtype=float),
        "truth_r": np.array(truth_r, dtype=float),
        "family": family,
    }
