"""THROWAWAY DIAGNOSTIC: checks whether onset alignment (zyldoa/onsets.py:align_trials) -- not
the post-onset analysis window -- is the real root cause of the huge raw miss (~90-115 deg,
consistent with DOA uncorrelated with truth) seen on the five new short-burst stimuli even
after the clip_duration_s window fix.

Two independent checks, run against REAL data (needs Teba access):

1. Trial spacing in the session's own clip-log CSV: onsets.align_trials's per-block offset
   search probes energy at onset-0.30s (expected silence) and onset+0.15s (expected signal).
   Those offsets were sized for the original ~1s-spaced trials. If the new 200ms-labeled
   stimuli are packed much more tightly, the "-0.30s" pre-onset probe can land on the PREVIOUS
   trial's reverb tail instead of true silence, confusing the energy-step search and shifting
   every onset in the block by a roughly constant amount -- which would explain both (a) why
   the raw miss is so large and uniform, and (b) why merely shortening the post-onset window
   (this session's earlier fix) made no difference: the window is still anchored to the same
   wrong onset.
2. Cached alignment confidence (zyldoa/onsets.py:align_trials's block_offset_confidence /
   block_repaired, written to <tag>_align.json by a prior run_analysis.py run): confidence
   below ~1.5 means that block's offset was too weak to trust and was interpolated from
   neighbours ("repaired") -- a direct, already-computed signal of an alignment problem,
   no audio access needed for this half of the check.

Usage (run on the Teba-connected machine, inside the zylia conda env):
    python diagnose_onset_alignment.py <session_folder> [--results-dir results/newstim_s12_mono_cc]

<session_folder> is one real session directory, e.g.:
    "T:\\JordanWashecheck\\Objective Localization\\applause_200ms_Setup1_mono_CC___2026__...".
--results-dir defaults to the condition whose session this is (inferred name not required --
pass the condition's results_dir explicitly, e.g. results/newstim_s12_mono_cc) so the cached
<tag>_align.json can be found; omit it to skip check 2 and run check 1 only.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from zyldoa.pipeline import folder_tag


def check_trial_spacing(folder: Path) -> None:
    csvs = sorted(folder.glob("*.csv"))
    if len(csvs) != 1:
        raise SystemExit(f"Expected exactly one clip-log CSV in {folder}, found {len(csvs)}")
    starts, durs = [], []
    with open(csvs[0], newline="") as fh:
        for row in csv.DictReader(fh):
            starts.append(float(row["ClipStart"]))
            durs.append(float(row["ClipEnd"]) - float(row["ClipStart"]))
    starts.sort()
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    print(f"=== trial spacing: {csvs[0].name} ===")
    print(f"  n_trials={len(starts)}  clip_duration: min={min(durs):.3f}s "
          f"median={sorted(durs)[len(durs)//2]:.3f}s max={max(durs):.3f}s")
    print(f"  inter-trial gap (ClipStart deltas): min={min(gaps):.3f}s "
          f"median={sorted(gaps)[len(gaps)//2]:.3f}s max={max(gaps):.3f}s")
    tight = sum(1 for g in gaps if g < 0.45)
    print(f"  gaps < 0.45s (would make the -0.30s pre-onset 'silence' probe land inside the "
          f"PREVIOUS trial's reverb tail): {tight}/{len(gaps)}"
          + ("  <-- LIKELY ROOT CAUSE" if tight else "  -- spacing looks safe"))


def check_alignment_confidence(results_dir: Path, folder: Path) -> None:
    tag = folder_tag(folder)
    meta_path = results_dir / f"{tag}_align.json"
    if not meta_path.exists():
        print(f"\n(no cached {meta_path} -- run run_analysis.py --force on this condition first "
              f"to generate it)")
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    conf = meta.get("block_offset_confidence", {})
    repaired = meta.get("block_repaired", {})
    n_weak = sum(1 for v in repaired.values() if v)
    print(f"\n=== cached alignment confidence: {meta_path.name} ===")
    print(f"  n_blocks={len(conf)}  n_repaired(weak, interpolated)={n_weak}")
    if n_weak:
        weak = {b: round(conf[b], 2) for b in conf if repaired.get(b)}
        print(f"  weak blocks (confidence, <1.5 triggers interpolation): {weak}"
              "  <-- LIKELY ROOT CAUSE if most/all blocks are weak")
    else:
        print("  all blocks locked with confidence >= 1.5 -- alignment itself looks fine "
              "(look elsewhere if raw miss is still large)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_folder", type=Path)
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="condition results dir holding the cached <tag>_align.json, e.g. "
                         "results/newstim_s12_mono_cc")
    args = ap.parse_args()

    check_trial_spacing(args.session_folder)
    if args.results_dir:
        check_alignment_confidence(args.results_dir, args.session_folder)
    else:
        print("\n(pass --results-dir to also check cached alignment confidence)")


if __name__ == "__main__":
    main()
