"""THROWAWAY DIAGNOSTIC: compares the MOUNT FIT ITSELF between pn_s12_mono_cc and pn_s12_vbap_cc
(the two conditions newstim_s12_mono_cc / newstim_s12_vbap_cc borrow their calibration from).

Prior diagnostics ruled out: window-length (clip_duration_s fix), onset-alignment (spacing +
cached block confidence both clean), and per-channel truth/survey error (error is UNIFORM
across every position/azimuth/elevation/setup for mono, not concentrated on a few channels).

A uniform-but-mono-only degradation that vanishes when the SAME mount is tested against the
data it was fit on (pink noise itself, both self-calibrated, both ~4-8 deg) but appears when
that mount is borrowed for a different dataset (applause/phone ringing/phonemes/1000 Hz tone)
is the signature of an OVER-FIT / badly-EXTRAPOLATING mount model: 'mount_model: auto' can pick
a rotation model with more free parameters than a narrow calibration position set actually
constrains (zyldoa/calibrate.py's own warnings flag exactly this: "narrow azimuth span:
azimuth scaling is unconstrained" / "rotation fit on a single elevation: tilt is
unconstrained"). Such a model can fit its own narrow calibration set near-perfectly while
diverging badly outside that set's azimuth/elevation span -- explaining why mono's OWN pink
noise looks fine, but applause/phonemes/phone ringing (likely swept across EVERY position,
not just whichever narrower set pink noise covered) do not.

This script prints, for pn_s12_mono_cc and pn_s12_vbap_cc side by side: the selected mount
model, calibration warnings, the azimuth/elevation span the fit was actually constrained by,
and the number of distinct positions. A narrower span / fewer positions / a rotation-model
warning on mono vs. vbap would confirm this as the root cause.

Run (needs real data / Teba access):
    python diagnose_mount_model.py
"""
from __future__ import annotations

import json

from compare_conditions import _load_registry, _results_dir

CONDITIONS = ["pn_s12_mono_cc", "pn_s12_vbap_cc"]


def _summary(registry: dict, name: str) -> dict:
    rd = _results_dir(registry, name)
    path = rd / "summary.json"
    if not path.exists():
        raise SystemExit(f"{name}: no {path} yet -- run `python run_analysis.py --condition "
                         f"{name}` first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    registry = _load_registry()
    for name in CONDITIONS:
        s = _summary(registry, name)
        az = s.get("calibration_inliers_az", [])
        print(f"=== {name} ===")
        print(f"  mounting: {s.get('mounting_description')}")
        print(f"  mount_model_selection: {s.get('mount_model_selection')}")
        print(f"  n_calibration_positions: {len(az)}")
        if az:
            print(f"  azimuth span used for the fit: min={min(az):.1f} max={max(az):.1f} "
                  f"span={max(az) - min(az):.1f} deg  distinct_az={len(set(round(a) for a in az))}")
        else:
            print("  azimuth span: n/a (no inlier positions recorded)")
        warnings = s.get("calibration_warnings", [])
        if warnings:
            print("  calibration warnings:")
            for w in warnings:
                print(f"    ! {w}")
        else:
            print("  calibration warnings: none")
        print()

    print("If mono shows a NARROWER azimuth span / fewer distinct positions / a 'rotation'-model "
          "selection + unconstrained-tilt or unconstrained-scaling warning relative to vbap, "
          "that confirms an over-fit/badly-extrapolating mono mount as the root cause: the "
          "borrowed mount fits pink noise (its own narrow calibration set) near-perfectly but "
          "diverges on the wider position sweep used by the new stimulus families.")


if __name__ == "__main__":
    main()
