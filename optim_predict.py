"""Per-direction VBAP predictions for the virtual-source comparison.

The virtual condition pans a phantom source with VBAP over a real 24-speaker array and records it on
the Zylia. We predict, for the SAME source directions, the angular error a phantom should have, using
the optimizer's own VBAP renderer (SpeakerOptimization/src/spatial_audio/vbap.py). Two layouts:

  DEPLOYED  (primary): the actually-installed array, AvLocalization3D/Assets/CaveSpeakerPositions.csv
            -- listener-relative spherical (Unity: az from +z/front toward +x/right, el up). This is
            what the virtual experiment really rendered, so its prediction is the apples-to-apples one.
  OPTIMIZED (reference): the optimizer's best_layout. NOTE the deployed array is NOT the optimized one
            (they share only the speaker count), so this is "what the optimizer promised", not what
            was built. Evaluated in the optimizer's room frame (z up, origin floor centre).

Per direction we report:
  loc_error  = angle(rV, target)   -- velocity vector. NOTE rV points at the target by construction
                                      within any VBAP triangle (gains solve L^T g = d, and energy/
                                      amplitude normalization is a positive scalar), so this is
                                      identically ~0 deg for every covered direction (the 0.0001 is a
                                      float floor). rV is therefore NOT a per-direction predictor.
  energy_err = angle(rE, target)   -- energy vector, the high-frequency blur. THE direction-dependent
                                      VBAP prediction of record.
  rV_mag, rE_mag                    -- 1 = sharp point source, <1 = diffuse/blurred phantom

The Zylia intensity DOA is a real, in-room low/mid-frequency probe; it follows neither ideal vector
exactly (empirically the measured miss exceeds both). rV and rE are reported as ideal references --
rE as the only non-trivial prediction -- not as quantities the measurement is expected to match.

`python optim_predict.py` writes results/optim/predictions.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
PROJECT = EXP_DIR.parent
OPT_DIR = PROJECT / "SpeakerOptimization"
DEPLOYED_CSV = PROJECT / "AvLocalization3D" / "Assets" / "CaveSpeakerPositions.csv"
DEFAULT_LAYOUT = OPT_DIR / "results" / "optimization_results_20260324_160148.json"
sys.path.insert(0, str(OPT_DIR))

from src.spatial_audio.vbap import VBAPRenderer  # noqa: E402

# June pilot directions (kept for the frozen predictions.json); the full July grid is
# 24 azimuths x 3 elevations, built in main().
EXP_AZ = [0.0, 15.0, 30.0, 45.0]
FULL_AZ = [float(a) for a in range(0, 360, 15)]
FULL_EL = [-25.0, 0.0, 25.0]


def spherical_to_cartesian(az_deg, el_deg, r=1.0) -> np.ndarray:
    """Unity convention (SpeakerPositioning.cs / TrialDefGenerator.cs): x=right, y=up, z=forward;
    azimuth 0 = +z (forward), +az toward +x (right, clockwise from above); elevation up."""
    az = np.radians(np.asarray(az_deg, float)); el = np.radians(np.asarray(el_deg, float))
    return np.stack([r * np.cos(el) * np.sin(az), r * np.sin(el), r * np.cos(el) * np.cos(az)], axis=-1)


def load_deployed(csv_path: Path = DEPLOYED_CSV) -> np.ndarray:
    """Installed array -> (24,3) listener-relative positions in metres, Unity frame."""
    pos = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            pos.append(spherical_to_cartesian(float(row["AZ_Degrees"]), float(row["EL_Degrees"]),
                                              float(row["Radius_Cm"]) / 100.0))
    return np.array(pos)


def load_optimized(path: Path = DEFAULT_LAYOUT) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    rec = data[0] if isinstance(data, list) else data
    return np.asarray(rec["best_layout"], dtype=float)


def exp_dirs_unity(az, el=0.0) -> np.ndarray:
    az = np.asarray(az, float)
    el = np.broadcast_to(np.asarray(el, float), az.shape)
    return spherical_to_cartesian(az, el, 1.0)


def exp_dirs_room(az, el=0.0, front_sign: float = -1.0) -> np.ndarray:
    """Experiment (az,el) -> unit vector in the optimizer room frame (z up). front_sign=-1 puts az=0
    along -Y (documented room 'front'); +1 puts it along +Y. Handedness preserved either way."""
    az = np.radians(np.asarray(az, float))
    el = np.radians(np.broadcast_to(np.asarray(el, float), az.shape))
    return np.stack([np.sin(az) * np.cos(el), front_sign * np.cos(az) * np.cos(el), np.sin(el)], axis=-1)


def _ang_deg(u, v) -> float:
    u = u / (np.linalg.norm(u) + 1e-12); v = v / (np.linalg.norm(v) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def predict(speakers: np.ndarray, listener: np.ndarray, dirs: np.ndarray) -> list[dict]:
    r = VBAPRenderer(speakers, listener)
    out = []
    for d in np.atleast_2d(dirs):
        d = d / (np.linalg.norm(d) + 1e-12)
        tri_idx, _ = r.find_enclosing_triangle(d)
        gains = r.compute_gains(d, normalize="energy")
        covered = bool(np.any(gains > 0))
        rV, rE = r.compute_rv_re(gains)
        out.append({
            "covered": covered,
            "loc_error_deg": round(_ang_deg(d, rV), 4) if covered else None,
            "energy_err_deg": round(_ang_deg(d, rE), 4) if covered else None,
            "rV_mag": round(float(np.linalg.norm(rV)), 4) if covered else 0.0,
            "rE_mag": round(float(np.linalg.norm(rE)), 4) if covered else 0.0,
            "triangle_cond": round(float(r.condition_numbers[tri_idx]), 3) if tri_idx >= 0 else None,
            "n_active_speakers": int(np.sum(gains > 1e-6)),
        })
    return out


def _block(name, speakers, listener, azel_list, dir_fn, sweep_az):
    az = np.array([a for a, _ in azel_list]); el = np.array([e for _, e in azel_list])
    exp = [{"az": float(a), "el": float(e), **p}
           for (a, e), p in zip(azel_list, predict(speakers, listener, dir_fn(az, el)))]
    sweep = [{"az": float(a), **p}
             for a, p in zip(sweep_az, predict(speakers, listener,
                                               dir_fn(sweep_az, np.zeros_like(sweep_az))))]
    cov = float(np.mean([s["covered"] for s in sweep]))
    return {"layout": name, "n_speakers": int(len(speakers)), "experiment_directions": exp,
            "azimuth_sweep_el0": sweep, "sweep_coverage_el0": cov}


def run(results_dir: Path = EXP_DIR / "results" / "optim", front_sign: float = -1.0,
        az_list=None, el_list=None, out_name: str = "predictions_full.json") -> dict:
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    sweep_az = np.arange(0.0, 360.0, 5.0)
    az_list = FULL_AZ if az_list is None else [float(a) for a in az_list]
    el_list = FULL_EL if el_list is None else [float(e) for e in el_list]
    azel = [(a, e) for e in el_list for a in az_list]

    deployed = _block("deployed", load_deployed(), np.zeros(3), azel, exp_dirs_unity, sweep_az)
    optimized = _block("optimized_best_layout", load_optimized(), np.array([0.0, 0.0, 1.2]),
                       azel, lambda a, e: exp_dirs_room(a, e, front_sign=front_sign), sweep_az)

    out = {
        "deployed": deployed, "optimized": optimized, "front_sign_optimized": front_sign,
        "grid": {"az": az_list, "el": el_list},
        "notes": "PRIMARY = deployed array (what the virtual experiment rendered; exact Unity-frame "
                 "directions, listener at origin). REFERENCE = optimizer best_layout (a DIFFERENT array, "
                 "shares only the 24-speaker count). loc_error=rV err is ~0 by VBAP construction (not a "
                 "predictor); energy_err=rE is the direction-dependent prediction. Compare measured miss "
                 "to rE; empirically it exceeds both rV and rE.",
    }
    (results_dir / out_name).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {results_dir/out_name}")
    for blk in (deployed, optimized):
        print(f"\n[{blk['layout']}]  sweep coverage el0 = {blk['sweep_coverage_el0']*100:.0f}%")
        errs = [e["energy_err_deg"] for e in blk["experiment_directions"] if e["covered"]]
        n_cov = sum(e["covered"] for e in blk["experiment_directions"])
        print(f"  {n_cov}/{len(blk['experiment_directions'])} directions covered; "
              f"median rE err = {np.median(errs):.2f} deg, p90 = {np.percentile(errs, 90):.2f} deg")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--front-sign", type=float, default=-1.0,
                    help="optimizer-frame 'front' for az=0: -1 = -Y (documented), +1 = +Y")
    ap.add_argument("--az", type=float, nargs="*", default=None,
                    help="azimuths to predict (default: full 24-point 15-degree grid)")
    ap.add_argument("--el", type=float, nargs="*", default=None,
                    help="elevations to predict (default: -25 0 25)")
    ap.add_argument("--out-name", default="predictions_full.json",
                    help="output file name under results/optim/ (June predictions.json is frozen)")
    args = ap.parse_args()
    run(front_sign=args.front_sign, az_list=args.az, el_list=args.el, out_name=args.out_name)


if __name__ == "__main__":
    main()
