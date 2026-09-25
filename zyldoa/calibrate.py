"""Mounting calibration and angular 'miss' between Zylia DOA and the true speaker direction.

The Zylia frame is rotated/handed relative to the room, so a raw azimuth comparison conflates a
fixed mounting offset with true detection error. With the speaker positions spanning only a
narrow azimuth range at one elevation, a full 3-D rotation fit is under-constrained, so we use an
interpretable constant-offset mount: a known azimuth handedness (+/-1) plus a constant azimuth
offset and a constant elevation offset. Handedness and offsets are fit on the self-consistent
recordings; the residual angle is the corrected miss.
"""
from __future__ import annotations

import numpy as np


def _circmean_deg(x) -> float:
    return float(np.degrees(np.arctan2(np.mean(np.sin(np.radians(x))),
                                       np.mean(np.cos(np.radians(x))))))


def _wrap180(a):
    return (np.asarray(a) + 180) % 360 - 180


def _circular_span_deg(angles: np.ndarray) -> float:
    """Angular extent covered by a set of azimuths (deg), correct across the +/-180 wrap."""
    a = np.sort(np.mod(np.asarray(angles, float), 360.0))
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(360.0 - gaps.max())


def fit_mounting_robust(positions: list[dict], tol_deg: float = 15.0,
                        min_az_span_deg: float = 10.0) -> dict:
    """Fit the mount from per-position broadband directions, auto-rejecting inconsistent ones.

    Generalizes to any set of speaker locations: tries each azimuth handedness, finds the largest
    cluster of positions agreeing on a constant offset, and treats the rest as out-of-frame
    anomalies. Positions are identified by `id` (folder name) so two locations sharing an azimuth
    do not merge. `positions` items: {id, truth_az, truth_el, meas_az, meas_el}. Returns the
    calibration, inlier/outlier ids+azimuths, per-position residual, and warnings about anything
    that makes the fit fragile (few positions, narrow span, ambiguous handedness)."""
    ids = [p["id"] for p in positions]
    if len(positions) < 2:
        p = positions[0] if positions else {"meas_az": 0, "truth_az": 0, "meas_el": 0, "truth_el": 0, "id": "?"}
        cal = {"az_sign": 1.0, "az_offset": float(p["truth_az"] - p["meas_az"]),
               "el_offset": float(p["truth_el"] - p["meas_el"])}
        return {"calibration": cal, "inlier_ids": ids, "outlier_ids": [],
                "inlier_az": [p["truth_az"]], "outlier_az": [], "residual_deg": {},
                "warnings": ["single position: mount offset is assumed, not measured"]}

    ta = np.array([p["truth_az"] for p in positions], float)
    te = np.array([p["truth_el"] for p in positions], float)
    ma = np.array([p["meas_az"] for p in positions], float)
    me = np.array([p["meas_el"] for p in positions], float)
    per_sign = {}
    for s in (1.0, -1.0):
        off = _wrap180(ta - s * ma)
        best_i = max(range(len(off)), key=lambda i: int((np.abs(_wrap180(off - off[i])) < tol_deg).sum()))
        per_sign[s] = np.abs(_wrap180(off - off[best_i])) < tol_deg
    s = max(per_sign, key=lambda k: int(per_sign[k].sum()))
    inl = per_sign[s]

    az_offset = _circmean_deg(_wrap180(ta[inl] - s * ma[inl]))
    el_offset = float(np.median(te[inl] - me[inl]))
    cal = {"az_sign": s, "az_offset": az_offset, "el_offset": el_offset}
    az_c, el_c = apply_mounting(cal, ma, me)
    resid = {ids[k]: round(float(miss_deg(az_c[k], el_c[k], ta[k], te[k])), 2) for k in range(len(positions))}

    warn = []
    if int(inl.sum()) < 3:
        warn.append(f"only {int(inl.sum())} self-consistent position(s): the mount is fit "
                    "in-sample, so corrected miss here is a lower bound on accuracy (it omits "
                    "held-out positions) and is not the same as within-position precision")
    span = _circular_span_deg(ta[inl]) if inl.sum() else 0.0
    if span < min_az_span_deg:
        warn.append(f"narrow azimuth span ({span:.0f} deg): azimuth scaling is unconstrained")
    if int(per_sign[-s].sum()) >= int(inl.sum()):
        warn.append("azimuth handedness is ambiguous (both signs fit comparably)")
    if len(np.unique(te)) <= 1:
        warn.append("all positions at one elevation: elevation accuracy is unvalidated and the "
                    "miss is effectively azimuth-only")
    return {"calibration": cal, "inlier_ids": [ids[k] for k in range(len(ids)) if inl[k]],
            "outlier_ids": [ids[k] for k in range(len(ids)) if not inl[k]],
            "inlier_az": [float(x) for x in ta[inl]], "outlier_az": [float(x) for x in ta[~inl]],
            "residual_deg": resid, "warnings": warn}


def _azel_to_unit(az, el):
    az = np.radians(np.asarray(az, float)); el = np.radians(np.asarray(el, float))
    return np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def _unit_to_azel(u):
    u = np.asarray(u, float)
    az = np.degrees(np.arctan2(u[..., 1], u[..., 0]))
    el = np.degrees(np.arcsin(np.clip(u[..., 2], -1.0, 1.0)))
    return _wrap180(az), el


def fit_mounting_rotation(positions: list[dict], tol_deg: float = 15.0,
                          max_iter: int = 10) -> dict:
    """Fit the mount as one orthogonal transform (rotation, or reflection for a handedness
    flip) via Procrustes on unit vectors, with the same robust inlier loop as the offset model.

    Unlike the constant-offset model, a rigid rotation represents a *tilted* mount exactly:
    a tilt produces azimuth-dependent elevation error that no constant el offset can absorb.
    Needs positions spanning >=2 elevations to constrain the tilt; with a single elevation the
    offset model is the safer choice (use mount_model='auto')."""
    ids = [p["id"] for p in positions]
    if len(positions) < 3:
        return {**fit_mounting_robust(positions, tol_deg=tol_deg),
                "warnings": ["too few positions for a rotation fit: fell back to the offset model"]}
    M = _azel_to_unit([p["meas_az"] for p in positions], [p["meas_el"] for p in positions])
    T = _azel_to_unit([p["truth_az"] for p in positions], [p["truth_el"] for p in positions])

    def _procrustes(mask):
        H = M[mask].T @ T[mask]
        U, _, Vt = np.linalg.svd(H)
        return (U @ Vt).T          # T ~= R @ M ; reflections allowed (det may be -1)

    inl = np.ones(len(positions), bool)
    for _ in range(max_iter):
        R = _procrustes(inl)
        resid = np.degrees(np.arccos(np.clip(np.sum((M @ R.T) * T, axis=1), -1.0, 1.0)))
        new_inl = resid < tol_deg
        if new_inl.sum() < 3 or np.array_equal(new_inl, inl):
            inl = new_inl if new_inl.sum() >= 3 else inl
            break
        inl = new_inl
    R = _procrustes(inl)
    resid = np.degrees(np.arccos(np.clip(np.sum((M @ R.T) * T, axis=1), -1.0, 1.0)))
    inl = resid < tol_deg if (resid < tol_deg).sum() >= 3 else inl

    cal = {"type": "rotation", "R": R.tolist(), "det": float(np.sign(np.linalg.det(R)))}
    ta = np.array([p["truth_az"] for p in positions], float)
    te = np.array([p["truth_el"] for p in positions], float)
    warn = []
    if len(np.unique(te[inl])) <= 1:
        warn.append("rotation fit on a single elevation: tilt is unconstrained; prefer the "
                    "offset model here")
    if int(inl.sum()) < 4:
        warn.append(f"only {int(inl.sum())} inlier position(s): rotation fit is fragile")
    if cal["det"] < 0:
        warn.append("mount transform is a reflection (azimuth handedness flipped)")
    return {"calibration": cal,
            "inlier_ids": [ids[k] for k in range(len(ids)) if inl[k]],
            "outlier_ids": [ids[k] for k in range(len(ids)) if not inl[k]],
            "inlier_az": [float(x) for x in ta[inl]],
            "outlier_az": [float(x) for x in ta[~inl]],
            "residual_deg": {ids[k]: round(float(resid[k]), 2) for k in range(len(ids))},
            "warnings": warn}


def select_mount_fit(positions: list[dict], mount_model: str = "auto",
                     tol_deg: float = 15.0) -> dict:
    """Fit the requested mount model, or (auto) pick between offset and rotation.

    Auto picks the rotation only when it is actually constrained (inliers span >=2 elevations)
    AND it beats the offset model's median inlier residual; otherwise the interpretable offset
    model wins. The returned fit carries a `model_selection` record with both residuals."""
    off = fit_mounting_robust(positions, tol_deg=tol_deg)
    if mount_model == "offset":
        off.setdefault("model_selection", {"model": "offset", "reason": "requested"})
        return off
    rot = fit_mounting_rotation(positions, tol_deg=tol_deg)
    if mount_model == "rotation":
        rot.setdefault("model_selection", {"model": "rotation", "reason": "requested"})
        return rot

    def _med_inlier_resid(fit):
        r = [v for k, v in fit["residual_deg"].items() if k in set(fit["inlier_ids"])]
        return float(np.median(r)) if r else float("inf")

    off_r, rot_r = _med_inlier_resid(off), _med_inlier_resid(rot)
    els = {p["truth_el"] for p in positions if p["id"] in set(rot["inlier_ids"])}
    pick_rot = (rot["calibration"].get("type") == "rotation" and len(els) >= 2
                and rot_r < off_r)
    chosen = rot if pick_rot else off
    chosen["model_selection"] = {
        "model": "rotation" if pick_rot else "offset",
        "offset_median_inlier_resid_deg": round(off_r, 2),
        "rotation_median_inlier_resid_deg": round(rot_r, 2) if np.isfinite(rot_r) else None,
        "reason": ("rotation constrained by >=2 elevations and lower residual" if pick_rot else
                   "offset model kept (rotation unconstrained or not better)")}
    return chosen


def fit_by_group(positions: list[dict], group_of: dict) -> dict:
    """Mount-stability diagnostic: fit the offset model separately per group (e.g. recording
    day) and report the spread of implied offsets. Large spread (>~2 deg) suggests the mic was
    remounted between groups and a single global mount is questionable."""
    groups: dict[str, list[dict]] = {}
    for p in positions:
        groups.setdefault(str(group_of.get(p["id"], "?")), []).append(p)
    out = {}
    for g, ps in sorted(groups.items()):
        if len(ps) < 2:
            out[g] = {"n": len(ps), "note": "too few positions to fit"}
            continue
        fit = fit_mounting_robust(ps)
        c = fit["calibration"]
        out[g] = {"n": len(ps), "az_sign": c["az_sign"],
                  "az_offset": round(c["az_offset"], 2), "el_offset": round(c["el_offset"], 2),
                  "n_inliers": len(fit["inlier_ids"])}
    offs = [v["az_offset"] for v in out.values() if "az_offset" in v]
    els = [v["el_offset"] for v in out.values() if "el_offset" in v]
    spread = {"az_offset_spread_deg": round(float(np.ptp(offs)), 2) if len(offs) > 1 else None,
              "el_offset_spread_deg": round(float(np.ptp(els)), 2) if len(els) > 1 else None}
    return {"by_group": out, **spread}


def apply_mounting(cal: dict, az, el):
    if cal.get("type") == "rotation" or "R" in cal:
        R = np.asarray(cal["R"], float)
        u = _azel_to_unit(az, el)
        return _unit_to_azel(u @ R.T)
    az_cal = _wrap180(cal["az_sign"] * np.asarray(az) + cal["az_offset"])
    el_cal = np.asarray(el, float) + cal["el_offset"]
    # fold elevation back into [-90, 90], flipping azimuth by 180 across a pole
    over = el_cal > 90; under = el_cal < -90
    el_cal = np.where(over, 180 - el_cal, np.where(under, -180 - el_cal, el_cal))
    az_cal = _wrap180(np.where(over | under, az_cal + 180, az_cal))
    return az_cal, el_cal


def miss_deg(az1, el1, az2, el2):
    """Great-circle angle(s) (deg) between directions given in degrees. Vectorized."""
    az1, el1, az2, el2 = map(lambda v: np.radians(np.asarray(v, float)), (az1, el1, az2, el2))
    dot = (np.cos(el1) * np.cos(el2) * np.cos(az1 - az2) + np.sin(el1) * np.sin(el2))
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def describe_mounting(cal: dict) -> str:
    if cal.get("type") == "rotation" or "R" in cal:
        R = np.asarray(cal["R"], float)
        f_az, f_el = _unit_to_azel(R @ np.array([1.0, 0.0, 0.0]))
        det = cal.get("det", float(np.sign(np.linalg.det(R))))
        return (f"rigid-rotation mount: Zylia front (az 0) -> room az {float(f_az):.1f} deg, "
                f"el {float(f_el):+.1f} deg; handedness "
                f"{'preserved' if det > 0 else 'flipped (reflection)'}")
    return (f"Zylia front (az 0) -> room az {cal['az_offset']:.1f} deg; "
            f"azimuth handedness {'preserved' if cal['az_sign'] > 0 else 'flipped'}; "
            f"elevation offset {cal['el_offset']:+.1f} deg")
