"""Zylia ground-truth localization analysis: separate trials, classify stimuli from their
spectrograms, estimate per-trial direction of arrival, and compare to the known speaker position.

The Zylia stays fixed; a single speaker is moved to each position (one folder per position, named
azimuth,elevation,radius). The pipeline streams each 19-channel Zylia WAV, separates the ~310
trials by aligning to the Unity trial table, classifies each block's sound from its spectrum,
estimates DOA via rigid-sphere first-order Ambisonic intensity, fits one Zylia mounting offset
from the self-consistent positions, and reports the angular 'miss' (raw and mount-corrected).

`python run_analysis.py` reproduces everything; it generalizes to any set of positions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from zyldoa import pipeline, calibrate, viz, stimid, unity, quality
from zyldoa.wavio import list_wavs as wavio_list_wavs

EXP_DIR = Path(__file__).resolve().parent
DATA_ROOT = EXP_DIR.parent
RESULTS = EXP_DIR / "results"
FIGURES = EXP_DIR / "figures"
RELIABLE_SNR_DB = 6.0       # below this a trial's DOA is untrustworthy
# A coarse high-frequency gate to drop the 6 kHz tone (kr~4.5, far past the order-1 ceiling). It is
# NOT the order-1 validity limit itself -- that is kr~1.1 (~1.1 kHz), enforced by the analysis band
# in doa.py. No analysed tone lies between 1.2 and 6 kHz, so the exact cap value is immaterial here.
MAX_RELIABLE_HZ = 5000.0


def is_broadband(stim_type: str) -> bool:
    return stim_type not in stimid.TONE_TYPES


def _session_stamp(folder_name: str) -> str:
    """Sortable timestamp from the session folder name ('' if none).

    Format is `..._YYYY__MM_DD__HH_MM_SS` (double underscore before month and hour only)."""
    import re as _re
    m = _re.search(r"(\d{4})__(\d{2})_(\d{2})__(\d{2})_(\d{2})_(\d{2})", folder_name)
    return "".join(m.groups()) if m else ""


def discover_sessions(data_root: Path, only: str | list[str] | None = None,
                      exclude: str | list[str] | None = None,
                      duplicates: str = "latest") -> list[Path]:
    """Session folders under data_root that can actually be analysed: a Zylia WAV plus either a
    parseable legacy `Subject_*` position or the newer single clip-log CSV (position fully
    data-driven, no folder-name pattern required). data_root is assumed dedicated to one
    condition's sessions, so every subfolder is a candidate -- no longer restricted to a
    `Subject_*` glob, since clip-log sessions use a different naming scheme entirely.

    Handles legacy single-position sessions, legacy packed multi-azimuth sessions
    (`Subject_0-345,...`), and the current clip-log format (one CSV per folder, every speaker
    position swept in one recording). Folders missing the 19-channel WAV (an interrupted
    recording) or, for legacy names, whose name carries no position (an aborted session) are
    skipped with a printed reason, so a partial collection never silently drops trials or
    crashes the run. `only` restricts to folders whose name contains the given substring, or --
    for multi-axis condition filtering (stimulus family AND speaker rig AND VBAP/mono AND curtain
    state, all out of one shared data_root) -- ALL substrings in a given list. `exclude` (single
    substring or list) drops any folder containing ANY of them, e.g. `exclude="_VBAP_"` to select
    the mono takes of a rig that also has VBAP-rendered sessions.

    When several sessions target the SAME cell (re-takes, e.g. the ',R'-flagged re-recordings,
    or a later timestamp on an identically-named clip-log session), `duplicates` decides:
    'latest' (default) keeps the newest take and prints what it superseded, 'earliest' keeps
    the original, 'all' keeps every take as its own session. The dedup cell is the parsed
    azimuth/elevation/radius for legacy `Subject_*` folders, or the folder name with its
    trailing timestamp stripped (`unity.session_key`) for the newer clip-log sessions (which
    carry no position in their name at all -- truth is fully per-trial, from the CSV)."""
    onlys = [only] if isinstance(only, str) else list(only or [])
    excludes = [exclude] if isinstance(exclude, str) else list(exclude or [])
    found = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if onlys and not all(tok in p.name for tok in onlys):
            continue
        if excludes and any(tok in p.name for tok in excludes):
            continue
        if not wavio_list_wavs(p):
            print(f"  [skip] {p.name}: no Zylia WAV"); continue
        csvs = list(p.glob("*.csv"))
        if len(csvs) == 1:
            cell = unity.session_key(p.name)
        else:
            try:
                gt = unity.parse_ground_truth(p.name)
            except ValueError:
                print(f"  [skip] {p.name}: no position in folder name"); continue
            cell = (tuple(gt["azimuth_range_deg"]) if gt.get("packed") else gt["azimuth_deg"],
                    gt["elevation_deg"], gt["radius_m"])
        found.append((cell, _session_stamp(p.name), p))
    if duplicates == "all":
        return [p for _, _, p in found]
    by_cell: dict = {}
    for cell, stamp, p in found:
        by_cell.setdefault(cell, []).append((stamp, p))
    out = []
    for cell, takes in sorted(by_cell.items(), key=lambda kv: str(kv[0])):
        takes.sort()
        keep = takes[-1] if duplicates == "latest" else takes[0]
        for stamp, p in takes:
            if p is not keep[1]:
                print(f"  [duplicate] {p.name}: superseded by {keep[1].name} "
                      f"(duplicates={duplicates})")
        out.append(keep[1])
    return sorted(out, key=lambda p: p.name)


discover_positions = discover_sessions  # historical name


def _session_day(folder_name: str) -> str:
    """Recording day parsed from the session timestamp, for mount-stability grouping."""
    import re as _re
    m = _re.search(r"(\d{4})__(\d{2})_(\d{2})__", folder_name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else "?"


def _external_fit(cal: dict, positions: list[dict]) -> dict:
    """Build a calibration 'fit' that *applies* a pre-measured mounting (e.g. fit on the ground
    truth) to a new condition rather than re-fitting it. Every position is treated as a valid
    measurement against that fixed mount; residuals are the resulting per-position miss."""
    ids = [p["id"] for p in positions]
    if positions:
        ma = np.array([p["meas_az"] for p in positions]); me = np.array([p["meas_el"] for p in positions])
        ta = np.array([p["truth_az"] for p in positions]); te = np.array([p["truth_el"] for p in positions])
        az_c, el_c = calibrate.apply_mounting(cal, ma, me)
        resid = {ids[k]: round(float(calibrate.miss_deg(az_c[k], el_c[k], ta[k], te[k])), 2)
                 for k in range(len(positions))}
        inlier_az = [float(x) for x in ta]
    else:
        resid, inlier_az = {}, []
    return {"calibration": cal, "inlier_ids": ids, "outlier_ids": [],
            "inlier_az": inlier_az, "outlier_az": [], "residual_deg": resid,
            "warnings": ["mounting applied from an external (ground-truth) calibration, not re-fit "
                         "on this condition's data"]}


def reliable(t: dict) -> bool:
    hz = float(t["stim_hz"]) if t["stim_hz"] else 0.0
    return float(t["inband_snr_db"]) >= RELIABLE_SNR_DB and hz < MAX_RELIABLE_HZ


def _circmedian(deg) -> float:
    a = np.radians(deg)
    c = np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    rel = (np.asarray(deg) - c + 180) % 360 - 180
    return float((c + np.median(rel) + 180) % 360 - 180)


def run(force: bool = False, figures: bool = True, data_root: Path = DATA_ROOT,
        results_dir: Path = RESULTS, figures_dir: Path = FIGURES,
        external_calibration: dict | None = None,
        stim_classification: dict | None = None,
        calibration_label: str | None = None,
        mount_model: str = "auto",
        per_session: bool = False,
        only: str | list[str] | None = None,
        exclude: str | list[str] | None = None,
        duplicates: str = "latest") -> dict:
    data_root = Path(data_root); results_dir = Path(results_dir); figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
    folders = discover_sessions(data_root, only=only, exclude=exclude, duplicates=duplicates)
    if not folders:
        raise SystemExit(f"No analysable Subject_* folders under {data_root}"
                         + (f" matching --only {only!r}" if only else "")
                         + (f" excluding {exclude!r}" if exclude else ""))
    print(f"Analysing {len(folders)} session(s) under {data_root}")

    # Stage 1: separate trials + classify stimuli (per folder), then reconcile to canonical labels.
    # A supplied stim_classification (e.g. the ground truth's) overrides reconciliation so a
    # condition with few positions inherits the same stimulus identities -- the source material is
    # identical across conditions, only the playback differs.
    metas = [pipeline.align_and_classify(f, results_dir, force=force) for f in folders]
    if stim_classification is not None:
        canonical = {int(b): v for b, v in stim_classification.items()
                     if str(b).lstrip("-").isdigit()}
    else:
        canonical = pipeline.reconcile_classification(metas)
    (results_dir / "stimulus_classification.json").write_text(json.dumps(canonical, indent=2))

    # Stage 1.5: raw-signal integrity BEFORE any DOA — capsule health / clipping / stimulus
    # verification per session (cached). Capsules flagged hot or dead are EXCLUDED from the
    # SH encoding for that session (the remaining capsules still resolve first order), so one
    # faulty sensor degrades gracefully instead of silently poisoning the direction estimates.
    per_quality, exclude_map = {}, {}
    for meta, folder in zip(metas, folders):
        qpath = results_dir / f"{meta['tag']}_quality.json"
        if qpath.exists() and not force:
            q = json.loads(qpath.read_text())
        else:
            q = quality.session_quality(folder, np.load(meta["npz_path"]), canonical)
            qpath.write_text(json.dumps(q, indent=2))
        per_quality[meta["folder"]] = q
        bad = sorted({c["capsule"] for c in q.get("capsules", {}).get("capsule_flags", [])
                      if {"hot", "dead"} & set(c["tags"])})
        exclude_map[meta["folder"]] = bad
        if bad:
            print(f"  [quality] {meta['folder']}: excluding capsule(s) {bad} from the encoding")
        elif not q.get("ok", True):
            print(f"  [quality] {meta['folder']}: " + "; ".join(q["issues"][:3]))

    # Stage 2: per-trial DOA using the canonical labels.
    # Truth comes from the folder name (one speaker per session) unless the TrialData carried
    # per-trial target positions (packed multi-azimuth sessions); `position` identifies one
    # physical/phantom source location within one session, the unit of calibration/stats.
    combined = []
    by_folder = {}
    for meta, folder in zip(metas, folders):
        rows = pipeline.estimate_doa(folder, canonical, results_dir, force=force,
                                     exclude_capsules=exclude_map[meta["folder"]])
        gt = meta["ground_truth"]
        for t in rows:
            t["folder"] = meta["folder"]
            if str(t.get("trial_truth_az", "")) != "":
                t["truth_az"] = float(t["trial_truth_az"])
                t["truth_el"] = float(t["trial_truth_el"])
                t["radius_m"] = float(t["trial_truth_r"])
                t["position"] = f"{meta['folder']}|az{t['truth_az']:g}"
            else:
                if gt["azimuth_deg"] is None:
                    raise SystemExit(f"{meta['folder']}: packed session without per-trial "
                                     "truth in TrialData -- cannot assign target directions")
                t["truth_az"] = gt["azimuth_deg"]; t["truth_el"] = gt["elevation_deg"]
                t["radius_m"] = gt["radius_m"]
                t["position"] = meta["folder"]
            t["reliable"] = reliable(t)
        by_folder[meta["folder"]] = (meta, rows)
        combined += rows

    # Robust mounting calibration from per-position broadband directions (auto-flags outliers).
    # Positions are identified by folder so two locations sharing an azimuth never merge.
    # Fit the mount on the cleanest wideband reference. With semantic stimuli the "broadband" bucket
    # mixes pink noise + phone + applause + speech (each with its own bias), so a designated reference
    # (pink noise) gives an interpretable mount against which every stimulus's deviation is measured.
    def _cal_trials(rows):
        if calibration_label:
            sel = [t for t in rows if t["stim_label"] == calibration_label and t["reliable"]]
            if sel:
                return sel
        return [t for t in rows if is_broadband(t["stim_type"]) and t["reliable"]]
    positions = []
    day_of = {}
    for meta, rows in by_folder.values():
        per_pos: dict[str, list] = {}
        for t in rows:
            per_pos.setdefault(t["position"], []).append(t)
        for pid, prows in sorted(per_pos.items()):
            bb = _cal_trials(prows)
            if not bb:
                continue
            positions.append({"id": pid,
                              "truth_az": float(prows[0]["truth_az"]),
                              "truth_el": float(prows[0]["truth_el"]),
                              "meas_az": _circmedian([float(t["doa_az"]) for t in bb]),
                              "meas_el": float(np.median([float(t["doa_el"]) for t in bb]))})
            day_of[pid] = _session_day(meta["folder"])
    # Per-session calibration: when the microphone frame differs between separately-recorded
    # packed sessions (each a single-elevation block), one global mount cannot register all of
    # them. Fit and apply a mount per session (folder). This is the per-session analog of the
    # self-mount and, like it, absorbs any per-session rendering rotation, so report it as such.
    per_session_cals = None
    if per_session and external_calibration is None:
        sess_positions: dict[str, list] = {}
        for p in positions:
            sess_positions.setdefault(p["id"].split("|")[0], []).append(p)
        per_session_cals = {}
        inlier_ids = set()
        for sess, ps in sorted(sess_positions.items()):
            sfit = calibrate.select_mount_fit(ps, mount_model)
            per_session_cals[sess] = sfit["calibration"]
            inlier_ids |= set(sfit["inlier_ids"])
        fit = calibrate.select_mount_fit(positions, mount_model)  # global fit: summary/diagnostic only
    else:
        fit = (_external_fit(external_calibration, positions) if external_calibration is not None
               else calibrate.select_mount_fit(positions, mount_model))
        inlier_ids = set(fit["inlier_ids"])
    cal = fit["calibration"]
    mount_by_day = (calibrate.fit_by_group(positions, day_of)
                    if len(set(day_of.values())) > 1 else None)

    for t in combined:
        az, el = float(t["doa_az"]), float(t["doa_el"])
        tcal = per_session_cals[t["folder"]] if per_session_cals else cal
        az_c, el_c = calibrate.apply_mounting(tcal, az, el)
        t["doa_az_cal"] = round(float(az_c), 2); t["doa_el_cal"] = round(float(el_c), 2)
        t["in_calibration"] = t["position"] in inlier_ids
        t["miss_raw"] = round(float(calibrate.miss_deg(az, el, t["truth_az"], t["truth_el"])), 2)
        t["miss_corrected"] = round(float(calibrate.miss_deg(az_c, el_c, t["truth_az"], t["truth_el"])), 2)

    _prune_stale_artifacts({m["tag"] for m in metas}, results_dir, figures_dir)
    _write_master_csv(results_dir / "all_trials.csv", combined)
    summary = summarize(combined, fit, canonical, mount_by_day=mount_by_day)
    if per_session_cals is not None:
        summary["per_session_mounts"] = {s: {k: (round(v, 2) if isinstance(v, float) else v)
                                              for k, v in c.items() if k != "R"}
                                         for s, c in per_session_cals.items()}
        summary["calibration_note"] = ("mount fit and applied per session (folder); global "
                                        "mount in 'mounting' is diagnostic only")
    summary["signal_quality"] = quality.aggregate(per_quality)
    summary["signal_quality"]["capsules_excluded_by_session"] = {
        f: bad for f, bad in exclude_map.items() if bad}
    summary["positions"] = [{"folder": m["folder"], "tag": m["tag"],
                             "ground_truth": m["ground_truth"]} for m in metas]
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results_dir / "report.md").write_text(generate_report(summary, canonical, fit), encoding="utf-8")

    if figures:
        def _safe(fn, *a):
            try:
                fn(*a)
            except Exception as e:                       # a plot must never discard the analysis
                print(f"  [warn] figure {fn.__name__} skipped: {e}")
        for meta in metas:
            npz = dict(np.load(meta["npz_path"]))
            folder = data_root / meta["folder"]
            _safe(viz.spectrogram_montage, folder, npz, canonical, figures_dir / f"{meta['tag']}_spectrograms.png")
            _safe(viz.onset_validation, meta["folder"], npz, figures_dir / f"{meta['tag']}_onsets.png")
        _safe(viz.doa_vs_truth, combined, figures_dir / "doa_vs_truth.png")
        _safe(viz.miss_by_stim, combined, figures_dir / "miss_by_stimulus.png")
        _safe(viz.miss_map, combined, figures_dir / "miss_map.png")
        _safe(viz.miss_map, combined, figures_dir / "miss_map_tones.png",
              ("250 Hz tone", "1000 Hz tone"), 60.0)

    print_report(summary)
    return summary


def _pos_label(r):
    return f"az{r['truth_az']:g}_el{r['truth_el']:g}_r{r['radius_m']:g}"


def precision_by_stim(combined: list[dict]) -> dict:
    """Within-position angular scatter (repeatability), independent of calibration: for each
    position+stimulus, the median deviation of each trial's DOA from that group's median DOA.
    Grouped by position id so packed sessions and repeated sessions stay distinct."""
    groups = {}
    for r in combined:
        if r["reliable"] and r["in_calibration"]:
            groups.setdefault((r["position"], r["stim_label"]), []).append(r)
    by_stim = {}
    for (_, lab), rows in groups.items():
        az = np.array([float(r["doa_az"]) for r in rows]); el = np.array([float(r["doa_el"]) for r in rows])
        maz, mel = _circmedian(az), float(np.median(el))
        dev = calibrate.miss_deg(az, el, maz, mel)
        by_stim.setdefault(lab, []).extend(dev.tolist())
    return {k: round(float(np.median(v)), 2) for k, v in by_stim.items()}


def summarize(combined: list[dict], fit: dict, canonical: dict,
              mount_by_day: dict | None = None) -> dict:
    def stats(rows, key):
        v = np.array([float(r[key]) for r in rows]) if rows else np.array([])
        return {"n": len(v), "median": round(float(np.median(v)), 2) if len(v) else None,
                "mean": round(float(np.mean(v)), 2) if len(v) else None,
                "p90": round(float(np.percentile(v, 90)), 2) if len(v) else None}
    by_pos = {}
    for pos in sorted({r["position"] for r in combined}):
        rel = [r for r in combined if r["position"] == pos and r["reliable"]]
        if not rel:
            continue
        bb = [r for r in rel if is_broadband(r["stim_type"])]
        tn = [r for r in rel if not is_broadband(r["stim_type"])]
        by_pos[pos] = {
            "label": _pos_label(rel[0]), "in_calibration": bool(rel[0]["in_calibration"]),
            "truth_az": float(rel[0]["truth_az"]), "truth_el": float(rel[0]["truth_el"]),
            "raw": stats(rel, "miss_raw"), "corrected": stats(rel, "miss_corrected"),
            "corrected_broadband": stats(bb, "miss_corrected"),
            "corrected_tones": stats(tn, "miss_corrected"),
        }
    by_stim = {}
    for r in combined:
        if r["reliable"] and r["in_calibration"]:
            by_stim.setdefault(r["stim_label"], []).append(float(r["miss_corrected"]))
    by_stim = {k: {"median": round(float(np.median(v)), 2), "n": len(v)} for k, v in by_stim.items()}

    # Elevation-resolved views: overall and per stimulus, over calibration-consistent trials.
    by_el: dict[str, list] = {}
    by_stim_el: dict[str, dict[str, list]] = {}
    for r in combined:
        if r["reliable"] and r["in_calibration"]:
            ek = f"{float(r['truth_el']):g}"
            by_el.setdefault(ek, []).append(float(r["miss_corrected"]))
            by_stim_el.setdefault(r["stim_label"], {}).setdefault(ek, []).append(
                float(r["miss_corrected"]))
    by_elevation = {k: {"median": round(float(np.median(v)), 2), "n": len(v)}
                    for k, v in sorted(by_el.items(), key=lambda kv: float(kv[0]))}
    miss_by_stim_el = {lab: {k: {"median": round(float(np.median(v)), 2), "n": len(v)}
                             for k, v in sorted(d.items(), key=lambda kv: float(kv[0]))}
                       for lab, d in by_stim_el.items()}

    out = {"mounting": fit["calibration"], "mounting_description": calibrate.describe_mounting(fit["calibration"]),
           "mount_model_selection": fit.get("model_selection"),
           "calibration_inlier_positions": fit["inlier_ids"], "calibration_outlier_positions": fit["outlier_ids"],
           "calibration_inliers_az": fit["inlier_az"], "calibration_outliers_az": fit["outlier_az"],
           "calibration_warnings": fit["warnings"],
           "per_position_calibration_residual_deg": fit["residual_deg"],
           "stimuli": {int(b): canonical[b]["label"] for b in canonical},
           "by_position": by_pos, "corrected_miss_by_stimulus": by_stim,
           "by_elevation": by_elevation,
           "miss_by_stimulus_and_elevation": miss_by_stim_el,
           "within_position_precision_by_stimulus": precision_by_stim(combined),
           "reliable_snr_db": RELIABLE_SNR_DB}
    if mount_by_day is not None:
        mount_by_day = dict(mount_by_day)
        mount_by_day["_caveat"] = ("recording day and elevation are confounded in the July "
                                   "ground-truth collection (one elevation per day): a day-to-day "
                                   "offset difference may be a true remount OR an elevation-"
                                   "dependent mount error the offset model cannot represent")
        out["mounting_by_day"] = mount_by_day
    return out


def generate_report(summary: dict, canonical: dict, fit: dict) -> str:
    L = ["# Zylia ground-truth localization report", ""]
    L += ["The Zylia ZM-1 is fixed; a single speaker is moved to each labelled position. For every",
          "sound played at each location we estimate the Zylia's perceived direction and compare it",
          "to the true speaker direction -- the **angular miss**.", ""]
    L += ["## Stimuli (classified from the spectrogram)", "", "| block | sound |", "|---|---|"]
    for b in sorted(canonical):
        L.append(f"| {b} | {canonical[b]['label']} |")
    L += ["", "## Zylia mounting", "",
          f"- {summary['mounting_description']}",
          f"- Calibration positions (self-consistent): {fit['inlier_az']}",
          f"- Out-of-frame / anomalous positions: {fit['outlier_az'] or 'none'}", ""]
    if fit["warnings"]:
        L += ["**Calibration caveats:**"] + [f"- {w}" for w in fit["warnings"]] + [""]
    L += ["## Localization miss (mount-corrected, median deg)", "",
          "| position | broadband | tones | all | in calibration |", "|---|---|---|---|---|"]
    for s in summary["by_position"].values():
        L.append(f"| {s['label']} | {s['corrected_broadband']['median']} | {s['corrected_tones']['median']} "
                 f"| {s['corrected']['median']} | {s['in_calibration']} |")
    L += ["", "## Per sound, at the calibration positions (median deg)", "",
          "| sound | corrected miss | within-position precision | n |", "|---|---|---|---|"]
    prec = summary["within_position_precision_by_stimulus"]
    for k, v in summary["corrected_miss_by_stimulus"].items():
        L.append(f"| {k} | {v['median']} | {prec.get(k, '-')} | {v['n']} |")
    L += ["", "## How to read these numbers", "",
          "- **Within-position precision** = how tightly the Zylia points at one fixed source "
          "across repeats. It measures **repeatability only and says nothing about correctness** -- "
          "a tone can be 0.1 deg precise yet 15 deg wrong (consistently pointing the wrong way).",
          "- **Mount-corrected miss** = absolute error after removing one fitted Zylia orientation. "
          f"With only {len(fit['inlier_az'])} self-consistent position(s) at one elevation, this is an "
          "**in-sample, azimuth-only** fit: a lower bound on accuracy (it omits held-out positions), "
          "not the same quantity as precision.",
          "- `tests/test_doa.py` confirms the first-order encoding + intensity inversion is "
          "self-consistent and correctly co-phased (synthetic plane wave recovered to <1 deg). This "
          "validates the *signal-processing*, not real-room performance, which the data above measures.",
          "- **Broadband sounds localize far better than pure tones.** Candidate causes for the tone "
          "degradation (not disentangled here): room standing-waves at a single mic point, first-order "
          "mode-strength conditioning, and spatial aliasing -- order-1 validity ends near kr=1 "
          "(~1.1 kHz for this array). Tones above ~5 kHz (e.g. the 6 kHz block) are excluded entirely "
          "as past the array's usable band.",
          f"- Out-of-frame positions ({fit['outlier_az'] or 'none'}) are inconsistent with the others "
          "under a single fixed mounting and should be checked against lab notes.", ""]
    return "\n".join(L)


def print_report(summary: dict) -> None:
    print("\n" + "=" * 74)
    print("ZYLIA GROUND-TRUTH LOCALIZATION - SUMMARY")
    print("=" * 74)
    print("Stimuli:", ", ".join(f"b{b}={lab}" for b, lab in summary["stimuli"].items()))
    print("Mounting:", summary["mounting_description"])
    print("Calibration inliers:", summary["calibration_inliers_az"],
          "| outliers:", summary["calibration_outliers_az"] or "none")
    for w in summary["calibration_warnings"]:
        print("  ! caveat:", w)
    print(f"\n{'position':>16} | {'cal':>5} | corrected miss median deg (broadband / tones / all)")
    print("-" * 74)
    for s in summary["by_position"].values():
        bb, tn, c = s["corrected_broadband"], s["corrected_tones"], s["corrected"]
        print(f"{s['label']:>16} | {str(s['in_calibration']):>5} | "
              f"{bb['median']!s:>7}({bb['n']:>3}) / {tn['median']!s:>6} / {c['median']!s:>6}")
    print("\nPer sound at calibration positions:  corrected miss  |  within-position precision")
    prec = summary["within_position_precision_by_stimulus"]
    for k, v in summary["corrected_miss_by_stimulus"].items():
        print(f"   {k:>16}: {v['median']:>6} deg     |  {prec.get(k, '-')!s:>6} deg   (n={v['n']})")
    print("=" * 74)


GLOBAL_OUTPUTS = {"all_trials.csv", "summary.json", "report.md", "stimulus_classification.json"}


def _prune_stale_artifacts(current_tags: set, results_dir: Path, figures_dir: Path) -> None:
    """Delete per-position cache/figure files whose tag no longer matches a current folder, so the
    notebook and report never display orphans from an earlier run or tagging scheme. Global
    deliverables are never touched. `_meta.json` is an old-scheme orphan suffix kept for cleanup."""
    patterns = [(results_dir, ["_align.npz", "_align.json", "_trials.csv", "_quality.json",
                               "_meta.json"]),
                (figures_dir, ["_spectrograms.png", "_onsets.png"])]
    for d, suffixes in patterns:
        for p in d.glob("*"):
            if p.name in GLOBAL_OUTPUTS:
                continue
            for suf in suffixes:
                prefix = p.name[:-len(suf)]
                if p.name.endswith(suf) and prefix and prefix not in current_tags:
                    p.unlink()


def _write_master_csv(path: Path, rows: list[dict]) -> None:
    import csv
    cols = ["folder", "position", "trial", "block", "stim_index", "stim_type", "stim_label",
            "stim_hz", "truth_az", "truth_el", "radius_m", "onset_s", "inband_snr_db",
            "reliable", "in_calibration",
            "doa_az", "doa_el", "doa_az_cal", "doa_el_cal", "miss_raw", "miss_corrected"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def load_condition(name: str, conditions_file: Path | str = "conditions.json") -> dict:
    """Resolve a named condition from the registry into `run()` keyword arguments.

    External calibration ('mode': 'external') reads the mount already fit for the source
    condition; run that source first."""
    cf = Path(conditions_file)
    if not cf.is_absolute():
        cf = EXP_DIR / cf
    reg = json.loads(cf.read_text(encoding="utf-8"))
    conds = reg["conditions"]
    if name not in conds:
        raise SystemExit(f"Unknown condition {name!r}; available: {', '.join(sorted(conds))}")
    c = conds[name]
    if not c.get("enabled", True):
        raise SystemExit(f"Condition {name!r} is disabled in {cf.name} "
                         "(no data recorded yet?). Flip 'enabled' to true once it exists.")

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else (EXP_DIR / p)

    kwargs = {"data_root": _resolve(c["data_root"]),
              "results_dir": _resolve(c["results_dir"]),
              "figures_dir": _resolve(c["figures_dir"]),
              "duplicates": c.get("duplicates", "latest")}
    if c.get("only"):
        kwargs["only"] = c["only"]
    if c.get("exclude"):
        kwargs["exclude"] = c["exclude"]
    if c.get("stim_json"):
        kwargs["stim_classification"] = json.loads(_resolve(c["stim_json"]).read_text(encoding="utf-8"))
    cal = c.get("calibration", {})
    kwargs["mount_model"] = cal.get("mount_model", "auto")
    kwargs["per_session"] = cal.get("per_session", False)
    if cal.get("mode") == "external":
        src = conds[cal["source"]]
        src_summary = _resolve(src["results_dir"]) / "summary.json"
        if not src_summary.exists():
            raise SystemExit(f"Condition {name!r} needs the mount from {cal['source']!r}; "
                             f"run that condition first ({src_summary} missing)")
        kwargs["external_calibration"] = json.loads(src_summary.read_text(encoding="utf-8"))["mounting"]
    else:
        kwargs["calibration_label"] = cal.get("label")
    return kwargs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="recompute, bypass cache")
    ap.add_argument("--no-figures", action="store_true", help="skip figure generation")
    ap.add_argument("--condition", default=None,
                    help="run a named condition from the registry (data root, output dirs, "
                         "stimulus labels and calibration all come from conditions.json)")
    ap.add_argument("--conditions-file", default="conditions.json",
                    help="condition registry used by --condition")
    ap.add_argument("--only", default=None,
                    help="restrict to Subject_* folders whose name contains this substring "
                         "(smoke tests, e.g. --only 'Subject_30,25')")
    ap.add_argument("--mount-model", default=None, choices=["offset", "rotation", "auto"],
                    help="Zylia mount model (default: condition setting, else auto)")
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="folder containing the Subject_* position folders")
    ap.add_argument("--results-dir", default=str(RESULTS), help="where to write results")
    ap.add_argument("--figures-dir", default=str(FIGURES), help="where to write figures")
    ap.add_argument("--calibration-json", default=None,
                    help="apply the Zylia mounting from this summary.json instead of re-fitting "
                         "(use the ground-truth summary when scoring virtual sources)")
    ap.add_argument("--stim-json", default=None,
                    help="use the stimulus labels from this stimulus_classification.json instead of "
                         "reconciling (use the ground truth's so identical source material matches)")
    ap.add_argument("--calibration-label", default=None,
                    help="fit the Zylia mounting only on this stimulus label (e.g. 'pink noise'), the "
                         "clean wideband reference, instead of all broadband-typed sounds")
    args = ap.parse_args()

    if args.condition:
        kwargs = load_condition(args.condition, args.conditions_file)
        if args.mount_model:
            kwargs["mount_model"] = args.mount_model
        cond_only = kwargs.pop("only", None)
        if args.only and cond_only:
            only = (cond_only if isinstance(cond_only, list) else [cond_only]) + [args.only]
        else:
            only = args.only or cond_only
        run(force=args.force, figures=not args.no_figures, only=only,
            exclude=kwargs.pop("exclude", None), **kwargs)
        return

    cal = None
    if args.calibration_json:
        cal = json.loads(Path(args.calibration_json).read_text(encoding="utf-8"))["mounting"]
    stim = None
    if args.stim_json:
        stim = json.loads(Path(args.stim_json).read_text(encoding="utf-8"))
    run(force=args.force, figures=not args.no_figures, data_root=Path(args.data_root),
        results_dir=Path(args.results_dir), figures_dir=Path(args.figures_dir),
        external_calibration=cal, stim_classification=stim,
        calibration_label=args.calibration_label,
        mount_model=args.mount_model or "auto", only=args.only)


if __name__ == "__main__":
    main()
