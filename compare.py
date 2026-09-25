"""Combine the ground-truth and virtual-source localization results and the speaker-array
optimization into one comparison + figures.

Three measurements, one fixed Zylia ZM-1:
  - GROUND TRUTH: a real loudspeaker moved around the listener (full azimuth circle).
  - VIRTUAL: a VBAP phantom panned over the installed 24-speaker array to the same directions,
    scored with the SAME mounting (fit on the ground-truth pink noise), so the only added error is
    the array's spatial reproduction.
  - OPTIMIZATION: the VBAP renderer's predicted rV/rE angular error at those directions.

The stimuli are SEMANTIC (pink noise, tones, phone ringing, applause, speech); pink noise is the
clean wideband reference used for the mount and for the broadband-vs-prediction comparison.
Outputs results/comparison.json and figures/compare_*.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
RESULTS = EXP_DIR / "results"
FIGURES = EXP_DIR / "figures"
BLUE, ORANGE, GREEN, GRAY = "#5B7FA5", "#E8873C", "#4C9A6B", "#9aa0a6"

# The mount is fit on pink noise (cleanest single wideband reference). For the broadband category in
# the comparisons we POOL pink noise + applause ("noise-like"): the two non-tonal real sounds agree
# closely (GT 6.0 vs 6.6 deg; virtual 12.2 vs 12.0) and together give full azimuth coverage where
# pink noise alone has SNR gaps.
NOISE_LIKE = ["pink noise", "applause"]
CAL_REF = "pink noise"
# display order: noise-like, phone, tones, speech (6 kHz tone excluded — no reliable trials)
STIM_ORDER = ["pink noise", "applause", "phone ringing", "250 Hz tone", "1000 Hz tone",
              "male speech", "female speech"]


def _nl(d: pd.DataFrame) -> pd.DataFrame:
    return d[d["stim_label"].isin(NOISE_LIKE)]


def _circmedian(deg) -> float:
    a = np.radians(np.asarray(deg, float))
    c = np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    rel = (np.asarray(deg, float) - c + 180) % 360 - 180
    return float((c + np.median(rel) + 180) % 360 - 180)


def _order(labels):
    return [s for s in STIM_ORDER if s in labels] + [s for s in labels if s not in STIM_ORDER]


def load_condition(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(results_dir / "all_trials.csv")
    for c in ("reliable", "in_calibration"):
        if df[c].dtype == object:
            df[c] = df[c].map(lambda x: str(x).strip().lower() == "true")
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    return df, summary


def is_tone(s: pd.Series) -> pd.Series:
    return s.isin(["tone", "complex_tone"])


def _boot_ci_median(x, n_boot: int = 2000, seed: int = 12345):
    """Percentile bootstrap 95% CI for the median (deterministic seed). None if n<3."""
    x = np.asarray(x, float)
    if len(x) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    return [round(float(np.percentile(meds, 2.5)), 1), round(float(np.percentile(meds, 97.5)), 1)]


def _boot_ci_delta(v, g, n_boot: int = 2000, seed: int = 777):
    """Bootstrap 95% CI for (median(v) - median(g)) by independently resampling each group."""
    v = np.asarray(v, float); g = np.asarray(g, float)
    if len(v) < 3 or len(g) < 3:
        return [None, None]
    rng = np.random.default_rng(seed)
    dv = np.median(v[rng.integers(0, len(v), size=(n_boot, len(v)))], axis=1)
    dg = np.median(g[rng.integers(0, len(g), size=(n_boot, len(g)))], axis=1)
    d = dv - dg
    return [round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]


def _stat(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)),
            "median": round(float(np.median(x)), 2) if len(x) else None,
            "p90": round(float(np.percentile(x, 90)), 2) if len(x) else None,
            "ci95": _boot_ci_median(x)}


def summarize_condition(df: pd.DataFrame) -> dict:
    d = df[df["reliable"]]
    ref = _nl(d)
    by_stim = {lab: _stat(g["miss_corrected"]) for lab, g in d.groupby("stim_label")}
    ref_by_az = {float(a): round(float(np.median(g["miss_corrected"])), 2)
                 for a, g in ref.groupby("truth_az")}
    return {"reference_stim": "noise-like (pink noise + applause)",
            "reference": _stat(ref["miss_corrected"]),
            "reference_miss_by_az": ref_by_az,
            "by_stimulus": by_stim}


def compare(gt_dir: Path, virt_dir: Path, optim_json: Path, out_results: Path,
            out_figures: Path) -> dict:
    out_results.mkdir(parents=True, exist_ok=True); out_figures.mkdir(parents=True, exist_ok=True)
    gt, gt_sum = load_condition(gt_dir)
    vt, vt_sum = load_condition(virt_dir)
    pred = json.loads(Path(optim_json).read_text(encoding="utf-8"))

    gt_s = summarize_condition(gt)
    vt_s = summarize_condition(vt)

    # Matched-position virtual-vs-GT, per stimulus (azimuths present in virtual).
    virt_az = sorted(vt["truth_az"].unique().tolist())
    matched = []
    labels = _order(set(vt["stim_label"]) | set(gt["stim_label"]))
    for az in virt_az:
        for lab in labels:
            gsel = gt[(gt.truth_az == az) & (gt.stim_label == lab) & gt.reliable]
            vsel = vt[(vt.truth_az == az) & (vt.stim_label == lab) & vt.reliable]
            if len(gsel) == 0 and len(vsel) == 0:
                continue
            gm = float(np.median(gsel["miss_corrected"])) if len(gsel) else None
            vm = float(np.median(vsel["miss_corrected"])) if len(vsel) else None
            dci = (_boot_ci_delta(vsel["miss_corrected"], gsel["miss_corrected"])
                   if (len(gsel) and len(vsel)) else [None, None])
            matched.append({"az": float(az), "stim": lab,
                            "gt_miss": round(gm, 2) if gm is not None else None,
                            "virtual_miss": round(vm, 2) if vm is not None else None,
                            "delta": round(vm - gm, 2) if (gm is not None and vm is not None) else None,
                            "delta_ci95": dci,
                            "delta_significant": bool(dci[0] is not None and (dci[0] > 0 or dci[1] < 0)),
                            "gt_n": int(len(gsel)), "virtual_n": int(len(vsel))})

    # Matched-azimuth noise-like baseline: the fair broadband comparison for the 3 front virtual az.
    def ref_by_az(df, az):
        g = _nl(df[(df.truth_az == az) & df.reliable])
        return round(float(np.median(g["miss_corrected"])), 2) if len(g) else None
    gt_ref_matched = {float(az): ref_by_az(gt, az) for az in virt_az}
    vt_ref_matched = {float(az): ref_by_az(vt, az) for az in virt_az}
    both = [az for az in virt_az if gt_ref_matched[az] is not None and vt_ref_matched[az] is not None]
    def _ratio(azs):
        if not azs:
            return None
        gm = float(np.mean([gt_ref_matched[a] for a in azs]))
        vm = float(np.mean([vt_ref_matched[a] for a in azs]))
        return round(vm / gm, 2) if gm else None
    both_excl0 = [a for a in both if a != 0.0]

    # Optimization / VBAP prediction at the matched azimuths vs the measured noise-like virtual miss.
    dep_by_az = {float(e["az"]): e for e in pred["deployed"]["experiment_directions"]}
    opt_by_az = {float(e["az"]): e for e in pred["optimized"]["experiment_directions"]}
    optim_match = []
    for az in virt_az:
        dp, op = dep_by_az.get(float(az)), opt_by_az.get(float(az))
        optim_match.append({"az": float(az),
                            "measured_virtual_noiselike_miss": vt_ref_matched[az],
                            "deployed_rE_err": dp["energy_err_deg"] if dp else None,
                            "deployed_rV_err": dp["loc_error_deg"] if dp else None,
                            "optimized_rE_err": op["energy_err_deg"] if op else None})

    result = {
        "reference_stim": "noise-like (pink noise + applause)", "calibration_stim": CAL_REF,
        "ground_truth": {"mounting": gt_sum["mounting"],
                         "mounting_description": gt_sum["mounting_description"],
                         "calibration_inliers_az": gt_sum["calibration_inliers_az"],
                         "calibration_outliers_az": gt_sum["calibration_outliers_az"],
                         "calibration_warnings": gt_sum["calibration_warnings"],
                         "summary": gt_s},
        "virtual": {"mounting_applied": vt_sum["mounting"], "summary": vt_s},
        "matched_baseline": {
            "virtual_azimuths": virt_az, "matched_azimuths": both,
            "gt_reference_by_az": gt_ref_matched, "virtual_reference_by_az": vt_ref_matched,
            "virtual_reference_overall": vt_s["reference"]["median"],
            "full_circle_gt_reference_median": gt_s["reference"]["median"],
            "ratio_all_matched_az": _ratio(both),
            "ratio_excl_0deg": _ratio(both_excl0)},
        "virtual_vs_ground_truth": matched,
        "optimization_vs_measured": optim_match,
        "virtual_mount_consistency": _mount_consistency(gt_sum, virt_dir),
        "unreliable_in_virtual": _unreliable(vt),
    }
    (out_results / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    _fig_gt_circle(gt, out_figures / "compare_gt_accuracy_circle.png")
    _fig_stim_bar(gt, out_figures / "compare_gt_miss_by_sound.png")
    _fig_virtual_vs_gt(matched, out_figures / "compare_virtual_vs_gt.png")
    _fig_optim_vs_measured(pred, gt_s, vt_s, out_figures / "compare_optim_vs_measured.png")
    return result


def _unreliable(vt: pd.DataFrame) -> dict:
    """Stimuli with no reliable trials in the virtual condition (so they drop out of comparison)."""
    out = {}
    for lab, g in vt.groupby("stim_label"):
        rel = int(g["reliable"].sum())
        if rel == 0:
            out[lab] = {"total": int(len(g)), "reliable": 0}
    return out


def _mount_consistency(gt_sum: dict, virt_dir: Path) -> dict:
    """Did the Zylia move between sessions? Compare the GT mounting to a mounting re-fit on the
    virtual data alone (all broadband sounds; results/virtual_selfcal/summary.json). The 2 usable
    front positions make this fit weak; the primary evidence for 'no azimuth rotation' is that the
    GT mount applied to virtual yields coherent (~12 deg), not absurd, misses."""
    vfit = virt_dir.parent / "virtual_selfcal" / "summary.json"
    out = {"gt_mounting": gt_sum["mounting"], "virtual_selffit_mounting": None,
           "virtual_selffit_inliers_az": None, "virtual_selffit_outliers_az": None}
    if vfit.exists():
        s = json.loads(vfit.read_text(encoding="utf-8"))
        vm = s.get("mounting", {})
        out["virtual_selffit_mounting"] = vm
        out["virtual_selffit_inliers_az"] = s.get("calibration_inliers_az")
        out["virtual_selffit_outliers_az"] = s.get("calibration_outliers_az")
        gm = gt_sum["mounting"]
        out["az_offset_diff_deg"] = round(abs(gm["az_offset"] - vm.get("az_offset", gm["az_offset"])), 2)
        out["el_offset_diff_deg"] = round(abs(gm["el_offset"] - vm.get("el_offset", gm["el_offset"])), 2)
        out["handedness_matches"] = (gm["az_sign"] == vm.get("az_sign"))
    return out


def _fig_gt_circle(gt: pd.DataFrame, out: Path) -> None:
    """Ground-truth miss by azimuth: pink noise (clean wideband reference) vs the pure tones."""
    d = gt[(gt.reliable) & (gt.truth_el == 0)].copy()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    series = [(_nl(d), BLUE, "noise-like (pink noise + applause)"),
              (d[is_tone(d.stim_type)], ORANGE, "pure tones")]
    for sub, color, lab in series:
        if len(sub) == 0:
            continue
        agg = sub.groupby("truth_az")["miss_corrected"].agg(["median", "count"])
        agg = agg[agg["count"] >= 5]
        if len(agg) == 0:
            continue
        th = np.radians(agg.index.to_numpy(float)); r = agg["median"].to_numpy(float)
        th = np.append(th, th[0]); r = np.append(r, r[0])
        ax.plot(th, r, "o-", color=color, label=lab, lw=1.5, ms=4)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_title("Ground-truth miss by azimuth (mount-corrected median; n≥5/az)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.16, 1.1))
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def _fig_stim_bar(gt: pd.DataFrame, out: Path) -> None:
    """Per-sound ground-truth localization miss (median + IQR), the central result of the study."""
    d = gt[gt.reliable]
    labs = _order([l for l in d["stim_label"].unique() if l != "6000 Hz tone"])
    data = [d[d.stim_label == l]["miss_corrected"].to_numpy(float) for l in labs]
    ns = [len(x) for x in data]
    cat_color = {"pink noise": BLUE, "applause": BLUE, "phone ringing": GREEN,
                 "250 Hz tone": ORANGE, "1000 Hz tone": ORANGE,
                 "male speech": "#A05195", "female speech": "#A05195"}
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=[f"{l}\n(n={n})" for l, n in zip(labs, ns)],
                    showfliers=False, patch_artist=True, medianprops=dict(color="k"))
    for patch, l in zip(bp["boxes"], labs):
        patch.set_facecolor(cat_color.get(l, GRAY)); patch.set_alpha(0.65)
    meds = [np.median(x) for x in data]
    for i, m in enumerate(meds, 1):
        ax.text(i, m, f" {m:.0f}", va="bottom", fontsize=8)
    ax.set_ylabel("ground-truth localization miss (deg)")
    ax.set_title("How different sounds localize (real source, full azimuth circle)")
    ax.grid(axis="y", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.65) for c in [BLUE, GREEN, ORANGE, "#A05195"]]
    ax.legend(handles, ["noise-like", "phone", "pure tone", "speech"], fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def _fig_virtual_vs_gt(matched: list[dict], out: Path) -> None:
    rows = [m for m in matched if m["gt_miss"] is not None and m["virtual_miss"] is not None]
    if not rows:
        return
    azs = sorted({m["az"] for m in rows}); stims = _order({m["stim"] for m in rows})
    fig, axs = plt.subplots(1, len(azs), figsize=(4.4 * len(azs), 4.7), sharey=True, squeeze=False)
    for ax, az in zip(axs[0], azs):
        sub = {m["stim"]: m for m in rows if m["az"] == az}
        labs = [s for s in stims if s in sub]
        x = np.arange(len(labs)); w = 0.38
        ax.bar(x - w/2, [sub[s]["gt_miss"] for s in labs], w, color=BLUE, label="ground truth")
        ax.bar(x + w/2, [sub[s]["virtual_miss"] for s in labs], w, color=ORANGE, label="virtual (VBAP)")
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
        ax.set_title(f"az {az:g}°", fontsize=10); ax.grid(axis="y", alpha=0.3)
    axs[0][0].set_ylabel("median corrected miss (deg)"); axs[0][0].legend(fontsize=8)
    fig.suptitle("Virtual (VBAP) vs ground-truth miss per sound, matched positions", y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def _fig_optim_vs_measured(pred: dict, gt_s: dict, vt_s: dict, out: Path) -> None:
    sweep = pred["deployed"]["azimuth_sweep_el0"]
    az = np.array([s["az"] for s in sweep])
    rE = np.array([s["energy_err_deg"] if s["energy_err_deg"] is not None else np.nan for s in sweep])
    rV = np.array([s["loc_error_deg"] if s["loc_error_deg"] is not None else np.nan for s in sweep])
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(az, rV, rE, color=GRAY, alpha=0.22,
                    label="VBAP ideal band: rV (≡0, by construction) → rE (high-freq blur)")
    ax.plot(az, rE, "-", color=GRAY, lw=1.2, label="VBAP predicted rE error (deployed array)")
    ax.plot(az, rV, "--", color=GREEN, lw=1.2)
    vbb = vt_s["reference_miss_by_az"]
    if vbb:
        xs = sorted(float(a) for a in vbb)
        ax.plot(xs, [vbb[str(a) if str(a) in vbb else a] for a in xs], "s-", color=ORANGE, ms=10,
                label="MEASURED VIRTUAL pink-noise miss (the VBAP-predicted quantity)")
        if 0.0 in [float(a) for a in vbb]:
            y0 = list(vbb.values())[[float(a) for a in vbb].index(0.0)]
            ax.annotate("0° session\nflagged outlier", (0.0, y0), fontsize=7, color=ORANGE,
                        xytext=(20, y0 - 6), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.8))
    gbb = gt_s["reference_miss_by_az"]
    xs = sorted(float(a) for a in gbb)
    ax.plot(xs, [gbb[str(a) if str(a) in gbb else a] for a in xs], "o-", color=BLUE, ms=4, alpha=0.7,
            label="real pink-noise floor (reference only — NOT a VBAP-predicted quantity)")
    ax.set_xlabel("azimuth (deg)"); ax.set_ylabel("angular error / miss (deg)")
    ax.set_title("VBAP prediction applies to the PHANTOM (orange): measured pink-noise miss "
                 "exceeds the rE blur", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper center"); ax.grid(alpha=0.3); ax.set_xlim(-5, 360)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", default=str(RESULTS / "gt"))
    ap.add_argument("--virt-dir", default=str(RESULTS / "virtual"))
    ap.add_argument("--optim-json", default=str(RESULTS / "optim" / "predictions.json"))
    args = ap.parse_args()
    res = compare(Path(args.gt_dir), Path(args.virt_dir), Path(args.optim_json), RESULTS, FIGURES)
    print("Optimization vs measured (pink noise):")
    print(json.dumps(res["optimization_vs_measured"], indent=2))
    print("\nVirtual vs GT (delta = virtual - ground truth):")
    for m in res["virtual_vs_ground_truth"]:
        if m["delta"] is not None:
            print(f"  az={m['az']:5g} {m['stim']:16s} gt={m['gt_miss']:6} virt={m['virtual_miss']:6} "
                  f"d={m['delta']:+7} (vn={m['virtual_n']})")
    print("\nUnreliable in virtual:", res["unreliable_in_virtual"])


if __name__ == "__main__":
    main()
