"""Build the AES-paper comparison figures from existing comparison outputs.

Reads results/comparisons/<name>/comparison.json and the per-condition
all_trials.csv (no re-computation), and writes print-safe versions into
Paper/figures/: series are distinguished by hatching / line style / marker,
not by color alone, so the figures survive black-and-white printing
(AES convention guideline). Pipeline figures under figures/comparisons/
are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_conditions import load_condition_results, _cells, _usable, _load_predictions
from zyldoa import stats

EXP_DIR = Path(__file__).resolve().parent
PAPER_FIGS = EXP_DIR.parent / "Paper" / "figures"

ORANGE, BLUE, GRAY = "#E8873C", "#5B7FA5", "#6a6f76"
EDGE = "#2b2b2b"


def _err(stat_d):
    ci = stat_d.get("ci95")
    if not ci or ci[0] is None or stat_d["median"] is None:
        return [0.0, 0.0]
    return [max(0.0, stat_d["median"] - ci[0]), max(0.0, ci[1] - stat_d["median"])]


def bars(comp_name: str, label_a: str, label_b: str, out_png: Path) -> None:
    comp = json.loads((EXP_DIR / "results" / "comparisons" / comp_name /
                       "comparison.json").read_text(encoding="utf-8"))
    la, lb = comp["a"], comp["b"]
    labs = list(comp["per_stimulus"].keys())
    ma = [comp["per_stimulus"][s][la]["median"] for s in labs]
    mb = [comp["per_stimulus"][s][lb]["median"] for s in labs]
    ea = np.array([_err(comp["per_stimulus"][s][la]) for s in labs]).T
    eb = np.array([_err(comp["per_stimulus"][s][lb]) for s in labs]).T
    x = np.arange(len(labs))
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.bar(x - 0.2, ma, 0.38, yerr=ea, capsize=3, label=label_a,
           facecolor=ORANGE, edgecolor=EDGE, linewidth=0.7, hatch="///")
    ax.bar(x + 0.2, mb, 0.38, yerr=eb, capsize=3, label=label_b,
           facecolor=BLUE, edgecolor=EDGE, linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("median corrected miss (deg)", fontsize=11)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(frameon=False, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"wrote {out_png}")


def azimuth_profile(comp_name: str, label_a: str, label_b: str, out_png: Path) -> None:
    comp = json.loads((EXP_DIR / "results" / "comparisons" / comp_name /
                       "comparison.json").read_text(encoding="utf-8"))
    reg = json.loads((EXP_DIR / "conditions.json").read_text(encoding="utf-8"))
    spec = next(c for c in reg["comparisons"] if c["name"] == comp_name)
    df_a, _ = load_condition_results(EXP_DIR / reg["conditions"][comp["a"]]["results_dir"])
    df_b, _ = load_condition_results(EXP_DIR / reg["conditions"][comp["b"]]["results_dir"])
    pred = None
    if spec.get("predictions"):
        pred = _load_predictions(EXP_DIR / spec["predictions"])

    a = _cells(_usable(df_a))
    b = _cells(_usable(df_b))
    els = comp["shared_elevations"]
    fig, axes = plt.subplots(1, len(els), figsize=(4.2 * len(els), 4.0),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    series = ((a, dict(color=ORANGE, lw=1.8, ls="-", marker="o", ms=3.5), label_a),
              (b, dict(color=BLUE, lw=1.8, ls="-.", marker="s", ms=3.2), label_b))
    for ax, el in zip(axes[0], els):
        for df, style, lab in series:
            sub = df[df["stim_label"].isin(stats.NOISE_LIKE) & (df["cell_el"] == el)]
            med = sub.groupby("cell_az")["miss_corrected"].median().sort_index()
            if len(med):
                th = np.radians(np.append(med.index.values, med.index.values[0]))
                rr = np.append(med.values, med.values[0])
                ax.plot(th, rr, label=lab, **style)
        if pred is not None:
            p = pred[(pred["cell_el"] == el) & pred["covered"]].sort_values("cell_az")
            if len(p):
                th = np.radians(np.append(p["cell_az"].values, p["cell_az"].values[0]))
                rr = np.append(p["rE_err_deg"].values, p["rE_err_deg"].values[0])
                ax.plot(th, rr, color=GRAY, lw=1.3, ls="--", label="rE prediction")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(f"elevation {el:+d}\N{DEGREE SIGN}", fontsize=12)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.06), fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")


import csv as _csv
from matplotlib import gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch

SPEAKERS_CSV = EXP_DIR.parent / "AvLocalization3D" / "Assets" / "CaveSpeakerPositions.csv"
NOISE_LIKE = ("pink noise", "applause")
PAPER_RC = {"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
            "xtick.labelsize": 8, "ytick.labelsize": 8}


def _comp(name):
    return json.loads((EXP_DIR / "results" / "comparisons" / name /
                       "comparison.json").read_text(encoding="utf-8"))


def _speaker_dirs():
    rows = list(_csv.DictReader(open(SPEAKERS_CSV)))
    az = [float(r["AZ_Degrees"]) % 360 for r in rows]
    el = [float(r["EL_Degrees"]) for r in rows]
    return az, el


def content_ladder(out_png: Path) -> None:
    """Fig 3: three-rung slopegraph, anechoic physical -> CAVE physical -> CAVE phantom."""
    room = _comp("room_effect")["per_stimulus"]
    vg = _comp("vbap_gtmount_vs_gt")["per_stimulus"]

    def node(src, stim, cond):
        d = src.get(stim, {}).get(cond)
        return None if not d or d.get("median") is None else (d["median"], d["ci95"])

    # style: class = line style, member = marker; Okabe-Ito color layer on top
    S = {
        "pink noise":    dict(ls="-",  lw=1.2, m="o", fill=True,  c="#0072B2", lab="pink noise",    dy=18.5),
        "applause":      dict(ls="-",  lw=1.2, m="s", fill=True,  c="#0072B2", lab="applause",      dy=13.5),
        "250 Hz tone":   dict(ls=(0, (4, 1.5)), lw=1.2, m="^", fill=True, c="#D55E00", lab="250 Hz tone", dy=56),
        "1000 Hz tone":  dict(ls=(0, (4, 1.5)), lw=1.2, m="D", fill=True, c="#D55E00", lab="1 kHz tone",  dy=42),
        "male speech":   dict(ls="-.", lw=1.1, m="o", fill=False, c="#009E73", lab="male speech", dy=28),
        "female speech": dict(ls="-.", lw=1.1, m="s", fill=False, c="#009E73", lab="female speech", dy=21),
        "phone ringing": dict(ls=":",  lw=1.1, m="D", fill=False, c="#777777", lab="phone ringing", dy=33),
    }

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.35, 3.4))
        ax.axhspan(2, 6, color="0.94", zorder=0)
        ax.text(2.55, 2.3, "anechoic range 2–6°", fontsize=6.5, color="0.45",
                ha="center", va="center")
        for stim, st in S.items():
            rungs = [node(room, stim, "gt_anechoic"),
                     node(vg, stim, "gt_cave") or node(room, stim, "gt_cave"),
                     node(vg, stim, "vbap_gtmount")]
            xs = [i for i, r in enumerate(rungs) if r]
            ys = [rungs[i][0] for i in xs]
            mfc = st["c"] if st["fill"] else "white"
            ax.plot(xs, ys, ls=st["ls"], lw=st["lw"], color=st["c"], zorder=3)
            for x in xs:
                lo, hi = rungs[x][1]
                if lo is not None:
                    ax.vlines(x, lo, hi, lw=0.8, color=st["c"], zorder=3)
            ax.plot(xs, ys, ls="none", marker=st["m"], ms=4.5, mfc=mfc,
                    mec=st["c"], mew=0.7, zorder=4)
            ax.text(xs[-1] + 0.08, st["dy"], st["lab"], fontsize=7,
                    color=st["c"], va="center")
        for x, y, t in ((0.5, 12, "tones ×4–7.5"), (0.5, 3.05, "broadband ×2"),
                        (1.5, 8.5, "broadband ×3–3.5"), (1.5, 38, "tones ×1.6–2.1")):
            ax.text(x, y, t, fontsize=7, style="italic", color="0.35",
                    ha="center", va="center", zorder=5,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
        ax.set_yscale("log")
        ax.set_ylim(1.6, 62)
        ax.set_yticks([2, 5, 10, 20, 50])
        ax.set_yticklabels(["2", "5", "10", "20", "50"])
        ax.minorticks_off()
        ax.set_ylabel("median angular miss (°)")
        ax.set_xlim(-0.15, 3.05)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Anechoic\nphysical", "CAVE\nphysical", "CAVE\nphantom (VBAP)"])
        ax.grid(axis="y", which="major", lw=0.4, color="0.90")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.4)
        fig.savefig(out_png, dpi=600)
        plt.close(fig)
    print(f"wrote {out_png}")


def miss_maps(out_png: Path) -> None:
    """Fig 4: stacked physical/phantom per-cell miss maps with speaker positions."""
    from zyldoa import stats as zstats
    reg = json.loads((EXP_DIR / "conditions.json").read_text(encoding="utf-8"))

    def cellmap(cond):
        df, _ = load_condition_results(EXP_DIR / reg["conditions"][cond]["results_dir"])
        d = _cells(_usable(df))
        d = d[d["stim_label"].isin(NOISE_LIKE)]
        return d.groupby(["cell_az", "cell_el"])["miss_corrected"].median()

    maps = {"Physical loudspeaker (CAVE)": cellmap("gt_cave"),
            "Optimized-VBAP phantom (CAVE)": cellmap("vbap_gtmount")}
    spk_az, spk_el = _speaker_dirs()
    vmax = 40.0
    cmap = plt.get_cmap("viridis")

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
        for ax, (title, cm) in zip(axes, maps.items()):
            for az in range(0, 360, 15):
                for el in (-25, 0, 25):
                    v = cm.get((az, el))
                    if v is None:
                        ax.add_patch(Rectangle((az - 7.5, el - 8), 15, 16, fc="white",
                                               ec="0.8", lw=0.4, hatch="///", zorder=1))
                        continue
                    ax.add_patch(Rectangle((az - 7.5, el - 8), 15, 16,
                                           fc=cmap(min(v, vmax) / vmax),
                                           ec="white", lw=0.5, zorder=1))
                    ax.text(az, el, f"{v:.0f}", fontsize=6, ha="center", va="center",
                            color="white" if v > 0.55 * vmax else "black", zorder=2)
            ax.scatter(spk_az, spk_el, s=25, fc="white", ec="black", lw=0.9, zorder=3)
            ax.set_xlim(-7.5, 352.5)
            ax.set_ylim(-40, 70)
            ax.set_yticks([-25, 0, 25, 50])
            ax.set_ylabel("elevation (°)")
            ax.text(0.006, 0.97, title, transform=ax.transAxes, fontsize=7.5,
                    fontweight="bold", va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
        hi_i = max(range(len(spk_el)), key=lambda i: spk_el[i])
        axes[0].annotate("loudspeakers (n=24)", xy=(spk_az[hi_i], spk_el[hi_i]),
                         xytext=(spk_az[hi_i] - 115, 57), fontsize=7,
                         arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4"))
        axes[1].text(2, -38.5, "no speakers below −23°", fontsize=6.5, color="0.35")
        axes[1].set_xticks(range(0, 360, 30))
        axes[1].set_xlabel("true azimuth (°)")
        fig.subplots_adjust(left=0.075, right=0.905, top=0.985, bottom=0.115, hspace=0.12)
        cax = fig.add_axes([0.918, 0.115, 0.014, 0.87])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
        cb = fig.colorbar(sm, cax=cax, ticks=[0, 10, 20, 30, 40])
        cb.set_label("median miss (°)", fontsize=7.5)
        cb.ax.set_yticklabels(["0", "10", "20", "30", "≥40"], fontsize=7)
        fig.savefig(out_png, dpi=600)
        plt.close(fig)
    print(f"wrote {out_png}")


def elevation_opposition(out_png: Path) -> None:
    """Fig 5: opposite elevation gradients (a) and the upward pull transfer (b)."""
    reg = json.loads((EXP_DIR / "conditions.json").read_text(encoding="utf-8"))
    room_el = _comp("room_effect")["noise_like_by_elevation"]
    rend_el = _comp("vbap_gtmount_vs_gt")["noise_like_by_elevation"]

    def transfer(cond):
        df, _ = load_condition_results(EXP_DIR / reg["conditions"][cond]["results_dir"])
        d = _usable(df)
        d = d[d["stim_label"].isin(NOISE_LIKE)]
        g = d.groupby(d["truth_el"].round().astype(int))["doa_el_cal"].median()
        return [float(g.get(el)) for el in (-25, 0, 25)]

    phys, phan = transfer("gt_cave"), transfer("vbap_gtmount")
    _, spk_el = _speaker_dirs()
    spk_mean = sum(spk_el) / len(spk_el)

    with plt.rc_context(PAPER_RC):
        fig = plt.figure(figsize=(3.35, 4.4))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.6], hspace=0.52,
                               left=0.16, right=0.97, top=0.93, bottom=0.09)

        # (a) horizontal grouped bars
        axa = fig.add_subplot(gs[0])
        ypos = {25: 2, 0: 1, -25: 0}
        for el_s, dd in room_el.items():
            el = int(el_s)
            d, ci = dd["delta"]["delta"], dd["delta"]["ci95"]
            axa.barh(ypos[el] + 0.18, d, height=0.32, fc="white", ec="black",
                     lw=0.8, hatch="///",
                     xerr=[[max(0, d - ci[0])], [max(0, ci[1] - d)]],
                     error_kw=dict(lw=0.8, capsize=2),
                     label="room (CAVE physical vs anechoic)" if el == 25 else None)
            axa.text(ci[1] + 0.7, ypos[el] + 0.18, f"+{d:.1f}", fontsize=6.5, va="center")
        for el_s, dd in rend_el.items():
            el = int(el_s)
            d, ci = dd["delta"]["delta"], dd["delta"]["ci95"]
            axa.barh(ypos[el] - 0.18, d, height=0.32, fc="#D55E00", ec="black", lw=0.5,
                     xerr=[[max(0, d - ci[0])], [max(0, ci[1] - d)]],
                     error_kw=dict(lw=0.8, capsize=2),
                     label="rendering (phantom vs physical)" if el == 25 else None)
            axa.text(ci[1] + 0.7, ypos[el] - 0.18, f"+{d:.1f}", fontsize=6.5, va="center")
        axa.text(9.3, ypos[25] + 0.18, "room hurts up-targets", fontsize=7,
                 style="italic", color="0.35", va="center")
        axa.text(12.0, ypos[-25] + 0.34, "rendering hurts down-targets", fontsize=7,
                 style="italic", color="0.35", va="center")
        axa.set_yticks(list(ypos.values()))
        axa.set_yticklabels([f"{k:+d}°" for k in ypos])
        axa.set_ylabel("target elevation")
        axa.set_xlim(0, 31.5)
        axa.set_xticks([0, 10, 20])
        axa.set_xlabel("added median error (°)")
        axa.grid(axis="x", lw=0.4, color="0.85")
        axa.set_axisbelow(True)
        axa.spines[["top", "right"]].set_visible(False)
        axa.legend(loc="lower left", bbox_to_anchor=(0.16, 1.01), frameon=False,
                   fontsize=6.5, handlelength=1.4, borderpad=0)
        axa.text(-0.26, 1.02, "(a)", transform=axa.transAxes, fontsize=9,
                 fontweight="bold")

        # (b) elevation transfer + speaker strip
        gsb = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],
                                               width_ratios=[1, 0.09], wspace=0.06)
        axb = fig.add_subplot(gsb[0])
        strip = fig.add_subplot(gsb[1], sharey=axb)
        xs = [-25, 0, 25]
        axb.plot([-33, 33], [-33, 33], ls=":", color="0.6", lw=0.8)
        axb.text(27.0, 16.5, "measured = true", fontsize=6.5, color="0.5",
                 rotation=38, ha="center")
        axb.plot(xs, phys, ls="-", lw=1.2, color="black", marker="s", ms=5,
                 mfc="white", mew=0.9)
        axb.plot(xs, phan, ls="-.", lw=1.6, color="black", marker="o", ms=5)
        axb.text(-31, -30.5, "physical speaker", fontsize=6.5, va="top")
        axb.text(-31, 7.5, "phantom (VBAP)", fontsize=6.5)
        for x, py, lab, ly in ((-25, phan[0], f"+{phan[0] + 25:.0f}°", -14),
                               (0, phan[1], f"+{phan[1]:.0f}°", 6)):
            axb.add_patch(FancyArrowPatch((x + 2.0, x), (x + 2.0, py),
                                          arrowstyle="-|>", mutation_scale=7,
                                          lw=1.0, color="0.25"))
            axb.text(x + 3.6, ly, lab, fontsize=6.5, color="0.25")
        axb.axhline(spk_mean, ls=":", lw=0.8, color="0.45")
        axb.text(-31, spk_mean + 1.5, f"speaker mean +{spk_mean:.1f}°",
                 fontsize=6.5, color="0.45")
        axb.set_xlim(-33, 33)
        axb.set_ylim(-33, 68)
        axb.set_xticks(xs)
        axb.set_yticks([-25, 0, 25, 50])
        axb.set_xlabel("true elevation (°)")
        axb.set_ylabel("measured elevation (°)")
        axb.spines[["top", "right"]].set_visible(False)
        axb.set_title("(b)", loc="left", fontsize=9, fontweight="bold")

        rng = np.random.default_rng(7)
        strip.scatter(rng.uniform(0.3, 0.7, len(spk_el)), spk_el, marker="<", s=9,
                      color="black")
        strip.plot([0.15, 0.85], [spk_mean, spk_mean], lw=1.6, color="black")
        strip.set_xlim(0, 1)
        strip.set_xticks([])
        strip.tick_params(labelleft=False, left=False)
        for sp in ("top", "right", "left", "bottom"):
            strip.spines[sp].set_visible(False)
        strip.set_title("speakers", fontsize=6.5, rotation=0, pad=2)

        fig.savefig(out_png, dpi=600)
        plt.close(fig)
    print(f"wrote {out_png}")


def build_all() -> list:
    """Build every publication figure into Paper/figures/; returns the paths."""
    outs = [PAPER_FIGS / "content-ladder.png",
            PAPER_FIGS / "miss-maps-physical-vs-phantom.png",
            PAPER_FIGS / "elevation-opposition.png"]
    content_ladder(outs[0])
    miss_maps(outs[1])
    elevation_opposition(outs[2])
    return outs


if __name__ == "__main__":
    build_all()
