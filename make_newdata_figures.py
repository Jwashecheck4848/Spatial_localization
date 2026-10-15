"""Build paper-style comparison figures for the new-data (2026-10) baseline run.

BRANCH newdata-first-pass: `make_paper_figures.py` is specific to the July
gt_cave/gt_anechoic/vbap study (its 3-rung anechoic/CAVE-physical/CAVE-phantom ladder and
24-speaker CAVE miss-map) and can't run against this isolated dataset, which has no anechoic
condition and a different physical-speaker layout. This module is the 2-stage analog:

  - content_ladder(): mono -> VBAP slopegraph across all 7 stimuli (pink noise, phonemes,
    250/1000/5000 Hz tone, phone ringing, applause), reading the comparison.json already
    written by compare_all_sounds.py.
  - miss_map(): stacked per-cell median-miss heatmaps (mono top, VBAP bottom), cell grid
    derived from the data itself rather than the old hardcoded 24-speaker layout, pooling the
    broadband-type stimuli (read from stimuli.json, not hardcoded) the way the original figure
    pooled pink noise + applause.

Prereq: compare_all_sounds.py must already have written
results/comparisons/all_sounds_mono_vs_vbap/comparison.json (run it first, or let the
pipeline notebook's Section 4 do it).

Run:  python3 make_newdata_figures.py [--out-base PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import compare_all_sounds as cas
from compare_conditions import EXP_DIR, _cells, _usable, _load_registry

PAPER_RC = {"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
            "xtick.labelsize": 8, "ytick.labelsize": 8}
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#777777", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def _broadband_labels() -> list[str]:
    """Stimulus labels whose stimuli.json `type` is broadband_noise (pink noise, phonemes,
    phone ringing, applause for this dataset) -- read from the registry, not hardcoded, so it
    stays correct if the stimulus set changes."""
    stim = json.loads((EXP_DIR / "stimuli.json").read_text(encoding="utf-8"))
    return [v["label"] for k, v in stim.items()
            if not k.startswith("_") and v.get("type") == "broadband_noise"]


def _nice_log_ticks(values: list[float]) -> list[int]:
    """Round tick values spanning `values` on a log scale (mirrors the hand-picked
    [2, 5, 10, 20, 50] ticks in the July study's content_ladder(), but computed from this
    dataset's actual range instead of hardcoded to one study's numbers)."""
    candidates = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500]
    lo, hi = min(values) / 1.3, max(values) * 1.3
    ticks = [c for c in candidates if lo <= c <= hi]
    return ticks if len(ticks) >= 3 else candidates


def content_ladder(comp: dict, out_png: Path) -> None:
    """Mono -> VBAP slopegraph across every stimulus in the comparison (2-rung analog of the
    July study's 3-rung anechoic/CAVE-physical/CAVE-phantom ladder)."""
    per_stim = comp["per_stimulus"]
    la, lb = comp["a"], comp["b"]
    labs = list(per_stim.keys())

    def node(lab, cond):
        d = per_stim[lab].get(cond)
        return None if not d or d.get("median") is None else (d["median"], d.get("ci95"))

    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        finals = []
        all_y = []
        for i, lab in enumerate(labs):
            c = PALETTE[i % len(PALETTE)]
            m = MARKERS[i % len(MARKERS)]
            rungs = [node(lab, la), node(lab, lb)]
            xs = [x for x, r in enumerate(rungs) if r]
            ys = [rungs[x][0] for x in xs]
            if not xs:
                continue
            all_y.extend(ys)
            ax.plot(xs, ys, ls="-", lw=1.2, color=c, zorder=3)
            for x in xs:
                ci = rungs[x][1]
                if ci and ci[0] is not None:
                    ax.vlines(x, ci[0], ci[1], lw=0.8, color=c, zorder=3)
                    all_y.extend(ci)
            ax.plot(xs, ys, ls="none", marker=m, ms=4.5, mfc=c, mec=c, mew=0.7, zorder=4)
            finals.append((ys[-1], lab, c))

        # end-of-line labels, nudged apart on the log scale so they don't overlap
        finals.sort()
        last_y = None
        for y, lab, c in finals:
            y_show = y if last_y is None else max(y, last_y + (max(y, last_y) * 0.08 + 0.5))
            ax.text(1.08, y_show, lab, fontsize=7, color=c, va="center")
            last_y = y_show

        ax.set_yscale("log")
        ticks = _nice_log_ticks(all_y)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_ylim(min(all_y) / 1.3, max(all_y) * 1.3)
        ax.set_ylabel("median angular miss (\N{DEGREE SIGN})")
        ax.set_xlim(-0.15, 1.75)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CAVE\nphysical (mono)", "CAVE\nphantom (VBAP)"])
        ax.grid(axis="y", which="major", lw=0.4, color="0.90")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.4)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    print(f"wrote {out_png}")


LOW_N_THRESHOLD = 3  # cells with fewer usable trials than this get flagged, not just colored


def miss_map(df_mono, df_vbap, out_png: Path, stim_filter=None, vmax: float = 40.0) -> None:
    """Stacked mono/VBAP per-cell median-miss heatmaps. The (az, el) grid is derived from the
    data itself (unique truth cells actually recorded), not assumed to match the old
    24-speaker CAVE layout.

    Cells with zero usable trials are hatched (not left blank) so "no data" is visually
    distinct from "low error". Cells are also annotated with their trial count n, and any
    cell with n < LOW_N_THRESHOLD gets a red outline -- a high median miss from 1-2 trials is
    far more likely to be one bad trial (onset slip, front-back flip) than a real effect, and
    this makes that distinction visible directly on the figure instead of hiding it."""

    def cellmap(df):
        d = _cells(_usable(df))
        if stim_filter:
            d = d[d["stim_label"].isin(stim_filter)]
        g = d.groupby(["cell_az", "cell_el"])["miss_corrected"]
        return g.median(), g.size()

    maps = {"CAVE physical (mono)": cellmap(df_mono),
            "CAVE phantom (VBAP)": cellmap(df_vbap)}
    cmap = plt.get_cmap("viridis")
    all_az = sorted({az for med, _ in maps.values() for az, _ in med.index})
    all_el = sorted({el for med, _ in maps.values() for _, el in med.index})
    if not all_az or not all_el:
        print(f"[skip] miss_map: no usable cells for {out_png.name}")
        return
    az_diffs = sorted(set(round(b - a, 3) for a, b in zip(all_az, all_az[1:])))
    az_w = max(8.0, (az_diffs[0] if az_diffs else 30.0) * 0.9)
    el_diffs = sorted(set(round(b - a, 3) for a, b in zip(all_el, all_el[1:])))
    el_h = max(10.0, (el_diffs[0] if el_diffs else 20.0) * 0.85)
    low_n_cells = []

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
        for ax, (title, (med, cnt)) in zip(axes, maps.items()):
            for az in all_az:
                for el in all_el:
                    v = med.get((az, el))
                    n = int(cnt.get((az, el), 0))
                    if v is None:
                        ax.add_patch(Rectangle((az - az_w / 2, el - el_h / 2), az_w, el_h,
                                               fc="white", ec="0.8", lw=0.4, hatch="///",
                                               zorder=1))
                        continue
                    low_n = n < LOW_N_THRESHOLD
                    if low_n:
                        low_n_cells.append((title, az, el, round(v, 1), n))
                    ax.add_patch(Rectangle((az - az_w / 2, el - el_h / 2), az_w, el_h,
                                           fc=cmap(min(v, vmax) / vmax),
                                           ec="#d62728" if low_n else "white",
                                           lw=1.3 if low_n else 0.5, zorder=1))
                    txt_color = "white" if v > 0.55 * vmax else "black"
                    ax.text(az, el + el_h * 0.12, f"{v:.0f}", fontsize=6, ha="center",
                            va="center", color=txt_color, zorder=2)
                    ax.text(az, el - el_h * 0.28, f"n={n}", fontsize=4.5, ha="center",
                            va="center", color=txt_color, alpha=0.85, zorder=2)
            ax.set_xlim(min(all_az) - az_w, max(all_az) + az_w)
            ax.set_ylim(min(all_el) - el_h, max(all_el) + el_h)
            ax.set_ylabel("elevation (\N{DEGREE SIGN})")
            ax.text(0.006, 0.97, title, transform=ax.transAxes, fontsize=7.5,
                    fontweight="bold", va="top",
                    bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))
        axes[1].set_xlabel("true azimuth (\N{DEGREE SIGN})")
        fig.subplots_adjust(left=0.08, right=0.9, top=0.985, bottom=0.115, hspace=0.12)
        cax = fig.add_axes((0.918, 0.115, 0.014, 0.87))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
        cb = fig.colorbar(sm, cax=cax, ticks=[0, 10, 20, 30, 40])
        cb.set_label("median miss (\N{DEGREE SIGN})", fontsize=7.5)
        cb.ax.set_yticklabels(["0", "10", "20", "30", "\u226540"], fontsize=7)
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
    print(f"wrote {out_png}")
    if low_n_cells:
        print(f"  [note] {len(low_n_cells)} cell(s) outlined red: n < {LOW_N_THRESHOLD} usable "
              "trial(s), treat their median as unreliable until more data comes in:")
        for title, az, el, v, n in low_n_cells:
            print(f"           {title}: az={az} el={el} -> {v}\N{DEGREE SIGN} (n={n})")


def build_all(out_base: Path | None = None) -> list[Path]:
    """Build both new-data figures into <out_base>/figures/comparisons/newdata_paper_figures/
    (or the repo root if out_base is None); returns the output paths."""
    base = Path(out_base) if out_base is not None else EXP_DIR
    comp_path = base / "results" / "comparisons" / cas.NAME / "comparison.json"
    if not comp_path.exists():
        raise SystemExit(f"{comp_path} not found -- run compare_all_sounds.main() first "
                         "(the pipeline notebook's Section 4 does this).")
    comp = json.loads(comp_path.read_text(encoding="utf-8"))

    registry = _load_registry()
    df_mono = cas._load_and_concat(registry, cas.MONO_CONDITIONS)
    df_vbap = cas._load_and_concat(registry, cas.VBAP_CONDITIONS)

    fig_dir = base / "figures" / "comparisons" / "newdata_paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    outs = [fig_dir / "content-ladder.png", fig_dir / "miss-map-mono-vs-vbap.png"]
    content_ladder(comp, outs[0])
    miss_map(df_mono, df_vbap, outs[1], stim_filter=_broadband_labels())
    return outs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-base", default=None,
                    help="read results/comparisons and write figures/comparisons under this "
                         "folder instead of the repo root")
    args = ap.parse_args()
    build_all(Path(args.out_base) if args.out_base else None)
