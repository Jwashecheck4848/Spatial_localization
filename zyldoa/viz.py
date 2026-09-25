"""Figures: per-block spectrograms, onset-alignment validation, and DOA-vs-truth / miss plots."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

from .wavio import ZyliaWav, open_recording

BLUE, ORANGE, GRAY = "#5B7FA5", "#E8873C", "#9aa0a6"


def spectrogram_montage(folder: Path, npz: dict, block_stim: dict, out_path: Path) -> None:
    """One representative-trial spectrogram per stimulus, with the placed onset marked."""
    wav = open_recording(Path(folder)); sr = wav.sample_rate
    osamp = npz["onset_samples"]
    # group by stimulus identity (stim_index) so packed multi-azimuth sessions show one
    # panel per sound, not one per azimuth block
    blocks = npz["stim_index"] if "stim_index" in getattr(npz, "files", npz) else npz["blocks"]
    ubl = np.unique(blocks); n = len(ubl)
    fig, axs = plt.subplots((n + 1) // 2, 2, figsize=(13, 2.0 * n))
    for ax, b in zip(axs.flat, ubl):
        idx = np.where(blocks == b)[0]; k = idx[len(idx) // 2]; on = int(osamp[k])
        seg = wav.read(on - int(0.2 * sr), int(1.4 * sr)).mean(axis=1)
        f, t, S = spectrogram(seg, fs=sr, nperseg=2048, noverlap=1536)
        ax.pcolormesh(t - 0.2, f, 10 * np.log10(S + 1e-12), shading="gouraud",
                      cmap="magma", vmin=-120, vmax=-40)
        ax.axvline(0, color="cyan", lw=1.2, ls="--")
        ax.set_ylim(0, 8000); ax.set_title(f"block {int(b)}: {block_stim[int(b)]['label']}", fontsize=9)
        ax.set_ylabel("Hz")
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.suptitle(f"{Path(folder).name} — stimulus spectrograms (cyan = detected onset)", y=1.0)
    fig.tight_layout(); fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)
    wav.close()


def onset_validation(folder_name: str, npz: dict, out_path: Path) -> None:
    """Envelope with onset markers over an early block and the final stretch of trials."""
    env = npz["env"]; rate = float(npz["env_rate"]); on = npz["onset_s"]
    tt = np.arange(len(env)) / rate
    edb = 20 * np.log10(env + 1e-9)
    fig, axs = plt.subplots(2, 1, figsize=(14, 5))
    tail = on[max(0, int(0.95 * len(on)))]      # last ~5% of trials, robust to trial count
    spans = [(on[0] - 2, on[0] + 25), (tail - 4, on[-1] + 6)]
    titles = ["early trials", "final trials"]
    for ax, (lo, hi), title in zip(axs, spans, titles):
        m = (tt >= lo) & (tt <= hi)
        ax.plot(tt[m], edb[m], lw=0.5, color=BLUE)
        for s in on[(on >= lo) & (on <= hi)]:
            ax.axvline(s, color=ORANGE, lw=0.8, alpha=0.8)
        ax.set_xlim(lo, hi); ax.set_xlabel("time (s)"); ax.set_ylabel("env (dB)")
        ax.set_title(title, fontsize=9)
    fig.suptitle(f"{folder_name} — onset alignment (orange = trial onset)", y=1.0)
    fig.tight_layout(); fig.savefig(out_path, dpi=110, bbox_inches="tight"); plt.close(fig)


def doa_vs_truth(trials: list[dict], out_path: Path) -> None:
    """Measured (calibrated) azimuth/elevation vs true speaker direction, colored by position."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    truths = sorted({float(t["truth_az"]) for t in trials})
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {tv: cmap[i % 10] for i, tv in enumerate(truths)}
    for ax, key, lab, unit in [(axs[0], "doa_az_cal", "azimuth", "deg"),
                               (axs[1], "doa_el_cal", "elevation", "deg")]:
        for tv in truths:
            sub = [t for t in trials if float(t["truth_az"]) == tv and t["reliable"]]
            tk = "truth_az" if key == "doa_az_cal" else "truth_el"
            x = [float(t[tk]) for t in sub]; y = [float(t[key]) for t in sub]
            ax.scatter(x, y, s=10, alpha=0.5, color=colors[tv], label=f"speaker az {tv:g} deg")
        lo, hi = (-180, 180) if key == "doa_az_cal" else (-90, 90)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.set_xlabel(f"true {lab} ({unit})"); ax.set_ylabel(f"Zylia {lab}, calibrated ({unit})")
        ax.set_title(f"{lab}: calibrated DOA vs truth", fontsize=10)
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def miss_by_stim(trials: list[dict], out_path: Path) -> None:
    """Corrected miss grouped by stimulus, for the calibration-consistent folders."""
    rel = [t for t in trials if t["reliable"] and t["in_calibration"]]
    groups = {}
    for t in rel:
        groups.setdefault(t["stim_label"], []).append(float(t["miss_corrected"]))
    labels = sorted(groups, key=lambda k: np.median(groups[k]))  # ascending by median miss
    data = [groups[k] for k in labels]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE); patch.set_alpha(0.6)
    meds = [np.median(d) for d in data]
    ax.plot(range(1, len(labels) + 1), meds, "o-", color=ORANGE, label="median miss")
    cal_az = sorted({float(t["truth_az"]) for t in rel})
    ax.set_ylabel("angular miss (deg)"); ax.set_xlabel("stimulus")
    ax.set_title(f"Calibrated localization miss by stimulus (calibration az {cal_az} deg)", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close(fig)


def miss_map(trials: list[dict], out_path: Path,
             stims=("pink noise", "applause"), vmax: float = 25.0) -> None:
    """Where the misses are. Top: azimuth x elevation heatmap of the median corrected miss.
    Bottom: per elevation, an arrow on the azimuth circle from each TRUE direction (gray dot)
    to the median MEASURED direction — the tangential component is azimuth error, the radial
    component elevation error (outward = heard too high). Noise-like stimuli by default (the
    calibrated wideband reference); tones' systematic biases get their own figures."""
    from . import stats

    def _b(x):
        return x is True or str(x).strip().lower() == "true"

    rel = [t for t in trials if _b(t["reliable"]) and _b(t["in_calibration"])
           and t["stim_label"] in stims]
    if not rel:
        return
    cells: dict[tuple, list] = {}
    for t in rel:
        key = (int(round(float(t["truth_az"]))) % 360, int(round(float(t["truth_el"]))))
        cells.setdefault(key, []).append(t)
    az_vals = sorted({k[0] for k in cells})
    el_vals = sorted({k[1] for k in cells})
    agg = {}
    for key, ts in cells.items():
        agg[key] = {
            "miss": float(np.median([float(t["miss_corrected"]) for t in ts])),
            "az": stats.circmedian([float(t["doa_az_cal"]) for t in ts]),
            "el": float(np.median([float(t["doa_el_cal"]) for t in ts])),
            "n": len(ts),
        }

    n_el = max(len(el_vals), 1)
    fig = plt.figure(figsize=(3.9 * max(n_el, 2) + 1.2, 8.4))
    gs = fig.add_gridspec(2, n_el, height_ratios=[1.0, 1.7], hspace=0.38)

    axh = fig.add_subplot(gs[0, :])
    M = np.full((len(el_vals), len(az_vals)), np.nan)
    for (a, e), v in agg.items():
        M[el_vals.index(e), az_vals.index(a)] = v["miss"]
    im = axh.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, origin="lower")
    axh.set_xticks(range(len(az_vals)))
    axh.set_xticklabels(az_vals, fontsize=7)
    axh.set_yticks(range(len(el_vals)))
    axh.set_yticklabels([f"{e:+d}" for e in el_vals], fontsize=8)
    axh.set_xlabel("true azimuth (deg)")
    axh.set_ylabel("elevation (deg)")
    for (a, e), v in agg.items():
        dark = v["miss"] > 0.6 * vmax
        axh.text(az_vals.index(a), el_vals.index(e), f"{v['miss']:.0f}",
                 ha="center", va="center", fontsize=7, color="black" if dark else "white")
    axh.set_title(f"median corrected miss (deg) — {' + '.join(stims)}")
    fig.colorbar(im, ax=axh, fraction=0.02, pad=0.01, label="miss (deg)")

    for j, e in enumerate(el_vals):
        ax = fig.add_subplot(gs[1, j], projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for a in az_vals:
            v = agg.get((a, e))
            if v is None:
                continue
            th0 = np.radians(a)
            dth = np.radians(((v["az"] - a + 180) % 360) - 180)
            dr = float(np.clip((v["el"] - e) / 90.0, -0.45, 0.45))
            col = plt.cm.viridis(min(v["miss"], vmax) / vmax)
            ax.annotate("", xy=(th0 + dth, 1.0 + dr), xytext=(th0, 1.0),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
            ax.plot([th0], [1.0], "o", ms=3, color=GRAY, zorder=1)
        ax.set_ylim(0, 1.55)
        ax.set_yticks([])
        ax.set_xticks(np.radians(range(0, 360, 45)))
        ax.tick_params(labelsize=7)
        ax.set_title(f"el {e:+d} deg", fontsize=10)
    fig.text(0.5, 0.015,
             "arrows: true direction (gray dot) -> median measured direction; "
             "tangential = azimuth error, radial = elevation error (outward = heard too high)",
             ha="center", fontsize=8, color="0.35")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
