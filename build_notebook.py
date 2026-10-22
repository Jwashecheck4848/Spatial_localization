"""Generate the canonical pipeline notebook (pipeline.ipynb) — a thin wrapper over the scripts.

Run `python build_notebook.py` to regenerate the notebook from this single source of truth.

The notebook drives every enabled condition in conditions.json through run_analysis.run()
(authoritative stimulus labels from stimuli.json, pink-noise mount calibration, per-condition
output dirs), then the cross-condition comparisons via compare_conditions.run_all().
"""
from pathlib import Path
import nbformat as nbf

EXP_DIR = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
C = nb.cells


def md(text):
    C.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    C.append(nbf.v4.new_code_cell(text.strip("\n")))


md(r"""
# CAVE speaker-array validation — unified Zylia pipeline

> **Branch `newdata-fix`** -- analyses the 2026-09/10 clip-log recordings (curtains closed,
> Setup1 + Setup2 pooled, no Outer speakers). What was wrong with the first/second pass and what
> changed: `NEWDATA_FIX.md`. The condition table below describes the July study; the conditions
> actually run here are the six in `conditions.json` (pink noise / phonemes / five 200 ms sounds,
> each physical and VBAP).

One notebook, every condition. The **Zylia ZM-1** measures direction-of-arrival (rigid-sphere
first-order Ambisonic active intensity) for each recorded condition:

| condition | data | render | room |
|---|---|---|---|
| `gt_cave` | real speaker moved over 24 az × 3 el | physical | CAVE |
| `gt_anechoic` | real speaker, 12 az × 3 el | physical | anechoic chamber |
| `vbap` | packed sessions, 24 phantom az × 3 el | Optimized VBAP | CAVE |
| `vbap_gtmount` | same data, ground-truth mount applied | Optimized VBAP | CAVE |
| `atmos`, `ambi` | *(enable in `conditions.json` when recorded)* | Dolby Atmos / Ambisonics | CAVE |

Stimulus identity comes from the **authoritative `stimuli.json`** (Unity playlist), never from
audio inference; the mic mount is calibrated on **pink noise** only. Re-taken sessions
(`,R`-flagged) supersede their originals via the `duplicates: latest` policy. Everything is
cached — **Run All** after dropping in new data, or flip `FORCE` to recompute from the WAVs.
""")

md("## 0. Setup + CONFIG")
code(r"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import Image, Markdown, display

EXP_DIR = Path.cwd()                      # the zylia_analysis/ folder (where this notebook lives)
sys.path.insert(0, str(EXP_DIR))
import run_analysis as ra
import compare_conditions as cc

# ---- CONFIG: all knobs in one place ----
FORCE              = False                 # master switch: recompute everything from the WAVs
SHOW_FIGURES       = True
ONLY               = None                  # substring filter for smoke tests (e.g. "Subject_30,25")
ra.RELIABLE_SNR_DB = 6.0                   # min in-band SNR (dB) for a trial's DOA to count
ra.MAX_RELIABLE_HZ = 5000.0                # exclude tones above the array's usable band (Hz)

registry = json.loads((EXP_DIR / "conditions.json").read_text())
CONDITIONS_TO_RUN = [name for name, c in registry["conditions"].items() if c.get("enabled", True)]
print("enabled conditions :", CONDITIONS_TO_RUN)
print("comparisons        :", [c["name"] for c in registry["comparisons"]])
""")

md(r"""
### Stimuli (authoritative)

The block→sound map comes from the Unity `TrialDefGenerator` playlist (`stimuli.json`), not
from audio classification, so phone ringing / applause / speech keep their semantic identity in
every condition.
""")
code(r"""
stims = json.loads((EXP_DIR / "stimuli.json").read_text())
display(pd.DataFrame([{"stim_index": int(k), "label": v["label"], "type": v["type"],
                       "dominant_hz": v["dominant_hz"]}
                      for k, v in stims.items() if k.isdigit()]).set_index("stim_index"))
""")

md(r"""
### How a direction is computed from the Zylia recording

Every number downstream comes from one estimator (`zyldoa/doa.py`; full derivation in
`METHODS_zylia_doa.md`). Per trial:

1. **Isolate the trial** (`onsets.py`). The Unity clock and the audio clock drift, so a
   robust envelope-burst detector plus a per-block Theil–Sen clock fit places each trial's
   onset in the recording; the analysis window is a fixed 0.10–0.40 s after onset for tones,
   pink noise, and applause, and a stimulus-adaptive early window (the earliest 0.40 s span
   clearing the SNR gate) for speech and the telephone ring, whose energy arrives in bursts.
   Trials whose windows leave the recording (a take stopped early) are flagged `truncated`
   and excluded.
2. **STFT** the 19 capsule signals (2048-sample frames, 50% overlap).
3. **Encode to first-order ambisonics.** Project the 19 pressures onto real spherical
   harmonics (ACN/N3D) to get $[W, X, Y, Z](f)$ — an omni channel plus three orthogonal
   dipoles.
4. **Undo the rigid sphere.** The ZM-1 body filters each order with a frequency-dependent
   *mode strength* $b_n(kr)$ (second-kind spherical Hankel functions, $r=4.9$ cm); dividing
   it out (Tikhonov-regularized) restores the free-field relationship between $W$ and the
   dipoles. `tests/test_doa.py` verifies the whole chain recovers synthetic plane waves to
   $<1°$.
5. **Active intensity → direction.** $\mathbf{I} = \sum_f \mathrm{Re}\{W^*(f)\,[X,Y,Z](f)\}$
   over the analysis band points along net acoustic energy flow; azimuth =
   $\mathrm{atan2}(I_y, I_x)$, elevation = $\arcsin(I_z/\lVert\mathbf{I}\rVert)$.
6. **Band choice is physics, not preference.** First-order encoding is valid up to
   $kr \approx 1$ (≈1.1 kHz for this array). Broadband sounds use 400–1200 Hz; tones use
   ±1/6 octave around their frequency; the 6 kHz tone ($kr \approx 4.5$) is excluded a
   priori. Trials with in-band SNR < 6 dB are dropped (`reliable = False`).
7. **Mount calibration → miss.** The Zylia's orientation in the room is fit *per condition
   from pink-noise trials only* — either a constant azimuth offset + handedness, or a full
   rigid rotation (Procrustes), chosen automatically (`mount_model: auto`). Positions
   inconsistent with one mount are flagged outliers (`in_calibration = False`). The
   **corrected miss** is the great-circle angle between the mount-corrected DOA and the true
   direction. **Within-position precision** (median scatter of repeats about their own
   median direction) is calibration-free: sub-degree precision with a large miss means a
   *systematic, repeatable* bias — the tones' signature — not measurement noise.

### Is the DOA running on true signal? (raw-signal integrity)

Before any direction estimate, every session passes three checks the SNR gate cannot do
(`zyldoa/quality.py`): **capsule health** (per-capsule RMS and coherence against the robust
array median — a hot or dead capsule silently corrupts the SH encoding while total power
stays high), **clipping**, and **stimulus verification** (the measured average spectrum of
each stimulus group must match its authoritative label — catches wrong playback, not just
wrong bookkeeping). Capsules flagged *hot* or *dead* are **excluded from the encoding for
that session** — the remaining capsules still resolve first order — and every exclusion is
recorded in `summary.json`. This caught a real fault: capsule 16 emitted ~45 dB of garbage
through five July-9 CAVE sessions (az 0–60, el −25); excluding it recovers those cells from
~60° miss to the normal ~5°.

As an estimator-independence check, `doa.srp_doa` provides a second, independent DOA method:
an SH-domain steered-response-power scan (orders 0–3, valid to ~3.3 kHz) with optional
PHAT-style per-bin whitening — the classic SRP-PHAT reverberation trick, done in the SH
domain because the rigid sphere breaks free-field pairwise TDOAs. Section 2 cross-checks it
against the intensity estimator on sampled trials.
""")

md(r"""
## 1. Run all conditions

Each condition runs through the same engine into its own `results/<name>/` + `figures/<name>/`.
Externally-calibrated conditions (e.g. `vbap_gtmount`) automatically run after their mount
source. First full pass streams ~150 GB off the share (hours); re-runs are instant.
""")
code(r"""
def _cal_order(names):
    reg = registry["conditions"]
    return sorted(names, key=lambda n: reg[n].get("calibration", {}).get("mode") == "external")

summaries = {}
for name in _cal_order(CONDITIONS_TO_RUN):
    print("=" * 74); print("CONDITION:", name)
    kwargs = ra.load_condition(name)
    cond_only = kwargs.pop("only", None)
    if ONLY and cond_only:
        only = (cond_only if isinstance(cond_only, list) else [cond_only]) + [ONLY]
    else:
        only = ONLY or cond_only
    summaries[name] = ra.run(force=FORCE, figures=SHOW_FIGURES, only=only,
                             exclude=kwargs.pop("exclude", None), **kwargs)

frames = []
for name in summaries:
    d = pd.read_csv(ra.load_condition(name)["results_dir"] / "all_trials.csv")
    d.insert(0, "condition", name)
    frames.append(d)
trials = pd.concat(frames, ignore_index=True)
print(len(trials), "trials across", trials.condition.nunique(), "conditions")
""")

md(r"""
## 2. Per-condition QC

Inventory, trial-separation/onset lock, stimulus spectra (grouped by true stimulus), and the
fitted mic mount — including the model chosen (constant offset vs rigid rotation) and the
per-day stability diagnostic (a large day-to-day spread suggests a remount; note that day and
elevation are confounded in the July ground-truth collection).
""")
code(r"""
inv = (trials.groupby(["condition"])
       .agg(sessions=("folder", "nunique"), positions=("position", "nunique"),
            n_trials=("trial", "size"), n_reliable=("reliable", "sum"),
            elevations=("truth_el", lambda e: sorted(set(np.round(e).astype(int)))))
       .reset_index())
display(inv)
""")
code(r"""
for name, s in summaries.items():
    sel = s.get("mount_model_selection") or {}
    display(Markdown(
        f"**{name}** — {s['mounting_description']}  \n"
        f"mount model: `{sel.get('model', 'offset')}` ({sel.get('reason', '')})  \n"
        f"calibration outliers: {s['calibration_outlier_positions'] or 'none'}"))
    if s.get("mounting_by_day"):
        d = s["mounting_by_day"]
        display(pd.DataFrame(d["by_group"]).T)
        display(Markdown(f"az-offset spread {d['az_offset_spread_deg']}°, "
                         f"el-offset spread {d['el_offset_spread_deg']}° — {d['_caveat']}"))
""")
code(r"""
# Raw-signal integrity roll-up: flagged sessions, capsule exclusions, playback verification
for name, s in summaries.items():
    q = s.get("signal_quality", {})
    display(Markdown(
        f"**{name}** — {q.get('n_flagged', '?')}/{q.get('n_sessions', '?')} sessions flagged; "
        f"persistent capsule suspects: {q.get('persistent_capsule_suspects') or 'none'}; "
        f"capsules excluded: {q.get('capsules_excluded_by_session') or 'none'}"))
    for sess, issues in list(q.get("flagged_sessions", {}).items())[:10]:
        display(Markdown(f"- `{sess}`: " + "; ".join(issues[:4])))
""")
code(r"""
# Trial-alignment QC (clip-log sessions). Every number downstream depends on each trial being
# matched to the right speaker position. `alignment_qc` comes from the global schedule aligner
# (onsets.align_schedule): n_on_sound should equal n_trials (180) and shift_margin >= 1.
# `label_shift_audit` is an independent check from the DOAs themselves: within-position
# scatter and a mount-free rigid-fit residual must be smallest at shift 0. Any `suspect`
# session must be inspected (its *_onsets.png figure) before its numbers are used.
rows = []
for name, s in summaries.items():
    for sess, q in (s.get("alignment_qc") or {}).items():
        a = (s.get("label_shift_audit") or {}).get(sess, {})
        rows.append({"condition": name, "session": sess, "n_on_sound": q.get("n_on_sound"),
                     "n_trials": q.get("n_trials"), "shift_margin": q.get("shift_margin_trials"),
                     "ambiguous": q.get("ambiguous"), "audit_suspect": a.get("suspect"),
                     "scatter_by_shift": a.get("scatter_deg"),
                     "rigid_resid_by_shift": a.get("rigid_fit_resid_by_shift_deg")})
if rows:
    qc = pd.DataFrame(rows)
    display(qc)
    bad = qc[(qc.n_on_sound < 0.9 * qc.n_trials) | qc.ambiguous | qc.audit_suspect.fillna(False)]
    display(Markdown(f"**{len(bad)} session(s) need a look** (weak/ambiguous alignment or audit flag)"
                     if len(bad) else "**All sessions aligned cleanly.**"))
""")
code(r"""
# Estimator cross-check: SH-domain SRP(-PHAT) vs the production intensity estimator on a
# sample of reliable pink-noise trials per condition. Large systematic disagreement on
# healthy broadband trials would indicate an estimator artifact; agreement (within the
# grid/reverb noise of each method) means conclusions are not an artifact of one algorithm.
import numpy as np
from zyldoa import doa as _doa, pipeline as _pl, geometry as _geo
from zyldoa.wavio import open_recording as _open

N_TRIALS = 12
for name in summaries:
    kwargs = ra.load_condition(name)
    d = pd.read_csv(kwargs["results_dir"] / "all_trials.csv")
    d = d[d.reliable & d.in_calibration & (d.stim_label == "pink noise")]
    if not len(d):
        continue
    take = d.sample(min(N_TRIALS, len(d)), random_state=0)
    diffs = []
    for folder_name, grp in take.groupby("folder"):
        folder = kwargs["data_root"] / folder_name
        npz = np.load(kwargs["results_dir"] / f"{_pl.folder_tag(folder)}_align.npz")
        wav = _open(folder); sr = wav.sample_rate
        onset_of = dict(zip(npz["trial"].tolist(), npz["onset_samples"].tolist()))
        for _, row in grp.iterrows():
            w0 = float(row.win_start_s) if "win_start_s" in row and pd.notna(row.win_start_s) else 0.10
            w1 = float(row.win_end_s) if "win_end_s" in row and pd.notna(row.win_end_s) else 0.40
            seg = wav.read(int(onset_of[int(row.trial)]) + int(w0 * sr), int((w1 - w0) * sr))
            saz, sel = _doa.srp_doa(seg, sr, phat=True)
            a = _geo.azel_to_unit(row.doa_az, row.doa_el); b = _geo.azel_to_unit(saz, sel)
            diffs.append(float(np.degrees(np.arccos(np.clip(a @ b, -1, 1)))))
        wav.close()
    print(f"{name:>14}: median |SRP - intensity| = {np.median(diffs):5.1f} deg "
          f"(p90 {np.percentile(diffs, 90):5.1f}, n={len(diffs)})")
""")
code(r"""
# One representative onset/spectrogram QC figure per condition (all sessions are in figures/<name>/)
for name, s in summaries.items():
    figdir = ra.load_condition(name)["figures_dir"]
    tag0 = s["positions"][0]["tag"]
    for suffix in ("onsets", "spectrograms"):
        p = figdir / f"{tag0}_{suffix}.png"
        if p.exists():
            display(Markdown(f"**{name}** — `{tag0}` {suffix}")); display(Image(str(p)))
""")

md(r"""
## 3. Results — accuracy by stimulus and elevation

Mount-corrected angular miss (median °) over calibration-consistent trials.
""")
code(r"""
# Where the misses are: heatmap (az x el) + per-elevation arrow maps.
# Arrows run from each TRUE direction to the median MEASURED direction:
# tangential component = azimuth error, radial = elevation error (outward = heard too high).
# White heatmap cells = no calibration-consistent reliable trials there (outlier positions).
for name in summaries:
    for fname, sub in (("miss_map.png", "noise-like (pink noise + applause)"),
                       ("miss_map_tones.png", "tones (250 Hz + 1000 Hz; note the wider scale)")):
        p = ra.load_condition(name)["figures_dir"] / fname
        if p.exists():
            display(Markdown(f"**{name}** — {sub}")); display(Image(str(p)))
""")
code(r"""
rows = []
for name, s in summaries.items():
    for lab, v in s["corrected_miss_by_stimulus"].items():
        rows.append(dict(condition=name, stimulus=lab, median_miss_deg=v["median"], n=v["n"]))
by_stim = pd.DataFrame(rows).pivot(index="stimulus", columns="condition", values="median_miss_deg")
display(by_stim)

rows = []
for name, s in summaries.items():
    for el, v in s.get("by_elevation", {}).items():
        rows.append(dict(condition=name, elevation_deg=float(el),
                         median_miss_deg=v["median"], n=v["n"]))
display(pd.DataFrame(rows).pivot(index="elevation_deg", columns="condition",
                                 values="median_miss_deg"))
""")
code(r"""
# Precision vs accuracy: sub-degree precision with a large miss = a systematic bias, not noise.
rows = []
for name, s in summaries.items():
    prec = s["within_position_precision_by_stimulus"]
    for lab, v in s["corrected_miss_by_stimulus"].items():
        rows.append(dict(condition=name, stimulus=lab,
                         miss_deg=v["median"], precision_deg=prec.get(lab)))
display(pd.DataFrame(rows).set_index(["stimulus", "condition"]).sort_index())
""")

md(r"""
## 4. Cross-condition comparisons

Matched (azimuth, elevation) cells only; per-stimulus deltas carry bootstrap 95% CIs.
`room_effect` = anechoic vs CAVE ground truth (12 shared azimuths × 3 elevations);
`vbap_vs_gt` = rendering penalty vs the physical-speaker floor, with the optimizer's
energy-vector (r_E) prediction overlaid.
""")
code(r"""
comparisons = cc.run_all(registry, force=FORCE)
for name, comp in comparisons.items():
    display(Markdown(f"### {name}"))
    display(pd.DataFrame(comp["per_stimulus"]).T)
    for fig in comp.get("figures", []):
        if Path(fig).exists():
            display(Image(fig))
""")

md("## 5. Report")
code(r"""
for name in summaries:
    p = ra.load_condition(name)["results_dir"] / "report.md"
    display(Markdown(f"--- \n# Condition: {name}"))
    display(Markdown(p.read_text(encoding="utf-8")))
""")

md(r"""
## 6. All sounds — physical (mono) vs VBAP, Setup1 + Setup2, curtains closed

One matched-cell comparison across every stimulus family (pink noise, phonemes, 250/1000/5000 Hz
tones, phone ringing, applause), pooled from the six conditions (`compare_all_sounds.py`).
Positive delta = VBAP localizes worse than the physical speaker at the same (az, el).
(The July paper layer -- `paper_stats.py` / `make_paper_figures.py` -- lives on `main` /
`legacy-output-redirect`; it needs the July conditions and is not run here.)
""")
code(r"""
import compare_all_sounds as cas
cas.main()
comp = json.loads((EXP_DIR / "results" / "comparisons" / cas.NAME / "comparison.json").read_text())
display(pd.DataFrame(comp["per_stimulus"]).T)
for fig in comp.get("figures", []):
    if Path(fig).exists():
        display(Image(fig))
""")

md(r"""
## 7. Alignment check (same table as `python check_alignment.py`)

Every session should show ~180/180 trials on sound, `ambig` False, and no `EXCLUDED` / `CHECK`
flag. An EXCLUDED session needs a look at its `*_onsets.png` before it is accepted or given an
`alignment_shift_override` in `conditions.json`.
""")
code(r"""
import check_alignment
check_alignment.report_current()
""")

out = EXP_DIR / "pipeline.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(C), "cells")
