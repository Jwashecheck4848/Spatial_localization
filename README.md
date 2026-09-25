# Zylia localization analysis — unified multi-condition pipeline

Measures how accurately a Zylia ZM-1 (19-mic rigid sphere) detects a sound's direction versus
the known target direction, for every recorded condition of the CAVE validation study:
ground-truth speaker (CAVE + anechoic chamber), Optimized VBAP, and (when recorded) Dolby
Atmos and Ambisonics phantom sources. Conditions, data roots, output dirs, and calibration
policy live in **`conditions.json`**; stimulus identity comes from the authoritative
**`stimuli.json`** (Unity playlist), never from audio inference.

Session layouts handled: single-position folders (`Subject_<az>,<el>,<r>[,A|,R]_...`) and
packed multi-azimuth sessions (`Subject_0-345,<el>,<r>,V_...`, 9-part >4 GB WAVs, per-trial
truth from the TrialData CSV). Re-taken sessions (`,R`) supersede their originals
(`duplicates: latest`).

## Run

```
pip install -r requirements.txt
python run_analysis.py --condition gt_cave        # one condition (cached)
python run_analysis.py --condition vbap --only "0-345,0"   # smoke test one session
python compare_conditions.py                       # all cross-condition comparisons
python optim_predict.py                            # VBAP rE predictions, full 24az x 3el grid
python paper_stats.py                              # every derived number quoted in the paper
python make_paper_figures.py                       # publication figures -> ../Paper/figures/
for t in tests/test_*.py; do python "$t"; done     # physics + parsing + calibration tests
python build_notebook.py                           # regenerate pipeline.ipynb (source of truth)
jupyter lab pipeline.ipynb                         # the canonical notebook (Run All)
```

The notebook must run with `zylia_analysis/` as its working directory (it resolves paths
from `Path.cwd()`), and the data volume must be mounted. With `FORCE = False` (default) a
Run All reloads the cached per-trial results and reproduces every table, statistic, and
figure in minutes; bootstrap confidence intervals use fixed seeds, so numbers are
deterministic. Setting `FORCE = True` recomputes everything from the raw WAVs (alignment,
QC, DOA; roughly 2–3.5 h for gt_cave, 1–2 h for gt_anechoic, 45–75 min for vbap).

Per-condition outputs land in `results/<condition>/` and `figures/<condition>/`;
cross-condition results in `results/comparisons/<name>/`; paper-level derived statistics in
`results/paper_stats.json`. The June pilot outputs (`results/`, `results/gt`,
`results/virtual*`, `results/optim/predictions.json`) are frozen artifacts — nothing
writes there.

## What it does

1. **Separate trials** (`onsets.py`). The Unity frame clock and the Zylia audio clock drift
   and jump across inter-block breaks, so each block gets its own offset (robust envelope
   onsets + Theil–Sen clock fit + per-block refinement). Trials whose analysis windows leave
   the recording (a truncated take) are excluded automatically.
2. **Label + verify** (`quality.py`). Stimulus identity comes from the Unity logs via
   `stimuli.json`. A per-session signal-integrity layer checks each capsule's RMS and
   coherence against the robust array median (faulty capsules are excluded from the encoding
   and reported), detects clipping, verifies each stimulus group's measured spectrum against
   its logged identity (speech/phone exempted), and gates trials on in-band SNR (≥6 dB in
   the analysis band).
3. **Estimate DOA** (`doa.py`, `geometry.py`): first-order Ambisonic active-intensity on the
   ZM-1 geometry with rigid-sphere mode-strength compensation (second-kind Hankel
   convention, Tikhonov-regularized). Broadband uses 400–1200 Hz (below the kr≈1 limit);
   tones use ±1/6 octave; 6 kHz is excluded a priori. An independent SH-domain
   steered-response-power scan with PHAT whitening (orders 0–3) cross-checks sampled trials.
   `tests/test_doa.py` recovers synthetic plane waves to <1°.
4. **Calibrate + miss** (`calibrate.py`): the mount is fit per condition from pink-noise
   trials only — constant azimuth-offset model or full rigid rotation (Procrustes),
   auto-selected; mount-inconsistent positions are flagged; a per-day diagnostic guards
   against remounts. Per-trial angular miss is reported raw and mount-corrected, plus a
   calibration-free **within-position precision** (repeatability).
5. **Compare** (`compare_conditions.py`): matched-cell contrasts only (shared azimuth ×
   elevation cells), per-stimulus and per-elevation medians with percentile-bootstrap 95%
   CIs (2000 resamples, fixed seeds), plus the optimizer's per-direction energy-vector
   predictions (`optim_predict.py`).

## Headline results (July 2026 main study; see the AES paper for full context)

- Anechoic instrument floor **2.1°** (noise-like), within-position precision ≤0.1°.
- Room adds **+2.6°** for broadband but **+20–25°** for sustained tones (tails to 112°);
  tone errors stay sub-degree repeatable — *precisely wrong*, i.e. room acoustics, not noise.
- Optimized-VBAP phantoms vs the physical speaker: **+2.8°** at the optimized +25°
  elevation, +12.4° at ear level, +26.2° at −25° (below the optimizer's declared test
  floor); failed phantoms keep azimuth and are pulled upward toward the speaker mass.

## Layout

```
zylia_analysis/
  pipeline.ipynb          # canonical notebook (built by build_notebook.py); viz layer only
  run_analysis.py         # orchestrator: CLI, caching, QC, calibration, miss, report
  compare_conditions.py   # matched-cell cross-condition comparisons + bootstrap stats
  optim_predict.py        # optimizer rE predictions over the measured grid
  paper_stats.py          # derived stats quoted in the paper -> results/paper_stats.json
  make_paper_figures.py   # publication figures -> ../Paper/figures/
  build_notebook.py       # regenerates pipeline.ipynb (single source of truth)
  conditions.json         # condition registry + comparisons list
  stimuli.json            # authoritative stimulus playlist (Unity)
  requirements.txt        # pinned, verified environment
  zyldoa/                 # geometry, wavio (streaming/multi-part reader), unity, onsets,
                          #   quality, doa (intensity + SH-SRP), calibrate, pipeline, stats, viz
  tests/                  # DOA physics, wavio, unity parsing, calibration, truncation, SRP, QC
  results/  figures/      # per-condition + comparison outputs (June pilot dirs frozen)
```
