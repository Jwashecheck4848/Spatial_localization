# New-data (2026-09/10 clip-log) analysis: what was wrong and what changed

Branch `newdata-fix` (built on `newdata-second-pass`). Scope: curtains closed (CC), Setup1 +
Setup2 pooled, no Outer speakers, no male phonemes.

## 1. Root cause: trials were matched to the wrong speaker positions

In the new recordings each clip plays exactly every 3.2 s (phonemes ~3.46 s) with no pause
between speaker positions. The legacy aligner (`onsets.align_trials`) was written for the July
sessions, which had long pauses between blocks, and it fixes the absolute offset from those
pauses. With no pauses it falls back to "first detected burst = trial 0". Its burst detector
was tuned for ~1 s stimuli and misses many 200 ms bursts. The per-block search can also slip by
one inter-trial interval. As a result, whole sessions were mapped **N trials late**, and every
trial took the truth position of a different trial.

Measured on the cached envelopes of the second-pass run (`python check_alignment.py --audit-old ...`):

| session | trial shift in the old run |
|---|---|
| Pink_Noise_Setup1_CC (mount reference) | **11** (every position mislabelled) |
| Pink_noise_Setup2_CC (mount reference) | **10** (every position mislabelled) |
| Pink_Noise_VBAP_Setup1_CC (9/13) | 184 (all trials mapped past the end of the WAV, so it looked "truncated") |
| Pink_Noise_VBAP_Setup2_CC (9/13) | 28 |
| 01_pink_noise_setup1_vbap_CC | 3 |
| applause S2 vbap / phone S1 mono / 1000 Hz S1 mono / applause S1 mono | 7 / 5-6 / 2 / 1 |
| applause S2 mono, phone S2 mono, 250 Hz S2 mono, pink S2 vbap (10/2), ... | 0 |

The mount was fitted on the mislabelled pink noise and then applied to every other condition,
so the error spread everywhere. The later band-aids (elevation-sign search, per-rig mounts,
hardened rotation selection, `in_calibration` fallback) were all fitting to this mislabelling.

With the labels corrected (cached DOAs re-scored, no new audio processing):

* The Zylia mount is clean: front at 0.9°, tilt < 1°. The only "reflection" is Unity's
  clockwise azimuth vs. the Zylia's counter-clockwise azimuth.
* Setup1 and Setup2 agree within about 1.5°, and the mount did not change between 9/13 and
  10/2 (the applause sessions from 9/28 and 10/2 fit within ~2.5° of the 9/13 mount). One
  mount is enough.
* Median corrected miss, old → corrected: pink mono 22.7 → 6.4°, applause mono 48 → 6.3°,
  phonemes mono 44 → 7.2°, VBAP pink 14.6 → 11°, VBAP applause 19 → 7.9°.

## 2. Other bugs fixed

* **Position key merged two speakers.** `position = folder|az` put el −20 and el 0 at the same
  azimuth into one "position" (mixed DOAs, truth taken from whichever trial came first). It is
  now `folder|az|el`.
* **Pink noise used the wrong window.** Pink noise is a 200 ms clip too (ClipEnd − ClipStart =
  0.200 s). It still used the 1 s-clip windows: DOA 0.10–0.40 s, and SNR/QC 0.15–0.55 s, which
  is mostly post-stimulus reverb. Many good trials failed the SNR gate. Every trial now uses its
  own logged duration: `(0.02, dur − 0.02)` s for the DOA, the SNR gate, the QC spectra and the
  capsule-health check.
* **Envelope cancellation.** The envelope was the RMS of the channel *mean*. Above ~1.5 kHz the
  capsules cancel each other, so the 5 kHz tone and weak VBAP renders nearly vanished. The
  aligner now uses the channel-*power* envelope.
* **Stimulus family from CSV filenames.** One CSV filename is wrong (the phone-ringing Setup1
  *mono* CSV is named `..._vbap_...`). The family now comes from the `ClipName` column when it
  can.
* **Stale caches.** `CACHE_SCHEMA` was raised to 4 and `*_trials.csv` carries it, so old,
  shifted caches are never reused. Output also goes to a fresh `newdata_fix_output` folder.

## 3. What changed (files)

* `zyldoa/onsets.py`: `align_schedule` / `align_schedule_env`. Fits one global clock
  (scale, offset) by counting trials that land on a real energy rise. Resolves whole-trial
  ambiguity explicitly. Refines each onset to its rising edge. Reports QC (`n_on_sound`,
  `shift_margin_trials`, `ambiguous`).
* `zyldoa/pipeline.py`: clip-log sessions use `align_schedule`; per-trial windows
  (`clip_window`); cache schema 4.
* `zyldoa/unity.py`: family from `ClipName`; per-trial `clip_dur_s`, `SourceType`, speaker channel.
* `zyldoa/stimid.py`, `zyldoa/quality.py`: QC windows inside the clip.
* `run_analysis.py`: elevation-aware position key; `tol_deg` calibration option;
  `label_shift_audit`. The audit is an independent check from the DOAs themselves
  (within-position scatter and a mount-free rigid-fit residual vs. label shift). Any session
  whose labels look shifted is printed as a `[LABEL-SHIFT WARNING]` and stored in
  `summary.json`.
* `conditions.json`: only the six CC, Setup1+2 conditions (plus one optional, disabled 9/13
  VBAP pink-noise take). One rigid-rotation mount fitted on physical pink noise
  (`pn_s12_mono_cc`, `tol_deg` 30, so all 36 positions are kept), applied externally to every
  other condition, VBAP included (so VBAP error is rendering error, as in the July
  `vbap_gtmount` comparison). Comparisons are VBAP vs. mono for pink noise, phonemes, and the
  five 200 ms sounds.
* `stimuli.json`: pink noise `clip_duration_s` = 0.2 (fallback only; the log value is used).
* `check_alignment.py`: alignment QC table after a run, and `--audit-old` to measure the shifts
  in an old output folder from its cached envelopes.
* `tests/test_align_schedule.py`: synthetic periodic schedule with a quiet first block. The
  legacy aligner lands 11 trials late; the new one is within 5 ms on every trial.
* `tests/test_newdata_end_to_end.py`: synthetic rigid-sphere plane waves written as 24-bit
  19-channel WAVs plus clip logs, run through the real `run_analysis.run()`. Fixed pipeline:
  0.16° miss, identity mount. Legacy aligner patched back in: 22° miss with sub-degree
  "precision", a bogus mount, and both sessions flagged by the audit.
* `build_notebook.py` / `pipeline.ipynb`: alignment-QC cell; the SRP cross-check uses each
  trial's own window.

## 4. How to run

```
git checkout newdata-fix
python tests/test_align_schedule.py          # ~10 s, no data needed
python tests/test_newdata_end_to_end.py      # ~1 min, no data needed
python run_analysis.py --condition pn_s12_mono_cc    # mount source first
python run_analysis.py --condition fphon_s12_mono_cc # ... or just Run All in pipeline.ipynb
python check_alignment.py                    # every session should be 180/180, margin >= 1, no flags
python compare_conditions.py
```

## 5. Still to look at (measurements, not code)

* **Physical el-0 speakers read 10–15° low in elevation**, with azimuth correct. El ±20 read
  correctly, and VBAP el-0 phantoms read 0°, so the Zylia frame is fine. Check the el-0 speaker
  heights and aim. Also check for a reflection or diffraction off the el −20 speaker that sits
  directly below and in front of each el-0 speaker at the same azimuth. A quick test is to play
  one el-0 speaker with the −20 speaker under it removed.
* **5000 Hz tone**: excluded from statistics (`MAX_RELIABLE_HZ`). kr ≈ 4.5 is far beyond the
  first-order model.
* **Weak sessions**: phone ringing VBAP, 1000 Hz VBAP, female phonemes VBAP and 5 kHz could not
  be verified from the old (channel-mean) envelopes. Confirm them in `check_alignment.py` after
  the re-run.
