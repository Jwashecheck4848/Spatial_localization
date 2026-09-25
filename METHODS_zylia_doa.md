# How the Zylia signal is turned into a direction (DOA methods)

This documents the signal-processing chain that converts the Zylia ZM-1's 19 microphone signals into an
estimated **direction of arrival (DOA)** — the perceived azimuth/elevation of a sound — and then into the
**angular miss** versus the true speaker direction. Code lives in `analysis/zyldoa/`.

**One-sentence summary.** We turn the 19 capsule pressures into a 4-channel first-order Ambisonic
(B-format) signal, divide out the rigid sphere's frequency-dependent scattering, form the **acoustic
active-intensity vector** (which points along the net flow of sound energy, i.e. back toward the source),
and read off its azimuth and elevation; a one-time mounting calibration rotates the Zylia frame into the
room frame.

---

## 0. Hardware and coordinate frame  (`geometry.py`)

- The **Zylia ZM-1** is a rigid sphere of radius **r = 49 mm** with **19 capsules** flush on its surface.
  The capsule look-directions are the SPARTA / Spatial_Audio_Framework `Zylia1D` table (`ZM1_COORDS_RAD`).
- Convention: unit vector `u = (cos el·cos az, cos el·sin az, sin el)`, azimuth CCW from the array front
  (+x), elevation up (+z). Speed of sound `c = 343 m/s`.
- "Rigid sphere" matters: the sphere **scatters** the incoming wave, so a capsule does not measure the
  free-field pressure — it measures pressure plus the sphere's scattered field. That scattering is undone
  analytically in Step 4; ignoring it is the single biggest way to get the direction wrong.

The dimensionless frequency variable throughout is **kr = 2π f r / c** (sphere circumference in
wavelengths). kr = 1 at ≈ 1.11 kHz for this sphere; it sets where the first-order model stays valid.

---

## 1. Isolate one sound in time  (`onsets.py`, `pipeline.py`)

The recording is one continuous ~1.1 GB, 19-channel, 24-bit WAV streamed with a random-access reader
(`wavio.py`). Each trial's stimulus onset is located by aligning the Unity trial table to the audio with a
**per-block clock model** (the Unity frame clock and the audio hardware clock drift ~0.2 % and jump across
inter-block breaks, so a single global offset fails). DOA is then computed on a short window
**0.10–0.40 s after the onset** (`ANALYSIS_WIN` in `pipeline.py`) — inside the steady part of the
stimulus, after the transient.

So the input to the DOA stage is a single window `x[t, m]`, `t` = samples, `m = 0…18` = capsules.

---

## 2. Time–frequency decomposition  (`doa.intensity_vector`)

Each capsule channel is STFT'd (`nperseg = 2048`, 50 % overlap) to get a complex spectrum per
time–frequency bin:

```
Z[f, m, τ]   = STFT of capsule m,  frequency bin f, frame τ
```

Working in the frequency domain is essential because the sphere's scattering, and the pressure↔velocity
relationship, are both **frequency-dependent** — they must be handled bin by bin.

> Phasor convention: scipy's STFT uses an `e^{+jωt}` convention. This fixes which spherical Hankel
> function is physically correct in Step 4 (it must be the **second kind**); the test in Step 8 fails if
> the wrong kind is used.

---

## 3. Spherical-harmonic (Ambisonic) encoding: 19 pressures → 1 pressure + 3 velocity components

We project the 19 capsule pressures onto the **first-order real spherical harmonics** (ACN channel order,
N3D normalization). The encoding matrix `Y` (19×4) has, for each capsule direction `u_m`, the row

```
[ 1 ,  √3·u_y ,  √3·u_z ,  √3·u_x ]      # channels  [ W , Y , Z , X ]
```

`W` is the omnidirectional (pressure-like) channel; `X, Y, Z` are three orthogonal figure-of-eight
(dipole, velocity-like) channels. Because there are 19 capsules but only 4 unknowns, the system is
over-determined and inverted in the least-squares sense with the pseudo-inverse `Y⁺` (4×19, `_YINV`):

```
a[c, f, τ] = Σ_m Y⁺[c, m] · Z[f, m, τ]          # c ∈ {W, Y, Z, X}
```

Intuition: `W` ≈ the sound **pressure** at the array center; `(X, Y, Z)` ≈ the three components of the
**particle velocity** (which way the air is moving). DOA from these two is then a physics problem, not a
geometry problem.

---

## 4. Undo the sphere — rigid-sphere mode-strength compensation  (`doa._mode_strength`)  ← the crux

On a rigid sphere, the order-`n` Ambisonic component is scaled by a complex, frequency-dependent
**mode strength** `b_n(kr)` (a magnitude *and a phase* imposed by scattering):

```
b_n(kr) = i^n · [ j_n(kr) − (j_n'(kr) / h_n'(kr)) · h_n(kr) ]
          with the second-kind Hankel  h_n = j_n − i·y_n
```

(`j_n`, `y_n` are spherical Bessel functions of the first/second kind, `'` their derivatives.)

- The **pressure** channel `W` carries `b_0(kr)`; the **velocity** channels `X, Y, Z` carry `b_1(kr)`.
- Critically, `b_0` and `b_1` have **different phases**. If you skip this step, the pressure and velocity
  estimates sit roughly **90° out of phase** at higher kr, and the active intensity (Step 5), which lives
  in the *real* part of their product, **collapses toward zero / points nowhere**. This compensation is
  what makes the method work.

The code forms the combined denominator and divides it out of the pressure×velocity product, with a small
**Tikhonov regularizer** so that band edges where `b_1(kr) → 0` (low kr) do not amplify noise:

```
d(f)    = conj(b_0(kr)) · b_1(kr)
comp(f) = conj(d) / ( |d|² + ε ),     ε = 0.01 · median_band(|d|²)
```

---

## 5. The active-intensity vector → a direction  (`doa.intensity_vector`, `geometry.unit_to_azel`)

The **acoustic active intensity** is the time-averaged product of pressure and particle velocity; it
points along the **net flow of acoustic energy**, which for a single dominant source is the line **back to
the source**. With `W ≈ pressure` and `(X, Y, Z) ≈ velocity`, and the sphere undone by `comp`, the
intensity vector is accumulated over the analysis band and all STFT frames:

```
I_x = Σ_{f∈band, τ}  Re{ conj(W) · X · comp }
I_y = Σ_{f∈band, τ}  Re{ conj(W) · Y · comp }
I_z = Σ_{f∈band, τ}  Re{ conj(W) · Z · comp }
```

Writing the measured channels as `W = b_0·P`, `X = b_1·V_x`, … and recalling `comp = conj(d)/|d|²` with
`d = conj(b_0)·b_1` (so the numerator is `conj(d) = b_0·conj(b_1)` — the conjugate falls only on the
velocity mode strength), the mode strengths cancel exactly:

```
conj(W)·X·comp = [conj(b_0)·P*] · [b_1·V_x] · [b_0·conj(b_1)] / |b_0·b_1|²
               = |b_0|²·|b_1|² · P*·V_x / |b_0·b_1|²  =  P*·V_x
```

so each bin's `conj(W)·X·comp` recovers the **true** free-field pressure–velocity product `P*·V_x`,
sphere removed; summing its *real part* over the band keeps only the **active** (propagating) intensity
and rejects the reactive (standing) part.

The source direction is then just the direction of `I`:

```
azimuth   = atan2(I_y, I_x)
elevation = asin( I_z / |I| )
```

(`+I` points at the source; any residual frame flip is fixed in Step 7.) This is the **raw** Zylia-frame
DOA reported per trial.

---

## 6. Which frequencies to use  (`doa.analysis_band`, `doa.BROADBAND_HZ`)

The band over which the intensity is summed is chosen per stimulus:

- **Broadband / noise-like sounds** → **400–1200 Hz** (`BROADBAND_HZ`). The lower edge keeps `b_1(kr)`
  well-conditioned (not vanishing); the upper edge stays under the order-1 ceiling **kr ≈ 1** (≈ 1.1 kHz),
  above which the rigid sphere carries significant 2nd/3rd-order scattered energy a first-order model
  cannot resolve.
- **Pure tones** → a narrow **±1/6 octave** band around the tone's frequency, so the estimate uses the
  bins that actually contain energy.
- A coarse SNR gate (in-band post/pre power ≥ 6 dB) and a >5 kHz cut drop trials with no usable signal
  (e.g. the 6 kHz tone, which is far past the array's order-1 band).

---

## 7. Zylia frame → room frame, and the "miss"  (`calibrate.py`, `run_analysis.py`)

The raw DOA is in the **Zylia's** frame, which is rotated and possibly handedness-flipped relative to the
room. Because the speaker positions span a full azimuth circle at essentially one elevation, a full 3-D
rotation is under-determined, so we fit an interpretable **constant-offset mounting**:

```
az_room = az_sign · az_zylia + az_offset            (az_sign = ±1, a handedness flip)
el_room = el_zylia + el_offset
```

The three parameters are fit **robustly on the clean wideband reference (pink noise)** across all
ground-truth positions, auto-rejecting positions inconsistent with one fixed mounting. The reported
**angular miss** is the **great-circle angle** between the mount-corrected DOA and the true speaker
direction:

```
miss = arccos( û_measured · û_truth )      (in degrees)
```

The same ground-truth mounting is then applied to the virtual (VBAP) recordings, so any extra error there
is reproduction error, not a different calibration.

---

## 8. Validity, assumptions, and what is tested

**Assumptions baked into the method**

1. **One dominant source per window.** Active intensity points at the *net* energy flow; with multiple
   coherent sources (e.g. a VBAP phantom from several speakers) it points at an energy-weighted average,
   not a single true direction — exactly why panned/virtual sources are harder.
2. **First-order / kr ≤ 1.** Only the n = 0 and n = 1 modes are used, valid below kr ≈ 1 (≈ 1.1 kHz). The
   broadband band and the >5 kHz cut enforce this.
3. **Plane-wave, far-field, sphere at the sweet spot.** The mode-strength model assumes a plane wave on an
   ideal rigid sphere; real near-field curvature, the room, and capsule mismatch are residual error.
4. **Correct phasor/Hankel convention.** `e^{+jωt}` ⇒ second-kind Hankel `h^(2)`; the wrong kind dephases
   pressure vs velocity and destroys the estimate.
5. **Stationary within the window.** The 0.10–0.40 s window is treated as one direction; strongly
   time-varying sounds (speech) violate this and localize worse.

**What is validated** (`tests/test_doa.py`). A synthetic plane wave is generated on the 19 capsules with
the *physically correct* mode strengths (independent of the code under test) and the pipeline must
(a) recover the direction to **< 1°** across frequencies and directions, and (b) leave the compensated
pressure-vs-dipole cross-spectrum **real and positive** (modes co-phased) — which only holds with the
correct Hankel kind. This validates the **signal processing**; it does not certify real-room accuracy,
which is what the ground-truth experiment measures.

---

### Pipeline at a glance

```
19-ch WAV ─▶ onset align ─▶ window 0.10–0.40 s ─▶ STFT per capsule
         ─▶ SH encode (Y⁺): pressure W + velocity X,Y,Z
         ─▶ ÷ rigid-sphere mode strengths b_0,b_1 (Tikhonov)
         ─▶ active intensity  I = Σ_band Re{W*·[X,Y,Z]·comp}
         ─▶ (az,el) = direction of I            [raw, Zylia frame]
         ─▶ constant-offset mounting (fit on pink noise)
         ─▶ great-circle miss vs true speaker direction
```

*Files: `geometry.py` (array + frames), `wavio.py` (streaming reader), `onsets.py` (trial alignment),
`doa.py` (encoding + mode compensation + intensity), `calibrate.py` (mounting + miss),
`pipeline.py`/`run_analysis.py` (orchestration), `tests/test_doa.py` (physics regression).*
