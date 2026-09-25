"""First-order Ambisonic intensity-vector direction-of-arrival for the rigid ZM-1 sphere.

Per trial: STFT the 19 capsules, project onto first-order real spherical harmonics, divide out
the rigid-sphere mode strengths b_0(kr), b_1(kr) (without this the pressure and dipole modes sit
~90 deg apart and the active intensity collapses), then form the active intensity vector
I = Re{ W* . [X, Y, Z] } summed over the analysis band. With the modes correctly co-phased, +I
points at the source (verified across directions/frequencies in tests/test_doa.py); any residual
frame rotation or azimuth handedness is removed downstream by the mounting calibration.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import stft
from scipy.special import spherical_jn, spherical_yn

from .geometry import capsule_unit_vectors, ZM1_RADIUS_M, SPEED_OF_SOUND

# First-order real SH (ACN order, N3D): columns [Y00, Y1-1(y), Y10(z), Y11(x)].
_U = capsule_unit_vectors()
_Y = np.column_stack([
    np.ones(len(_U)),
    np.sqrt(3.0) * _U[:, 1],
    np.sqrt(3.0) * _U[:, 2],
    np.sqrt(3.0) * _U[:, 0],
])
_YINV = np.linalg.pinv(_Y)  # (4, 19): capsule pressures -> SH coefficients


def _mode_strength(n: int, kr: np.ndarray) -> np.ndarray:
    """Rigid-sphere mode strength b_n(kr) at the surface (constant 4*pi dropped).

    Uses the second-kind spherical Hankel h^(2) = j_n - i*y_n, matching the e^{+jwt} phasor
    convention of the STFT here; with h^(1) the pressure (n=0) and dipole (n=1) modes are left
    ~90 deg out of phase at higher kr and the compensated intensity de-phases (verified: residual
    cross-spectrum phase grows to >100 deg by 4 kHz with h^(1), but stays ~0 with h^(2))."""
    kr = np.maximum(kr, 1e-8)
    jn = spherical_jn(n, kr); jnp = spherical_jn(n, kr, derivative=True)
    yn = spherical_yn(n, kr); ynp = spherical_yn(n, kr, derivative=True)
    hn = jn - 1j * yn; hnp = jnp - 1j * ynp
    return (1j ** n) * (jn - (jnp / hnp) * hn)


_YINV_MASKED: dict = {}


def _first_order_decoder(capsule_mask=None) -> np.ndarray:
    """Pressure->SH decoder, optionally with capsules excluded (e.g. a hot/dead capsule
    flagged by the quality layer): the pinv is recomputed over the healthy subset, so one
    bad sensor never has to poison the whole array."""
    if capsule_mask is None:
        return _YINV
    key = tuple(np.flatnonzero(~np.asarray(capsule_mask)))
    if key not in _YINV_MASKED:
        _YINV_MASKED[key] = np.linalg.pinv(_Y[np.asarray(capsule_mask)])
    return _YINV_MASKED[key]


def intensity_vector(window: np.ndarray, sr: int, band: tuple[float, float],
                     nperseg: int = 2048, capsule_mask=None) -> np.ndarray:
    """Active-intensity DOA vector (3,) for one 19-channel window. Source is along +/- this."""
    yinv = _first_order_decoder(capsule_mask)
    if capsule_mask is not None:
        window = window[:, np.asarray(capsule_mask)]
    nperseg = min(nperseg, window.shape[0])
    f, _, Z = stft(window, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, axis=0)
    a = np.einsum("cj,fjt->cft", yinv, Z)             # (4, nf, nt): [W, Y, Z, X]
    kr = 2 * np.pi * f / SPEED_OF_SOUND * ZM1_RADIUS_M
    # Tikhonov-regularize the radial inversion so band edges where b_1(kr)->0 (low kr) don't
    # over-amplify noise: 1/d -> conj(d)/(|d|^2 + reg).
    d = np.conj(_mode_strength(0, kr)) * _mode_strength(1, kr)
    m = (f >= band[0]) & (f <= band[1])
    reg = 1e-2 * float(np.median(np.abs(d[m]) ** 2)) if m.any() else 0.0
    comp = np.conj(d) / (np.abs(d) ** 2 + reg)
    cw = np.conj(a[0])[m]                              # W* over band, (nf_b, nt)
    cf = comp[m][:, None]
    ix = np.sum(np.real(cw * a[3][m] * cf))
    iy = np.sum(np.real(cw * a[1][m] * cf))
    iz = np.sum(np.real(cw * a[2][m] * cf))
    return np.array([ix, iy, iz])


# Order-1 validity ends near kr=1 (~1.1 kHz for r=49 mm); above it the rigid sphere carries
# significant 2nd/3rd-order scattered energy a first-order model cannot resolve. The broadband
# band is kept inside both that validity ceiling and the well-conditioned range (b_1 not
# vanishing at the low edge): 400 Hz (kr~0.36) to 1200 Hz (kr~1.08).
BROADBAND_HZ = (400.0, 1200.0)


def analysis_band(stim_info: dict) -> tuple[float, float]:
    """Frequency band to localize in: +/-1/6 octave around a (complex) tone, else broadband."""
    if stim_info["type"] in ("tone", "complex_tone") and stim_info.get("dominant_hz"):
        fc = stim_info["dominant_hz"]
        return (fc * 2 ** (-1 / 6), fc * 2 ** (1 / 6))
    return BROADBAND_HZ


# ---------------------------------------------------------------------------
# Independent cross-check estimator: SH-domain steered-response power (SRP).
#
# Complements the first-order intensity vector with a different estimator family:
# a plane-wave-decomposition power scan over a direction grid, using spherical-harmonic
# orders 0..3 (19 capsules resolve 16 coefficients; order n is valid to kr ~ n, so order 3
# reaches ~3.3 kHz on the 49 mm sphere -- triple the intensity band). The optional
# per-bin magnitude whitening is the PHAT trick from classic SRP-PHAT: every
# time-frequency bin votes with unit power, so a handful of strong (standing-wave) bins
# cannot dominate the map. Classic pairwise SRP-PHAT itself assumes free-field TDOAs,
# which the rigid sphere violates; doing SRP in the SH domain keeps the scattering model
# exact while retaining the whitening idea.
# ---------------------------------------------------------------------------

def _real_sh_n3d(U: np.ndarray) -> np.ndarray:
    """Real spherical harmonics, ACN order, N3D scaling (Y00 = 1), orders 0..3.
    U: (..., 3) unit vectors -> (..., 16)."""
    x, y, z = U[..., 0], U[..., 1], U[..., 2]
    s3, s5, s15, s7 = np.sqrt(3.0), np.sqrt(5.0), np.sqrt(15.0), np.sqrt(7.0)
    s35_8, s105, s21_8 = np.sqrt(35.0 / 2.0) / 2.0, np.sqrt(105.0), np.sqrt(21.0 / 2.0) / 2.0
    cols = [
        np.ones_like(x),                       # (0, 0)
        s3 * y, s3 * z, s3 * x,                # (1,-1) (1,0) (1,1)
        s15 * x * y,                           # (2,-2)
        s15 * y * z,                           # (2,-1)
        s5 * 0.5 * (3 * z ** 2 - 1),           # (2, 0)
        s15 * x * z,                           # (2, 1)
        s15 * 0.5 * (x ** 2 - y ** 2),         # (2, 2)
        s35_8 * y * (3 * x ** 2 - y ** 2),     # (3,-3)
        s105 * x * y * z,                      # (3,-2)
        s21_8 * y * (5 * z ** 2 - 1),          # (3,-1)
        s7 * 0.5 * (5 * z ** 3 - 3 * z),       # (3, 0)
        s21_8 * x * (5 * z ** 2 - 1),          # (3, 1)
        s105 * 0.5 * z * (x ** 2 - y ** 2),    # (3, 2)
        s35_8 * x * (x ** 2 - 3 * y ** 2),     # (3, 3)
    ]
    return np.stack(cols, axis=-1)


_Y3 = _real_sh_n3d(_U)                 # (19, 16)
_Y3INV = np.linalg.pinv(_Y3)           # (16, 19)
_ORDER_OF = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3])
SRP_BAND_HZ = (300.0, 3300.0)          # order-3 validity ceiling (kr ~ 3)


def _fibonacci_grid(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta),
                            np.sin(phi) * np.sin(theta), np.cos(phi)])


def srp_map(window: np.ndarray, sr: int, band: tuple[float, float] = SRP_BAND_HZ,
            phat: bool = True, order: int = 3, n_grid: int = 1500,
            nperseg: int = 2048, grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Steered-power over a direction grid (Fibonacci by default). Returns (grid, power)."""
    nperseg = min(nperseg, window.shape[0])
    f, _, Z = stft(window, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, axis=0)
    a = np.einsum("cj,fjt->cft", _Y3INV, Z)       # (16, nf, nt) SH coefficients
    kr = 2 * np.pi * f / SPEED_OF_SOUND * ZM1_RADIUS_M
    m = (f >= band[0]) & (f <= band[1])
    if not m.any():
        m = np.ones_like(f, bool)
    nsh = (order + 1) ** 2
    A = a[:nsh, m, :]
    comp = np.empty((nsh, int(m.sum())), dtype=complex)
    for n in range(order + 1):
        b = _mode_strength(n, kr[m])
        reg = 1e-2 * float(np.median(np.abs(b) ** 2))
        cn = np.conj(b) / (np.abs(b) ** 2 + reg)
        comp[_ORDER_OF[:nsh] == n] = cn
    Ahat = A * comp[:, :, None]
    if phat:
        w = 1.0 / (np.sum(np.abs(Ahat) ** 2, axis=0) + 1e-18)   # (nf, nt) unit-power bins
    else:
        w = np.ones(Ahat.shape[1:])
    if grid is None:
        grid = _fibonacci_grid(n_grid)
    Yg = _real_sh_n3d(grid)[:, :nsh]              # (g, nsh)
    y = np.einsum("gc,cft->gft", Yg, Ahat)
    power = np.einsum("gft,gft,ft->g", y, np.conj(y), w).real
    return grid, power


def _local_grid(center: np.ndarray, half_angle_deg: float = 5.0, n: int = 400) -> np.ndarray:
    """Dense cap of directions around `center` for peak refinement."""
    c = center / np.linalg.norm(center)
    a = np.array([1.0, 0.0, 0.0]) if abs(c[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(c, a); t1 /= np.linalg.norm(t1)
    t2 = np.cross(c, t1)
    rho = np.radians(half_angle_deg) * np.sqrt(np.random.default_rng(0).uniform(0, 1, n))
    th = np.random.default_rng(1).uniform(0, 2 * np.pi, n)
    pts = (np.cos(rho)[:, None] * c[None, :]
           + np.sin(rho)[:, None] * (np.cos(th)[:, None] * t1 + np.sin(th)[:, None] * t2))
    return np.vstack([c, pts])


def srp_doa(window: np.ndarray, sr: int, band: tuple[float, float] = SRP_BAND_HZ,
            phat: bool = True, order: int = 3, n_grid: int = 1500) -> tuple[float, float]:
    """(azimuth, elevation) of the steered-power peak — the SRP cross-check estimate.
    Two-stage: coarse Fibonacci scan, then a dense refinement cap around the peak (the
    coarse grid alone quantizes to ~2-3 deg)."""
    grid, power = srp_map(window, sr, band=band, phat=phat, order=order, n_grid=n_grid)
    peak = grid[int(np.argmax(power))]
    fine = _local_grid(peak)
    fgrid, fpower = srp_map(window, sr, band=band, phat=phat, order=order, grid=fine)
    from .geometry import unit_to_azel
    az, el = unit_to_azel(fgrid[int(np.argmax(fpower))])
    return float(az), float(el)
