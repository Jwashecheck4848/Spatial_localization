"""Zylia ZM-1 array geometry and first-order Ambisonic encoding directions.

The 19 capsule directions are the SPARTA / Spatial_Audio_Framework `__Zylia1D_coords_rad`
table (rigid sphere, radius 49 mm). Convention: azimuth CCW from front (+x), elevation up,
unit vector = (cos el cos az, cos el sin az, sin el).
"""
from __future__ import annotations

import numpy as np

# (azimuth, elevation) in radians, 19 capsules.
# Source: leomccormack/Spatial_Audio_Framework saf_utility_sensorarray_presets.c
ZM1_COORDS_RAD = np.array([
    [ 0.0,                1.57079632679490],
    [ 0.00305809444245928, 0.840254037451382],
    [ 2.09600986753364,   0.840126252832125],
    [-2.09336058192593,   0.840886905122138],
    [-1.43409959239697,   0.338967177556435],
    [-0.656487391713457,  0.339152933310760],
    [ 0.661232814211584,  0.338858655681573],
    [ 1.43624308141539,   0.339058915910358],
    [ 2.75545932621978,   0.339167630604397],
    [-2.75063229463181,   0.339281599533891],
    [-2.48035983937821,  -0.338858655681573],
    [-1.70534957217440,  -0.339058915910358],
    [-0.386133327370014, -0.339167630604397],
    [ 0.390960358957982, -0.339281599533891],
    [ 1.70749306119282,  -0.338967177556435],
    [ 2.48510526187634,  -0.339152933310760],
    [-3.13853455914733,  -0.840254037451382],
    [-1.04558278605616,  -0.840126252832125],
    [ 1.04823207166387,  -0.840886905122138],
], dtype=np.float64)

ZM1_RADIUS_M = 0.049
N_CAPSULES = 19
SPEED_OF_SOUND = 343.0


def capsule_unit_vectors() -> np.ndarray:
    """Return (19, 3) array of capsule look directions as unit vectors."""
    az = ZM1_COORDS_RAD[:, 0]
    el = ZM1_COORDS_RAD[:, 1]
    return np.column_stack([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])


def azel_to_unit(az_deg: float, el_deg: float) -> np.ndarray:
    """(azimuth, elevation) in degrees -> unit direction vector (x front, y left, z up)."""
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def unit_to_azel(vec: np.ndarray) -> tuple[float, float]:
    """Unit (or any) vector -> (azimuth_deg, elevation_deg)."""
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return 0.0, 0.0
    v = v / n
    az = np.rad2deg(np.arctan2(v[1], v[0]))
    el = np.rad2deg(np.arcsin(np.clip(v[2], -1.0, 1.0)))
    return float(az), float(el)


def angular_distance_deg(az1, el1, az2, el2) -> float:
    """Great-circle angle (deg) between two (az, el) directions in degrees."""
    a = azel_to_unit(az1, el1)
    b = azel_to_unit(az2, el2)
    return float(np.rad2deg(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
