"""
Lightweight LiDAR-only ego-motion baseline.

This is deliberately the first temporal layer of the SIH system.  It does NOT
use KITTI Odometry poses or SemanticKITTI labels.  Consecutive LiDAR scans are
converted to a bird's-eye occupancy image and phase correlation estimates the
2-D translation of the scene.  Because static scene points dominate the scan,
the dominant translation is the inverse of the ego vehicle's translation.

Stage 1 intentionally estimates planar translation only.  Rotation, robust
point-level residuals, object association, and learned motion classification
are separate stages and should not be mixed into this first diagnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class EgoMotion:
    """Previous-scan -> current-scan translation estimate."""
    tx: float
    ty: float
    dt: float
    speed_mps: float
    confidence: float
    shift_px: tuple[int, int]
    resolution: float

    @property
    def speed_kmh(self) -> float:
        return self.speed_mps * 3.6

    @property
    def T_prev_to_curr(self) -> np.ndarray:
        """Homogeneous transform for the translation-only baseline."""
        T = np.eye(4, dtype=np.float64)
        T[0, 3] = self.tx
        T[1, 3] = self.ty
        return T


def bev(points: np.ndarray, resolution: float = 0.25,
        max_range: float = 50.0, z_min: float = -1.0,
        z_max: float = 2.5) -> np.ndarray:
    """Build a smoothed BEV occupancy/count image from Nx3 or Nx4 points."""
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] < 3:
        raise ValueError("points must have shape (N,3+)" )
    if resolution <= 0 or max_range <= 0:
        raise ValueError("resolution and max_range must be positive")

    m = ((np.hypot(p[:, 0], p[:, 1]) < max_range) &
         (p[:, 2] >= z_min) & (p[:, 2] <= z_max))
    q = p[m, :3]
    n = int(np.ceil(2.0 * max_range / resolution))
    ix = np.floor((q[:, 0] + max_range) / resolution).astype(np.int64)
    iy = np.floor((q[:, 1] + max_range) / resolution).astype(np.int64)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)

    out = np.zeros((n, n), dtype=np.float32)
    np.add.at(out, (iy[ok], ix[ok]), 1.0)

    # Smooth occupancy makes the correlation less sensitive to individual
    # LiDAR sampling differences while retaining metre-scale structure.
    return ndimage.gaussian_filter(out, sigma=1.0, mode="constant")


def _phase_shift(previous: np.ndarray, current: np.ndarray):
    """Return the integer image shift mapping previous scene -> current scene.

    The Fourier phase-correlation peak convention is the displacement needed
    to align `current` back onto `previous`; hence the returned physical
    translation below is the negative of the image shift.
    """
    A = np.fft.fft2(previous)
    B = np.fft.fft2(current)
    cross = A * np.conj(B)
    cross /= np.maximum(np.abs(cross), 1e-9)
    corr = np.fft.ifft2(cross).real

    iy, ix = np.unravel_index(np.argmax(corr), corr.shape)
    sy = int(iy if iy < corr.shape[0] / 2 else iy - corr.shape[0])
    sx = int(ix if ix < corr.shape[1] / 2 else ix - corr.shape[1])

    peak = float(corr[iy, ix])
    flat = np.abs(corr).ravel()
    if len(flat) > 1:
        second = float(np.partition(flat, -2)[-2])
    else:
        second = 0.0
    confidence = peak / max(second, 1e-9)
    return sx, sy, confidence


def estimate(previous: np.ndarray, current: np.ndarray, dt: float = 0.1,
             resolution: float = 0.25, max_range: float = 50.0,
             z_min: float = -1.0, z_max: float = 2.5) -> EgoMotion:
    """Estimate planar ego translation from two consecutive LiDAR scans.

    The transform returned is *previous sensor frame -> current sensor frame*.
    For a forward-moving vehicle this normally has a negative x translation,
    because stationary world geometry appears to move backwards in the current
    sensor frame.  Speed uses the magnitude, so the sign convention cannot
    affect the scalar speed estimate.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    A = bev(previous, resolution, max_range, z_min, z_max)
    B = bev(current, resolution, max_range, z_min, z_max)
    sx, sy, conf = _phase_shift(A, B)
    tx = -sx * resolution
    ty = -sy * resolution
    speed = float(np.hypot(tx, ty) / dt)
    return EgoMotion(tx=tx, ty=ty, dt=dt, speed_mps=speed,
                     confidence=conf, shift_px=(sx, sy),
                     resolution=resolution)
