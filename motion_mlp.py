"""Tiny object-level learned motion refinement for Stage 6.

The model does NOT replace LiDAR registration, residual computation, or
tracking. It consumes the compact geometric features already produced by
Stage 5 and returns a dynamic-object probability. If no trained checkpoint is
present, the pipeline keeps the Stage-5 geometric decision unchanged.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np

FEATURE_NAMES = [
    "class_car", "class_pedestrian", "class_cyclist",
    "log_points", "range_m", "height_m", "length_m", "width_m",
    "median_residual_m", "p75_residual_m", "moving_fraction",
    "relative_speed_mps", "trajectory_speed_mps", "velocity_std_mps",
    "direction_consistency", "trajectory_rmse_m", "track_age",
    "history_len", "ego_speed_mps", "ego_confidence",
]
FEATURE_VERSION = 1


def _finite(x, default=0.0):
    try:
        x = float(x)
    except Exception:
        return float(default)
    return x if np.isfinite(x) else float(default)


def object_features(obj: dict, ego: dict | None = None) -> np.ndarray:
    """Build the fixed-length feature vector used by the Stage-6 MLP."""
    cls = int(obj.get("class_id", -1))
    c = np.asarray(obj.get("center", [0, 0, 0]), dtype=np.float64)
    size = np.asarray(obj.get("size", [0, 0, 0]), dtype=np.float64)
    traj = obj.get("trajectory", {}) or {}
    r = float(np.hypot(c[0], c[1])) if len(c) >= 2 else 0.0
    h = _finite(size[2] if len(size) > 2 else 0.0)
    length = _finite(size[0] if len(size) > 0 else 0.0)
    width = _finite(size[1] if len(size) > 1 else 0.0)
    e = ego or {}
    x = np.array([
        1.0 if cls == 1 else 0.0,
        1.0 if cls == 2 else 0.0,
        1.0 if cls == 3 else 0.0,
        np.log1p(max(0, _finite(obj.get("points", 0)))),
        r, h, length, width,
        _finite(obj.get("median_m", 0.0)),
        _finite(obj.get("p75_m", 0.0)),
        _finite(obj.get("moving_fraction", 0.0)),
        _finite(obj.get("relative_speed_mps", 0.0)),
        _finite(traj.get("speed_mps", 0.0)),
        _finite(traj.get("velocity_std_mps", 0.0)),
        _finite(traj.get("direction_consistency", 0.0)),
        _finite(traj.get("trajectory_rmse_m", 0.0)),
        float(max(0, int(obj.get("age", 1) or 1))),
        float(max(0, int(traj.get("history_len", 0) or 0))),
        _finite(e.get("speed_mps", 0.0)),
        _finite(e.get("confidence", 0.0)),
    ], dtype=np.float32)
    return x


def batch_features(objects, ego=None) -> np.ndarray:
    if not objects:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.stack([object_features(o, ego) for o in objects]).astype(np.float32)


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def make_model(n_features: int = len(FEATURE_NAMES)):
    torch, nn = _torch()
    class MotionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)
    return MotionMLP()


class MotionMLP:
    """CPU inference wrapper with checkpoint + normalization metadata."""
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        torch, _ = _torch()
        self.torch = torch
        self.device = torch.device(device)
        ckpt = torch.load(str(checkpoint), map_location=self.device, weights_only=False)
        self.feature_version = int(ckpt.get("feature_version", 0))
        if self.feature_version != FEATURE_VERSION:
            raise ValueError(f"motion MLP feature version {self.feature_version} != {FEATURE_VERSION}")
        self.mean = torch.as_tensor(ckpt["mean"], dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(ckpt["std"], dtype=torch.float32, device=self.device)
        self.model = make_model(len(FEATURE_NAMES)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def predict_proba(self, objects, ego=None) -> np.ndarray:
        X = batch_features(objects, ego)
        if len(X) == 0:
            return np.empty(0, dtype=np.float32)
        with self.torch.inference_mode():
            x = self.torch.from_numpy(X).to(self.device)
            x = (x - self.mean) / self.std.clamp_min(1e-6)
            p = self.torch.sigmoid(self.model(x)).cpu().numpy()
        return p.astype(np.float32)


def refine(objects, ego, predictor: MotionMLP | None,
           low: float = 0.35, high: float = 0.65):
    """Apply conservative learned refinement while preserving geometry.

    p >= high => DYNAMIC, p <= low => STATIC. In the middle, retain the
    Stage-5 decision and mark the result as learned-ambiguous.
    """
    if predictor is None or not objects:
        return objects
    probs = predictor.predict_proba(objects, ego)
    out = []
    for o, p in zip(objects, probs):
        q = dict(o)
        p = float(p)
        old = q.get("state", "UNKNOWN")
        if p >= high:
            state = "DYNAMIC"
        elif p <= low:
            state = "STATIC"
        else:
            state = old
        # Combine learned evidence with the existing geometric confidence;
        # this is intentionally not presented as a calibrated probability.
        geom = _finite(q.get("confidence", 0.0))
        learned_conf = abs(p - 0.5) * 2.0
        q["mlp_dynamic_probability"] = p
        q["mlp_confidence"] = float(learned_conf)
        q["mlp_state"] = state
        q["mlp_override"] = bool(state != old)
        q["mlp_geometry_state"] = old
        if state != old:
            q["state"] = state
            q["confidence"] = float(max(geom, 0.55 + 0.40 * learned_conf))
        out.append(q)
    return out
