# Stage 6 — Tiny Object-Level MLP Motion Refinement

Stage 6 adds a **small optional MLP** after the existing Stage-5 geometry-first
pipeline. It does not replace ego-motion estimation, ego compensation,
nearest-neighbour residuals, object association, or trajectory fitting.

## Runtime architecture

```text
LiDAR(t-1), LiDAR(t)
       ↓
SE(2) ego registration
       ↓
ego compensation
       ↓
object residuals
       ↓
Hungarian tracking
       ↓
short trajectory + relative velocity
       ↓
20 compact object features
       ↓
32 → 16 → 1 MLP
       ↓
conservative dynamic/static refinement
       ↓
adaptive 2.5D map
```

The MLP runs **once per detected object**, not once per LiDAR point. A missing
`motion_mlp.pt` checkpoint leaves the Stage-5 behaviour unchanged.

## Features

The model consumes class indicators, object point count/range/size, residual
statistics, moving fraction, relative speed, trajectory speed and consistency,
trajectory RMSE, track age/history length, and ego speed/confidence.

## Decision rule

- `p >= 0.65`: DYNAMIC
- `p <= 0.35`: STATIC
- otherwise: keep the Stage-5 geometric state

The output records both the geometric state and the learned probability so the
MLP remains inspectable and can be disabled without changing the core system.

## Training

`train_motion_mlp.py` is an optional supervised trainer. It uses the existing
PointNet detector/Stage-5 features and SemanticKITTI `moving-*` flags to create
object-level labels. The runtime itself does **not** require labels.

Training data should contain consecutive `.bin` and `.label` files in
`cache/raw`. The existing fetch/pipeline mechanism can populate these; no KITTI
Odometry pose file is required.

Example:

```bash
python train_motion_mlp.py --seq 00 --start 1 --count 300 --epochs 30
```

This creates `motion_mlp.pt`. Restart the server after creating the checkpoint.

## Why this is low-latency

The network has only 20 inputs and 2 small hidden layers. Inference is batched
across detected objects and is expected to be tiny relative to the existing
PointNet detector and LiDAR registration. It is deliberately not a point-wise
or range-image CNN.

## Important limitation

A learned classifier cannot recover information that bad ego-motion removed.
The Stage-5 geometric signals remain the physical backbone and fallback. The
MLP is a refinement layer, not the foundation of motion estimation.
