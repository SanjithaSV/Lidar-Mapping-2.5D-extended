# Stage 3 — Ego-Compensated Object Motion Residual

Stage 3 adds the first static/dynamic signal without KITTI Odometry or SemanticKITTI supervision.

## Pipeline

```text
LiDAR(t-1) + LiDAR(t)
        |
        v
Stage 2 SE(2) registration
        |
        v
Ego transform T(t-1 -> t)
        |
        v
Transform previous scan into current frame
        |
        v
Nearest-neighbour XY residuals
        |
        v
Existing PointNet object clusters
        |
        v
Per-cluster robust residual statistics
        |
        +----> STATIC / DYNAMIC / UNKNOWN
        |
        v
Motion state projected into adaptive 2.5D cells
```

## Current geometric baseline

For every current object point, the module finds the nearest point in the ego-compensated previous scan. Cluster statistics include:

- median residual
- 75th percentile residual
- fraction of points above the motion threshold
- state
- diagnostic confidence

The default baseline threshold is 0.35 m, with a small range-scaled tolerance. This is a tunable geometric baseline, not a learned probability.

## Important limitations

- Current motion residual is planar (XY), matching the Stage 2 SE(2) ego estimator.
- The nearest-neighbour residual is deliberately simple and should be treated as a baseline.
- A moving object can be missed if it has strong overlap with static geometry or severe occlusion.
- Absolute/world object velocity is not claimed yet; this stage primarily establishes motion relative to the ego-compensated scene.
- No SemanticKITTI or KITTI Odometry labels are used by this stage.

## Viewer

The `motion` colour mode displays object cells as:

- static
- dynamic
- unknown

The information panel reports dynamic/static/unknown object clusters and point counts.

## Next stage

Stage 4 should improve temporal object association and estimate object displacement/relative velocity from persistent tracks. Only after this geometric baseline is behaving sensibly should a learned temporal classifier be considered.
