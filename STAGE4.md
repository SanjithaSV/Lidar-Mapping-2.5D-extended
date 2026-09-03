# Stage 4 — Temporal Object Tracking and Relative Velocity

Stage 4 builds on the Stage-3 ego-compensated residual baseline. It assigns persistent track IDs to detector clusters, transforms the previous object center into the current LiDAR frame using the estimated SE(2) ego transform, associates same-class objects with Hungarian assignment, and estimates relative object displacement/velocity.

## Output per object

- `track_id`: persistent ID within a sequential job
- `age`: number of associated frames
- `displacement_xy_m`: ego-compensated displacement over the previous frame interval
- `relative_velocity_xy_mps`: displacement / dt in the current ego frame
- `relative_speed_mps`, `relative_speed_kmh`
- `association_distance_m`
- Stage-3 residual statistics and static/dynamic state

## Important interpretation

The velocity is **relative to the ego vehicle**. It is not absolute world velocity. Absolute/world velocity requires an additional global reference or a globally consistent ego trajectory.

The tracker is intentionally lightweight and CPU-friendly. It is a geometric baseline; it does not use KITTI Odometry, SemanticKITTI motion labels, or a learned temporal network.

## Stage 4 flow

```text
LiDAR(t-1) + LiDAR(t)
        ↓
SE(2) ego registration
        ↓
ego compensation
        ↓
Stage-3 object residuals
        +
previous object tracks
        ↓
same-class gated association
        ↓
persistent track IDs
        ↓
ego-compensated displacement
        ↓
relative velocity
        ↓
static / dynamic refinement
```
