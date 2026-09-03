# Stage 5 — Multi-frame Motion Validation and Trajectory Estimation

Stage 5 builds on the Stage-4 tracker. A matched object's previous trajectory is
re-expressed in the current ego frame using the newly estimated SE(2) transform,
then the current center is appended. A short least-squares trajectory fit provides
robust relative velocity and direction-consistency statistics.

## New per-track quantities

- `trajectory_xy`: recent object centers, all expressed in the current ego frame
- `trajectory.history_len`
- `trajectory.displacement_m`
- `trajectory.velocity_xy_mps`
- `trajectory.speed_mps` / `speed_kmh`
- `trajectory.velocity_std_mps`
- `trajectory.direction_consistency`
- `trajectory.trajectory_rmse_m`
- `motion_validation`: `initial` or `multi_frame`

## State validation

A static residual classification can be promoted to DYNAMIC only after at least
three trajectory samples (two intervals) and a coherent multi-frame velocity.
A residual-based DYNAMIC state is only demoted after at least four samples when
both the fitted trajectory speed and current residual are small.

This is intentionally a geometric validation stage. It still uses no KITTI
Odometry poses, SemanticKITTI labels, or learned temporal network.

## Pipeline

```text
LiDAR(t-1) + LiDAR(t)
        ↓
SE(2) ego registration
        ↓
ego compensation
        ↓
object association
        ↓
re-express previous trajectory in current frame
        ↓
append current object center
        ↓
short-window trajectory fit
        ↓
relative velocity + direction consistency
        ↓
static/dynamic validation
        ↓
motion-aware 2.5D map
```
