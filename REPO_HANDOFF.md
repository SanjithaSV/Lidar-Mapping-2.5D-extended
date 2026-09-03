# Stage 6C repository handoff

This package contains the Stage 6C source tree for the adaptive 2.5D LiDAR
mapping + temporal motion pipeline.

Included:
- adaptive 2.5D mapping
- LiDAR-only planar SE(2) ego-motion
- ego-compensated object residual motion
- temporal object tracking and relative velocity
- multi-frame trajectory validation
- optional tiny object-level motion MLP
- SemanticKITTI Ground Truth temporal-motion path
- server and web UI integration
- PointNet detector source

Intentionally excluded:
- downloaded KITTI/SemanticKITTI scans and labels
- trained model checkpoints
- generated viewer payloads
- caches and large generated artifacts

See STAGE6B_GT_FIX.md and the stage documentation for implementation notes.
