# Ground Truth Temporal Motion Fix

Ground Truth mode now feeds SemanticKITTI instance labels into the same temporal motion pipeline used by detector mode.

## What changed

- Reads SemanticKITTI low-16-bit semantic IDs and high-16-bit instance IDs.
- Builds dense object instance IDs for car/pedestrian/VRU instances.
- Converts those instance classes to the motion module's detector-compatible IDs (1=car, 2=pedestrian, 3=cyclist).
- Runs the existing ego-motion -> object residual -> tracking -> trajectory pipeline in `source=truth` mode.
- Carries the SemanticKITTI moving flag as `gt_state` on each object for evaluation; it does not replace the geometric motion decision.
- Adds GT dynamic/static object counts to the motion payload.
- Ground-truth class counts now appear in the normal frame panel.
- Temporal builds bypass the per-frame cache because motion/tracking depends on the previous frame and previous track state. Standalone/random frame caches remain usable.
- Cache version bumped to 11.

## Expected behavior

Frame 0 has no previous scan, so it has no temporal motion result. Frame 1 and later in a sequential run should contain object clusters, geometric STATIC/DYNAMIC/UNKNOWN states, track IDs, and relative velocity. In Ground Truth mode each object additionally has `gt_state` for comparison.

This is an evaluation/oracle mode: it answers how the temporal motion system behaves when object identity/classification comes from ground truth rather than the PointNet detector. Detector mode remains the deployed path.
