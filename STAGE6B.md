# Stage 6B — Dataset, Evaluation and Speed Validation

## What was actually validated

The Stage-6 training/evaluation path was exercised on the labeled LiDAR files packaged with this artifact. They contain 11 labeled frames from sequence 00, but the frames are sparse rather than consecutive (mostly 50–150 frame gaps). The resulting diagnostic set contains 132 detected-object samples and only 1 positive moving-object sample.

Therefore these results are useful for validating the code path, feature extraction, training, and runtime—not for claiming dynamic-object generalization.

## Diagnostic results

| System | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Stage-5 geometry | 58.33% | 0.00 | 0.00 | 0.00 |
| MLP high-confidence | 99.24% | 0.00 | 0.00 | 0.00 |
| Conservative hybrid | 72.73% | 0.00 | 0.00 | 0.00 |

The apparent accuracy increase is misleading because the validation set has only one positive example. No accuracy improvement is claimed from this experiment.

## Speed

The tiny MLP was benchmarked separately with six detected objects per inference call:

- median: **169.8 µs/call**
- p95: **259.2 µs/call**
- median: **28.3 µs/object**

For the full prototype pipeline, steady-state measurements on the available sparse frame pairs were approximately:

- MLP enabled: **5.377 s/frame**
- MLP disabled: **5.429 s/frame**

This difference is within normal runtime noise, so the correct conclusion is that the object-level MLP adds **negligible latency** relative to the existing detector/registration/grid pipeline.

Python's `timeit` guidance recommends repeated measurements and using the fastest/steady measurements for timing small code paths; deterministic profilers are better used to find bottlenecks than as benchmarks. citeturn0search3turn0search0

## Current recommendation

Keep the MLP **optional and disabled by default** until a genuinely consecutive labeled sequence is available. The trained checkpoint in `artifacts/diagnostic_motion_mlp.pt` is retained only for experimentation and is not placed at the runtime path `motion_mlp.pt`.

The next meaningful evaluation should use consecutive frames with moving-object labels and compare Stage 5 vs Stage 6 using object-level precision, recall, F1, false positives/frame, ID stability, and end-to-end FPS/latency.
