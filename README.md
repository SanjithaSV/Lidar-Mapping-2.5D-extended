# Adaptive 2.5D LiDAR Mapping with Stage 6C Temporal Motion Pipeline

An end-to-end, high-performance LiDAR perception and 2.5D adaptive mapping framework featuring real-time SE(2) ego-motion estimation, ego-compensated temporal object tracking, multi-frame trajectory validation, detector integration, and interactive web visualization.

---

## Overview

This repository provides the complete, authoritative Stage 6C implementation for adaptive 2.5D LiDAR elevation mapping and temporal motion reasoning:

- **Adaptive 2.5D Elevation Mapping**: Converts raw 3D LiDAR point clouds into memory-efficient, multi-resolution 2.5D surface grids with drivability classification and terrain modeling.
- **LiDAR-Only Planar SE(2) Ego-Motion**: Computes high-precision planar sensor ego-motion ($\Delta x, \Delta y, \Delta \theta$) directly from raw point cloud scans without requiring external IMU/GNSS.
- **Ego-Compensated Object Motion Residuals**: Projects past detected objects into the current frame coordinate frame to compute motion residuals and filter out sensor movement.
- **Multi-Frame Temporal Tracking & Trajectory Fit**: Tracks objects across consecutive sweeps using Hungarian association, estimating relative velocities ($\text{m/s}$ and $\text{km/h}$) and least-squares trajectory consistency over a sliding window.
- **Dynamic / Static Classification**: Disambiguates moving objects from stationary obstacles and detector jitter using directional trajectory consistency, speed thresholds, and optional learned MLP refinement.
- **Stage 6C SemanticKITTI Ground-Truth Integration**: Supports both trained PointNet detector inference and SemanticKITTI ground-truth labels for evaluation and benchmarking.
- **Consecutive Sequential Processing**: Enforces strictly consecutive frame execution ($000000 \to 000001 \to 000002 \dots$) for temporal motion continuity across runs.
- **Interactive Web Interface & Streaming Backend**: FastAPI + Uvicorn server providing Server-Sent Events (SSE) streaming, asynchronous downloading/processing, and 2.5D canvas visualization.

---

## Project Structure

```
.
├── server/
│   ├── app.py                   # FastAPI application & REST/SSE endpoints
│   └── pipeline.py              # Core Stage 6C processing pipeline & caching
├── web/
│   ├── index.html               # Interactive web visualization client
│   ├── app.js                   # Client-side UI & canvas renderer
│   └── style.css                # Interface styling
├── trail/
│   ├── best.pt                  # Trained PointNet 3D detector checkpoint (~3.28 MB)
│   ├── pointnet-det/            # PointNet detector training and proposal pipeline
│   ├── scripts/                 # Analysis and export scripts
│   └── reports/                 # Evaluation reports and ablations
├── predict.py                   # Detector inference module (loads trail/best.pt)
├── grid25.py                    # 2.5D elevation grid construction and drivability
├── motion.py                    # SE(2) ego-motion, association, and multi-frame tracking
├── motion_mlp.py                # Optional object-level motion MLP refinement
├── train_motion_mlp.py          # Training script for object motion MLP
├── evaluate_motion_mlp.py       # Evaluation metrics for motion classification
├── build_motion_dataset.py      # Dataset extraction for motion training
├── benchmark_stage6.py          # Benchmarking script for Stage 6 pipeline
├── scene.py                     # Scene representation utilities
├── kitti.py                     # KITTI dataset parser and projection helpers
├── fetch_kitti.py               # Asynchronous remote scan fetching
├── requirements.txt             # Python package dependencies
├── STAGE3.md - STAGE6B_GT_FIX.md# Detailed stage design and validation documentation
└── REPO_HANDOFF.md              # Stage 6C handoff notes
```

---

## Installation & Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11 / 3.12 / 3.14)
- Modern web browser (Chrome, Edge, Firefox)

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/SanjithaSV/Lidar-Mapping-2.5D-extended.git
cd Lidar-Mapping-2.5D-extended

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Web Application

Start the pipeline server with Uvicorn:

```bash
python -m uvicorn app:app --app-dir server --host 127.0.0.1 --port 8011
```

Open your browser and navigate to:
```
http://127.0.0.1:8011/
```

### Modes of Operation
- **Sequential Mode**: Processes consecutive LiDAR sweeps ($t, t+1, t+2, \dots$) to track real motion and estimate object trajectories. The stride is automatically locked to 1.
- **Random Mode**: Samples frames randomly across the sequence to test spatial generalization and detector coverage across varied scenes.

---

## Pipeline Evaluation & Benchmarks

Run the Stage 6 pipeline benchmark:
```bash
python benchmark_stage6.py
```

Run motion MLP evaluation:
```bash
python evaluate_motion_mlp.py
```

Validate motion tracking on consecutive scans:
```bash
python test_motion.py
```

---

## Trained Weights & Checkpoints

- `trail/best.pt`: The pretrained PointNet 3D bounding box detector checkpoint (~3.28 MB) is bundled directly in the repository and loaded by `predict.py` during detector mode.
