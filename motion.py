"""Lightweight LiDAR-only ego-motion estimation.

Stage 1 provided a BEV phase-correlation translation baseline.  Stage 2 adds
an iterative robust SE(2) registration step so consecutive scans can explain
both planar translation and yaw.  No KITTI Odometry or SemanticKITTI labels
are required.

The transform represented by :class:`EgoMotion` maps points from the previous
LiDAR frame into the current LiDAR frame.  For stationary world geometry this
is the apparent motion induced by the ego vehicle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class EgoMotion:
    """Previous-scan -> current-scan planar rigid transform."""
    tx: float
    ty: float
    yaw: float
    dt: float
    speed_mps: float
    confidence: float
    shift_px: tuple[int, int]
    resolution: float
    iterations: int = 0
    rmse: float = float("nan")
    inlier_ratio: float = 0.0
    method: str = "bev_phase_translation"

    @property
    def speed_kmh(self) -> float:
        return self.speed_mps * 3.6

    @property
    def yaw_deg(self) -> float:
        return float(np.degrees(self.yaw))

    @property
    def T_prev_to_curr(self) -> np.ndarray:
        """4x4 homogeneous transform, previous sensor frame -> current."""
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        T = np.eye(4, dtype=np.float64)
        T[:2, :2] = ((c, -s), (s, c))
        T[0, 3] = self.tx
        T[1, 3] = self.ty
        return T


def _xy(points: np.ndarray, max_range: float = 50.0,
        z_min: float = -1.0, z_max: float = 2.5) -> np.ndarray:
    p = np.asarray(points)
    if p.ndim != 2 or p.shape[1] < 3:
        raise ValueError("points must have shape (N,3+)")
    m = ((np.hypot(p[:, 0], p[:, 1]) < max_range) &
         (p[:, 2] >= z_min) & (p[:, 2] <= z_max))
    return p[m, :2].astype(np.float64, copy=False)


def bev(points: np.ndarray, resolution: float = 0.25,
        max_range: float = 50.0, z_min: float = -1.0,
        z_max: float = 2.5) -> np.ndarray:
    """Build a smoothed BEV occupancy/count image from Nx3 or Nx4 points."""
    q = _xy(points, max_range, z_min, z_max)
    n = int(np.ceil(2.0 * max_range / resolution))
    ix = np.floor((q[:, 0] + max_range) / resolution).astype(np.int64)
    iy = np.floor((q[:, 1] + max_range) / resolution).astype(np.int64)
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    out = np.zeros((n, n), dtype=np.float32)
    np.add.at(out, (iy[ok], ix[ok]), 1.0)
    return ndimage.gaussian_filter(out, sigma=1.0, mode="constant")


def _phase_shift(previous: np.ndarray, current: np.ndarray):
    """Return integer image shift for previous scene -> current scene."""
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
    second = float(np.partition(flat, -2)[-2]) if len(flat) > 1 else 0.0
    confidence = peak / max(second, 1e-9)
    return sx, sy, confidence


def _sample_bev_points(points: np.ndarray, resolution: float,
                       max_range: float, z_min: float, z_max: float,
                       max_points: int) -> np.ndarray:
    """Turn a scan into one representative XY point per BEV cell."""
    q = _xy(points, max_range, z_min, z_max)
    if len(q) == 0:
        return q
    ix = np.floor((q[:, 0] + max_range) / resolution).astype(np.int64)
    iy = np.floor((q[:, 1] + max_range) / resolution).astype(np.int64)
    n = int(np.ceil(2.0 * max_range / resolution))
    ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
    q, ix, iy = q[ok], ix[ok], iy[ok]
    key = iy.astype(np.int64) * n + ix
    order = np.argsort(key, kind="mergesort")
    key = key[order]
    q = q[order]
    first = np.r_[True, key[1:] != key[:-1]]
    idx = np.flatnonzero(first)
    # Cell representatives are the first point; this is deterministic and
    # avoids a costly full point-cloud ICP while preserving scene structure.
    q = q[idx]
    if len(q) > max_points:
        stride = int(np.ceil(len(q) / max_points))
        q = q[::stride]
    return q


def _apply_se2(p: np.ndarray, yaw: float, tx: float, ty: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return p @ np.array([[c, s], [-s, c]]) + np.array([tx, ty])


def _icp_se2(previous_xy: np.ndarray, current_xy: np.ndarray,
             init_yaw: float = 0.0, init_tx: float = 0.0,
             init_ty: float = 0.0, max_corr: float = 2.0,
             trim_fraction: float = 0.70, max_iter: int = 25,
             tol: float = 1e-4):
    """Robust point-to-point 2-D ICP with a fixed trim fraction."""
    if len(previous_xy) < 10 or len(current_xy) < 10:
        return init_yaw, init_tx, init_ty, float("inf"), 0, 0.0

    yaw, tx, ty = float(init_yaw), float(init_tx), float(init_ty)
    tree = cKDTree(current_xy)
    prev_rmse = float("inf")
    used = 0

    for it in range(1, max_iter + 1):
        moved = _apply_se2(previous_xy, yaw, tx, ty)
        dist, idx = tree.query(moved, k=1, distance_upper_bound=max_corr)
        valid = np.isfinite(dist) & (dist < max_corr)
        if valid.sum() < 10:
            break

        ids = np.flatnonzero(valid)
        keep_n = max(10, int(len(ids) * trim_fraction))
        order = np.argpartition(dist[ids], keep_n - 1)[:keep_n]
        ids = ids[order]
        src = moved[ids]
        dst = current_xy[idx[ids]]
        used = len(ids)

        # Solve the incremental rigid alignment src -> dst, then compose it
        # with the current transform.
        cs = src.mean(axis=0)
        cd = dst.mean(axis=0)
        X = src - cs
        Y = dst - cd
        H = X.T @ Y
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        dtheta = np.arctan2(R[1, 0], R[0, 0])
        dtrans = cd - cs @ R.T

        # Compose old T with incremental T: p' = Rinc (Rold p+told)+dtrans.
        c0, s0 = np.cos(yaw), np.sin(yaw)
        R0 = np.array([[c0, -s0], [s0, c0]])
        Rnew = R @ R0
        t0 = np.array([tx, ty])
        tnew = R @ t0 + dtrans
        yaw = float(np.arctan2(Rnew[1, 0], Rnew[0, 0]))
        tx, ty = map(float, tnew)

        rmse = float(np.sqrt(np.mean(np.sum((src @ R.T + dtrans - dst) ** 2, axis=1))))
        delta = abs(prev_rmse - rmse)
        prev_rmse = rmse
        if abs(dtheta) < tol and delta < tol:
            return yaw, tx, ty, rmse, it, used / max(len(previous_xy), 1)

    return yaw, tx, ty, prev_rmse, it if 'it' in locals() else 0, used / max(len(previous_xy), 1)


def estimate(previous: np.ndarray, current: np.ndarray, dt: float = 0.1,
             resolution: float = 0.25, max_range: float = 50.0,
             z_min: float = -1.0, z_max: float = 2.5,
             max_icp_points: int = 12000, max_corr: float = 2.0,
             trim_fraction: float = 0.70, max_iter: int = 25) -> EgoMotion:
    """Estimate planar SE(2) ego motion from two consecutive LiDAR scans.

    Phase correlation supplies a coarse translation. Robust ICP then refines
    translation and estimates yaw.  The transform is previous sensor frame ->
    current sensor frame, i.e. the apparent motion of stationary scene points.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    A = bev(previous, resolution, max_range, z_min, z_max)
    B = bev(current, resolution, max_range, z_min, z_max)
    sx, sy, conf = _phase_shift(A, B)
    init_tx, init_ty = -sx * resolution, -sy * resolution

    prev_xy = _sample_bev_points(previous, resolution, max_range, z_min, z_max,
                                 max_icp_points)
    curr_xy = _sample_bev_points(current, resolution, max_range, z_min, z_max,
                                 max_icp_points)
    yaw, tx, ty, rmse, iters, inlier_ratio = _icp_se2(
        prev_xy, curr_xy, init_tx=init_tx, init_ty=init_ty,
        max_corr=max_corr, trim_fraction=trim_fraction, max_iter=max_iter)

    speed = float(np.hypot(tx, ty) / dt)
    # Registration quality is deliberately a diagnostic score, not a calibrated
    # probability. A low residual and a healthy inlier ratio are desirable.
    quality = inlier_ratio * np.exp(-rmse / max(max_corr, 1e-6)) if np.isfinite(rmse) else 0.0
    confidence = float(max(0.0, min(1.0, 0.5 * min(conf / 2.0, 1.0) + 0.5 * quality)))
    return EgoMotion(tx=tx, ty=ty, yaw=yaw, dt=dt, speed_mps=speed,
                     confidence=confidence, shift_px=(sx, sy),
                     resolution=resolution, iterations=iters, rmse=rmse,
                     inlier_ratio=inlier_ratio, method="bev_phase_init_robust_icp_se2")


def object_motion(previous: np.ndarray, current: np.ndarray,
                  ego: EgoMotion, cluster_ids: np.ndarray,
                  cluster_classes: np.ndarray,
                  base_threshold: float = 0.35,
                  range_fraction: float = 0.015,
                  min_points: int = 8):
    """Estimate object motion after ego-motion compensation.

    `cluster_ids` is aligned with the full current scan and uses -1 for points
    that are not part of a detector cluster. `cluster_classes` is indexed by
    cluster id and uses the detector's class ids: 0=background, 1=car,
    2=pedestrian, 3=cyclist.

    The geometric signal is deliberately simple in Stage 3: for every current
    object point, find the nearest point in the ego-compensated previous scan
    in XY. A cluster is dynamic when its robust residual is substantially
    larger than the expected registration/measurement tolerance.
    """
    p = np.asarray(previous, dtype=np.float64)
    c = np.asarray(current, dtype=np.float64)
    cid = np.asarray(cluster_ids, dtype=np.int64)
    cls = np.asarray(cluster_classes, dtype=np.int64)
    if len(c) != len(cid):
        raise ValueError("cluster_ids must be aligned with current scan")

    T = ego.T_prev_to_curr
    prev_xy = (p[:, :2] @ T[:2, :2].T) + T[:2, 3]
    curr_xy = c[:, :2]

    # A full-scan tree is still cheap compared with the detector. Keep it in
    # XY because Stage 2 estimates planar SE(2), and vertical differences are
    # dominated by road/surface sampling rather than object motion.
    tree = cKDTree(prev_xy)
    residual = np.full(len(c), np.nan, dtype=np.float32)
    valid = cid >= 0
    object_mask = valid & (cid < len(cls)) & np.isin(cls[np.maximum(cid, 0)], [1, 2, 3])
    oi = np.flatnonzero(object_mask)
    if len(oi):
        d, _ = tree.query(curr_xy[oi], k=1, distance_upper_bound=2.5)
        residual[oi] = np.where(np.isfinite(d), d, np.nan).astype(np.float32)

    states = np.full(len(c), 255, np.uint8)  # 0 static, 1 dynamic, 255 unknown
    clusters = []
    for k in range(len(cls)):
        if cls[k] not in (1, 2, 3):
            continue
        idx = np.flatnonzero(cid == k)
        r = residual[idx]
        r = r[np.isfinite(r)]
        if len(r) < min_points:
            clusters.append(dict(id=int(k), class_id=int(cls[k]), state="UNKNOWN",
                                 confidence=0.0, points=int(len(r)),
                                 median_m=float('nan'), p75_m=float('nan'),
                                 moving_fraction=0.0))
            continue
        # Range-scaled tolerance accounts for sparser correspondences at
        # distance without making the threshold proportional enough to hide
        # genuine nearby object motion.
        rr = np.linalg.norm(c[idx, :2], axis=1)
        threshold = np.maximum(base_threshold, range_fraction * rr)
        rv = residual[idx]
        good = np.isfinite(rv)
        moving = good & (rv > threshold)
        frac = float(moving.sum() / max(good.sum(), 1))
        med = float(np.nanmedian(rv))
        p75 = float(np.nanpercentile(rv, 75))
        # Require both a broad residual and a substantial fraction of points.
        dynamic = (p75 > base_threshold and frac >= 0.25)
        state = "DYNAMIC" if dynamic else "STATIC"
        margin = (p75 - base_threshold) / max(base_threshold, 1e-6)
        conf = float(np.clip(0.5 + 0.35 * np.tanh(margin) +
                             0.15 * np.tanh((frac - 0.25) * 4.0), 0.0, 1.0))
        states[idx] = 1 if dynamic else 0
        clusters.append(dict(id=int(k), class_id=int(cls[k]), state=state,
                             confidence=conf, points=int(len(r)),
                             median_m=med, p75_m=p75, moving_fraction=frac))

    # Attach a compact geometric descriptor used by Stage 4 tracking.
    # This stays separate from the classifier state so the tracker can use
    # motion history without changing the Stage-3 residual rule.
    for o in clusters:
        idx = np.flatnonzero(cid == o['id'])
        q = c[idx, :3]
        if len(q):
            o['center'] = q.mean(axis=0).astype(float).tolist()
            lo = q.min(axis=0); hi = q.max(axis=0)
            o['bbox_min'] = lo.astype(float).tolist()
            o['bbox_max'] = hi.astype(float).tolist()
            o['size'] = (hi - lo).astype(float).tolist()
        else:
            o['center'] = [float('nan')]*3
            o['bbox_min'] = [float('nan')]*3
            o['bbox_max'] = [float('nan')]*3
            o['size'] = [float('nan')]*3
    return residual, states, clusters


def _transform_xy(xy: np.ndarray, ego: EgoMotion) -> np.ndarray:
    c, s = np.cos(ego.yaw), np.sin(ego.yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return np.asarray(xy, dtype=np.float64) @ R.T + np.array([ego.tx, ego.ty])


def _trajectory_stats(history_xy, dt: float, window: int = 6):
    """Robust multi-frame relative-motion statistics.

    `history_xy` contains object centers from oldest to newest, all expressed
    in the *current* ego frame.  Older points are re-expressed in the current
    frame by :func:`track_objects` before this function is called.  This makes
    a stationary object stay spatially coherent while a moving object leaves
    a coherent trajectory.
    """
    h = np.asarray(history_xy, dtype=np.float64)
    if h.ndim != 2 or h.shape[1] != 2 or len(h) < 2:
        return dict(history_len=int(len(h)), displacement_m=0.0,
                    velocity_xy_mps=[0.0, 0.0], speed_mps=0.0,
                    speed_kmh=0.0, velocity_std_mps=0.0,
                    direction_consistency=0.0, trajectory_rmse_m=0.0)
    h = h[-max(2, int(window)):]
    n = len(h)
    # Fit a straight line independently in x/y.  This is less sensitive to
    # one-frame detector jitter than simply dividing the first/last delta.
    t = np.arange(n, dtype=np.float64) * float(dt)
    tc = t - t.mean()
    denom = float(np.dot(tc, tc))
    if denom <= 1e-12:
        vel = np.zeros(2, dtype=np.float64)
    else:
        vel = (tc[:, None] * (h - h.mean(axis=0))).sum(axis=0) / denom
    pred = h.mean(axis=0) + t[:, None] * vel
    # Centre the fitted line at the observed time origin for the residual.
    pred += h.mean(axis=0) - (h.mean(axis=0) + t.mean() * vel)
    err = np.linalg.norm(h - pred, axis=1)
    # Segment speeds give a useful robustness diagnostic.  A dynamic object
    # should usually have a coherent direction; random detector jitter does
    # not.
    seg = np.diff(h, axis=0) / float(dt)
    seg_speed = np.linalg.norm(seg, axis=1)
    nonzero = seg_speed > 1e-4
    if nonzero.any() and np.linalg.norm(vel) > 1e-4:
        unit = seg[nonzero] / seg_speed[nonzero, None]
        vunit = vel / np.linalg.norm(vel)
        consistency = float(np.mean(unit @ vunit))
        consistency = float(np.clip(consistency, -1.0, 1.0))
    else:
        consistency = 0.0
    disp = float(np.linalg.norm(h[-1] - h[0]))
    v_std = float(np.std(seg_speed)) if len(seg_speed) else 0.0
    if math.isnan(v_std) or math.isinf(v_std):
        v_std = 0.0
    t_rmse = float(np.sqrt(np.mean(err ** 2))) if len(err) else 0.0
    if math.isnan(t_rmse) or math.isinf(t_rmse):
        t_rmse = 0.0
    return dict(history_len=n, displacement_m=disp,
                velocity_xy_mps=vel.astype(float).tolist(),
                speed_mps=float(np.linalg.norm(vel)),
                speed_kmh=float(np.linalg.norm(vel) * 3.6),
                velocity_std_mps=v_std,
                direction_consistency=consistency,
                trajectory_rmse_m=t_rmse)


def track_objects(previous_objects, current_objects, ego: EgoMotion, dt: float,
                  next_track_id: int = 1, gate_m: float = 3.0,
                  dynamic_speed_mps: float = 0.75,
                  history_window: int = 6):
    """Associate objects and validate motion over multiple frames.

    Stage 5 extends Stage 4 in two ways. First, every matched track carries a
    short trajectory whose old centers are transformed into the current ego
    frame before the current center is appended. Second, velocity is estimated
    from a least-squares fit over that history rather than one frame only.

    The trajectory is therefore a *relative* trajectory. A static object should
    collapse to a small spatial cluster after ego compensation; a genuinely
    moving object should form a coherent trajectory. No external odometry or
    semantic labels are required.
    """
    prev = list(previous_objects or [])
    cur = list(current_objects or [])
    if not cur:
        return [], next_track_id
    if dt <= 0:
        raise ValueError('dt must be positive')

    pp = np.array([o.get('center', [np.nan]*3) for o in prev], dtype=np.float64)
    cc = np.array([o.get('center', [np.nan]*3) for o in cur], dtype=np.float64)
    valid_p = np.isfinite(pp[:, :2]).all(axis=1) if len(pp) else np.zeros(0, bool)
    valid_c = np.isfinite(cc[:, :2]).all(axis=1)
    pred = _transform_xy(pp[:, :2], ego) if len(pp) else np.empty((0,2))

    matches=[]
    if len(prev) and valid_p.any() and valid_c.any():
        cost=np.full((len(prev),len(cur)), 1e6, dtype=np.float64)
        D=np.linalg.norm(pred[:,None,:]-cc[None,:,:2],axis=2)
        for i in range(len(prev)):
            if not valid_p[i]: continue
            for j in range(len(cur)):
                if not valid_c[j]: continue
                if int(prev[i].get('class_id',-1)) != int(cur[j].get('class_id',-2)):
                    continue
                size = np.asarray(prev[i].get('size',[0,0,0]),float)
                gate = max(gate_m, 0.5*float(np.linalg.norm(size[:2])))
                if D[i,j] <= gate:
                    cost[i,j]=D[i,j]
        ri,cj=linear_sum_assignment(cost)
        matches=[(i,j,float(cost[i,j])) for i,j in zip(ri,cj) if cost[i,j] < 1e5]

    by_cur={j:(i,d) for i,j,d in matches}
    out=[]
    for j,o0 in enumerate(cur):
        o=dict(o0)
        if j in by_cur:
            i,d=by_cur[j]
            track_id=int(prev[i].get('track_id',0) or 0)
            if track_id <= 0:
                track_id=next_track_id; next_track_id+=1

            # Previous history is already expressed in the previous current
            # frame. Re-express the entire history in this current frame.
            old_hist = prev[i].get('trajectory_xy', [])
            if old_hist:
                oh=np.asarray(old_hist, dtype=np.float64)
                if oh.ndim == 2 and oh.shape[1] == 2 and np.isfinite(oh).all():
                    hist=_transform_xy(oh, ego)
                else:
                    hist=np.asarray([pred[i]], dtype=np.float64)
            else:
                hist=np.asarray([pred[i]], dtype=np.float64)
            hist=np.vstack([hist, cc[j,:2]])
            if len(hist) > history_window:
                hist=hist[-history_window:]
            stats=_trajectory_stats(hist, dt, history_window)

            disp=cc[j,:2]-pred[i]
            vel=disp/dt
            age=int(prev[i].get('age',1))+1
            o['track_id']=track_id
            o['age']=age
            o['association_distance_m']=d
            o['predicted_center_xy']=pred[i].astype(float).tolist()
            o['displacement_xy_m']=disp.astype(float).tolist()
            o['relative_velocity_xy_mps']=vel.astype(float).tolist()
            o['relative_speed_mps']=float(np.linalg.norm(vel))
            o['relative_speed_kmh']=float(np.linalg.norm(vel)*3.6)
            o['trajectory_xy']=hist.astype(float).tolist()
            o['trajectory']=stats

            # Multi-frame validation. Require two intervals before using the
            # trajectory to override a low-residual static decision. A strong,
            # directionally coherent trajectory is much less likely to be
            # detector jitter than a one-frame displacement.
            intervals=max(0, stats['history_len']-1)
            coherent=(stats['direction_consistency'] >= 0.45 and
                      stats['speed_mps'] >= dynamic_speed_mps)
            if o.get('state') == 'STATIC' and intervals >= 2 and coherent:
                o['state']='DYNAMIC'
                o['confidence']=float(max(o.get('confidence',0.0),
                    min(0.99, 0.55 + 0.20*stats['direction_consistency'] +
                    0.20*min(stats['speed_mps']/dynamic_speed_mps, 2.0)/2.0)))
            elif o.get('state') == 'DYNAMIC' and intervals >= 3:
                # Do not immediately demote a residual-based dynamic object.
                # Demote only when the multi-frame trajectory is consistently
                # stationary and the current residual is also small.
                if stats['speed_mps'] < 0.45*dynamic_speed_mps and o.get('p75_m', 0.0) < 0.5:
                    o['state']='STATIC'
                    o['confidence']=float(max(0.0, min(0.95, 1.0-stats['speed_mps']/max(dynamic_speed_mps,1e-6))))
            o['motion_validation']='multi_frame'
        else:
            track_id=next_track_id; next_track_id+=1
            o['track_id']=track_id
            o['age']=1
            o['association_distance_m']=None
            o['predicted_center_xy']=None
            o['displacement_xy_m']=[0.0,0.0]
            o['relative_velocity_xy_mps']=[0.0,0.0]
            o['relative_speed_mps']=0.0
            o['relative_speed_kmh']=0.0
            center=np.asarray(o.get('center',[np.nan]*3),float)
            o['trajectory_xy']=[center[:2].astype(float).tolist()] if np.isfinite(center[:2]).all() else []
            o['trajectory']=_trajectory_stats(o['trajectory_xy'], dt, history_window)
            o['motion_validation']='initial'
        out.append(o)
    return out, next_track_id
