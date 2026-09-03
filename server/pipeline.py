"""
one frame, end to end: fetch -> label -> grid -> surface.

nothing here is precomputed. a frame is pulled from the remote SemanticKITTI
archives on demand (a few MB of HTTP range reads, not the 80 GB zip), labelled
by the network, converted by grid25, and reduced to a surface the browser can
draw. results are cached on disk so a second request for the same frame is
instant, but the cache is an optimisation, not the source of truth.
"""

from __future__ import annotations

import base64, io, json, os, queue, sys, threading, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import grid25 as g            # noqa: E402
import kitti                  # noqa: E402
import motion as ego_motion  # noqa: E402
import motion_mlp  # noqa: E402
import fetch_kitti as FK      # noqa: E402
from export_viewer import surface_json, CLASSES   # noqa: E402

RAW = ROOT / 'cache' / 'raw'
OUT = ROOT / 'cache' / 'frames'
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SURF_MULT = 4          # surface base = 5 cm x this. 4 -> 20/40/80/160 cm.
                       # a car is 0.7 m wide, so at 20 cm it spans 3 nodes and
                       # renders as a flat slab. 10 cm gives 7 and it starts to
                       # look like a car, for 4x the payload.
CACHE_V = 11            # bump when the frame payload changes shape, so stale
                       # cache entries are rebuilt instead of silently served

# the left colour camera that rode along with the laser. it is FORWARD ONLY,
# about 90 degrees wide, while the map is the full 360 -- so the photo shows
# roughly the top-right quadrant of the map, not all of it.
CAM = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_color.zip'
CALIB = 'https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_calib.zip'
_calib: dict[str, dict] = {}
_model_lock = threading.Lock()
_model = None
_motion_mlp = None
_motion_mlp_lock = threading.Lock()
MOTION_MLP_PATH = ROOT / 'motion_mlp.pt'
# a pool rather than one shared handle: ZipFile is not thread safe, and
# building one costs a read of a 44k-entry central directory, so they are
# reused rather than recreated. fetches can then overlap.
_pool: dict[str, queue.Queue] = {}
_pool_lock = threading.Lock()
POOL_MAX = 3

# how long each frame's download actually took, recorded by whichever thread
# did it. without this the reported fetch time is 0 for every prefetched
# frame -- true of the build() call, but a lie about the pipeline.
_fetch_ms: dict[tuple, int] = {}


def motion_model():
    """Load the optional Stage-6 object MLP once."""
    global _motion_mlp
    if not MOTION_MLP_PATH.exists():
        return None
    with _motion_mlp_lock:
        if _motion_mlp is None:
            try:
                _motion_mlp = motion_mlp.MotionMLP(MOTION_MLP_PATH)
            except Exception as exc:
                print(f'[motion-mlp] disabled: {exc}')
                _motion_mlp = False
        return None if _motion_mlp is False else _motion_mlp


def model():
    """load the detector once, on first use."""
    global _model
    with _model_lock:
        if _model is None:
            import predict
            _model = predict.load()
        return _model


class _borrow:
    """lend a ZipFile from the pool for the duration of one read."""

    def __init__(self, url):
        self.url = url

    def __enter__(self):
        with _pool_lock:
            q = _pool.setdefault(self.url, queue.Queue())
        try:
            self.z = q.get_nowait()
        except queue.Empty:
            import zipfile
            self.z = zipfile.ZipFile(FK.httpfile(self.url))
        return self.z

    def __exit__(self, *exc):
        q = _pool[self.url]
        if q.qsize() < POOL_MAX:
            q.put(self.z)


def fetch(seq: str, frame: str, want_truth: bool):
    """make sure the raw files exist locally; pull them if not."""
    b = RAW / f'{seq}_{frame}.bin'
    lb = RAW / f'{seq}_{frame}.label'
    got = []
    t0 = time.perf_counter()
    if not b.exists() or b.stat().st_size == 0:
        with _borrow(FK.VEL) as z:
            data = z.read(f'dataset/sequences/{seq}/velodyne/{frame}.bin')
        b.write_bytes(data)          # read fully first: a failed read must not
        got.append('velodyne')       # leave a 0-byte file behind
    if want_truth and (not lb.exists() or lb.stat().st_size == 0):
        try:
            with _borrow(FK.LAB) as z:
                data = z.read(f'dataset/sequences/{seq}/labels/{frame}.label')
            lb.write_bytes(data)
            got.append('labels')
        except KeyError:
            pass                     # sequences 11-21 have no public labels
    if got:
        _fetch_ms[(seq, frame)] = round((time.perf_counter() - t0) * 1000)
    return b, (lb if lb.exists() else None), got


def cell_provenance(m, x, y, prov):
    """
    the detector's own verdict per cell, not per point.

    a cell that contains a car or a person takes that, whatever else is in it --
    same priority rule the class histogram uses, and for the same reason. what
    is left over resolves by majority, which is how "the network rejected this"
    stays distinguishable from "the network never saw this".
    """
    ix = np.floor(x / g.res0).astype(np.int64)
    iy = np.floor(y / g.res0).astype(np.int64)
    lv = g.blocklevel(ix, iy)
    pk = ((lv << 62) | (((ix >> lv) & 0x7fffffff) << 31) | ((iy >> lv) & 0x7fffffff))
    ck = ((m['lvl'].astype(np.int64) << 62)
          | ((m['ix'] & 0x7fffffff) << 31) | (m['iy'] & 0x7fffffff))
    cid = np.searchsorted(ck, pk)
    h = np.zeros((len(ck), 6), np.int64)
    np.add.at(h, (cid, prov), 1)
    out = h.argmax(1).astype(np.uint8)
    for critical in (2, 3, 4):                 # car, pedestrian, cyclist
        out[h[:, critical] > 0] = critical
    return out


def calib(seq: str):
    """
    where the camera points and how wide it sees, in the laser's own frame.

    all four KITTI cameras are one forward-facing stereo rig -- they differ
    only by a sideways offset of up to 54 cm -- so there is no rear or side
    view to be had. this exists to draw the slice the photo DOES cover onto
    the map, rather than describing it in a caption.
    """
    if seq in _calib:
        return _calib[seq]
    import zipfile, urllib.request
    p = RAW / 'calib.zip'
    if not p.exists() or not p.stat().st_size:
        p.write_bytes(urllib.request.urlopen(CALIB, timeout=120).read())
    with zipfile.ZipFile(p) as z:
        txt = z.read(f'dataset/sequences/{seq}/calib.txt').decode()
    v = {}
    for line in txt.strip().splitlines():
        k, rest = line.split(':', 1)
        v[k.strip()] = np.array([float(x) for x in rest.split()])
    fx = v['P2'].reshape(3, 4)[0, 0]
    fov = float(2*np.degrees(np.arctan(1241/(2*fx))))
    R = v['Tr'].reshape(3, 4)[:, :3]
    fwd = R.T @ np.array([0.0, 0.0, 1.0])        # camera +z, in laser coords
    yaw = float(np.degrees(np.arctan2(fwd[1], fwd[0])))
    _calib[seq] = dict(fov=round(fov, 1), yaw=round(yaw, 1))
    return _calib[seq]


def image(seq: str, frame: str):
    """
    the camera frame that goes with this sweep, pulled the same way as the
    scan. cheaper than the scan (~850 KB, ~5 s) but the first call pays a
    one-off ~50 s to read that archive's 87k-entry directory.
    """
    p = RAW / f'{seq}_{frame}_cam.png'
    if p.exists() and p.stat().st_size:
        return p
    with _borrow(CAM) as z:
        data = z.read(f'dataset/sequences/{seq}/image_2/{frame}.png')
    p.write_bytes(data)
    return p


def project(seq, xyz, lab, w=1241, h=376):
    """
    put the laser points onto the camera image.

    velodyne -> rectified camera (Tr) -> pixels (P2), then keep only what is
    in front of the lens and inside the frame. that is about 15% of a sweep:
    the camera sees 82 degrees of the laser's 360.
    """
    import zipfile
    with zipfile.ZipFile(RAW / 'calib.zip') as z:
        txt = z.read(f'dataset/sequences/{seq}/calib.txt').decode()
    V = {}
    for line in txt.strip().splitlines():
        k, r = line.split(':', 1)
        V[k.strip()] = np.array([float(x) for x in r.split()])
    P2, Tr = V['P2'].reshape(3, 4), V['Tr'].reshape(3, 4)
    cam = (Tr @ np.c_[xyz, np.ones(len(xyz))].T).T
    hom = (P2 @ np.c_[cam, np.ones(len(cam))].T).T
    d = hom[:, 2]
    ok = d > 0.1
    u = np.where(ok, hom[:, 0] / np.where(ok, d, 1), -1)
    v = np.where(ok, hom[:, 1] / np.where(ok, d, 1), -1)
    m = ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return dict(w=w, h=h, n=int(m.sum()),
                u=base64.b64encode(u[m].astype(np.uint16).tobytes()).decode(),
                v=base64.b64encode(v[m].astype(np.uint16).tobytes()).decode(),
                cls=base64.b64encode(lab[m].astype(np.uint8).tobytes()).decode())


def _safe_project(seq, xyz, lab):
    try:
        calib(seq)                      # makes sure calib.zip is on disk
        return project(seq, xyz, lab)
    except Exception:
        return None


def _safe_calib(seq):
    try:
        return calib(seq)
    except Exception:
        return None


def prefetch_image(seq: str, frame: str):
    try:
        image(seq, frame)
    except Exception:
        pass          # no photo is not an error; the map stands on its own


def prefetch(seq: str, frame: str, want_truth: bool):
    """pull the raw files only. safe to run several of these at once."""
    try:
        fetch(seq, frame, want_truth)
    except Exception:
        pass          # the worker will retry and report properly


def truth_objects(pts4: np.ndarray, labp: Path):
    """Build detector-compatible object instances from SemanticKITTI labels.

    SemanticKITTI stores semantic id in the low 16 bits and instance id in
    the high 16 bits.  The motion module intentionally consumes the same
    compact object-class ids as the detector (1=car, 2=pedestrian,
    3=cyclist), so Ground Truth can exercise the exact same temporal code.
    The mapped grid labels remain in ``lab`` for the 2.5D renderer, while the
    original moving flag is carried as ``gt_state`` for evaluation.
    """
    raw = np.fromfile(labp, np.uint32)
    if len(raw) != len(pts4):
        raise ValueError(f'ground-truth label count {len(raw)} != point count {len(pts4)}')
    sem = (raw & 0xffff).astype(np.int64)
    inst = (raw >> 16).astype(np.int64)
    mapped = kitti.LUT[sem]
    moving = kitti.MOV[sem]

    # SemanticKITTI object semantics -> detector-compatible motion ids.
    # Moving-* ids are already represented by kitti.MOV, but their semantic
    # ids are included here so the same instance can be grouped correctly.
    car_sem = {10, 11, 13, 15, 16, 18, 20, 252, 256, 257, 258, 259}
    ped_sem = {30, 253, 254}
    cyc_sem = {31, 32, 255}
    det_cls = np.zeros(len(pts4), np.int64)
    det_cls[np.isin(sem, list(car_sem))] = 1
    det_cls[np.isin(sem, list(ped_sem))] = 2
    det_cls[np.isin(sem, list(cyc_sem))] = 3

    obj = (det_cls > 0) & (inst > 0)
    idx = np.flatnonzero(obj)
    full_cl = np.full(len(pts4), -1, np.int64)
    if not len(idx):
        return mapped.astype(np.int64), full_cl, np.empty(0, np.int64), [], dict(
            clusters=0, counts={'Car': 0, 'Pedestrian': 0, 'Cyclist': 0},
            moving_instances=0, static_instances=0)

    # Group by detector class + instance id.  Dense ids make this independent
    # of the sparse instance numbers used by SemanticKITTI.
    keys = np.stack([det_cls[idx], inst[idx]], axis=1)
    _, dense = np.unique(keys, axis=0, return_inverse=True)
    dense = dense.astype(np.int64)
    full_cl[idx] = dense
    ncl = int(dense.max()) + 1
    cluster_classes = np.empty(ncl, np.int64)
    clusters_meta = []
    moving_instances = static_instances = 0
    for k in range(ncl):
        q = idx[dense == k]
        cls = int(det_cls[q[0]])
        cluster_classes[k] = cls
        gt_dyn = bool(np.mean(moving[q]) >= 0.5)
        if gt_dyn:
            moving_instances += 1
        else:
            static_instances += 1
        clusters_meta.append(dict(
            id=k, class_id=cls, gt_state='DYNAMIC' if gt_dyn else 'STATIC',
            gt_moving_fraction=float(np.mean(moving[q])), points=int(len(q))))

    counts = {
        'Car': int(np.sum(cluster_classes == 1)),
        'Pedestrian': int(np.sum(cluster_classes == 2)),
        'Cyclist': int(np.sum(cluster_classes == 3)),
    }
    return mapped.astype(np.int64), full_cl, cluster_classes, clusters_meta, dict(
        clusters=ncl, counts=counts, moving_instances=moving_instances,
        static_instances=static_instances)


def build(seq: str, frame: str, source: str = 'model', mult: int = SURF_MULT,
          prev_frame: str | None = None, motion_dt: float = 0.1,
          previous_objects=None, next_track_id: int = 1):
    """
    source 'model'  -> labels from the PointNet detector
           'truth'  -> SemanticKITTI ground truth
    returns the dict the browser draws.
    """
    key = OUT / f'{seq}_{frame}_{source}_m{mult}.json'
    # A temporal result depends on the previous scan and previous track state,
    # so a per-frame cache entry is not sufficient.  Cache remains useful for
    # standalone/random frames, but sequential motion frames are rebuilt.
    if prev_frame is None and key.exists():
        d = json.loads(key.read_text())
        if d.get('v') == CACHE_V:
            d['cached'] = True
            return d

    t0 = time.perf_counter()
    binp, labp, fetched = fetch(seq, frame, want_truth=(source == 'truth'))
    # if a prefetch thread already downloaded it, credit that thread's time
    t_fetch = max(time.perf_counter() - t0,
                  _fetch_ms.pop((seq, frame), 0) / 1000)

    pts4 = np.fromfile(binp, np.float32).reshape(-1, 4)

    # Temporal layer: estimate ego translation directly from consecutive
    # LiDAR scans.  No KITTI pose file and no semantic labels are involved.
    # This first stage is intentionally planar/translation-only; later stages
    # will add yaw/6-DoF registration and object residuals.
    ego = None
    e = None
    if prev_frame is not None:
        prev_bin = RAW / f'{seq}_{prev_frame}.bin'
        if prev_bin.exists() and prev_bin.stat().st_size:
            prev_pts4 = np.fromfile(prev_bin, np.float32).reshape(-1, 4)
            e = ego_motion.estimate(prev_pts4, pts4, dt=motion_dt,
                                    resolution=0.25, max_range=g.maxrange)
            ego = dict(tx=e.tx, ty=e.ty, yaw=e.yaw, yaw_deg=e.yaw_deg,
                        speed_mps=e.speed_mps, speed_kmh=e.speed_kmh,
                        confidence=e.confidence, shift_px=list(e.shift_px),
                        resolution=e.resolution, iterations=e.iterations,
                        rmse=e.rmse, inlier_ratio=e.inlier_ratio,
                        method=e.method)

    info, prov = {}, None
    cluster_ids = None
    cluster_classes = None
    gt_meta = []
    t_motion = 0.0
    t0 = time.perf_counter()
    if source == 'truth':
        if labp is None:
            raise ValueError(f'no ground-truth labels published for sequence {seq}')
        lab, cluster_ids, cluster_classes, gt_meta, gt_info = truth_objects(pts4, labp)
        info.update(gt_info)
    else:
        import predict
        m_, cfg = model()
        lab, info, prov, cluster_ids, cluster_classes = predict.predict(
            pts4, m_, cfg, with_prov=True, with_clusters=True)
    t_label = time.perf_counter() - t0

    # Stage 3: ego-compensated temporal residual. This is intentionally a
    # geometric baseline: no SemanticKITTI labels and no motion network.
    object_motion = None
    if ego is not None and cluster_ids is not None and cluster_classes is not None:
        prev_bin = RAW / f'{seq}_{prev_frame}.bin'
        if prev_bin.exists() and prev_bin.stat().st_size:
            prev_pts4 = np.fromfile(prev_bin, np.float32).reshape(-1, 4)
            tm0 = time.perf_counter()
            eobj = ego_motion.object_motion(
                prev_pts4, pts4, e, cluster_ids, cluster_classes)
            t_motion = time.perf_counter() - tm0
            residual, point_motion, object_motion = eobj
            if source == 'truth':
                gt_by_id = {int(o['id']): o for o in gt_meta}
                for o in object_motion:
                    gm = gt_by_id.get(int(o['id']))
                    if gm:
                        o['gt_state'] = gm['gt_state']
                        o['gt_moving_fraction'] = gm['gt_moving_fraction']
            # Stage 4: add persistent IDs and ego-compensated relative velocity.
            object_motion, next_track_id = ego_motion.track_objects(
                previous_objects, object_motion, e, motion_dt, next_track_id=next_track_id)
            # Stage 6: optional tiny learned refinement. Geometry remains the
            # fallback and supplies the features; the MLP only runs per object.
            mlp = motion_model()
            if mlp is not None:
                object_motion = motion_mlp.refine(object_motion, ego, mlp)
        else:
            residual = np.full(len(pts4), np.nan, np.float32)
            point_motion = np.full(len(pts4), 255, np.uint8)
    else:
        residual = np.full(len(pts4), np.nan, np.float32)
        point_motion = np.full(len(pts4), 255, np.uint8)

    x, y, z = (pts4[:, i].astype(float) for i in range(3))
    k = np.hypot(x, y) < g.maxrange
    x, y, z, lab = x[k], y[k], z[k], lab[k].astype(np.int64)

    t0 = time.perf_counter()
    m = g.build(np.stack([x, y, z], 1), lab)
    t_grid = time.perf_counter() - t0

    extra = {}
    if prov is not None:
        extra['det'] = cell_provenance(m, x, y, prov[k])

    motion_cells = np.full(len(m['n']), 255, np.uint8)
    if object_motion is not None:
        pm = point_motion[k]
        objp = np.isin(lab, [g.car, g.ped]) & (pm != 255)
        if np.any(objp):
            ix = np.floor(x / g.res0).astype(np.int64)
            iy = np.floor(y / g.res0).astype(np.int64)
            lv = g.blocklevel(ix, iy)
            pk = ((lv << 62) | (((ix >> lv) & 0x7fffffff) << 31)
                  | ((iy >> lv) & 0x7fffffff))
            ck = ((m['lvl'].astype(np.int64) << 62)
                  | ((m['ix'] & 0x7fffffff) << 31) | (m['iy'] & 0x7fffffff))
            ci = np.searchsorted(ck, pk)
            ok = objp & (ci < len(ck))
            ok &= ck[np.minimum(ci, len(ck)-1)] == pk
            dc = np.zeros(len(ck), np.int32); sc = np.zeros(len(ck), np.int32)
            np.add.at(dc, ci[ok & (pm == 1)], 1)
            np.add.at(sc, ci[ok & (pm == 0)], 1)
            motion_cells[(dc == 0) & (sc > 0)] = 0
            motion_cells[(dc > sc) & (dc > 0)] = 1
        extra['motion'] = motion_cells

    t0 = time.perf_counter()
    srf = surface_json(m, x, y, z, lab, mult=mult, quiet=True, extra=extra)
    t_surf = time.perf_counter() - t0

    s = g.memstats(m)
    out = dict(
        seq=seq, frame=frame, source=source,
        tiers=srf['tiers'], zlo=srf['zlo'], zhi=srf['zhi'],
        zglo=float(min(np.frombuffer(base64.b64decode(t['zgnd']), np.int16).min()
                       for t in srf['tiers'])) / 1000,
        zghi=float(max(np.frombuffer(base64.b64decode(t['zgnd']), np.int16).max()
                       for t in srf['tiers'])) / 1000,
        npts=int(k.sum()), ncells=int(len(m['n'])), fine=int(s['fine']),
        uniform=int(s['uniform']),
        drivable=round(100 * float(m['trav'].mean()), 1),
        lvlcount=[int((m['lvl'] == i).sum()) for i in range(4)],
        clscount=[int(c) for c in np.bincount(m['cls'], minlength=8)],
        provcount=info.get('provcount', {}),
        clusters=info.get('clusters', 0),
        cars=info.get('counts', {}).get('Car', 0),
        vru=(info.get('counts', {}).get('Pedestrian', 0)
             + info.get('counts', {}).get('Cyclist', 0)),
        ms=dict(fetch=round(t_fetch*1000), label=round(t_label*1000),
                motion=round(t_motion*1000), grid=round(t_grid*1000), surface=round(t_surf*1000)),
        fetched=fetched, cached=False,
        ego=ego,
        objects=object_motion or [],
        motion=dict(dynamic_points=int(np.sum(point_motion[k] == 1)),
                    static_points=int(np.sum(point_motion[k] == 0)),
                    unknown_points=int(np.sum(point_motion[k] == 255)),
                    dynamic_objects=int(sum(o['state'] == 'DYNAMIC' for o in (object_motion or []))),
                    static_objects=int(sum(o['state'] == 'STATIC' for o in (object_motion or []))),
                    unknown_objects=int(sum(o['state'] == 'UNKNOWN' for o in (object_motion or []))),
                    mlp_enabled=bool(motion_model() is not None),
                    mlp_dynamic_objects=int(sum(o.get('mlp_state') == 'DYNAMIC' for o in (object_motion or []))),
                    mlp_overrides=int(sum(bool(o.get('mlp_override')) for o in (object_motion or []))),
                    gt_dynamic_objects=int(sum(o.get('gt_state') == 'DYNAMIC' for o in (object_motion or []))),
                    gt_static_objects=int(sum(o.get('gt_state') == 'STATIC' for o in (object_motion or [])))),
        next_track_id=next_track_id,
        v=CACHE_V,
        mult=mult,
        cam=_safe_calib(seq),
        proj=_safe_project(seq, np.stack([x, y, z], 1), lab),
    )
    key.write_text(json.dumps(out, separators=(',', ':')))
    return out


def frame_ids(seq: str, mode: str, start: int, count: int, stride: int, seed: int = 0):
    """sequential = consecutive motion; random = scattered across the sequence."""
    n = SEQ_LEN.get(seq, 1000)
    if mode == 'random':
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(n, size=min(count, n), replace=False).tolist())
    else:
        ids = [start + i*stride for i in range(count) if start + i*stride < n]
    return [f'{i:06d}' for i in ids]


# frame counts for the odometry sequences
SEQ_LEN = {'00': 4541, '01': 1101, '02': 4661, '03': 801, '04': 271, '05': 2761,
           '06': 1101, '07': 1101, '08': 4071, '09': 1591, '10': 1201}
