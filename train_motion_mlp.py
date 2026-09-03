"""Train the Stage-6 object-level motion MLP from labeled frame pairs.

This script deliberately keeps data preparation separate from inference. It
uses the existing detector + Stage-5 geometry to create object features, then
assigns a ground-truth object label from SemanticKITTI's moving-* semantic
IDs. It is an optional training/evaluation utility; the runtime does not need
labels or this script once motion_mlp.pt exists.

Example:
  python train_motion_mlp.py --seq 00 --start 0 --count 300 --epochs 30

The script expects consecutive raw .bin/.label files in cache/raw. Use the
existing pipeline/fetch mechanism to populate them first.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import motion
import motion_mlp
import kitti
import predict

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'cache' / 'raw'
OUT = ROOT / 'motion_mlp.pt'


def load_frame(seq, fid):
    b = RAW / f'{seq}_{fid:06d}.bin'
    l = RAW / f'{seq}_{fid:06d}.label'
    if not b.exists() or not l.exists():
        return None
    return np.fromfile(b, np.float32).reshape(-1,4), np.fromfile(l, np.uint32)


def make_samples(seq, start, count, dt, allow_gaps=False):
    model, cfg = predict.load()
    rows, ys, meta = [], [], []
    prev_objects = []
    next_id = 1
    if allow_gaps:
        ids = sorted(int(p.stem.split('_')[-1]) for p in RAW.glob(f'{seq}_*.bin'))
        ids = [f for f in ids if f >= start][:count]
        pairs = list(zip(ids[:-1], ids[1:]))
    else:
        ids = list(range(start, start + count))
        pairs = [(f-1, f) for f in ids]
    for f0, f in pairs:
        pair0 = load_frame(seq, f0)
        pair1 = load_frame(seq, f)
        if pair0 is None or pair1 is None:
            continue
        p0, _ = pair0; p1, raw1 = pair1
        pair_dt = dt * max(1, f - f0) if allow_gaps else dt
        # Detector output and Stage-5 geometric features.
        lab, info, prov, cids, cclasses = predict.predict(
            p1, model, cfg, with_prov=True, with_clusters=True)
        e = motion.estimate(p0, p1, dt=pair_dt, resolution=0.25, max_range=50.0)
        _, _, objs = motion.object_motion(p0, p1, e, cids, cclasses)
        objs, next_id = motion.track_objects(prev_objects, objs, e, pair_dt, next_track_id=next_id)
        ego = dict(speed_mps=e.speed_mps, confidence=e.confidence)
        # SemanticKITTI moving flag per point. Match a predicted cluster by
        # its point membership; >=50% moving points => moving object label.
        mov = kitti.MOV[(raw1 & 0xffff).astype(np.int64)]
        for o in objs:
            cid = int(o['id'])
            idx = np.flatnonzero(cids == cid)
            if len(idx) < 8:
                continue
            target = float(np.mean(mov[idx]) >= 0.5)
            rows.append(motion_mlp.object_features(o, ego))
            ys.append(target)
            meta.append((f0, f, int(o.get('track_id', -1)), int(target), str(o.get('state', 'UNKNOWN'))))
        prev_objects = objs
        print(f'pair {f0}->{f}: {len(objs)} objects, samples={len(rows)}')
    if not rows:
        raise RuntimeError('No samples. Populate consecutive .bin/.label files in cache/raw.')
    return np.stack(rows).astype(np.float32), np.asarray(ys, np.float32), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='00')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--count', type=int, default=300)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--allow-gaps', action='store_true', help='pair available labeled frames even when frame IDs are not consecutive')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()

    X, y, meta = make_samples(args.seq, args.start, args.count, args.dt, allow_gaps=args.allow_gaps)
    # Preserve temporal order: the tail is the validation holdout.
    n = len(X); cut = max(1, int(0.8*n))
    Xtr, Xva, ytr, yva = X[:cut], X[cut:], y[:cut], y[cut:]
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-5] = 1.0
    xt = torch.from_numpy((Xtr-mean)/std); yt = torch.from_numpy(ytr)
    xv = torch.from_numpy((Xva-mean)/std); yv = torch.from_numpy(yva)
    model = motion_mlp.make_model(X.shape[1])
    # Balance the two classes when both are present.
    pos = max(float(ytr.sum()), 1.0); neg = max(float(len(ytr)-ytr.sum()), 1.0)
    pos_weight = torch.tensor([neg/pos], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs+1):
        model.train(); total=0.0
        for xb, yb in loader:
            opt.zero_grad()
            logits=model(xb)
            loss=F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            loss.backward(); opt.step(); total += float(loss.detach())*len(xb)
        model.eval()
        with torch.no_grad():
            pv=torch.sigmoid(model(xv)) if len(xv) else torch.empty(0)
            acc=float(((pv>=0.5)==(yv>=0.5)).float().mean()) if len(yv) else float('nan')
        print(f'epoch {epoch:02d} loss={total/len(xt):.4f} val_acc={acc:.3f}')
    ckpt={'feature_version':motion_mlp.FEATURE_VERSION,
          'feature_names':motion_mlp.FEATURE_NAMES,
          'mean':mean.tolist(),'std':std.tolist(),
          'model':model.state_dict(),
          'train_meta':vars(args),
          'sample_meta':meta}
    torch.save(ckpt, args.out)
    print(f'Wrote {args.out}')

if __name__ == '__main__':
    main()
