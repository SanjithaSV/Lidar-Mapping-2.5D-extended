from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from train_motion_mlp import make_samples

ap=argparse.ArgumentParser(); ap.add_argument('--seq',default='00'); ap.add_argument('--start',type=int,default=0); ap.add_argument('--count',type=int,default=11); ap.add_argument('--dt',type=float,default=.1); ap.add_argument('--out',default='artifacts/motion_stage6_dataset.npz'); args=ap.parse_args()
X,y,meta=make_samples(args.seq,args.start,args.count,args.dt,allow_gaps=True)
out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
pairs=np.array([[m[0],m[1]] for m in meta],np.int32)
track=np.array([m[2] for m in meta],np.int32)
geom=np.array([1 if m[4]=='DYNAMIC' else 0 for m in meta],np.int8)
np.savez_compressed(out,X=X,y=y,geom=geom,pair=pairs,track_id=track,feature_names=np.array(__import__('motion_mlp').FEATURE_NAMES))
summary={'samples':int(len(y)),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),'geometry_dynamic':int(geom.sum()),'pairs':sorted({tuple(x) for x in pairs.tolist()}),'source_note':'Available packaged SemanticKITTI-labeled frames are sparse (mostly 50-150 frame gaps); this dataset is diagnostic only, not a final training/validation benchmark.'}
out.with_suffix('.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2)); print(out)
