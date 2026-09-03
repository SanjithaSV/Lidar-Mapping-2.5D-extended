from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import motion_mlp
from train_motion_mlp import make_samples


def metrics(y, pred):
    y=np.asarray(y).astype(int); p=np.asarray(pred).astype(int)
    tp=int(((y==1)&(p==1)).sum()); tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum())
    return dict(n=len(y), positives=int(y.sum()), accuracy=(tp+tn)/max(1,len(y)),
                precision=tp/max(1,tp+fp), recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn), tp=tp,tn=tn,fp=fp,fn=fn)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seq',default='00'); ap.add_argument('--start',type=int,default=0); ap.add_argument('--count',type=int,default=11); ap.add_argument('--dt',type=float,default=.1); ap.add_argument('--ckpt',default='motion_mlp.pt'); ap.add_argument('--out',default='stage6_eval.json'); args=ap.parse_args()
    X,y,meta=make_samples(args.seq,args.start,args.count,args.dt,allow_gaps=True)
    geom=np.array([1 if m[4]=='DYNAMIC' else 0 for m in meta])
    predictor=motion_mlp.MotionMLP(args.ckpt)
    # Reconstruct object dictionaries only for inference timing is handled separately;
    # checkpoint inference is benchmarked in benchmark_stage6.py.
    # For evaluation, use the same normalized MLP by feeding stored feature rows directly.
    ck=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    model=motion_mlp.make_model(X.shape[1]); model.load_state_dict(ck['model']); model.eval()
    mean=np.asarray(ck['mean'],np.float32); std=np.asarray(ck['std'],np.float32)
    with torch.inference_mode():
        prob=torch.sigmoid(model(torch.from_numpy((X-mean)/np.maximum(std,1e-6)))).numpy()
    mlp=(prob>=.65).astype(int)
    # Conservative runtime rule: <=.35 static, >=.65 dynamic, otherwise retain geometry.
    final=geom.copy(); final[prob>=.65]=1; final[prob<=.35]=0
    result={'dataset':{'pairs':sorted(set((int(m[0]),int(m[1])) for m in meta)), 'samples':len(meta), 'note':'Sparse labeled frames are used because the packaged artifact has no consecutive raw sequence. This is a diagnostic benchmark, not a final generalization result.'},
            'geometry':metrics(y,geom),'mlp_high_confidence':metrics(y,mlp),'hybrid':metrics(y,final),
            'mlp':{'mean_probability':float(prob.mean()),'high_confidence_fraction':float(np.mean((prob<=.35)|(prob>=.65))), 'overrides':int(np.sum(final!=geom))}}
    Path(args.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
