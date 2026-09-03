from __future__ import annotations
import argparse, json, statistics, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import motion_mlp
from server import pipeline as P

def run_pipeline(pairs, base, enabled):
    rows=[]
    if not enabled:
        ck=P.MOTION_MLP_PATH
        bak=ck.with_suffix('.pt.bak_bench')
        if ck.exists(): ck.rename(bak)
    P._motion_mlp=None
    try:
        # Warm one frame so JIT/model load isn't mistaken for steady-state throughput.
        for i,(f,prev) in enumerate(pairs):
            t=time.perf_counter(); d=P.build('00',f,source='model',prev_frame=prev,motion_dt=max(.1,(int(f)-int(prev))*.1),mult=base+i); wall=(time.perf_counter()-t)*1000
            rows.append({'frame':f,'wall_ms':wall,'label_ms':d['ms']['label'],'motion_ms':d['ms']['motion'],'grid_ms':d['ms']['grid'],'surface_ms':d['ms']['surface'],'objects':len(d['objects']),'mlp_enabled':d['motion']['mlp_enabled']})
    finally:
        if not enabled:
            bak=P.MOTION_MLP_PATH.with_suffix('.pt.bak_bench')
            if bak.exists(): bak.rename(P.MOTION_MLP_PATH)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pairs',default='000100:000000,000150:000100,000300:000150'); ap.add_argument('--out',default='artifacts/stage6_speed.json'); args=ap.parse_args()
    pairs=[tuple(x.split(':')) for x in args.pairs.split(',')]
    enabled=run_pipeline(pairs,800,True)
    disabled=run_pipeline(pairs,900,False)
    # Direct object-level MLP timing.
    j=json.loads((ROOT/'cache/frames/00_000150_model_m700.json').read_text()); objs=j['objects']; ego=j['ego']; pred=motion_mlp.MotionMLP(ROOT/'motion_mlp.pt')
    for _ in range(20): pred.predict_proba(objs,ego)
    us=[]
    for _ in range(1000):
        t=time.perf_counter(); pred.predict_proba(objs,ego); us.append((time.perf_counter()-t)*1e6)
    result={'pipeline':{'mlp_enabled':enabled,'mlp_disabled':disabled,'steady_median_wall_ms_enabled':statistics.median(r['wall_ms'] for r in enabled[1:]),'steady_median_wall_ms_disabled':statistics.median(r['wall_ms'] for r in disabled[1:])},'mlp_inference':{'objects_per_call':len(objs),'median_us_per_call':statistics.median(us),'p95_us_per_call':sorted(us)[950],'median_us_per_object':statistics.median(us)/max(1,len(objs))},'note':'Pipeline pairs are sparse packaged frames, so wall time is a runtime benchmark, not a motion-accuracy benchmark.'}
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
