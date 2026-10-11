#!/usr/bin/env python3
"""Audit equal-luminance RGB recovery on independent crossing brushwork."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv
from impasto.stage2.raster import srgb_gray
from impasto.stage3.planner import AdaptiveMPCPlanner,LearnedCanvasDynamics
from scripts.run_stage3 import load_system

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default='reports/chromatic_overlap_audit')
    args=parser.parse_args()
    cfg,s2,s1,model,device,_=load_system('configs/stage3_coverage_order.json')
    out=ROOT/args.out_dir;out.mkdir(parents=True,exist_ok=True)
    dynamics=LearnedCanvasDynamics(model,device,s2,cfg)
    planner=AdaptiveMPCPlanner(dynamics,cfg)
    palette=np.asarray(cfg['palette_rgb'])/255
    gray=palette @ [.299,.587,.114]
    pairs=sorted((abs(gray[i]-gray[j]),i,j) for i in range(24) for j in range(i+1,24)
                 if np.max(abs(palette[i]-palette[j]))>.10)[:8]
    rng=np.random.default_rng(2026101144)
    rows=[];fig,axes=plt.subplots(8,4,figsize=(12,18))
    for k,(gap,c0,c1) in enumerate(pairs):
        env=PaintingEnv(s1,seed=2026101144+k,render=False)
        try:
            angle=rng.uniform(-np.pi,np.pi)
            center=rng.uniform(-.025,.025,2)
            tangent=np.array([np.cos(angle),np.sin(angle)])
            start=center-.03*tangent
            first=Stroke(*start,.06,float(rng.uniform(-.01,.01)),angle,.95,c0)
            result=env.execute_stroke(first)
            if not result['success']:raise RuntimeError(result)
            before=env.canvas.image().astype(np.float32)/255
            angle+=rng.uniform(.5,1.2);tangent=np.array([np.cos(angle),np.sin(angle)])
            start=center-.032*tangent
            second=Stroke(*start,.064,float(rng.uniform(-.012,.012)),angle,
                          float(rng.choice([.45,.7,1.3])),c1)
            action=np.array([second.x,second.y,second.length,second.bend,second.angle,second.force])
            pred,_,coverage=dynamics.rollout_batch(srgb_gray(before)[None],action[None,None],
                                                  np.array([[c1]]),return_coverage=True)
            previous=planner._rgb_step(before[None],srgb_gray(before)[None],pred,[c1])[0]
            proposed=planner._rgb_step(before[None],srgb_gray(before)[None],pred,[c1],coverage[0])[0]
            result=env.execute_stroke(second)
            if not result['success']:raise RuntimeError(result)
            actual=env.canvas.image().astype(np.float32)/255
            changed=np.max(abs(actual-before),axis=-1)>.015
            ambiguous=changed & (abs(gray[c1]-srgb_gray(before))<=.025)
            rows.append({'seed':2026101144+k,'colours':[c0,c1],
                         'pigment_gray_difference':float(gap),'action':second.to_dict(),
                         'changed_pixels':int(changed.sum()),'ambiguous_pixels':int(ambiguous.sum()),
                         **{label:{'changed_region_rgb_l1':float(abs(image-actual)[changed].mean()),
                                  'ambiguous_region_rgb_l1':float(abs(image-actual)[ambiguous].mean()) if ambiguous.any() else None}
                            for label,image in [('previous',previous),('proposed',proposed)]}})
            yy,xx=np.where(np.max(abs(actual-before),axis=-1)>.015)
            box=(max(0,xx.min()-15),min(512,xx.max()+16),max(0,yy.min()-15),min(512,yy.max()+16))
            for ax,image,title in zip(axes[k],[before,previous,proposed,actual],
                                     ['Before','Gray-ratio RGB','With learned coverage','MuJoCo execution']):
                ax.imshow(np.clip(image,0,1));ax.set_xlim(box[:2]);ax.set_ylim(box[3],box[2]);ax.axis('off')
                if k==0:ax.set_title(title)
        finally:env.close()
    summary={'protocol':'8 near-luminance pigment pairs; random crossing curves; no artwork; same grayscale prediction, optional learned blank-reference coverage',
             'seed':2026101144,'samples':rows,
             'means':{label:{key:float(np.mean([r[label][key] for r in rows if r[label][key] is not None]))
                            for key in ['changed_region_rgb_l1','ambiguous_region_rgb_l1']}
                      for label in ['previous','proposed']}}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=145);plt.close(fig)
    print(json.dumps(summary['means'],indent=2))


if __name__=='__main__':main()
