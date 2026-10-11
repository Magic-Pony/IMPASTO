#!/usr/bin/env python3
"""Audit color sampling on independent MuJoCo marks; no artwork inputs."""
from __future__ import annotations
import argparse
import copy
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
from impasto.stage3.planner import AdaptiveMPCPlanner, LearnedCanvasDynamics
from scripts.run_stage3 import load_system
from scripts.collect_selfplay import draw_stroke

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default='configs/stage3_refined.json')
    ap.add_argument('--out-dir',default='reports/color_sampling_audit')
    args=ap.parse_args()
    cfg,s2,s1,model,device,_=load_system(args.config)
    s1=json.loads((ROOT/'configs/stage1_stable.json').read_text())
    variants={}
    for name,dense in [('sparse',False),('footprint_disk',True)]:
        c=copy.deepcopy(cfg);c['planner']['color_footprint_disk']=dense
        variants[name]=AdaptiveMPCPlanner(LearnedCanvasDynamics(model,device,s2,c),c)
    rng=np.random.default_rng(2026101213)
    records=[];examples=[]
    for color in range(24):
        for force in [.45,.7,1.2]:
            seed=2026101213+len(records)
            stroke=draw_stroke(rng,s2,color)
            stroke=Stroke(stroke.x,stroke.y,stroke.length,stroke.bend,stroke.angle,force,color)
            env=PaintingEnv(s1,seed=seed,render=False)
            try:
                if len(records)%2:
                    center=stroke.points(np.array([.5]))[0]
                    # An independent, darker crossing mark under the test stroke.
                    a=stroke.angle+np.pi/2
                    start=center-.015*np.array([np.cos(a),np.sin(a)])
                    res=env.execute_stroke(Stroke(*start,.03,0.,a,.8,(color+5)%24))
                    if not res['success']:raise RuntimeError(res)
                res=env.execute_stroke(stroke)
                if not res['success']:raise RuntimeError(res)
                rgb=env.canvas.image()
                action=np.array([stroke.x,stroke.y,stroke.length,stroke.bend,stroke.angle,stroke.force])
                rec={'seed':seed,'stroke':stroke.to_dict(),'overlap':bool(len(records)%2),'classified':{}}
                for name,planner in variants.items():
                    planner.color_target=rgb.astype(np.float32)/255
                    rec['classified'][name]=int(planner._candidate_colors(action[None,None],[0])[0,0])
                records.append(rec)
                if any(c!=color for c in rec['classified'].values()):
                    pts=stroke.points(np.linspace(0,1,100))
                    center=stroke.points(np.array([.5]))[0]
                    px=(center[0]/env.canvas.width+.5)*511
                    py=(.5-center[1]/env.canvas.height)*511
                    examples.append((rgb,px,py,rec))
                print(f'{len(records):02d}/72 color={color} force={force}: {rec["classified"]}',flush=True)
            finally:env.close()
    summary={'protocol':'24 colors x 3 forces, random reachable geometry, alternating blank/overlap. Fixed seed, no artwork targets.',
        'samples':records,'accuracy':{name:sum(r['classified'][name]==r['stroke']['color'] for r in records)/len(records) for name in variants}}
    out=ROOT/args.out_dir;out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    if examples:
        selected=examples[:12]
        fig,axes=plt.subplots(3,4,figsize=(12,9))
        for ax,(rgb,x,y,r) in zip(axes.flat,selected):
            ax.imshow(rgb);ax.set_xlim(x-32,x+32);ax.set_ylim(y+32,y-32);ax.axis('off')
            ax.set_title(f"F={r['stroke']['force']:.2f} true={r['stroke']['color']}\n{r['classified']}",fontsize=8)
        for ax in list(axes.flat)[len(selected):]:ax.axis('off')
        fig.tight_layout();fig.savefig(out/'misclassified.png',dpi=150);plt.close(fig)
    print(json.dumps(summary['accuracy'],indent=2))

if __name__=='__main__':main()
