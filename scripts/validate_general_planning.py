#!/usr/bin/env python3
"""Compare local planner revisions on held-out random MuJoCo brushwork.

Target actions are used only to generate evaluation images, never supplied
as proposals or optimization seeds. This does not reproduce paper baselines.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import time
import sys
import hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv
from impasto.stage2.raster import srgb_gray
from impasto.stage3.planner import AdaptiveMPCPlanner, LearnedCanvasDynamics, _project_action
from scripts.run_stage3 import load_system, metrics
from scripts.audit_refinement import region_metrics, bounds, image_axis

ROOT = Path(__file__).resolve().parents[1]


def sample_target(seed, config, count):
    rng = np.random.default_rng(seed)
    actions = []
    for i in range(count):
        for _ in range(500):
            action = np.array([rng.uniform(-.105, .03), rng.uniform(-.1, .1),
                rng.uniform(.025,.085), rng.uniform(-.025,.025),
                rng.uniform(-np.pi,np.pi), rng.choice([.45,.65,.95,1.3])])
            projected = _project_action(action, config['action_bounds'], (.15,.16),
                    (config['canvas_width_m'],config['canvas_height_m']))
            if projected is not None:
                actions.append(Stroke(*map(float,projected),int(rng.integers(24))))
                break
        else:
            raise RuntimeError('Target sampler failed')
    return actions


def evaluate(cfg, stage2, stage1, model, device, target, seed, count, iterations):
    cfg = copy.deepcopy(cfg)
    cfg.pop('task_color_subsets', None)
    cfg.pop('active_color_subset', None)
    cfg['seed'] = seed
    cfg['inference_batch_size'] = 8
    cfg['planner']['iterations'] = iterations
    cfg['planner']['candidates'] = 32
    env = PaintingEnv({**stage1,'canvas_resolution':cfg['canvas_resolution']},seed=seed,render=False)
    dynamics = LearnedCanvasDynamics(model,device,stage2,cfg)
    planner = AdaptiveMPCPlanner(dynamics,cfg)
    executed = 0
    records = []
    initial = srgb_gray(env.canvas.image())
    try:
        while executed < count:
            planner.pc['horizon'] = min(5,count-executed)
            before = env.canvas.image()
            plan = planner.plan(srgb_gray(before),srgb_gray(target),color_images=(before,target))
            for action,color,opacity in zip(plan.actions[:min(2,count-executed)],
                        plan.colors,plan.opacities):
                stroke=Stroke(*map(float,action),int(color),float(opacity))
                result=env.execute_stroke(stroke)
                if not result['success']:raise RuntimeError(result)
                records.append({'stroke':stroke.to_dict(),'force_rmse_N':result['force_rmse_N'],
                                'trajectory_rmse_m':result['trajectory_rmse_m']})
                executed+=1
        actual=env.canvas.image()
        return actual, {'metrics':metrics(srgb_gray(actual),actual,target,srgb_gray(target),initial,cfg),
                        'region_metrics':region_metrics(actual,target),'records':records}
    finally:
        env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=8)
    parser.add_argument('--strokes',type=int,default=3)
    parser.add_argument('--iterations',type=int,default=16)
    parser.add_argument('--seed-start',type=int,default=2026101100)
    parser.add_argument('--out-dir',default='reports/general_validation')
    parser.add_argument('--previous-config',default='reports/stage3_rollback_1107014/resolved_config.json')
    parser.add_argument('--target-dir', help='Reuse independent target PNGs from an earlier validation')
    args=parser.parse_args()
    config,stage2,stage1,model,device,checkpoint=load_system()
    previous,prev_stage2,prev_stage1,prev_model,prev_device,prev_checkpoint=load_system(args.previous_config)
    out=ROOT/args.out_dir;out.mkdir(parents=True,exist_ok=True)
    summary={'protocol':'held-out random MuJoCo targets; same full palette, 32 candidates, equal iteration budget',
             'seed_start':args.seed_start,'samples':[],'iterations':args.iterations,
             'strokes':args.strokes,'proposed_config':config,'previous_config':previous,
             'models':{label:{'checkpoint':cfg['checkpoint'],
                 'sha256':hashlib.sha256((ROOT/cfg['checkpoint']).read_bytes()).hexdigest(),
                 'stage1':s1, 'stage2':s2} for label,cfg,s1,s2 in
                 [('previous',previous,prev_stage1,prev_stage2),('proposed',config,stage1,stage2)]},
             'target_source':args.target_dir or 'current simulator random actions'}
    started=time.perf_counter()
    fig,axes=plt.subplots(args.samples,3,figsize=(10,args.samples*3),squeeze=False)
    for i in range(args.samples):
        seed=args.seed_start+i
        strokes=sample_target(seed,config,args.strokes)
        if args.target_dir:
            target=np.asarray(Image.open(ROOT/args.target_dir/f'case_{i:02d}_target.png').convert('RGB'))
        else:
            env=PaintingEnv({**stage1,'canvas_resolution':config['canvas_resolution']},seed=seed,render=False)
            try:
                for stroke in strokes:
                    result=env.execute_stroke(stroke)
                    if not result['success']:raise RuntimeError(result)
                target=env.canvas.image()
            finally:env.close()
        Image.fromarray(target).save(out/f'case_{i:02d}_target.png')
        rec={'seed':seed,'target_actions_for_audit_only':[s.to_dict() for s in strokes]}
        box=bounds(target)
        image_axis(axes[i,0],target,f'Random target {i}',box)
        for col,(label,cfg,s2,s1,m,dev) in enumerate([
                ('previous',previous,prev_stage2,prev_stage1,prev_model,prev_device),
                ('proposed',config,stage2,stage1,model,device)],start=1):
            actual,details=evaluate(cfg,s2,s1,m,dev,target,seed,args.strokes,args.iterations)
            rec[label]=details
            Image.fromarray(actual).save(out/f'case_{i:02d}_{label}.png')
            image_axis(axes[i,col],actual,f'{label}: L1={details["metrics"]["weighted_gray_l1"]:.4f}',box)
            print(f'case {i} {label}: {details["metrics"]["weighted_gray_l1"]:.6f}',flush=True)
        summary['samples'].append(rec)
        summary['wall_seconds']=time.perf_counter()-started
        (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    summary['means']={label:{key:float(np.mean([r[label]['metrics'][key] for r in summary['samples']]))
                            for key in ('weighted_gray_l1','changed_region_gray_l1')}
                     for label in ('previous','proposed')}
    summary['wins']=sum(r['proposed']['metrics']['weighted_gray_l1']<r['previous']['metrics']['weighted_gray_l1']
                        for r in summary['samples'])
    (out/'metrics.json').write_text(json.dumps(summary,indent=2))
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({'means':summary['means'],'wins':summary['wins']},indent=2))


if __name__=='__main__':main()
