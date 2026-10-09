"""Visualize splitter seeds against unmodified public targets, without planning."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from impasto.stage3.targets import render_target_prefix
from impasto.stage3.splitter import split_difference
from impasto.sim.paint import Canvas
from impasto.stage2.raster import srgb_gray
from impasto.actions import Stroke
cfg=json.loads(Path('configs/stage3_refined.json').read_text())
base=Canvas().image()
fig,axes=plt.subplots(1,4,figsize=(16,5))
for ax,n in zip(axes,[5,10,15,17]):
 target=render_target_prefix('flower',n)
 ps=split_difference(srgb_gray(base),target.image_gray,components=5,
    threshold=.035, action_bounds=cfg['action_bounds'],seed=3102,
    color_subset=cfg['color_subset'],palette=cfg['palette_rgb'],
    base_rgb=base,target_rgb=target.image_rgb,opaque_colors=True)
 ax.imshow(target.image_rgb)
 for i,p in enumerate(ps):
  s=Stroke(*map(float,p.action))
  pts=s.points(np.linspace(0,1,100))
  px=np.c_[(pts[:,0]/.4572+.5)*511,(.5-pts[:,1]/.5334)*511]
  ax.plot(px[:,0],px[:,1],label=f'{i}: F={p.action[5]:.2f}')
  ax.text(*px[0],str(i))
 ax.set_xlim(115,370);ax.set_ylim(385,130);ax.set_title(f'Goal {n} / blank base')
 ax.legend(fontsize=8);ax.axis('off')
fig.tight_layout();fig.savefig('reports/stage3_refined/initialization_blank.png',dpi=150)
