"""Controlled comparison of legacy post-hoc opacity and actual deposition."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from impasto.actions import Stroke
from impasto.sim.mujoco_env import PaintingEnv
from impasto.stage2.model import ImpastoUNet
from impasto.stage2.raster import srgb_gray
from impasto.stage3.planner import LearnedCanvasDynamics
root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'configs/stage3.json').read_text());s2=json.loads((root/cfg['stage2_config']).read_text());s1=json.loads((root/cfg['stage1_config']).read_text())
model=ImpastoUNet(channels=tuple(s2['unet_channels']),action_embedding_channels=s2['action_embedding_channels'],dropout=s2['dropout_probability']).cuda()
model.load_state_dict(torch.load(root/cfg['checkpoint'],weights_only=False)['state_dict'])
d=LearnedCanvasDynamics(model,torch.device('cuda'),s2,cfg)
env=PaintingEnv(s1,seed=4491)
rows=[];fig,axes=plt.subplots(4,3,figsize=(9,10))
try:
 for i,opacity in enumerate([.25,.5,1.,5.]):
  env.canvas.rgb[:]=env.canvas.initial;env.canvas.film[:]=0;env.brush.clean()
  act=np.array([-.05,0,.07,.02,.1,1.0])
  before=srgb_gray(env.canvas.image())
  pred,*_=d.step(before,act,8,opacity)
  alpha=d.effective_opacity(opacity)
  result=env.execute_stroke(Stroke(*act,8,alpha))
  actual=srgb_gray(env.canvas.image());mask=abs(actual-before)>.02
  row=dict(ratio=opacity,effective_opacity=alpha,changed_region_l1=float(abs(pred-actual)[mask].mean()),actual_contrast=float((before-actual)[mask].mean()),predicted_contrast=float((before-pred)[mask].mean()),force_rmse=result['force_rmse_N']);rows.append(row)
  for ax,img,title in zip(axes[i],[pred,actual,abs(pred-actual)],[f'Predicted alpha={alpha:.2f}','Executed','Absolute error']):
   ax.imshow(img[215:285,180:300],cmap='gray' if title!='Absolute error' else 'magma',vmin=0,vmax=1 if title!='Absolute error' else .4);ax.set_title(title);ax.axis('off')
finally:env.close()
fig.tight_layout();fig.savefig(root/'reports/stage3_refined/legacy_opacity_audit.png',dpi=150)
(root/'reports/stage3_refined/legacy_opacity_audit.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows))
