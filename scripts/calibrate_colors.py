"""Measure a simulation-specific 25x25 opaque pigment patch database."""
from pathlib import Path
import json
import os
import numpy as np
import cv2
from PIL import Image
from impasto.sim.mujoco_env import PaintingEnv
from impasto.actions import Stroke
root=Path(__file__).resolve().parents[1]
config_path=Path(os.environ.get('IMPASTO_STAGE3_CONFIG','configs/stage3.json'))
if not config_path.is_absolute():
 config_path=root/config_path
cfg=json.loads(config_path.read_text())
if cfg.get('deployment_palette_rgb'):
 cfg['palette_rgb']=cfg['deployment_palette_rgb']
s1=json.loads((root/cfg['stage1_config']).read_text())
s1['paint']['palette_rgb']=cfg.get('deployment_palette_rgb',cfg['palette_rgb'])
env=PaintingEnv(s1,seed=6127)
out=root/cfg.get('report_dir','reports/stage3')
out.mkdir(parents=True,exist_ok=True)
means=[];covariances=[];patches=[];records=[]
try:
 for color in range(24):
  samples=[]
  for force in [.85,1.2]:
   env.canvas.rgb[:]=env.canvas.initial;env.canvas.film[:]=0;env.brush.clean()
   result=env.execute_stroke(Stroke(-.035,0,.06,0,0,force,color,1.))
   if not result['success']:raise RuntimeError(result)
   im=env.canvas.image()
   x,y=env.canvas.pixel([-.005,0])
   # 4 mm center window samples the interior pigment, excluding canvas.
   dx=.002/env.canvas.width*511;dy=.002/env.canvas.height*511
   patch=cv2.resize(im[round(y-dy):round(y+dy)+1,round(x-dx):round(x+dx)+1],(25,25))
   samples.append(patch)
   records.append(dict(color=color,force=force,execution=result))
  pixels=np.concatenate(samples).reshape(-1,3).astype(float)/255
  linear=np.where(pixels<=.04045,pixels/12.92,((pixels+.055)/1.055)**2.4)
  means.append(linear.mean(0));covariances.append(np.cov(linear,rowvar=False)+np.eye(3)*.002)
  patches.append(samples[0])
finally:env.close()
db={'kind':'simulated opaque central 25x25 patches','samples_per_color':2,
    'forces_N':[.85,1.2],'covariance_regularization':.002,
    'means_linear_rgb':np.asarray(means).tolist(),'covariances':np.asarray(covariances).tolist(),
    'palette_rgb':s1['paint']['palette_rgb'],'seed':6127,'records':records}
(out/'color_database.json').write_text(json.dumps(db,indent=2))
sheet=np.zeros((4*75,6*75,3),dtype=np.uint8)
for i,patch in enumerate(patches):sheet[(i//6)*75:(i//6+1)*75,(i%6)*75:(i%6+1)*75]=cv2.resize(patch,(75,75),interpolation=cv2.INTER_NEAREST)
Image.fromarray(sheet).save(out/'color_database.png')
print(f'color database: {out / "color_database.json"}')
print('48 opaque patch measurements complete')
