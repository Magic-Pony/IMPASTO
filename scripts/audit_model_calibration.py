#!/usr/bin/env python3
"""Compare raw and train-only BatchNorm calibration without artwork inputs.

Calibrate layers in forward order using population moments of unaugmented
training images. All weights, dropout, architecture and validation data stay
fixed; validation selects among reported variants, so is not an untouched test.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from impasto.stage2.data import Stage2Dataset
from impasto.stage2.model import ImpastoUNet

ROOT=Path(__file__).resolve().parents[1]


@torch.no_grad()
def calibrate(model, loader, device):
    model.eval()
    for layer in model.modules():
        if not isinstance(layer, nn.BatchNorm2d):
            continue
        total=torch.zeros(layer.num_features, device=device, dtype=torch.float64)
        square=total.clone()
        count=0
        def accumulate(module, inputs):
            nonlocal count
            x=inputs[0].double()
            total.add_(x.sum(dim=(0,2,3)))
            square.add_((x*x).sum(dim=(0,2,3)))
            count+=x.shape[0]*x.shape[2]*x.shape[3]
        handle=layer.register_forward_pre_hook(accumulate)
        try:
            for batch in loader:
                model(batch['image'].to(device),batch['action'].to(device))
        finally:
            handle.remove()
        mean=total/count
        var=(square/count-mean*mean).clamp_min(0)
        layer.running_mean.copy_(mean.to(layer.running_mean.dtype))
        layer.running_var.copy_(var.to(layer.running_var.dtype))


@torch.no_grad()
def evaluate(model, loader, device, threshold):
    errors=[]; masks=[]; unions=[]
    for batch in loader:
        image=batch['image'].to(device); target=batch['target'].to(device)
        pred=model(image,batch['action'].to(device))
        mask=(target-image[:,0:1]).abs()>threshold
        union=mask | ((pred-image[:,0:1]).abs()>threshold)
        errors.append((pred-target).abs().cpu().numpy())
        masks.append(mask.cpu().numpy());unions.append(union.cpu().numpy())
    error=np.concatenate(errors);mask=np.concatenate(masks);union=np.concatenate(unions)
    return {'full_patch_l1':float(error.mean()),
            'changed_region_l1':float(error[mask].mean()),
            'union_region_l1':float(error[union].mean()),
            'background_l1':float(error[~mask].mean())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/stage2_stable.json')
    parser.add_argument('--out-dir',default='reports/model_calibration_audit')
    args=parser.parse_args()
    cfg=json.loads((ROOT/args.config).read_text())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train=Stage2Dataset(ROOT/cfg['dataset_dir']/'stage2_selfplay.npz',cfg['action_bounds'],False,
                        config=cfg['training'],episode_ids=cfg['split']['train_episode_ids'])
    val=Stage2Dataset(ROOT/cfg['dataset_dir']/'stage2_selfplay.npz',cfg['action_bounds'],False,
                      config=cfg['training'],episode_ids=cfg['split']['validation_episode_ids'])
    train_loader=DataLoader(train,batch_size=120,shuffle=False)
    val_loader=DataLoader(val,batch_size=64,shuffle=False)
    out=ROOT/args.out_dir;out.mkdir(parents=True,exist_ok=True)
    summary={'protocol':'unaugmented training-only population BatchNorm moments; dropout disabled; no gradient or artwork; validation comparison, not independent test',
             'train_samples':len(train),'validation_samples':len(val),'variants':{}}
    for name,path in [('old','outputs/stage2_refined/impasto_unet_best.pt'),
                      ('best','outputs/stage2_stable/impasto_unet_best.pt'),
                      ('last','outputs/stage2_stable/impasto_unet_last.pt')]:
        ckpt=torch.load(ROOT/path,map_location=device,weights_only=False)
        model=ImpastoUNet(channels=tuple(cfg['unet_channels']),
                         action_embedding_channels=cfg['action_embedding_channels'],
                         dropout=cfg['dropout_probability']).to(device)
        model.load_state_dict(ckpt['state_dict']);model.eval()
        for calibrated in (False,True):
            label=name+('_calibrated' if calibrated else '_raw')
            if calibrated:
                calibrate(model,train_loader,device)
                ckpt=copy.deepcopy(ckpt)
                ckpt['state_dict']=model.state_dict()
                ckpt['normalization_calibration']={'protocol':summary['protocol'],
                                                   'dataset_config':args.config,'source':path}
                dest=ROOT/'outputs/stage2_stable'/f'impasto_unet_{label}.pt'
                torch.save(ckpt,dest)
            else:
                dest=ROOT/path
            result=evaluate(model,val_loader,device,cfg['training']['changed_pixel_threshold'])
            result.update(checkpoint=str(dest.relative_to(ROOT)),
                          sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
            summary['variants'][label]=result
            (out/'metrics.json').write_text(json.dumps(summary,indent=2))
            print(label,json.dumps(result),flush=True)


if __name__=='__main__':main()
