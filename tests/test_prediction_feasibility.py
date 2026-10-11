"""Physical invariants of the learned grayscale-to-canvas adapter."""
import numpy as np
import torch
from impasto.stage2.raster import CropTransform
from impasto.stage3.planner import LearnedCanvasDynamics


def test_one_pigment_cannot_erase_or_overshoot_existing_luminance():
    cfg={'canvas_width_m':.4572,'canvas_height_m':.5334,'patch_size':100,
         'force_radius_m':{},'palette_rgb':[[102,102,102]],
         'residual_patch_paste':True,'convex_pigment_projection':True}
    dynamics=LearnedCanvasDynamics(torch.nn.Identity(),None,{'crop_margin_m':.025},cfg)
    canvas=np.full((128,128),.8,np.float32)
    crop=CropTransform(0,0,.1,100)
    erasure=dynamics.apply_local(canvas,np.full((100,100),.98,np.float32),crop,color=0)
    np.testing.assert_array_equal(erasure,canvas)
    overshoot=dynamics.apply_local(canvas,np.full((100,100),.1,np.float32),crop,color=0)
    assert abs(float(overshoot[64,64])-.4)<1e-6
    feasible=dynamics.apply_local(canvas,np.full((100,100),.6,np.float32),crop,color=0)
    # Residual paste retains the quantized model input contract.
    assert .59<float(feasible[64,64])<.61
    assert np.all((feasible>=.4)&(feasible<=canvas))
