"""Regression cases independent of the paper's demonstration artworks."""
import cv2
import numpy as np

from impasto.stage3.splitter import split_difference, _centerline_support
from impasto.actions import Stroke


def test_centerline_support_detects_a_bridge_between_separate_marks():
    mask = np.zeros((256,256),np.uint8)
    cv2.line(mask,(65,128),(95,128),1,5)
    cv2.line(mask,(155,128),(190,128),1,5)
    action=np.array([(65/255-.5)*.4572,0,(190-65)/255*.4572,0,0,.5])
    support,gap=_centerline_support(action,mask,.4572,.5334,support_radius_px=1)
    assert support < .72
    assert gap > 40


def test_topology_filter_keeps_thin_continuous_strokes_but_removes_specks():
    base=np.ones((128,128),np.float32)
    target=base.copy()
    target[65:67,25:75]=.4
    target[10,10]=.4
    assert not split_difference(base,target,components=1,min_pixels=10)
    proposals=split_difference(base,target,components=1,min_pixels=10,
        preserve_thin_regions=True,seed=91)
    assert len(proposals)==1
    assert proposals[0].mask.sum()>=70
    assert not proposals[0].mask[10,10]


def test_force_hypotheses_retain_broad_and_narrow_alternatives():
    base=np.full((256,256,3),[245,242,234],np.uint8)
    target=base.copy()
    cv2.line(target,(85,95),(130,95),(64,113,175),15)
    cv2.line(target,(80,165),(130,165),(80,160,80),3)
    gray=lambda rgb:np.mean(rgb,axis=-1).astype(np.float32)/255
    bounds={'x_m':[-.2,.2],'y_m':[-.2,.2],'length_m':[.004,.105],
            'bend_m':[-.05,.05],'force_N':[.4,1.5]}
    pool=split_difference(gray(base),gray(target),components=2,min_pixels=10,
        kmeans_iterations=6,ransac_iterations=12,seed=991,action_bounds=bounds,
        base_rgb=base,target_rgb=target,palette=[[64,113,175],[80,160,80]],
        color_subset=[0,1],opaque_colors=True,robust_candidates=True,
        candidate_quality_checks=True,return_pool=True,
        force_hypotheses=[.4,.7,1.,1.3],palette_regions=True)
    assert {p.color for p in pool}=={0,1}
    for color in (0,1):
        assert {round(float(p.action[5]),1) for p in pool if p.color==color} >= {.4,.7,1.,1.3}
    # No path may span the gap between the independent marks.
    for p in pool:
        path=Stroke(*map(float,p.action)).points(np.linspace(0,1,50))
        ys=(.5-path[:,1]/.5334)*255
        assert np.ptp(ys)<15


def test_rgb_region_support_is_not_discarded_by_a_second_luminance_gate():
    # Two colours with near-equal luminance still have a real RGB residual.
    target=np.full((128,128), .94, np.float32)
    mask=np.zeros(target.shape, bool)
    mask[60:62,30:85]=True
    fake_base=target.copy()
    fake_base[mask]+=.2
    proposals=split_difference(fake_base,target,components=1,min_pixels=10,
        preserve_thin_regions=True,region_mask=mask,seed=12)
    assert len(proposals)==1
    assert np.all(proposals[0].mask <= mask)


def test_learned_seed_ranking_uses_the_execution_color_classifier():
    from impasto.stage3.planner import AdaptiveMPCPlanner
    from impasto.stage3.splitter import StrokeProposal

    class Dynamics:
        canvas_width=.4572
        canvas_height=.5334
        workspace_limits=(.15,.16)
        def rollout_batch(self,initial,actions,colors,opacities=None):
            result=initial.copy()
            result[:,5:10,5:10]=np.where(colors[:,0,None,None]==1,.4,.8)
            return result,[result]
    cfg={'seed':10,'palette_rgb':[[200,200,200],[100,100,100]],
         'action_bounds':{'x_m':[-.2,.2],'y_m':[-.2,.2],'length_m':[.004,.1],
                          'bend_m':[-.05,.05],'force_N':[.4,1.5]},
         'planner':{'objective_threshold':.015,'objective_dilation_kernel':3,
                    'objective_changed_weight':4,'rgb_objective_weight':0}}
    planner=AdaptiveMPCPlanner(Dynamics(),cfg)
    planner.color_target=np.ones((16,16,3),np.float32)
    planner._candidate_colors=lambda candidates,default:np.ones(candidates.shape[:2],dtype=int)
    current=np.ones((16,16),np.float32);goal=current.copy();goal[5:10,5:10]=.4
    p=StrokeProposal(np.array([0.,0.,.04,0.,0.,.7]),0,1.,goal<.9,1.,'line')
    selected=planner._select_learned_seeds(current,goal,[p],1,None)
    assert selected[0].color==1
    assert p.color==0
