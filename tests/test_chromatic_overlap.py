"""Generic pigment/ordering cases, independent of demonstration images."""
import numpy as np
from impasto.stage3.planner import AdaptiveMPCPlanner


def test_equal_luminance_pigments_still_change_rgb_when_coverage_is_known():
    # Gray alone cannot distinguish a red-to-green colour transition here.
    palette=np.array([[.7,.2,.3],[.2,.2+.5*.299/.587,.3]],np.float32)
    planner=AdaptiveMPCPlanner(None,{'seed':1,'action_bounds':{},'planner':{},
                                    'palette_rgb':(palette*255).tolist()})
    rgb=np.broadcast_to(palette[0],(1,4,4,3)).copy()
    gray=rgb @ np.array([.299,.587,.114],np.float32)
    np.testing.assert_allclose(gray[0,0,0],palette[1] @ [.299,.587,.114],atol=1e-7)
    before=planner._rgb_step(rgb,gray,gray,[1])
    np.testing.assert_array_equal(before,rgb)
    coverage=np.zeros((1,4,4),np.float32);coverage[:,1:3,1:3]=.6
    after=planner._rgb_step(rgb,gray,gray,[1],coverage)
    np.testing.assert_allclose(after[0,1,1],.4*palette[0]+.6*palette[1],atol=1e-7)
    np.testing.assert_array_equal(after[0,0,0],rgb[0,0,0])


def test_seed_order_search_preserves_colours_and_selects_overlap_order():
    class Dynamics:
        canvas_width=.4572
        canvas_height=.5334
        workspace_limits=(.15,.16)
        def rollout_batch(self,initial,actions,colors,opacities=None):
            state=initial.copy();frames=[]
            for t in range(actions.shape[1]):
                state[:,3:7,3:7]=np.where(colors[:,t,None,None]==0,.3,.6)
                frames.append(state.copy())
            return state,frames
        def rollout(self,initial,actions,colors,opacities=None):
            result,frames=self.rollout_batch(initial[None],actions[None],np.array(colors)[None])
            return result[0],[frame[0] for frame in frames]
    cfg={'seed':1,'palette_rgb':[[80,80,80],[150,150,150]],'action_bounds':{},
         'planner':{'horizon':2,'iterations':0,'seed_order_search':True,
                    'update_colors_with_geometry':False,'initial_sigma':[.01]*6,
                    'objective_threshold':.015,'objective_dilation_kernel':3,
                    'objective_changed_weight':4}}
    planner=AdaptiveMPCPlanner(Dynamics(),cfg)
    initial=np.ones((10,10),np.float32);goal=initial.copy();goal[3:7,3:7]=.3
    actions=np.array([[0,0,.02,0,0,.7],[.03,0,.02,0,0,.7]],np.float32)
    plan=planner.plan(initial,goal,colors=[0,1],initial=actions)
    assert plan.colors==[1,0]
    np.testing.assert_array_equal(plan.actions,actions[::-1])
    assert plan.final_cost==0
