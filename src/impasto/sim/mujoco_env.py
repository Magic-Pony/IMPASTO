"""MuJoCo Panda painting scene and deterministic Cartesian joint-space servo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from impasto.actions import Stroke
from impasto.sim.paint import BrushState, Canvas


ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "assets/scenes/paint_scene.xml"


def _id(model, kind, name):
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return idx


@dataclass
class StepObservation:
    time: float
    tip_xyz: np.ndarray
    target_xyz: np.ndarray
    normal_force: float
    contacting: bool
    brush_tilt_deg: float
    q: np.ndarray


class PaintingEnv:
    """One independent Panda + scene + canvas instance (safe for process workers)."""

    def __init__(self, config=None, seed=0, render=False):
        self.config = dict(config or {})
        self.seed = int(seed)
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep) * int(self.config.get("control_decimation", 2))
        self.decimation = int(self.config.get("control_decimation", 2))
        self.tip_site = _id(self.model, mujoco.mjtObj.mjOBJ_SITE, "brush_tip_site")
        self.tip_geom = _id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "brush_tip")
        self.canvas_geom = _id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "canvas_surface")
        self.camera_ids = {
            name: _id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("top_camera", "canvas_camera", "overview_camera")
        }
        self.canvas = Canvas(
            float(self.config.get("canvas_width", .4572)),
            float(self.config.get("canvas_height", .5334)),
            int(self.config.get("canvas_resolution", 512)),
            seed=self.seed,
        )
        self.brush = BrushState()
        self.renderer = None
        self.render_enabled = bool(render)
        if self.render_enabled:
            w = int(self.config.get("camera_width", 960))
            h = int(self.config.get("camera_height", 540))
            self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self.canvas_tex = _id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "canvas_texture")
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.home_q = self.data.qpos[:7].copy()
        self.home_tip_rot = self.data.site_xmat[self.tip_site].reshape(3, 3).copy()
        self.tip_world_offset = self.data.site_xpos[self.tip_site].copy() - self.data.xpos[
            _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ]
        self.force_target = 0.0
        self.force_integral = 0.0
        self.phase = "IDLE"
        self.events = [{"time": float(self.data.time), "phase": self.phase}]
        self.observations: list[StepObservation] = []
        self.canvas.reset_path()
        self.brush = BrushState()
        return self.observe(np.array([np.nan, np.nan, np.nan]))

    def _set_phase(self, phase):
        if phase != self.phase:
            self.phase = phase
            self.events.append({"time": float(self.data.time), "phase": phase})

    def tip_position(self):
        return self.data.site_xpos[self.tip_site].copy()

    def brush_axis(self):
        """Brush local +Z axis in world coordinates; nominally points down."""
        return self.data.site_xmat[self.tip_site].reshape(3, 3)[:, 2].copy()

    def brush_tilt_deg(self):
        return float(np.degrees(np.arccos(np.clip(abs(self.brush_axis()[2]), -1.0, 1.0))))

    def normal_force(self):
        """Calibrated virtual F/T sensor reading in Newtons.

        MuJoCo's stiff penalty contact produces a large impulse at the first
        discrete contact for this small brush tip.  The configured 0.08 scale
        is the stage-1 sensor calibration used here so the reported force spans the
        paper's 0.1 N sensor floor and the action values used in demos.
        Raw contact forces remain available through ``normal_force_raw``.
        """
        return float(self.config.get("normal_force_scale", 0.08)) * self.normal_force_raw()

    def normal_force_raw(self):
        total = 0.0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if {int(c.geom1), int(c.geom2)} == {self.tip_geom, self.canvas_geom}:
                wrench = np.zeros(6, dtype=float)
                mujoco.mj_contactForce(self.model, self.data, i, wrench)
                total += abs(float(wrench[0]))
        return total

    def _ik(self, xyz, seed=None, max_iter=30):
        xyz = np.asarray(xyz, dtype=float)
        q0 = self.data.qpos[:7].copy()
        q = q0.copy() if seed is None else np.asarray(seed, dtype=float).copy()
        old_arm = self.data.qpos[:7].copy()
        old_finger = self.data.qpos[7:].copy()
        damping = 0.0025
        target_rot = Rotation.from_matrix(self.home_tip_rot)
        for _ in range(max_iter):
            self.data.qpos[:7] = q
            self.data.qpos[7:] = old_finger
            mujoco.mj_forward(self.model, self.data)
            current = self.data.site_xpos[self.tip_site].copy()
            current_rot = self.data.site_xmat[self.tip_site].reshape(3, 3).copy()
            pos_err = xyz - current
            rot_err = (target_rot * Rotation.from_matrix(current_rot).inv()).as_rotvec()
            if np.linalg.norm(pos_err) < 0.0007 and np.linalg.norm(rot_err) < 0.03:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tip_site)
            jac = np.vstack((jacp[:, :7], .45 * jacr[:, :7]))
            err = np.r_[pos_err, .45 * rot_err]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            # A weak null-space posture term avoids joint-limit drift without
            # changing the paper's task-space stroke action.
            jpinv = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), np.eye(6))
            null = np.eye(7) - jpinv @ jac
            dq += null @ (0.025 * (self.home_q - q))
            scale = min(1.0, .065 / max(float(np.max(np.abs(dq))), 1e-12))
            q += scale * dq
            for j in range(7):
                jid = self.model.joint(f"joint{j+1}").id
                lo, hi = self.model.jnt_range[jid]
                q[j] = np.clip(q[j], lo + .006, hi - .006)
        self.data.qpos[:7] = old_arm
        self.data.qpos[7:] = old_finger
        mujoco.mj_forward(self.model, self.data)
        return q

    def observe(self, target_xyz):
        tip = self.tip_position()
        return StepObservation(
            time=float(self.data.time), tip_xyz=tip,
            target_xyz=np.asarray(target_xyz, dtype=float).copy(),
            normal_force=self.normal_force(),
            contacting=self.normal_force() > 0.02,
            brush_tilt_deg=self.brush_tilt_deg(),
            q=self.data.qpos[:7].copy(),
        )

    def step(self, target_xyz, phase=None, paint=True):
        target_xyz = np.asarray(target_xyz, dtype=float)
        if phase:
            self._set_phase(phase)
        q_target = self._ik(target_xyz)
        self.data.ctrl[:7] = q_target
        # The Menagerie model's high-gain general actuators provide the Panda
        # servo behavior; this bounded Cartesian-to-joint correction removes
        # the small gravity/friction steady-state error that otherwise matters
        # at the 10^-4 m brush-to-canvas clearance scale.
        applied_gain = 1400.0
        self.data.qfrc_applied[:7] = np.clip(
            applied_gain * (q_target - self.data.qpos[:7]) - 26.0 * self.data.qvel[:7],
            -80.0, 80.0,
        )
        if self.model.nu > 7:
            self.data.ctrl[7] = 255.0
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        force = self.normal_force()
        tip = self.tip_position()
        contacting = force > .02
        if paint:
            self.canvas.update(tip[:2] - np.array([.50, 0.0]), force, self.brush, self.dt, contacting)
        obs = StepObservation(float(self.data.time), tip, target_xyz.copy(), force, contacting,
                             self.brush_tilt_deg(), self.data.qpos[:7].copy())
        self.observations.append(obs)
        return obs

    def move_linear(self, xyz, phase="MOVE", speed=.30, render_hook=None, paint=False):
        start = self.tip_position()
        xyz = np.asarray(xyz, dtype=float)
        duration = float(np.linalg.norm(xyz - start)) / max(speed, .005)
        n = max(1, int(np.ceil(duration / self.dt)))
        for i in range(1, n + 1):
            p = i / n
            obs = self.step(start * (1-p) + xyz * p, phase=phase, paint=paint)
            if render_hook:
                render_hook(self, obs)
        return self.tip_position()

    def _approach_force(self, target_xy, target_force, render_hook=None):
        # The canvas top is 0.059 m and the brush-tip radius is 0.010 m.
        # Start 5 mm above first contact, then acquire the requested force.
        surface_z = .057 + .002
        contact_center_z = surface_z + .010
        target = np.array([target_xy[0] + .50, target_xy[1], contact_center_z + .005])
        self.move_linear(target, "APPROACH", speed=.12, render_hook=render_hook)
        z_ref = contact_center_z + .005
        filtered = 0.0
        ema = float(self.config.get("force_ema", 0.90))
        stable = 0
        nmax = int(float(self.config.get("force_timeout", 4.0)) / self.dt)
        for _ in range(nmax):
            f = self.normal_force()
            filtered = ema * filtered + (1.0 - ema) * f
            err = float(target_force) - filtered
            # Filtered integral force acquisition: positive error lowers the tip.
            z_ref -= float(np.clip(0.00004 * err, -0.00006, 0.00008))
            z_ref = max(contact_center_z - .0015, z_ref)
            obs = self.step([target_xy[0] + .50, target_xy[1], z_ref], phase="FORCE_ACQUIRE")
            if render_hook:
                render_hook(self, obs)
            stable = stable + 1 if abs(filtered-target_force) <= float(self.config.get("force_tolerance", .12)) else 0
            if stable * self.dt >= float(self.config.get("force_settle_time", .16)):
                self._set_phase("STROKE")
                return z_ref, True
        return z_ref, False

    def execute_stroke(self, stroke: Stroke, render_hook=None):
        start_wall = time.perf_counter()
        force_start = len(self.observations)
        if self.brush.color != stroke.color or self.brush.load < .08:
            if self.brush.color >= 0:
                self.clean_brush(render_hook)
            self.dip_brush(stroke.color, render_hook)
        points = stroke.points(np.linspace(0, 1, 60))
        start = points[0]
        self.move_linear([start[0] + .50, start[1], .13], "MOVE_TO_START", speed=.24,
                         render_hook=render_hook)
        z, acquired = self._approach_force(start, stroke.force, render_hook=render_hook)
        if not acquired:
            self._set_phase("FORCE_TIMEOUT")
            return {"success": False, "reason": "force_acquisition_timeout", "force_acquired": False,
                    "wall_seconds": time.perf_counter()-start_wall}
        self.canvas.reset_path()
        # During the drawing phase the normal target remains at its acquired height.
        length = max(stroke.arc_length(), 1e-9)
        duration = length / float(self.config.get("stroke_speed", .045))
        n = max(2, int(np.ceil(duration / self.dt)))
        ss = np.linspace(0, 1, n+1)[1:]
        stroke_obs_start = len(self.observations)
        force_ema = self.normal_force()
        ema = float(self.config.get("force_ema", 0.90))
        z_track = z
        for s in ss:
            p = stroke.points(s)
            # The paper holds the acquired height in impedance mode. A small
            # low-bandwidth correction is retained here because the MuJoCo
            # penalty contact and joint servo have a measurable steady-state
            # error while the arm translates across the canvas.
            force_ema = ema * force_ema + (1.0 - ema) * self.normal_force()
            gain = float(self.config.get("stroke_force_admittance_gain", 0.000025))
            limit = float(self.config.get("stroke_force_admittance_step_limit", 0.000035))
            z_track += float(np.clip(gain * (force_ema - stroke.force), -limit, limit))
            z_track = float(np.clip(z_track, .0665, .085))
            obs = self.step([p[0] + .50, p[1], z_track], phase="STROKE", paint=True)
            if render_hook:
                render_hook(self, obs)
        stroke_obs_end = len(self.observations)
        stroke_obs = self.observations[stroke_obs_start:stroke_obs_end]
        ftrace = np.array([o.normal_force for o in stroke_obs])
        contact_ftrace = ftrace[ftrace > .02]
        self.canvas.reset_path()
        end = points[-1]
        self.move_linear([end[0] + .50, end[1], .13], "RETRACT", speed=.12,
                         render_hook=render_hook)
        actual = np.array([o.tip_xyz[:2] - [0.50, 0.0] for o in stroke_obs if o.contacting])
        desired = np.array([o.target_xyz[:2] - [0.50, 0.0] for o in stroke_obs if o.contacting])
        rmse = float(np.sqrt(np.mean(np.sum((actual-desired)**2, axis=1)))) if len(actual) else float("nan")
        return {
            "success": bool(len(actual) > 5 and np.count_nonzero(ftrace > .02) > 5),
            "reason": "ok" if len(actual) > 5 else "no_canvas_contact",
            "force_acquired": True,
            "target_force_N": float(stroke.force),
            "contact_samples": int(np.count_nonzero(ftrace > .02)),
            "force_samples": int(len(ftrace)),
            "force_mean_N": float(np.mean(contact_ftrace)) if len(contact_ftrace) else 0.0,
            "force_rmse_N": float(np.sqrt(np.mean((contact_ftrace-stroke.force)**2))) if len(contact_ftrace) else 0.0,
            "force_max_N": float(np.max(contact_ftrace)) if len(contact_ftrace) else 0.0,
            "force_std_N": float(np.std(contact_ftrace)) if len(contact_ftrace) else 0.0,
            "force_range_N": float(np.ptp(contact_ftrace)) if len(contact_ftrace) else 0.0,
            "brush_tilt_mean_deg": float(np.mean([o.brush_tilt_deg for o in stroke_obs])) if stroke_obs else 0.0,
            "brush_tilt_max_deg": float(np.max([o.brush_tilt_deg for o in stroke_obs])) if stroke_obs else 0.0,
            "trajectory_rmse_m": rmse,
            "wall_seconds": float(time.perf_counter()-start_wall),
        }

    def dip_brush(self, color, render_hook=None):
        self._set_phase("MOVE_TO_PALETTE")
        row, col = divmod(int(color), 8)
        px = .04 - .1225 + col*.035
        py = .235 - .0525 + row*.0525
        # Deposit wells are mapped to a reachable in-scene palette work area.
        target_xy = np.array([px, py])
        self.move_linear([target_xy[0], target_xy[1], .15], "MOVE_TO_PALETTE", .24, render_hook)
        # Approach the selected well from above and lower the tip until the
        # bristle sphere just enters the premixed paint surface.
        self.move_linear([target_xy[0], target_xy[1], .091], "DIP", .06, render_hook)
        self.move_linear([target_xy[0], target_xy[1], .15], "DIP_RETRACT", .10, render_hook)
        self.brush.dip(color)
        self._set_phase("IDLE")

    def clean_brush(self, render_hook=None):
        self._set_phase("MOVE_TO_CLEANER")
        # The cleaner is represented as a physical basin plus explicit paint-state wash.
        xy = np.array([.04, -.235])
        self.move_linear([xy[0], xy[1], .15], "MOVE_TO_CLEANER", .24, render_hook)
        self.move_linear([xy[0], xy[1], .099], "CLEAN", .06, render_hook)
        # A short horizontal swirl is the visible cleaning action. The brush
        # stays vertical while the bristles move through the water basin.
        swirl_radius = .022
        n = max(8, int(np.ceil(.75 / self.dt)))
        for theta in np.linspace(0.0, 4.0 * np.pi, n):
            self.step([xy[0] + swirl_radius*np.cos(theta),
                       xy[1] + swirl_radius*np.sin(theta), .099],
                      phase="CLEAN_SWIRL", paint=False)
        self.brush.clean()
        self.move_linear([xy[0], xy[1], .15], "CLEAN_RETRACT", .10, render_hook)
        self._set_phase("IDLE")

    def render(self, camera="overview_camera", width=None, height=None):
        if self.renderer is None or (width and height and (self.renderer.width != width or self.renderer.height != height)):
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(
                self.model,
                width=int(width or self.config.get("camera_width", 960)),
                height=int(height or self.config.get("camera_height", 540)),
            )
        # Keep the physical MuJoCo canvas and the contact-driven paint state in
        # the same image. This is the texture a camera would actually observe.
        rgb = np.asarray(np.flipud(self.canvas.image()), dtype=np.uint8)
        tex_adr = int(self.model.tex_adr[self.canvas_tex])
        tex_size = int(self.model.tex_width[self.canvas_tex] * self.model.tex_height[self.canvas_tex] * self.model.tex_nchannel[self.canvas_tex])
        self.model.tex_data[tex_adr:tex_adr + tex_size] = rgb.reshape(-1)
        mujoco.mjr_uploadTexture(self.model, self.renderer._mjr_context, self.canvas_tex)
        self.renderer.update_scene(self.data, camera=self.camera_ids[camera])
        return self.renderer.render().copy()

    def metadata(self):
        return {"seed": self.seed, "dt_control": self.dt, "sim_time": float(self.data.time),
                "phase_events": self.events, "mujoco_version": mujoco.__version__}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def write_episode(path, env: PaintingEnv, strokes, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"metadata": env.metadata(), "strokes": [s.to_dict() for s in strokes],
           "metrics": metrics, "observations": [
               {"time": o.time, "tip_xyz": o.tip_xyz.tolist(), "target_xyz": o.target_xyz.tolist(),
                "normal_force_N": o.normal_force, "contacting": o.contacting,
                "brush_tilt_deg": o.brush_tilt_deg, "q": o.q.tolist()}
               for o in env.observations]}
    path.write_text(json.dumps(doc, indent=2, allow_nan=True))
