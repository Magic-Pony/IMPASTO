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
        self.model.opt.timestep = float(self.config.get("timestep", self.model.opt.timestep))
        compression = self.model.joint("bristle_compression").id
        self.model.jnt_stiffness[compression] = float(self.config.get("brush_stiffness", 350.0))
        self.model.dof_damping[self.model.jnt_dofadr[compression]] = float(self.config.get("brush_damping", 2.5))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep) * int(self.config.get("control_decimation", 2))
        self.decimation = int(self.config.get("control_decimation", 2))
        self.tip_site = _id(self.model, mujoco.mjtObj.mjOBJ_SITE, "brush_tip_site")
        self.control_site = self.model.site("brush_nominal_site").id
        self.tip_geom = _id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "brush_tip")
        self.canvas_geom = _id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "canvas_surface")
        self.palette_well_geoms = [self.model.geom(f"well_{i:02d}").id for i in range(24)]
        self.washer_water_geom = self.model.geom("washer_water").id
        self.spinner_mocap = int(self.model.body_mocapid[self.model.body("washer_spinner").id])
        self.camera_ids = {
            name: _id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            for name in ("top_camera", "canvas_camera", "overview_camera")
        }
        self.canvas = Canvas(
            float(2 * self.model.geom_size[self.canvas_geom, 0]),
            float(2 * self.model.geom_size[self.canvas_geom, 1]),
            int(self.config.get("canvas_resolution", 512)),
            seed=self.seed,
        )
        self.brush = BrushState()
        self.renderer = None
        self.renderers = {}
        self.render_enabled = bool(render)
        self.canvas_tex = _id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "canvas_texture")
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.home_q = self.data.qpos[:7].copy()
        self.last_q_target = self.home_q.copy()
        self.home_tip_rot = self.data.site_xmat[self.tip_site].reshape(3, 3).copy()
        self.tip_world_offset = self.data.site_xpos[self.tip_site].copy() - self.data.xpos[
            _id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        ]
        self.force_target = 0.0
        self.force_integral = 0.0
        self.phase = "IDLE"
        self.events = [{"time": float(self.data.time), "phase": self.phase}]
        self.workstation_results = []
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
        """Actual simulated brush/canvas contact force in N, without rescaling.

        This contact-only signal excludes tool weight. The separate ideal
        six-axis mount sensor also includes weight and inertial loads.
        """
        return self.normal_force_raw()

    def tool_wrench(self):
        """Ideal F/T reading in sensor coordinates (N, Nm), including payload."""
        return np.r_[self.data.sensor("tool_force").data,
                     self.data.sensor("tool_torque").data].copy()

    def canvas_contact_height(self):
        return float(self.data.geom_xpos[self.canvas_geom, 2]
                     + self.model.geom_size[self.canvas_geom, 2]
                     + self.model.geom_size[self.tip_geom, 0])

    def canvas_xy(self, xy):
        return self.data.geom_xpos[self.canvas_geom, :2] + np.asarray(xy)

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
            current = self.data.site_xpos[self.control_site].copy()
            current_rot = self.data.site_xmat[self.control_site].reshape(3, 3).copy()
            pos_err = xyz - current
            rot_err = (target_rot * Rotation.from_matrix(current_rot).inv()).as_rotvec()
            if np.linalg.norm(pos_err) < 0.00005 and np.linalg.norm(rot_err) < 0.001:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.control_site)
            jac = np.vstack((jacp[:, :7], .45 * jacr[:, :7]))
            err = np.r_[pos_err, .45 * rot_err]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), err)
            # A weak null-space posture term avoids joint-limit drift without
            # changing the paper's task-space stroke action.
            jpinv = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(6), np.eye(6))
            null = np.eye(7) - jpinv @ jac
            dq += null @ (0.002 * (self.home_q - q))
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

    def step(self, target_xyz, phase=None, paint=True, velocity_feedforward=False):
        target_xyz = np.asarray(target_xyz, dtype=float)
        if phase:
            self._set_phase(phase)
        q_target = self._ik(target_xyz)
        target_qvel = np.clip(
            (q_target - self.last_q_target) / max(self.dt, 1e-9),
            -4.0,
            4.0,
        )
        self.last_q_target = q_target.copy()
        self.data.ctrl[:7] = q_target
        if velocity_feedforward:
            # Compensate the position actuators' velocity damping during free
            # space moves: K*(q_ref-q) - D*qdot + D*qdot_ref. Painting keeps
            # its separately validated servo and normal-force controller.
            stiffness = self.model.actuator_gainprm[:7, 0]
            damping = -self.model.actuator_biasprm[:7, 2]
            self.data.ctrl[:7] += (damping / stiffness) * target_qvel
        # The Menagerie model's high-gain general actuators provide the Panda
        # servo behavior; this bounded Cartesian-to-joint correction removes
        # the small gravity/friction steady-state error that otherwise matters
        # at the 10^-4 m brush-to-canvas clearance scale.
        applied_gain = float(self.config.get("joint_position_gain", 1400.0))
        applied_damping = float(self.config.get("joint_velocity_gain", 26.0))
        servo_torque = (
            applied_gain * (q_target - self.data.qpos[:7])
            + applied_damping * (target_qvel - self.data.qvel[:7])
        )
        gravity_comp = self.data.qfrc_bias[:7] if self.config.get("gravity_compensation", False) else 0.0
        self.data.qfrc_applied[:7] = np.clip(servo_torque + gravity_comp, -80.0, 80.0)
        if self.model.nu > 7:
            self.data.ctrl[7] = 114.75  # 36 mm gripper opening at grasping lugs
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        force = self.normal_force()
        tip = self.tip_position()
        contacting = force > .02
        if paint:
            self.canvas.update(tip[:2] - self.data.geom_xpos[self.canvas_geom, :2], force, self.brush, self.dt, contacting)
        obs = StepObservation(float(self.data.time), tip, target_xyz.copy(), force, contacting,
                             self.brush_tilt_deg(), self.data.qpos[:7].copy())
        self.observations.append(obs)
        return obs

    def move_linear(self, xyz, phase="MOVE", speed=.30, render_hook=None, paint=False,
                    acceleration=None, velocity_feedforward=False):
        start = self.tip_position()
        xyz = np.asarray(xyz, dtype=float)
        # Cubic time scaling starts/stops at zero velocity. Its peak derivative
        # is 1.5, included in duration so speed remains an actual upper bound.
        distance = float(np.linalg.norm(xyz - start))
        duration = 1.5 * distance / max(speed, .005)
        if acceleration is not None:
            # Quintic time scaling has zero velocity AND acceleration at both
            # ends. Bound its peak speed and acceleration for fast transfers.
            duration = max(1.875 * distance / max(speed, .005),
                           np.sqrt(5.773503 * distance / acceleration))
        n = max(1, int(np.ceil(duration / self.dt)))
        for i in range(1, n + 1):
            u = i / n
            p = (u**3 * (10 - 15*u + 6*u*u) if acceleration is not None
                 else u * u * (3 - 2 * u))
            obs = self.step(start * (1-p) + xyz * p, phase=phase, paint=paint,
                            velocity_feedforward=velocity_feedforward)
            if render_hook:
                render_hook(self, obs)
        return self.tip_position()

    def _approach_force(self, target_xy, target_force, render_hook=None):
        contact_center_z = self.canvas_contact_height()
        target = np.r_[self.canvas_xy(target_xy), contact_center_z + .005]
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
            z_ref -= float(np.clip(0.003 * self.dt * err, -0.00006, 0.00008))
            z_ref = max(contact_center_z - .011, z_ref)
            obs = self.step([*self.canvas_xy(target_xy), z_ref], phase="FORCE_ACQUIRE")
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
        self.move_linear([*self.canvas_xy(start), .13], "MOVE_TO_START", speed=.24,
                         render_hook=render_hook)
        z, acquired = self._approach_force(start, stroke.force, render_hook=render_hook)
        if not acquired:
            self._set_phase("FORCE_TIMEOUT")
            return {"success": False, "reason": "force_acquisition_timeout", "force_acquired": False,
                    "wall_seconds": time.perf_counter()-start_wall}
        self.canvas.reset_path()
        # The paper holds acquired height. This simulator retains the documented
        # continuous force-feedback height correction (not paper Eq. 4).
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
            gain = float(self.config.get("stroke_force_admittance_gain", 0.006)) * self.dt
            limit = float(self.config.get("stroke_force_speed_limit", 0.008)) * self.dt
            z_track += float(np.clip(gain * (force_ema - stroke.force), -limit, limit))
            z_track = float(np.clip(z_track, self.canvas_contact_height() - .011,
                                    self.canvas_contact_height() + .005))
            obs = self.step([*self.canvas_xy(p), z_track], phase="STROKE", paint=True)
            if render_hook:
                render_hook(self, obs)
        stroke_obs_end = len(self.observations)
        stroke_obs = self.observations[stroke_obs_start:stroke_obs_end]
        ftrace = np.array([o.normal_force for o in stroke_obs])
        contact_ftrace = ftrace[ftrace > .02]
        self.canvas.reset_path()
        end = points[-1]
        self.move_linear([*self.canvas_xy(end), .13], "RETRACT", speed=.12,
                         render_hook=render_hook)
        actual = np.array([o.tip_xyz[:2] - self.canvas_xy([0, 0]) for o in stroke_obs if o.contacting])
        desired = np.array([o.target_xyz[:2] - self.canvas_xy([0, 0]) for o in stroke_obs if o.contacting])
        rmse = float(np.sqrt(np.mean(np.sum((actual-desired)**2, axis=1)))) if len(actual) else float("nan")
        return {
            "success": bool(len(actual) > 5 and np.count_nonzero(ftrace > .02) > 5),
            "reason": "ok" if len(actual) > 5 else "no_canvas_contact",
            "force_acquired": True,
            "target_force_N": float(stroke.force),
            "contact_samples": int(np.count_nonzero(ftrace > .02)),
            "force_samples": int(len(ftrace)),
            "contact_fraction": float(np.mean(ftrace > .02)),
            "force_all_samples_rmse_N": float(np.sqrt(np.mean((ftrace-stroke.force)**2))),
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

    def _surface_target(self, geom_id, depth):
        """World TCP center for a sphere immersed by depth below a liquid top."""
        center = self.data.geom_xpos[geom_id].copy()
        center[2] += self.model.geom_size[geom_id, 1]
        center[2] += self.model.geom_size[self.tip_geom, 0] - depth
        return center

    def palette_target(self, color):
        if int(color) != color or not 0 <= color < 24:
            raise ValueError("Palette color must be an integer in [0, 23]")
        return self._surface_target(self.palette_well_geoms[int(color)], .005)

    def washer_target(self):
        return self._surface_target(self.washer_water_geom, .015)

    def _hold(self, target, duration, phase, render_hook=None, spin=False):
        for _ in range(max(1, int(np.ceil(duration / self.dt)))):
            if spin:
                angle = 2 * np.pi * 2 * self.data.time
                self.data.mocap_quat[self.spinner_mocap] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
            obs = self.step(target, phase=phase, paint=False)
            if render_hook:
                render_hook(self, obs)

    def _station_option(self, name, default):
        return float(self.config.get("workstation_motion", {}).get(name, default))

    def workstation_clearance_height(self):
        # Scene-derived height clears the tallest rim plus the contact radius.
        rims = [self.model.geom(i).id for i in range(self.model.ngeom)
                if (self.model.geom(i).name or "").startswith(
                    ("washer_wall_", "palette_divider_", "canvas_surface"))]
        top = max(self.data.geom_xpos[g, 2] + self.model.geom_size[g, 2] for g in rims)
        return float(top + self.model.geom_size[self.tip_geom, 0]
                     + self._station_option("clearance_m", .025))

    def _station_move(self, target, phase, speed, render_hook, acceleration=None):
        if np.linalg.norm(self.tip_position() - target) < .0001:
            return
        self.move_linear(target, phase, speed, render_hook,
                         acceleration=(self._station_option("acceleration_m_s2", .6)
                                       if acceleration is None else acceleration),
                         velocity_feedforward=True)

    def _settle_above_station(self, target, phase, render_hook):
        # Wait for measured alignment only when the faster transit needs it.
        for _ in range(int(np.ceil(self._station_option("settle_timeout_s", .8) / self.dt))):
            speed = float(np.linalg.norm(self.data.qvel[:7]))
            if (np.linalg.norm(self.tip_position() - target) < .001
                    and self.brush_tilt_deg() < .7 and speed < .12):
                return
            self._hold(target, self.dt, phase, render_hook)
        raise RuntimeError("Workstation transit did not settle before insertion")

    def _transfer_above(self, target, phase, render_hook=None):
        current = self.tip_position()
        safe_z = self.workstation_clearance_height()
        above = np.r_[target[:2], safe_z]
        # A low pose lifts clear of rims first. A high initial home pose can
        # take a direct free-space segment whose entire path is above safe_z.
        if current[2] < safe_z - .0001:
            self._station_move(np.r_[current[:2], safe_z], phase + "_HEIGHT",
                               self._station_option("vertical_speed_m_s", .25), render_hook,
                               self._station_option("vertical_acceleration_m_s2", .6))
        self._station_move(above, phase,
                           self._station_option("travel_speed_m_s", .45), render_hook,
                           self._station_option("travel_acceleration_m_s2", .6))
        self._settle_above_station(above, phase + "_SETTLE", render_hook)
        return above

    def _verify_immersion(self, geom, phase):
        pos = self.data.geom_xpos[geom]
        radius = float(self.model.geom_size[self.tip_geom, 0])
        tip = self.tip_position()
        xy_error = float(np.linalg.norm(tip[:2] - pos[:2]))
        top = float(pos[2] + self.model.geom_size[geom, 1])
        depth = float(top - (tip[2] - radius))
        result = {"phase": phase, "geom": self.model.geom(geom).name,
                  "time": float(self.data.time), "xy_error_m": xy_error,
                  "immersion_m": depth, "tilt_deg": self.brush_tilt_deg()}
        result["success"] = bool(
            xy_error + radius < self.model.geom_size[geom, 0]
            and .001 < depth < 2 * self.model.geom_size[geom, 1]
            and self.brush_axis()[2] < -np.cos(np.deg2rad(1.0)))
        self.workstation_results.append(result)
        if not result["success"]:
            raise RuntimeError(f"Workstation immersion check failed: {result}")
        return result

    def dip_brush(self, color, render_hook=None):
        started = float(self.data.time)
        target = self.palette_target(color)
        above = self._transfer_above(target, "MOVE_TO_PALETTE", render_hook)
        self._station_move(target, "DIP", self._station_option("insertion_speed_m_s", .09), render_hook)
        self._hold(target, self._station_option("dip_hold_s", .10), "DIP_HOLD", render_hook)
        result = self._verify_immersion(self.palette_well_geoms[int(color)], "DIP")
        self.brush.dip(color)
        self._station_move(above, "DIP_RETRACT", self._station_option("vertical_speed_m_s", .25), render_hook,
                           self._station_option("vertical_acceleration_m_s2", .6))
        result["duration_s"] = float(self.data.time) - started
        self._set_phase("IDLE")
        return result

    def clean_brush(self, render_hook=None):
        started = float(self.data.time)
        target = self.washer_target()
        above = self._transfer_above(target, "MOVE_TO_CLEANER", render_hook)
        self._station_move(target, "CLEAN", self._station_option("insertion_speed_m_s", .09), render_hook)
        # The paper's cleaner rotates water; hold the brush vertical and still.
        self._hold(target, self._station_option("clean_hold_s", .30), "CLEAN_SPIN", render_hook, spin=True)
        result = self._verify_immersion(self.washer_water_geom, "CLEAN")
        self.brush.clean()
        self._station_move(above, "CLEAN_RETRACT", self._station_option("vertical_speed_m_s", .25), render_hook,
                           self._station_option("vertical_acceleration_m_s2", .6))
        result["duration_s"] = float(self.data.time) - started
        self._set_phase("IDLE")
        return result

    def render(self, camera="overview_camera", width=None, height=None):
        size = (int(width or self.config.get("camera_width", 960)),
                int(height or self.config.get("camera_height", 540)))
        if size not in self.renderers:
            self.renderers[size] = mujoco.Renderer(self.model, width=size[0], height=size[1])
        self.renderer = self.renderers[size]
        # Keep the physical MuJoCo canvas and the contact-driven paint state in
        # the same image. This is the texture a camera would actually observe.
        # Canvas.pixel() already maps world +y to image row 0. MuJoCo's box
        # texture samples that row at the world +y edge, so flipping here
        # would mirror every stroke vertically in the camera image.
        rgb = np.asarray(self.canvas.image(), dtype=np.uint8)
        tex_adr = int(self.model.tex_adr[self.canvas_tex])
        tex_size = int(self.model.tex_width[self.canvas_tex] * self.model.tex_height[self.canvas_tex] * self.model.tex_nchannel[self.canvas_tex])
        self.model.tex_data[tex_adr:tex_adr + tex_size] = rgb.reshape(-1)
        mujoco.mjr_uploadTexture(self.model, self.renderer._mjr_context, self.canvas_tex)
        self.renderer.update_scene(self.data, camera=self.camera_ids[camera])
        return self.renderer.render().copy()

    def metadata(self):
        return {"seed": self.seed, "dt_control": self.dt, "sim_time": float(self.data.time),
                "phase_events": self.events, "workstation_results": self.workstation_results,
                "force_source": "unscaled MuJoCo brush/canvas contact force in N",
                "mujoco_version": mujoco.__version__}

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
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
