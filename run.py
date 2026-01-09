# run.py
# CPU-only PyBullet(DIRECT) + MeshCat visualization
# UR3e + Robotiq 2F-85: pick phone from cabinet mid-shelf -> place to workstation
#
# Key guarantees in this script:
# 1) Base placement: base_xy is chosen ON the cabinet-center(C) <-> workstation(W) line extension,
#    and base yaw always faces cabinet center => cabinet & workstation aligned in base-forward direction.
# 2) Reachability validation: candidate base positions are scored by IK error on critical waypoints.
# 3) Obstacle avoidance: joint trajectory is collision-checked against table/cabinet/workstation.
#    Planner retries with structured detours (x offsets, y-edge offsets, z-clear increments).
# 4) No random twisting: IK is seeded continuously + per-step delta cap + joint-space densify.
# 5) Stable idle: arm holds its last target (no drift before commands).
# 6) Robust grasp: close gripper -> attach constraint only if near phone and gripper sufficiently closed.
#
# NOTE:
# - We use airo_models URDFs; PyBullet physics + constraint-based grasp for reliability.
# - Visual arm is a smooth “industrial” look with fixed radii cylinders + joint spheres only at joints.

import math
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable

import numpy as np
import pybullet as p
import pybullet_data

import meshcat
import meshcat.geometry as g

import airo_models


# =========================
# Configuration
# =========================
@dataclass(frozen=True)
class SceneConfig:
    # Larger table to ensure reach & obstacle clearance
    table_w: float = 2.30
    table_d: float = 1.40
    table_h: float = 0.75
    table_top_t: float = 0.045
    table_leg_w: float = 0.075
    table_leg_inset: float = 0.14
    table_center_xy: Tuple[float, float] = (1.08, 0.00)

    # Robot pedestal (fixed)
    pedestal_size: Tuple[float, float, float] = (0.26, 0.26, 0.14)

    # Cabinet (two layers; mid shelf)
    cab_outer: Tuple[float, float, float] = (0.76, 0.54, 0.62)  # W, D, H
    cab_wall: float = 0.02
    cab_shelf_t: float = 0.02
    cab_offset_xy: Tuple[float, float] = (0.42, 0.32)

    # Workstation (target location)
    ws_plate: Tuple[float, float, float] = (0.32, 0.26, 0.015)
    ws_leg_h: float = 0.03
    ws_offset_xy: Tuple[float, float] = (0.78, -0.36)

    # Phone
    phone_size: Tuple[float, float, float] = (0.075, 0.150, 0.008)

    # placement constraints
    table_margin: float = 0.16

    # optional: snap on place if close (debug)
    snap_place: bool = False
    snap_xy_tol: float = 0.03


@dataclass(frozen=True)
class PlannerConfig:
    dt: float = 1.0 / 240.0
    realtime: bool = True

    # Cartesian interpolation density
    cart_step: float = 0.012
    ang_step_deg: float = 8.0

    # IK continuity + smoothing
    max_step_delta_rad: float = 0.16
    joint_damping: float = 0.06

    # Joint-space densify to avoid jumps
    densify_max_dq: float = 0.06  # rad

    # Collision checking
    obstacle_margin: float = 0.020
    exec_collision_margin: float = 0.007

    # replan
    max_replan_tries: int = 22

    # execution
    knot_ticks: int = 7
    settle_tol_rad: float = 0.018
    settle_timeout_s: float = 2.0

    # “near joint limit” warnings
    limit_warn_rad: float = math.radians(7.0)


@dataclass
class Waypoint:
    name: str
    pos: np.ndarray
    quat: np.ndarray


# =========================
# MeshCat Bridge
# =========================
class MeshcatBridge:
    def __init__(self):
        self.vis = meshcat.Visualizer().open()
        self.vis.delete()
        self._created = set()
        self.has_cylinder = hasattr(g, "Cylinder")

        self.add_frame("world", length=0.30, radius=0.01)
        self.set_transform("world", np.eye(4))

        # camera
        try:
            self.vis["/Cameras/default"].set_transform(self._T(t=[1.65, 0.25, 1.25]))
            self.vis["/Cameras/default/rotated"].set_transform(self._T(R=self._rot_x(-0.95)))
        except Exception:
            pass

        print("\n[MeshCat] 打开浏览器访问：", self.vis.url(), "\n")

    def _T(self, R=None, t=None):
        T = np.eye(4)
        if R is not None:
            T[:3, :3] = np.asarray(R, dtype=float)
        if t is not None:
            T[:3, 3] = np.asarray(t, dtype=float)
        return T

    def _rot_x(self, a):
        ca, sa = math.cos(a), math.sin(a)
        return np.array([[1, 0, 0],
                         [0, ca, -sa],
                         [0, sa, ca]], dtype=float)

    def _mat(self, color=0x999999, opacity=1.0):
        try:
            return g.MeshLambertMaterial(color=color, opacity=opacity, transparent=(opacity < 1.0))
        except Exception:
            return g.MeshPhongMaterial(color=color, opacity=opacity, transparent=(opacity < 1.0))

    def set_transform(self, path: str, T: np.ndarray):
        self.vis[path].set_transform(T)

    def add_box(self, path: str, size_xyz, color=0x999999, opacity=1.0):
        if path in self._created:
            return
        self._created.add(path)
        self.vis[path].set_object(g.Box(list(size_xyz)), self._mat(color, opacity))

    def add_sphere(self, path: str, radius: float, color=0xff8800, opacity=1.0):
        if path in self._created:
            return
        self._created.add(path)
        self.vis[path].set_object(g.Sphere(float(radius)), self._mat(color, opacity))

    def add_cylinder_or_box(self, path: str, length: float, radius: float, color=0xcccccc, opacity=1.0):
        if path in self._created:
            return
        self._created.add(path)
        if self.has_cylinder:
            self.vis[path].set_object(g.Cylinder(float(length), float(radius)), self._mat(color, opacity))
        else:
            self.vis[path].set_object(g.Box([radius * 2, length, radius * 2]), self._mat(color, opacity))

    def add_frame(self, path: str, length: float = 0.12, radius: float = 0.006):
        # x red, y green, z blue
        self.add_cylinder_or_box(f"{path}/y", length, radius, color=0x00ff00, opacity=1.0)
        self.set_transform(f"{path}/y", self._T(t=[0, length / 2, 0]))
        self.add_cylinder_or_box(f"{path}/x", length, radius, color=0xff0000, opacity=1.0)
        self.set_transform(f"{path}/x", self._T(R=np.array([[0, -1, 0],
                                                            [1, 0, 0],
                                                            [0, 0, 1]], dtype=float),
                                              t=[length / 2, 0, 0]))
        self.add_cylinder_or_box(f"{path}/z", length, radius, color=0x0000ff, opacity=1.0)
        self.set_transform(f"{path}/z", self._T(R=self._rot_x(math.pi / 2), t=[0, 0, length / 2]))

    def add_polyline(self, path: str, pts_xyz: np.ndarray, color=0xffaa00, opacity=1.0):
        if path in self._created:
            return
        self._created.add(path)
        try:
            pts = np.asarray(pts_xyz, dtype=float)
            geom = g.Line(g.PointsGeometry(pts.T), g.LineBasicMaterial(color=color, opacity=opacity))
            self.vis[path].set_object(geom)
        except Exception:
            pass


# =========================
# Math utils
# =========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        U[:, -1] *= -1
        R2 = U @ Vt
    return R2


def pose_to_T(pos, quat_xyzw) -> np.ndarray:
    R = np.array(p.getMatrixFromQuaternion(quat_xyzw), dtype=float).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def align_y_to_dir(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return np.eye(3)
    v = d / n
    y = np.array([0.0, 1.0, 0.0], dtype=float)
    c = float(np.clip(np.dot(y, v), -1.0, 1.0))
    if abs(c - 1.0) < 1e-8:
        return np.eye(3)
    if abs(c + 1.0) < 1e-8:
        return np.array([[1, 0, 0],
                         [0, -1, 0],
                         [0, 0, -1]], dtype=float)

    axis = np.cross(y, v)
    s = float(np.linalg.norm(axis))
    axis = axis / s

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3) + K + K @ K * ((1 - c) / (s * s))
    return orthonormalize(R)


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0, 0, 0, 1], dtype=float)
    return q / n


def quat_slerp(q0, q1, t: float):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = clamp(dot, -1.0, 1.0)
    if dot > 0.9995:
        return quat_normalize(q0 + t * (q1 - q0))

    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


def quat_angle_deg(q0, q1) -> float:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = abs(float(np.dot(q0, q1)))
    dot = clamp(dot, -1.0, 1.0)
    return math.degrees(2.0 * math.acos(dot))


def quat_from_R(R: np.ndarray) -> Tuple[float, float, float, float]:
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / S
        x = 0.25 * S
        y = (m[0, 1] + m[1, 0]) / S
        z = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / S
        x = (m[0, 1] + m[1, 0]) / S
        y = 0.25 * S
        z = (m[1, 2] + m[2, 1]) / S
    else:
        S = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / S
        x = (m[0, 2] + m[2, 0]) / S
        y = (m[1, 2] + m[2, 1]) / S
        z = 0.25 * S
    return (x, y, z, w)


# =========================
# Industrial-looking UR arm visual
# =========================
class URArmVisual:
    def __init__(self, viz: MeshcatBridge):
        self.viz = viz
        self.seg_len_nom: List[float] = []
        self.seg_rad: List[float] = []
        self.joint_rad: List[float] = []

    def create(self, seg_dist_nom: List[float], seg_rad: List[float], joint_rad: List[float]):
        assert len(seg_dist_nom) == 7 and len(seg_rad) == 7 and len(joint_rad) == 6
        self.seg_len_nom = [max(float(d), 0.02) for d in seg_dist_nom]
        self.seg_rad = list(seg_rad)
        self.joint_rad = list(joint_rad)

        for k in range(7):
            self.viz.add_cylinder_or_box(f"robot/ur/seg{k}",
                                         length=self.seg_len_nom[k],
                                         radius=self.seg_rad[k],
                                         color=0xd7d7d7,
                                         opacity=1.0)

        # joint spheres only at joints (6 joints)
        for k in range(6):
            self.viz.add_sphere(f"robot/ur/joint{k}",
                                radius=float(self.joint_rad[k]),
                                color=0xbcbcbc,
                                opacity=1.0)

        self.viz.add_cylinder_or_box("robot/ur/flange", length=0.038, radius=0.040,
                                     color=0x8f8f8f, opacity=1.0)

    def update(self, pts: List[np.ndarray], ee_T: np.ndarray):
        # pts length = 8 : base + 6 joints + ee
        for k in range(7):
            p0 = np.asarray(pts[k], dtype=float)
            p1 = np.asarray(pts[k + 1], dtype=float)
            v = p1 - p0
            dist = float(np.linalg.norm(v))
            if dist < 1e-9:
                continue
            d = v / dist
            R = align_y_to_dir(d)
            center = (p0 + p1) * 0.5

            # IMPORTANT: only scale along cylinder axis (y) => radius never changes
            sy = dist / max(self.seg_len_nom[k], 1e-6)
            S = np.diag([1.0, sy, 1.0])

            T = np.eye(4)
            T[:3, :3] = R @ S
            T[:3, 3] = center
            self.viz.set_transform(f"robot/ur/seg{k}", T)

        for k in range(6):
            pj = np.asarray(pts[1 + k], dtype=float)
            Tj = np.eye(4)
            Tj[:3, 3] = pj
            self.viz.set_transform(f"robot/ur/joint{k}", Tj)

        Rx90 = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=float)
        Tf = np.eye(4)
        Tf[:3, :3] = ee_T[:3, :3] @ Rx90
        Tf[:3, 3] = ee_T[:3, 3]
        self.viz.set_transform("robot/ur/flange", Tf)


# =========================
# Main Demo
# =========================
class UR3PickPlaceDemo:
    def __init__(self, scene: SceneConfig = SceneConfig(), plan: PlannerConfig = PlannerConfig()):
        self.scene = scene
        self.plan = plan

        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.plan.dt)
        p.setPhysicsEngineParameter(numSolverIterations=260)
        p.loadURDF("plane.urdf")

        self.viz = MeshcatBridge()
        self.arm_vis = URArmVisual(self.viz)

        self.phone_constraint: Optional[int] = None
        self.gripper_close_frac: float = 0.0

        self._build_scene()

        # Choose robot scale first (reach probe), then choose base placement on cab-ws line.
        self.ur_scale = self._choose_ur_scale()
        self.robot_base_pos, self.robot_base_quat, self.robot_base_yaw = self._choose_base_pose_strict_aligned()

        self._build_robot_and_mount()
        self._setup_control_and_visual()
        self.init_robot_state()

        for _ in range(30):
            p.stepSimulation()
            self._update_meshcat()

    # -----------------------
    # Scene: build table/cabinet/workstation/phone
    # -----------------------
    def _create_static_box(self, center_xyz, size_xyz, friction=0.9) -> int:
        he = (np.asarray(size_xyz, dtype=float) / 2.0).tolist()
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=he)
        bid = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col,
                                basePosition=np.asarray(center_xyz, dtype=float).tolist())
        p.changeDynamics(bid, -1, lateralFriction=friction)
        return bid

    def _add_box(self, path: str, center: np.ndarray, size: np.ndarray,
                 color: int, opacity: float, friction: float) -> int:
        bid = self._create_static_box(center, size, friction=friction)
        self.viz.add_box(path, tuple(size.tolist()), color=color, opacity=opacity)
        self.viz.set_transform(path, pose_to_T(center, (0, 0, 0, 1)))
        return bid

    def _build_scene(self):
        sc = self.scene

        self.table_center_xy = np.array(sc.table_center_xy, dtype=float)
        self.table_surface_z = sc.table_h

        # table top
        top_z = sc.table_h - sc.table_top_t / 2.0
        table_top_center = np.array([self.table_center_xy[0], self.table_center_xy[1], top_z], dtype=float)
        table_top_size = np.array([sc.table_w, sc.table_d, sc.table_top_t], dtype=float)
        self.table_top_id = self._create_static_box(table_top_center, table_top_size, friction=0.98)
        self.viz.add_box("env/table/top", tuple(table_top_size.tolist()), color=0x9a7b55, opacity=1.0)
        self.viz.set_transform("env/table/top", pose_to_T(table_top_center, (0, 0, 0, 1)))

        # legs
        leg_h = sc.table_h - sc.table_top_t
        leg_z = leg_h / 2.0
        leg_size = np.array([sc.table_leg_w, sc.table_leg_w, leg_h], dtype=float)
        x0 = self.table_center_xy[0] - sc.table_w / 2 + sc.table_leg_inset
        x1 = self.table_center_xy[0] + sc.table_w / 2 - sc.table_leg_inset
        y0 = self.table_center_xy[1] - sc.table_d / 2 + sc.table_leg_inset
        y1 = self.table_center_xy[1] + sc.table_d / 2 - sc.table_leg_inset
        leg_centers = [
            [x0, y0, leg_z],
            [x0, y1, leg_z],
            [x1, y0, leg_z],
            [x1, y1, leg_z],
        ]
        self.table_leg_ids = [self._create_static_box(c, leg_size, friction=0.98) for c in leg_centers]
        for i, c in enumerate(leg_centers):
            self.viz.add_box(f"env/table/leg{i}", tuple(leg_size.tolist()), color=0x7a5e3c, opacity=1.0)
            self.viz.set_transform(f"env/table/leg{i}", pose_to_T(c, (0, 0, 0, 1)))

        # cabinet
        W, D, H = sc.cab_outer
        w = sc.cab_wall
        shelf_t = sc.cab_shelf_t
        cab_xy = self.table_center_xy + np.array(sc.cab_offset_xy, dtype=float)
        self.cab_center = np.array([cab_xy[0], cab_xy[1], self.table_surface_z + H / 2.0], dtype=float)

        self.cabinet_bodies: List[int] = []
        self.cabinet_bodies.append(self._add_box("env/cab/back",
                                                 self.cab_center + np.array([0, +(D / 2 - w / 2), 0]),
                                                 np.array([W, w, H], dtype=float),
                                                 color=0xbdbdbd, opacity=0.85, friction=0.7))
        self.cabinet_bodies.append(self._add_box("env/cab/left",
                                                 self.cab_center + np.array([-(W / 2 - w / 2), 0, 0]),
                                                 np.array([w, D, H], dtype=float),
                                                 color=0xbdbdbd, opacity=0.85, friction=0.7))
        self.cabinet_bodies.append(self._add_box("env/cab/right",
                                                 self.cab_center + np.array([+(W / 2 - w / 2), 0, 0]),
                                                 np.array([w, D, H], dtype=float),
                                                 color=0xbdbdbd, opacity=0.85, friction=0.7))
        self.cabinet_bodies.append(self._add_box("env/cab/bottom",
                                                 self.cab_center + np.array([0, 0, -(H / 2 - w / 2)]),
                                                 np.array([W, D, w], dtype=float),
                                                 color=0xa0a0a0, opacity=0.95, friction=0.8))
        self.cabinet_bodies.append(self._add_box("env/cab/top",
                                                 self.cab_center + np.array([0, 0, +(H / 2 - w / 2)]),
                                                 np.array([W, D, w], dtype=float),
                                                 color=0xa0a0a0, opacity=0.95, friction=0.8))

        # mid shelf (creates 2 layers)
        shelf_width = W - 2 * w
        shelf_depth = D - w
        self.shelf_center = self.cab_center + np.array([0, 0, 0], dtype=float)
        self.cabinet_bodies.append(self._add_box("env/cab/shelf_mid",
                                                 self.shelf_center,
                                                 np.array([shelf_width, shelf_depth, shelf_t], dtype=float),
                                                 color=0x8f8f8f, opacity=0.98, friction=0.9))

        # front frame (visual only)
        frame_t = 0.01
        frame_y = self.cab_center[1] - (D / 2 - frame_t / 2)
        frame_center = np.array([self.cab_center[0], frame_y, self.cab_center[2]], dtype=float)
        self.viz.add_box("env/cab/front_frame", (W, frame_t, H), color=0xdddddd, opacity=0.12)
        self.viz.set_transform("env/cab/front_frame", pose_to_T(frame_center, (0, 0, 0, 1)))

        self.cab_top_z = float(self.cab_center[2] + H / 2.0)
        self.cab_front_y = float(self.cab_center[1] - D / 2.0)
        self.cab_inner_w = float(W - 2 * w)
        self.cab_inner_d = float(D - w)
        self.cab_inner_front_y = float(self.cab_center[1] - D / 2 + w)
        self.cab_inner_back_y = float(self.cab_center[1] + D / 2 - w)
        self.shelf_top_z = float(self.shelf_center[2] + shelf_t / 2.0)

        # workstation
        ws_xy = self.table_center_xy + np.array(sc.ws_offset_xy, dtype=float)
        ws_plate = np.array(sc.ws_plate, dtype=float)
        ws_leg_h = sc.ws_leg_h

        ws_leg_z = self.table_surface_z + ws_leg_h / 2.0
        ws_plate_z = self.table_surface_z + ws_leg_h + ws_plate[2] / 2.0

        ws_leg = np.array([0.03, 0.03, ws_leg_h], dtype=float)
        inset = 0.02
        lx0 = ws_xy[0] - ws_plate[0] / 2 + inset
        lx1 = ws_xy[0] + ws_plate[0] / 2 - inset
        ly0 = ws_xy[1] - ws_plate[1] / 2 + inset
        ly1 = ws_xy[1] + ws_plate[1] / 2 - inset

        self.workstation_bodies: List[int] = []
        for i, c in enumerate([
            np.array([lx0, ly0, ws_leg_z], dtype=float),
            np.array([lx0, ly1, ws_leg_z], dtype=float),
            np.array([lx1, ly0, ws_leg_z], dtype=float),
            np.array([lx1, ly1, ws_leg_z], dtype=float),
        ]):
            self.workstation_bodies.append(self._add_box(f"env/ws/leg{i}", c, ws_leg,
                                                         color=0x4f4f4f, opacity=1.0, friction=0.95))
        self.ws_plate_center = np.array([ws_xy[0], ws_xy[1], ws_plate_z], dtype=float)
        self.ws_plate_size = ws_plate
        self.workstation_bodies.append(self._add_box("env/ws/plate", self.ws_plate_center, self.ws_plate_size,
                                                     color=0x2f2f2f, opacity=1.0, friction=1.00))

        self.ws_top_z = float(self.ws_plate_center[2] + ws_plate[2] / 2.0)
        self.ws_target = np.array([self.ws_plate_center[0], self.ws_plate_center[1], self.ws_top_z], dtype=float)
        self.viz.add_sphere("env/ws/target", 0.02, color=0x00ff00, opacity=1.0)
        self.viz.set_transform("env/ws/target", pose_to_T(self.ws_target, (0, 0, 0, 1)))

        # phone on mid shelf (layer-1)
        phone_size = np.array(sc.phone_size, dtype=float)
        phone_z = self.shelf_top_z + float(phone_size[2] / 2.0)
        phone_y = self.cab_inner_front_y + float(phone_size[1] / 2.0) + 0.06
        phone_x = float(self.cab_center[0] + 0.02)
        self.phone_init_pos = np.array([phone_x, phone_y, phone_z], dtype=float)

        phone_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=(phone_size / 2.0).tolist())
        self.phone_id = p.createMultiBody(baseMass=0.18, baseCollisionShapeIndex=phone_col,
                                          basePosition=self.phone_init_pos.tolist())
        p.changeDynamics(self.phone_id, -1, lateralFriction=1.10,
                         rollingFriction=0.003, spinningFriction=0.003)

        self.viz.add_box("objects/phone", tuple(phone_size.tolist()), color=0x111111, opacity=1.0)
        self.viz.set_transform("objects/phone", pose_to_T(self.phone_init_pos, (0, 0, 0, 1)))

        # obstacles list
        self.obstacles = [self.table_top_id] + self.cabinet_bodies + self.workstation_bodies

        # pedestal & base frame (pose will be set after base chosen)
        self.ped_size = np.array(sc.pedestal_size, dtype=float)
        self.viz.add_box("robot/pedestal", tuple(self.ped_size.tolist()), color=0x555555, opacity=1.0)
        self.viz.add_frame("robot/base", length=0.18, radius=0.01)

    # -----------------------
    # Robot scale + base selection (STRICT aligned)
    # -----------------------
    def _guess_ee_link_tmp(self, rid: int) -> int:
        num = p.getNumJoints(rid)
        for j in range(num):
            link_name = p.getJointInfo(rid, j)[12].decode("utf-8")
            if any(k in link_name for k in ["tool0", "tcp", "ee", "wrist_3", "flange"]):
                return j
        rev = [j for j in range(num) if p.getJointInfo(rid, j)[2] == p.JOINT_REVOLUTE]
        return rev[-1] if rev else num - 1

    def _choose_ur_scale(self) -> float:
        # choose from a few scales to ensure reach (same as your previous iterations)
        ur_urdf = airo_models.get_urdf_path("ur3e")
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_IGNORE_VISUAL_SHAPES
        scales = [1.00, 1.10, 1.20, 1.30]
        best_s, best_err = scales[0], 1e9

        # probe target (near cabinet front, above phone)
        phone = self.phone_init_pos.copy()
        probe = phone + np.array([0.0, -0.10, 0.12], dtype=float)

        # temporary base (rough) for scale selection
        base_xy = self.cab_center[:2] + np.array([-0.65, -0.45], dtype=float)
        base_pos = np.array([base_xy[0], base_xy[1], self.table_surface_z + self.scene.pedestal_size[2]], dtype=float)
        yaw = float(math.atan2(self.cab_center[1] - base_xy[1], self.cab_center[0] - base_xy[0]))
        qbase = p.getQuaternionFromEuler([0, 0, yaw])

        for s in scales:
            rid = p.loadURDF(ur_urdf, basePosition=base_pos.tolist(),
                             baseOrientation=qbase, useFixedBase=True,
                             flags=flags, globalScaling=s)
            ee = self._guess_ee_link_tmp(rid)
            q = p.calculateInverseKinematics(rid, ee, probe.tolist(),
                                             maxNumIterations=220, residualThreshold=1e-4)
            for j in range(min(len(q), p.getNumJoints(rid))):
                p.resetJointState(rid, j, float(q[j]))
            st = p.getLinkState(rid, ee, computeForwardKinematics=True)
            ee_pos = np.array(st[4], dtype=float)
            err = float(np.linalg.norm(ee_pos - probe))
            if err < best_err:
                best_err, best_s = err, s
            p.removeBody(rid)

        print(f"[ReachScale] globalScaling={best_s:.2f} probe_err={best_err:.4f}m")
        return best_s

    def _base_yaw_for_xy(self, base_xy: np.ndarray) -> float:
        v = np.array([self.cab_center[0] - base_xy[0],
                      self.cab_center[1] - base_xy[1]], dtype=float)
        return float(math.atan2(v[1], v[0]))

    def _table_xy_bounds(self) -> Tuple[float, float, float, float]:
        sc = self.scene
        tx0 = self.table_center_xy[0] - sc.table_w / 2 + sc.table_margin
        tx1 = self.table_center_xy[0] + sc.table_w / 2 - sc.table_margin
        ty0 = self.table_center_xy[1] - sc.table_d / 2 + sc.table_margin
        ty1 = self.table_center_xy[1] + sc.table_d / 2 - sc.table_margin
        return tx0, tx1, ty0, ty1

    def _pick_base_on_cab_ws_line_candidates(self) -> List[np.ndarray]:
        """
        Generate base_xy candidates on (or near) the extension line of C<->W,
        such that both cabinet center and workstation align in the same forward direction.
        """
        C = self.cab_center[:2].astype(float)
        W = self.ws_target[:2].astype(float)

        u = (C - W)
        n = float(np.linalg.norm(u))
        if n < 1e-9:
            return []
        u = u / n  # direction from W->C

        # perpendicular for slight lateral offsets
        perp = np.array([-u[1], u[0]], dtype=float)

        tx0, tx1, ty0, ty1 = self._table_xy_bounds()

        # distances behind cabinet (meters)
        t_list = [0.55, 0.62, 0.70, 0.78, 0.86, 0.94]
        off_list = [0.00, +0.04, -0.04, +0.08, -0.08]

        cands = []
        for t in t_list:
            base_xy = C + u * t
            for off in off_list:
                b = base_xy + perp * off
                if (tx0 <= b[0] <= tx1) and (ty0 <= b[1] <= ty1):
                    cands.append(b)
        return cands

    def _critical_targets_for_scoring(self, base_xy: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Build key EE targets (pos, quat) that must be reachable from this base pose.
        We use the same orientation policy as planner.
        """
        phone = self.phone_init_pos.copy()
        ws = self.ws_target.copy()

        # pick orientation: forward +Y into cabinet, down -Z
        q_pick = self.build_ee_quat_from_world_fwd_down(
            fwd_w=np.array([0.0, 1.0, 0.0]),
            down_w=np.array([0.0, 0.0, -1.0]),
            ee_to_gripper_quat=np.array([0, 0, 0, 1], dtype=float)  # placeholder, replaced after mount in real
        )

        # place orientation: forward -Y (toward operator side), down -Z
        q_place = self.build_ee_quat_from_world_fwd_down(
            fwd_w=np.array([0.0, -1.0, 0.0]),
            down_w=np.array([0.0, 0.0, -1.0]),
            ee_to_gripper_quat=np.array([0, 0, 0, 1], dtype=float)
        )

        # safe heights (below cabinet top)
        z_clear = min(self.cab_top_z - 0.08, phone[2] + 0.18)
        y_edge = self.cab_front_y - 0.20

        targets = [
            (np.array([phone[0], y_edge, z_clear], dtype=float), q_pick),
            (np.array([phone[0], phone[1], z_clear], dtype=float), q_pick),
            (np.array([phone[0], phone[1], phone[2]], dtype=float), q_pick),
            (np.array([ws[0], ws[1], max(self.ws_top_z + 0.20, z_clear)], dtype=float), q_place),
        ]
        return targets

    def _score_base_candidate(self, base_xy: np.ndarray) -> Tuple[float, float]:
        """
        Score base candidate by IK error to critical targets.
        Returns (total_score, max_err). Lower is better.
        """
        ur_urdf = airo_models.get_urdf_path("ur3e")
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_IGNORE_VISUAL_SHAPES

        yaw = self._base_yaw_for_xy(base_xy)
        qbase = p.getQuaternionFromEuler([0, 0, yaw])
        base_pos = np.array([base_xy[0], base_xy[1], self.table_surface_z + self.scene.pedestal_size[2]], dtype=float)

        rid = p.loadURDF(ur_urdf, basePosition=base_pos.tolist(),
                         baseOrientation=qbase, useFixedBase=True,
                         flags=flags, globalScaling=self.ur_scale)
        try:
            ee = self._guess_ee_link_tmp(rid)
            total, max_err = 0.0, 0.0
            targets = self._critical_targets_for_scoring(base_xy)
            for tp, tq in targets:
                q = p.calculateInverseKinematics(rid, ee, tp.tolist(),
                                                 targetOrientation=tq.tolist(),
                                                 maxNumIterations=220, residualThreshold=1e-4)
                for j in range(min(len(q), p.getNumJoints(rid))):
                    p.resetJointState(rid, j, float(q[j]))
                st = p.getLinkState(rid, ee, computeForwardKinematics=True)
                ee_pos = np.array(st[4], dtype=float)
                err = float(np.linalg.norm(ee_pos - tp))
                max_err = max(max_err, err)
                total += 300.0 * err
            return total, max_err
        finally:
            p.removeBody(rid)

    def _choose_base_pose_strict_aligned(self) -> Tuple[np.ndarray, Tuple[float, float, float, float], float]:
        """
        Choose base_xy on cab-ws aligned line (or nearby) by scoring candidates.
        Then set yaw to face cabinet center (base forward points to cabinet).
        """
        cands = self._pick_base_on_cab_ws_line_candidates()
        if not cands:
            # fallback
            base_xy = self.cab_center[:2] + np.array([-0.70, -0.48], dtype=float)
            yaw = self._base_yaw_for_xy(base_xy)
            q = p.getQuaternionFromEuler([0, 0, yaw])
            base = np.array([base_xy[0], base_xy[1], self.table_surface_z + self.scene.pedestal_size[2]], dtype=float)
            print("[BaseSelect] fallback base_xy =", base_xy.tolist())
            return base, q, yaw

        best = None
        for b in cands:
            s, emax = self._score_base_candidate(b)
            if best is None or s < best[0]:
                best = (s, emax, b.copy())

        _, emax, base_xy = best
        yaw = self._base_yaw_for_xy(base_xy)
        q = p.getQuaternionFromEuler([0, 0, yaw])
        base = np.array([base_xy[0], base_xy[1], self.table_surface_z + self.scene.pedestal_size[2]], dtype=float)
        print(f"[BaseSelect] aligned base_xy={base_xy.tolist()} yaw(deg)={math.degrees(yaw):.1f} max_err≈{emax:.4f}m")
        return base, q, yaw

    # -----------------------
    # Robot load + mount
    # -----------------------
    def _build_robot_and_mount(self):
        ur_urdf = airo_models.get_urdf_path("ur3e")
        rq_urdf = airo_models.get_urdf_path("robotiq_2f_85")
        flags = (p.URDF_USE_INERTIA_FROM_FILE |
                 p.URDF_IGNORE_VISUAL_SHAPES |
                 p.URDF_USE_SELF_COLLISION)

        # load arm with strict base pose
        self.robot_id = p.loadURDF(ur_urdf,
                                   basePosition=self.robot_base_pos.tolist(),
                                   baseOrientation=self.robot_base_quat,
                                   useFixedBase=True, flags=flags, globalScaling=self.ur_scale)

        # load gripper near base pose (will be constrained)
        self.gripper_id = p.loadURDF(rq_urdf,
                                     basePosition=self.robot_base_pos.tolist(),
                                     baseOrientation=self.robot_base_quat,
                                     useFixedBase=False, flags=flags, globalScaling=self.ur_scale)

        self.ee_link = self._guess_ee_link_tmp(self.robot_id)

        # mount gripper rigidly to ee
        p.createConstraint(self.robot_id, self.ee_link,
                           self.gripper_id, -1,
                           p.JOINT_FIXED, [0, 0, 0],
                           [0, 0, 0], [0, 0, 0],
                           [0, 0, 0, 1], [0, 0, 0, 1])

        # update pedestal + base frames
        ped_center = np.array([self.robot_base_pos[0], self.robot_base_pos[1],
                               self.table_surface_z + self.scene.pedestal_size[2] / 2.0], dtype=float)
        self.viz.set_transform("robot/pedestal", pose_to_T(ped_center, self.robot_base_quat))
        self.viz.set_transform("robot/base", pose_to_T(self.robot_base_pos.tolist(), self.robot_base_quat))

    # -----------------------
    # Joints / Limits / Visual setup
    # -----------------------
    def _select_arm_joints(self) -> List[int]:
        num = p.getNumJoints(self.robot_id)
        movable = [j for j in range(num) if p.getJointInfo(self.robot_id, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        name_map = {j: p.getJointInfo(self.robot_id, j)[1].decode("utf-8") for j in movable}
        preferred = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                     "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        chosen = []
        for nm in preferred:
            for j, jn in name_map.items():
                if jn == nm:
                    chosen.append(j)
                    break
        if len(chosen) == 6:
            return chosen
        rev = [j for j in movable if p.getJointInfo(self.robot_id, j)[2] == p.JOINT_REVOLUTE]
        return rev[:6]

    def _estimate_link_radius(self, body_id: int, link_index: int) -> float:
        try:
            aabb_min, aabb_max = p.getAABB(body_id, link_index)
            ext = np.array(aabb_max, dtype=float) - np.array(aabb_min, dtype=float)
            s = np.sort(ext)
            thickness = float(max(s[0], s[1]))
            r = 0.5 * thickness
            return float(np.clip(r * 0.85, 0.015, 0.048))
        except Exception:
            return 0.028

    def _setup_control_and_visual(self):
        self.arm_joints = self._select_arm_joints()
        self.movable_joints = [j for j in range(p.getNumJoints(self.robot_id))
                               if p.getJointInfo(self.robot_id, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        self.movable_index = {j: i for i, j in enumerate(self.movable_joints)}

        # limits for IK: use URDF if present; if continuous, keep large but finite
        cont = math.radians(359.0)
        self.joint_limit_by_index: Dict[int, Tuple[float, float]] = {}
        self.lower, self.upper, self.ranges, self.damping = [], [], [], []
        for j in self.movable_joints:
            info = p.getJointInfo(self.robot_id, j)
            if info[2] not in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                continue
            lo, hi = float(info[8]), float(info[9])
            if abs(hi - lo) < 1e-9:
                lo, hi = -cont, +cont
            if (hi - lo) > math.radians(720.0):
                lo, hi = -cont, +cont
            self.joint_limit_by_index[j] = (lo, hi)
            self.lower.append(lo); self.upper.append(hi)
            self.ranges.append(max(hi - lo, 1e-6))
            self.damping.append(self.plan.joint_damping)

        # gripper joints
        gnum = p.getNumJoints(self.gripper_id)
        self.gripper_joints = [j for j in range(gnum)
                               if p.getJointInfo(self.gripper_id, j)[2] in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE)]
        self.gripper_limits = []
        for j in self.gripper_joints:
            lo, hi = float(p.getJointInfo(self.gripper_id, j)[8]), float(p.getJointInfo(self.gripper_id, j)[9])
            if abs(hi - lo) < 1e-9:
                lo, hi = 0.0, 0.8
            self.gripper_limits.append((lo, hi))

        # build arm visuals (probe distances)
        for j in self.arm_joints:
            p.resetJointState(self.robot_id, j, 0.0)
        p.stepSimulation()

        pts = self._get_visual_chain_points()
        seg_nom = [float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(7)]
        link_r = [self._estimate_link_radius(self.robot_id, j) for j in self.arm_joints]

        # fixed radii per segment (industrial tapering)
        seg_rad = [
            max(link_r[0] * 1.02, 0.020),
            max(min(link_r[0], link_r[1]) * 0.98, 0.018),
            max(min(link_r[1], link_r[2]) * 0.97, 0.017),
            max(min(link_r[2], link_r[3]) * 0.96, 0.016),
            max(min(link_r[3], link_r[4]) * 0.95, 0.0155),
            max(min(link_r[4], link_r[5]) * 0.94, 0.0150),
            max(link_r[5] * 0.94, 0.0150),
        ]
        joint_rad = [clamp(0.78 * max(seg_rad[i], seg_rad[min(i + 1, 6)]) + 0.002, 0.018, 0.036)
                     for i in range(6)]
        self.arm_vis.create(seg_nom, seg_rad, joint_rad)

        # simple gripper visuals
        self.viz.add_box("robot/gripper/base", (0.085, 0.045, 0.045), color=0x222222, opacity=1.0)
        self.viz.add_box("robot/gripper/fingerL", (0.018, 0.095, 0.02), color=0x333333, opacity=1.0)
        self.viz.add_box("robot/gripper/fingerR", (0.018, 0.095, 0.02), color=0x333333, opacity=1.0)

        # wrist camera frame (visual only)
        self.viz.add_frame("robot/wrist_rgbd", length=0.10, radius=0.006)

        # compute ee->gripper transform
        self.set_gripper(0.0)
        for _ in range(10):
            p.stepSimulation()
        st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        ee_pos, ee_quat = st[4], st[5]
        gpos, gquat = p.getBasePositionAndOrientation(self.gripper_id)
        inv_ee_pos, inv_ee_quat = p.invertTransform(ee_pos, ee_quat)
        rel_pos, rel_quat = p.multiplyTransforms(inv_ee_pos, inv_ee_quat, gpos, gquat)
        self.ee_to_gripper_pos = np.array(rel_pos, dtype=float)
        self.ee_to_gripper_quat = np.array(rel_quat, dtype=float)

        print("[Debug] base_pos:", self.robot_base_pos.tolist())
        print("[Debug] base_yaw(deg):", math.degrees(self.robot_base_yaw))
        print("[Debug] arm joints:", self.arm_joints)
        print("[Debug] ee link:", self.ee_link)

    def get_joint_limit(self, joint_index: int) -> Tuple[float, float]:
        return self.joint_limit_by_index.get(joint_index, (-math.radians(359), math.radians(359)))

    def get_arm_q(self) -> List[float]:
        return [p.getJointState(self.robot_id, j)[0] for j in self.arm_joints]

    def lock_arm_targets(self, q_arm: Optional[List[float]] = None):
        if q_arm is None:
            q_arm = self.get_arm_q()
        for ji, qv in zip(self.arm_joints, q_arm):
            lo, hi = self.get_joint_limit(ji)
            qv = clamp(float(qv), lo, hi)
            p.setJointMotorControl2(self.robot_id, ji, p.POSITION_CONTROL,
                                    targetPosition=qv, targetVelocity=0.0,
                                    positionGain=0.30, velocityGain=1.0,
                                    force=420)

    # -----------------------
    # Visual chain points
    # -----------------------
    def _joint_pivot_pos(self, joint_index: int) -> np.ndarray:
        st = p.getLinkState(self.robot_id, joint_index, computeForwardKinematics=True)
        return np.array(st[4], dtype=float)

    def _get_visual_chain_points(self) -> List[np.ndarray]:
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        base_anchor = np.array(base_pos, dtype=float)
        joint_pts = [self._joint_pivot_pos(j) for j in self.arm_joints]
        st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        ee_pos = np.array(st[4], dtype=float)
        return [base_anchor] + joint_pts + [ee_pos]

    # -----------------------
    # Gripper
    # -----------------------
    def set_gripper(self, close_fraction: float):
        close_fraction = clamp(close_fraction, 0.0, 1.0)
        self.gripper_close_frac = close_fraction
        for (j, (lo, hi)) in zip(self.gripper_joints, self.gripper_limits):
            tgt = lo + close_fraction * (hi - lo)
            p.setJointMotorControl2(self.gripper_id, j, p.POSITION_CONTROL, targetPosition=float(tgt), force=220)

    # -----------------------
    # Orientation helpers
    # -----------------------
    def _quat_mul(self, qa, qb):
        return p.multiplyTransforms([0, 0, 0], qa, [0, 0, 0], qb)[1]

    def _quat_inv(self, q):
        return p.invertTransform([0, 0, 0], q)[1]

    def build_ee_quat_from_world_fwd_down(self, fwd_w: np.ndarray, down_w: np.ndarray,
                                          ee_to_gripper_quat: np.ndarray) -> np.ndarray:
        """
        Build EE quaternion so that gripper forward axis aligns with fwd_w and gripper down aligns with down_w.
        We then compensate by EE->gripper quaternion so that after mounting, the GRIPPER has desired orientation.
        """
        fwd = fwd_w / max(np.linalg.norm(fwd_w), 1e-9)
        down = down_w / max(np.linalg.norm(down_w), 1e-9)
        left = np.cross(down, fwd)
        left = left / max(np.linalg.norm(left), 1e-9)
        fwd = np.cross(left, down)
        Rg = np.stack([left, fwd, down], axis=1)
        Rg = orthonormalize(Rg)
        qg = np.array(quat_from_R(Rg), dtype=float)
        q_ee = self._quat_mul(qg, self._quat_inv(ee_to_gripper_quat))
        return np.array(q_ee, dtype=float)

    # -----------------------
    # IK with continuity
    # -----------------------
    def _wrap_near(self, q: float, ref: float) -> float:
        two_pi = 2.0 * math.pi
        dq = q - ref
        dq = (dq + math.pi) % two_pi - math.pi
        return ref + dq

    def ik_limited(self, target_pos, target_quat_xyzw, seed_arm_q: List[float]) -> List[float]:
        # restPoses over movable joints
        rest_m = [0.0] * len(self.movable_joints)
        for k, j in enumerate(self.arm_joints):
            rest_m[self.movable_index[j]] = float(seed_arm_q[k])

        q_sol = p.calculateInverseKinematics(
            self.robot_id, self.ee_link,
            target_pos, targetOrientation=target_quat_xyzw,
            lowerLimits=self.lower, upperLimits=self.upper,
            jointRanges=self.ranges, restPoses=rest_m,
            jointDamping=[self.plan.joint_damping] * len(self.lower),
            solver=p.IK_DLS, maxNumIterations=360, residualThreshold=1e-4
        )
        q_sol = list(q_sol)
        numJ = p.getNumJoints(self.robot_id)

        if len(q_sol) == len(self.movable_joints):
            q_map = {j: q_sol[i] for i, j in enumerate(self.movable_joints)}
            q_arm = [float(q_map[self.arm_joints[k]]) for k in range(6)]
        elif len(q_sol) == numJ:
            q_arm = [float(q_sol[j]) for j in self.arm_joints]
        else:
            q_arm = list(seed_arm_q)

        # continuity + delta cap + clamp
        out = []
        for qi, ri in zip(q_arm, seed_arm_q):
            qi = self._wrap_near(float(qi), float(ri))
            d = qi - float(ri)
            if abs(d) > self.plan.max_step_delta_rad:
                qi = float(ri) + math.copysign(self.plan.max_step_delta_rad, d)
            out.append(float(qi))

        q_arm = out
        for i, j in enumerate(self.arm_joints):
            lo, hi = self.get_joint_limit(j)
            q_arm[i] = clamp(float(q_arm[i]), lo, hi)
        return q_arm

    # -----------------------
    # Collision helpers
    # -----------------------
    def _compute_gripper_world_from_ee(self) -> Tuple[np.ndarray, np.ndarray]:
        st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        ee_pos, ee_quat = st[4], st[5]
        gpos, gquat = p.multiplyTransforms(
            ee_pos, ee_quat,
            self.ee_to_gripper_pos.tolist(),
            self.ee_to_gripper_quat.tolist()
        )
        return np.array(gpos, dtype=float), np.array(gquat, dtype=float)

    def _set_arm_kinematic(self, q_arm: List[float]):
        for ji, qv in zip(self.arm_joints, q_arm):
            p.resetJointState(self.robot_id, ji, float(qv))
        gpos, gquat = self._compute_gripper_world_from_ee()
        p.resetBasePositionAndOrientation(self.gripper_id, gpos.tolist(), gquat.tolist())

    def _in_collision(self, q_arm: List[float], margin: float) -> bool:
        self._set_arm_kinematic(q_arm)
        for obs in self.obstacles:
            if p.getClosestPoints(self.robot_id, obs, distance=margin):
                return True
            if p.getClosestPoints(self.gripper_id, obs, distance=margin):
                return True
        return False

    def _exec_collision_now(self, margin: float) -> bool:
        for obs in self.obstacles:
            if p.getClosestPoints(self.robot_id, obs, distance=margin):
                return True
            if p.getClosestPoints(self.gripper_id, obs, distance=margin):
                return True
        return False

    # -----------------------
    # Interpolation (Cartesian + joint densify)
    # -----------------------
    def interpolate_pose_list(self, p0, q0, p1, q1) -> List[Tuple[np.ndarray, np.ndarray]]:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)

        dist = float(np.linalg.norm(p1 - p0))
        ang = float(quat_angle_deg(q0, q1))
        n_pos = max(1, int(math.ceil(dist / self.plan.cart_step)))
        n_ang = max(1, int(math.ceil(ang / self.plan.ang_step_deg)))
        n = max(n_pos, n_ang)

        out = []
        for i in range(n + 1):
            t = i / n
            pos = (1 - t) * p0 + t * p1
            quat = quat_slerp(q0, q1, t)
            out.append((pos, quat))
        return out

    def densify_joint_traj(self, q_traj: List[List[float]]) -> List[List[float]]:
        if not q_traj:
            return q_traj
        out = [q_traj[0]]
        max_dq = self.plan.densify_max_dq
        for qa, qb in zip(q_traj[:-1], q_traj[1:]):
            qa = np.array(qa, dtype=float)
            qb = np.array(qb, dtype=float)
            dq = float(np.max(np.abs(qb - qa)))
            n = max(1, int(math.ceil(dq / max_dq)))
            for k in range(1, n + 1):
                t = k / n
                out.append((1 - t) * qa + t * qb)
        return [list(map(float, x)) for x in out]

    # -----------------------
    # Planning (Cartesian -> IK -> collision check)
    # -----------------------
    def _viz_plan(self, name: str, wps: List[Waypoint], path_xyz: np.ndarray):
        for i, wp in enumerate(wps):
            key = f"plan/{name}/wp{i}_{wp.name}"
            self.viz.add_sphere(key, 0.012, color=0xffaa00, opacity=1.0)
            self.viz.set_transform(key, pose_to_T(wp.pos, (0, 0, 0, 1)))
        if path_xyz is not None and len(path_xyz) > 2:
            self.viz.add_polyline(f"plan/{name}/path", path_xyz, color=0xffaa00, opacity=0.85)

    def plan_cartesian_raw(self, waypoints: List[Waypoint]) -> Tuple[List[List[float]], np.ndarray]:
        q_seed = self.get_arm_q()
        q_traj: List[List[float]] = []
        ee_path: List[np.ndarray] = []

        for a, b in zip(waypoints[:-1], waypoints[1:]):
            interp = self.interpolate_pose_list(a.pos, a.quat, b.pos, b.quat)
            if ee_path:
                interp = interp[1:]
            for pos, quat in interp:
                q = self.ik_limited(pos.tolist(), quat.tolist(), q_seed)
                q_traj.append(q)
                ee_path.append(np.asarray(pos, dtype=float))
                q_seed = q

        q_traj = self.densify_joint_traj(q_traj)
        return q_traj, (np.array(ee_path, dtype=float) if ee_path else np.zeros((0, 3), dtype=float))

    def plan_cartesian_avoid(self, name: str, builder: Callable[[int], List[Waypoint]]) -> List[List[float]]:
        state_id = p.saveState()
        last = ([], None, None)
        try:
            for attempt in range(self.plan.max_replan_tries):
                p.restoreState(state_id)
                self.lock_arm_targets()

                wps = builder(attempt)
                traj, path = self.plan_cartesian_raw(wps)

                collision = False
                for q in traj:
                    if self._in_collision(q, self.plan.obstacle_margin):
                        collision = True
                        break

                last = (traj, wps, path)
                if (not collision) and len(traj) > 0:
                    self._viz_plan(name, wps, path)
                    return traj

            traj, wps, path = last
            if wps is not None and path is not None:
                self._viz_plan(name, wps, path)
            return traj
        finally:
            try:
                p.removeState(state_id)
            except Exception:
                pass

    # -----------------------
    # Execution
    # -----------------------
    def _warn_limits(self, q_traj: List[List[float]], tag: str):
        for k, ji in enumerate(self.arm_joints):
            lo, hi = self.get_joint_limit(ji)
            mn = min(q[k] for q in q_traj)
            mx = max(q[k] for q in q_traj)
            if (mn - lo) < self.plan.limit_warn_rad or (hi - mx) < self.plan.limit_warn_rad:
                name = p.getJointInfo(self.robot_id, ji)[1].decode("utf-8")
                print(f"[WARN][{tag}] joint '{name}' near limit: range=[{mn:.3f},{mx:.3f}] limits=[{lo:.3f},{hi:.3f}]")

    def wait_until_reached(self, q_target: List[float], timeout_s: float) -> bool:
        t0 = time.time()
        while True:
            cur = self.get_arm_q()
            err = max(abs(float(a - b)) for a, b in zip(cur, q_target))
            if err <= self.plan.settle_tol_rad:
                return True
            if (time.time() - t0) >= timeout_s:
                return False
            self.lock_arm_targets(q_target)
            p.stepSimulation()
            self._update_meshcat()
            if self.plan.realtime:
                time.sleep(self.plan.dt)

    def exec_q_traj(self, name: str, q_traj: List[List[float]]):
        print(f"\n=== Execute: {name} knots={len(q_traj)} ===")
        if not q_traj:
            raise RuntimeError(f"[{name}] Empty trajectory (planning failed).")

        self._warn_limits(q_traj, tag=name)

        for q in q_traj:
            for _ in range(self.plan.knot_ticks):
                self.lock_arm_targets(q)
                p.stepSimulation()
                self._update_meshcat()

                if self._exec_collision_now(self.plan.exec_collision_margin):
                    raise RuntimeError(f"[FATAL][{name}] collision during execution. Stop.")

                if self.plan.realtime:
                    time.sleep(self.plan.dt)

        ok = self.wait_until_reached(q_traj[-1], timeout_s=self.plan.settle_timeout_s)
        if not ok:
            print(f"[WARN][{name}] did not fully converge to final knot; locking anyway.")
        self.lock_arm_targets(q_traj[-1])

    def hold(self, seconds: float):
        # keep holding targets (no drift)
        self.lock_arm_targets()
        steps = max(1, int(seconds * 60))
        for _ in range(steps):
            p.stepSimulation()
            self._update_meshcat()
            if self.plan.realtime:
                time.sleep(1.0 / 60)

    # -----------------------
    # Phone attach/detach
    # -----------------------
    def try_attach_phone(self) -> bool:
        if self.phone_constraint is not None:
            return True
        if self.gripper_close_frac < 0.86:
            return False

        st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        ee_pos = np.array(st[4], dtype=float)

        opos, _ = p.getBasePositionAndOrientation(self.phone_id)
        opos = np.array(opos, dtype=float)
        if float(np.linalg.norm(ee_pos - opos)) > 0.095:
            return False

        gpos, gquat = p.getBasePositionAndOrientation(self.gripper_id)
        inv_gpos, inv_gquat = p.invertTransform(gpos, gquat)
        rel_pos, rel_quat = p.multiplyTransforms(inv_gpos, inv_gquat, opos.tolist(), [0, 0, 0, 1])

        self.phone_constraint = p.createConstraint(
            parentBodyUniqueId=self.gripper_id, parentLinkIndex=-1,
            childBodyUniqueId=self.phone_id, childLinkIndex=-1,
            jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos, childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_quat, childFrameOrientation=[0, 0, 0, 1]
        )
        return True

    def detach_phone(self):
        if self.phone_constraint is None:
            return
        p.removeConstraint(self.phone_constraint)
        self.phone_constraint = None

    def _zero_phone_velocity(self):
        p.resetBaseVelocity(self.phone_id, [0, 0, 0], [0, 0, 0])

    def snap_phone_to_ws_if_close(self):
        if not self.scene.snap_place:
            return
        pos, _ = p.getBasePositionAndOrientation(self.phone_id)
        pos = np.array(pos, dtype=float)
        dxy = float(np.linalg.norm(pos[:2] - self.ws_target[:2]))
        if dxy <= self.scene.snap_xy_tol:
            z = self.ws_top_z + self.scene.phone_size[2] / 2.0 + 0.002
            p.resetBasePositionAndOrientation(self.phone_id,
                                              [self.ws_target[0], self.ws_target[1], z],
                                              [0, 0, 0, 1])
            self._zero_phone_velocity()

    # -----------------------
    # MeshCat update
    # -----------------------
    def _update_meshcat(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        self.viz.set_transform("robot/base", pose_to_T(base_pos, base_quat))

        pts = self._get_visual_chain_points()
        st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        ee_T = pose_to_T(st[4], st[5])
        self.arm_vis.update(pts, ee_T)

        # gripper visuals
        gpos, gquat = p.getBasePositionAndOrientation(self.gripper_id)
        T_g = pose_to_T(gpos, gquat)
        self.viz.set_transform("robot/gripper/base", T_g)

        # simple finger animation by gap
        max_gap = 0.085
        min_gap = 0.006
        gap = (1.0 - self.gripper_close_frac) * (max_gap - min_gap) + min_gap
        dx = gap / 2.0 + 0.018 / 2.0
        T_L = np.eye(4); T_L[:3, 3] = np.array([+dx, 0.075, 0.0], dtype=float)
        T_R = np.eye(4); T_R[:3, 3] = np.array([-dx, 0.075, 0.0], dtype=float)
        self.viz.set_transform("robot/gripper/fingerL", T_g @ T_L)
        self.viz.set_transform("robot/gripper/fingerR", T_g @ T_R)

        # wrist rgbd camera frame: put it at flange forward a bit (visual)
        # (This is a visual frame only; you can add real p.getCameraImage if needed.)
        Tw = np.eye(4)
        Tw[:3, 3] = np.array([0.0, 0.06, 0.02], dtype=float)
        self.viz.set_transform("robot/wrist_rgbd", ee_T @ Tw)

        # phone
        phone_pos, phone_quat = p.getBasePositionAndOrientation(self.phone_id)
        self.viz.set_transform("objects/phone", pose_to_T(phone_pos, phone_quat))

    # -----------------------
    # Init
    # -----------------------
    def init_robot_state(self):
        # Base already faces cabinet; this init pose matches UR-style “natural bend”
        q_init = [0.0, -1.05, 1.55, -1.70, -1.57, 0.0]
        for i, ji in enumerate(self.arm_joints):
            lo, hi = self.get_joint_limit(ji)
            q_init[i] = clamp(float(q_init[i]), lo, hi)
            p.resetJointState(self.robot_id, ji, float(q_init[i]))

        self.detach_phone()
        self.set_gripper(0.0)
        self.lock_arm_targets(q_init)

        # settle
        for _ in range(int(0.6 / self.plan.dt)):
            p.stepSimulation()
            self._update_meshcat()
            if self.plan.realtime:
                time.sleep(self.plan.dt)

        print("[Init] base strictly aligned (cabinet+ws forward); arm stable; gripper opened.")

    # -----------------------
    # Pose helpers
    # -----------------------
    def ee_world_pos(self) -> np.ndarray:
        st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        return np.array(st[4], dtype=float)

    def is_inside_cabinet(self, pos_xyz: np.ndarray) -> bool:
        y = float(pos_xyz[1])
        x = float(pos_xyz[0])
        z = float(pos_xyz[2])
        inside_y = (self.cab_inner_front_y <= y <= self.cab_inner_back_y)
        inside_x = (abs(x - self.cab_center[0]) <= self.cab_inner_w / 2.0)
        inside_z = (self.table_surface_z <= z <= self.cab_top_z)
        return inside_y and inside_x and inside_z

    def _height_policy(self, phone_pos: np.ndarray, attempt: int) -> Tuple[float, float]:
        """
        Avoid overly high lift. Always clamp below cabinet top.
        """
        margin = 0.08
        inside_clear_z = min(self.cab_top_z - margin,
                             phone_pos[2] + 0.17 + 0.01 * attempt)
        transit_z = min(self.cab_top_z - margin,
                        max(inside_clear_z, self.ws_top_z + 0.18))
        return float(inside_clear_z), float(transit_z)

    # -----------------------
    # Planning (PICK/RETRACT/PLACE)
    # -----------------------
    def plan_pick(self) -> List[List[float]]:
        phone_pos = np.array(p.getBasePositionAndOrientation(self.phone_id)[0], dtype=float)

        ee_quat = self.build_ee_quat_from_world_fwd_down(
            fwd_w=np.array([0.0, 1.0, 0.0]),   # into cabinet
            down_w=np.array([0.0, 0.0, -1.0]),
            ee_to_gripper_quat=self.ee_to_gripper_quat
        )
        cur_pos = self.ee_world_pos()

        x_min = self.cab_center[0] - self.cab_inner_w / 2.0 + 0.07
        x_max = self.cab_center[0] + self.cab_inner_w / 2.0 - 0.07

        def builder(attempt: int) -> List[Waypoint]:
            # structured detours: shift x slightly to avoid side collisions
            detours = [0.0, +0.025, -0.025, +0.05, -0.05, +0.075, -0.075]
            dx = detours[min(attempt, len(detours) - 1)]
            x = float(clamp(phone_pos[0] + dx, x_min, x_max))

            # y at cabinet opening (outside), then inside at phone y
            y_edge = float(self.cab_front_y - (0.20 + 0.02 * attempt))
            y_in = float(phone_pos[1])

            inside_clear_z, _ = self._height_policy(phone_pos, attempt)
            z_down = float(phone_pos[2])

            # clear first (from current), then go to opening, then move inside, then down
            return [
                Waypoint("start", cur_pos, ee_quat),
                Waypoint("to_clear", np.array([cur_pos[0], cur_pos[1], inside_clear_z]), ee_quat),
                Waypoint("edge_clear", np.array([x, y_edge, inside_clear_z]), ee_quat),
                Waypoint("in_clear", np.array([x, y_in, inside_clear_z]), ee_quat),
                Waypoint("down_pick", np.array([x, y_in, z_down]), ee_quat),
            ]

        return self.plan_cartesian_avoid("PICK", builder)

    def plan_retract(self) -> List[List[float]]:
        phone_pos = np.array(p.getBasePositionAndOrientation(self.phone_id)[0], dtype=float)

        ee_quat = self.build_ee_quat_from_world_fwd_down(
            fwd_w=np.array([0.0, 1.0, 0.0]),
            down_w=np.array([0.0, 0.0, -1.0]),
            ee_to_gripper_quat=self.ee_to_gripper_quat
        )
        cur_pos = self.ee_world_pos()

        x_min = self.cab_center[0] - self.cab_inner_w / 2.0 + 0.07
        x_max = self.cab_center[0] + self.cab_inner_w / 2.0 - 0.07
        x = float(clamp(phone_pos[0], x_min, x_max))

        def builder(attempt: int) -> List[Waypoint]:
            y_edge = float(self.cab_front_y - (0.22 + 0.02 * attempt))
            y_in = float(phone_pos[1])

            inside_clear_z, transit_z = self._height_policy(phone_pos, attempt)

            return [
                Waypoint("start", cur_pos, ee_quat),
                Waypoint("lift_clear", np.array([x, y_in, inside_clear_z]), ee_quat),
                Waypoint("back_edge", np.array([x, y_edge, inside_clear_z]), ee_quat),
                Waypoint("edge_transit", np.array([x, y_edge, transit_z]), ee_quat),
            ]

        return self.plan_cartesian_avoid("RETRACT", builder)

    def plan_place(self) -> List[List[float]]:
        ws = self.ws_target.copy()
        phone_pos = np.array(p.getBasePositionAndOrientation(self.phone_id)[0], dtype=float)

        ee_quat = self.build_ee_quat_from_world_fwd_down(
            fwd_w=np.array([0.0, -1.0, 0.0]),  # toward outside/operator
            down_w=np.array([0.0, 0.0, -1.0]),
            ee_to_gripper_quat=self.ee_to_gripper_quat
        )
        cur_pos = self.ee_world_pos()

        def builder(attempt: int) -> List[Waypoint]:
            inside_clear_z, transit_z = self._height_policy(phone_pos, attempt)
            y_exit = float(self.cab_front_y - (0.24 + 0.02 * attempt))

            # do NOT lift too high: clamp below cabinet top, and only a bit above workstation
            z_pre = float(min(self.cab_top_z - 0.08, self.ws_top_z + 0.20 + 0.01 * attempt))
            z_touch = float(self.ws_top_z + self.scene.phone_size[2] / 2.0 + 0.002)

            wps = [Waypoint("start", cur_pos, ee_quat)]

            if self.is_inside_cabinet(cur_pos):
                wps.append(Waypoint("exit_y", np.array([cur_pos[0], y_exit, inside_clear_z]), ee_quat))
                wps.append(Waypoint("exit_transit", np.array([cur_pos[0], y_exit, transit_z]), ee_quat))
            else:
                wps.append(Waypoint("to_transit", np.array([cur_pos[0], cur_pos[1], transit_z]), ee_quat))

            wps += [
                Waypoint("ws_transit", np.array([ws[0], ws[1], transit_z]), ee_quat),
                Waypoint("ws_pre", np.array([ws[0], ws[1], z_pre]), ee_quat),
                Waypoint("ws_touch", np.array([ws[0], ws[1], z_touch]), ee_quat),
            ]
            return wps

        return self.plan_cartesian_avoid("PLACE", builder)

    # -----------------------
    # Main flow (guarantee pick&place or raise clear error)
    # -----------------------
    def run(self):
        print("\n[Flow] 1) init hold (arm must not move)...")
        self.hold(0.8)

        print("\n[Flow] 2) plan pick path (collision-checked)...")
        pick_traj = self.plan_pick()
        print("[Flow] 3) execute pick...")
        self.set_gripper(0.0)
        self.hold(0.2)
        self.exec_q_traj("PICK", pick_traj)

        # close gripper & attach
        print("[Flow] 4) close gripper & attach phone...")
        attached = False
        steps = int(1.0 / self.plan.dt)
        for i in range(steps):
            frac = (i + 1) / max(1, steps)
            self.set_gripper(frac)
            p.stepSimulation()
            self._update_meshcat()
            if (not attached) and frac > 0.86:
                attached = self.try_attach_phone()
            if self.plan.realtime:
                time.sleep(self.plan.dt)

        print("[Pick] grasp_success =", attached)
        if not attached:
            raise RuntimeError("Pick failed: gripper could not attach phone (pose too far or not closed).")

        print("\n[Flow] 5) retract out of cabinet (collision-checked)...")
        ret_traj = self.plan_retract()
        self.exec_q_traj("RETRACT", ret_traj)
        self.hold(0.20)

        print("\n[Flow] 6) plan+execute place trajectory...")
        place_traj = self.plan_place()
        self.exec_q_traj("PLACE", place_traj)

        print("\n[Flow] 7) release phone on workstation...")
        self.detach_phone()
        self._zero_phone_velocity()

        # open gripper slowly
        for i in range(steps):
            frac = 1.0 - (i + 1) / max(1, steps)
            self.set_gripper(frac)
            p.stepSimulation()
            self._update_meshcat()
            if self.plan.realtime:
                time.sleep(self.plan.dt)

        self.set_gripper(0.0)
        self._zero_phone_velocity()
        self.snap_phone_to_ws_if_close()

        print("\n[Flow] Done. Ctrl+C to exit.")
        while True:
            self.lock_arm_targets()
            p.stepSimulation()
            self._update_meshcat()
            time.sleep(1 / 60)


if __name__ == "__main__":
    UR3PickPlaceDemo().run()
