# run.py
# PyBullet(DIRECT) + MeshCat
# UR3e + Robotiq 2F-85: pick phone from cabinet mid-shelf -> place on workstation
#
# Key fix for "joint disconnect" (visual gap/segment drifting in MeshCat):
# - Use unit primitives (unit cylinder / unit sphere / unit box fallback)
# - Apply full scale via transform matrix each frame: S=diag([r, L, r])
# - Strong overlap so cylinders always penetrate joint spheres => never visible gaps
#
# CPU-only, no GUI, view in MeshCat browser.

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
# Config
# =========================
@dataclass(frozen=True)
class SceneConfig:
    # bigger table (for reach + avoidance)
    table_w: float = 3.00
    table_d: float = 1.80
    table_h: float = 0.75
    table_top_t: float = 0.045
    table_leg_w: float = 0.075
    table_leg_inset: float = 0.20
    table_center_xy: Tuple[float, float] = (1.12, 0.00)

    pedestal_size: Tuple[float, float, float] = (0.26, 0.26, 0.14)

    # cabinet (two levels)
    cab_outer: Tuple[float, float, float] = (0.92, 0.68, 0.72)  # W, D, H
    cab_wall: float = 0.02
    cab_shelf_t: float = 0.02
    cab_offset_xy: Tuple[float, float] = (0.50, 0.40)

    # workstation
    ws_plate: Tuple[float, float, float] = (0.36, 0.30, 0.015)
    ws_leg_h: float = 0.03
    ws_offset_xy: Tuple[float, float] = (0.92, -0.46)

    # phone
    phone_size: Tuple[float, float, float] = (0.075, 0.150, 0.008)

    table_margin: float = 0.20

    snap_place: bool = True
    snap_xy_tol: float = 0.05
    snap_z_tol: float = 0.08


@dataclass(frozen=True)
class PlannerConfig:
    dt: float = 1.0 / 240.0
    realtime: bool = True

    cart_step: float = 0.010
    ang_step_deg: float = 6.5

    max_step_delta_rad: float = 0.14
    joint_damping: float = 0.06
    densify_max_dq: float = 0.05

    obstacle_margin: float = 0.022
    exec_collision_margin: float = 0.008

    max_replan_tries: int = 28

    knot_ticks: int = 7
    settle_tol_rad: float = 0.020
    settle_timeout_s: float = 2.0

    kp: float = 0.85
    kd: float = 1.25
    max_force: float = 2600.0

    auto_boost_steps: int = 140
    auto_boost_force: float = 5200.0
    auto_boost_kp: float = 1.20

    # shortest-path cost weights
    cost_w_ee: float = 1.0
    cost_w_joint: float = 0.20


@dataclass
class Waypoint:
    name: str
    pos: np.ndarray
    quat: np.ndarray


# =========================
# Math helpers
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


# =========================
# MeshCat Bridge
# =========================
class MeshcatBridge:
    def __init__(self):
        self.vis = meshcat.Visualizer().open()
        self.vis.delete()
        self._created = set()

        self.add_frame("world", length=0.30, radius=0.01)
        self.set_transform("world", np.eye(4))

        try:
            self.vis["/Cameras/default"].set_transform(self._T(t=[1.75, 0.25, 1.35]))
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
        if hasattr(g, "Cylinder"):
            try:
                self.vis[path].set_object(g.Cylinder(float(length), float(radius)), self._mat(color, opacity))
                return
            except Exception:
                pass
        self.vis[path].set_object(g.Box([radius * 2, length, radius * 2]), self._mat(color, opacity))

    def add_unit_cylinder_like(self, path: str, color=0xcccccc, opacity=1.0) -> bool:
        """
        Create a unit-length, unit-radius cylinder-like primitive at `path`.
        Returns: uses_box_fallback (True if Box is used).
        """
        if path in self._created:
            return False
        self._created.add(path)

        if hasattr(g, "Cylinder"):
            try:
                # Unit primitive: avoid Cylinder(height,radius) vs Cylinder(radius,height) ambiguity.
                # We'll always scale with matrix to the desired dimensions.
                self.vis[path].set_object(g.Cylinder(1.0, 1.0), self._mat(color, opacity))
                return False
            except Exception:
                pass

        self.vis[path].set_object(g.Box([1.0, 1.0, 1.0]), self._mat(color, opacity))
        return True

    def add_frame(self, path: str, length: float = 0.12, radius: float = 0.006):
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
# UR Industrial Visual (ROBUST, NO GAPS)
# =========================
class URArmVisual:
    """
    Robust, gap-free visual arm:
    - segments: unit cylinder (or unit box fallback) scaled by matrix S=diag([r, L, r])
    - joints: unit sphere scaled by matrix S=diag([R, R, R])
    - cylinders extend into joint spheres (overlap) so there is never any visible gap.
    """
    def __init__(self, viz: MeshcatBridge):
        self.viz = viz

        self.seg_rad: List[float] = []
        self.joint_rad: List[float] = []

        # if cylinder creation falls back to box, x/z scale must be 2r (box width=1 => half extent=0.5)
        self._seg_xz_factor: float = 1.0
        self._flange_xz_factor: float = 1.0

        # Tune for “never show gaps”
        self.joint_cover_scale: float = 1.08   # sphere slightly bigger than neighbor segment radius
        self.overlap_frac: float = 0.85        # cylinder penetrates sphere deeply => never gap

    def create(self, seg_rad: List[float]):
        assert len(seg_rad) == 7
        self.seg_rad = [float(r) for r in seg_rad]

        # joint radius derived from adjacent segments for smooth look
        self.joint_rad = []
        for j in range(6):
            r = max(self.seg_rad[j], self.seg_rad[j + 1])
            self.joint_rad.append(float(r * self.joint_cover_scale))

        # segments as unit cylinder-like
        uses_box = self.viz.add_unit_cylinder_like("robot/ur/seg0", color=0xd7d7d7, opacity=1.0)
        for k in range(1, 7):
            self.viz.add_unit_cylinder_like(f"robot/ur/seg{k}", color=0xd7d7d7, opacity=1.0)
        self._seg_xz_factor = 2.0 if uses_box else 1.0

        # joints as unit spheres (Sphere API is stable)
        for k in range(6):
            path = f"robot/ur/joint{k}"
            if path in self.viz._created:
                continue
            self.viz._created.add(path)
            self.viz.vis[path].set_object(g.Sphere(1.0), self.viz._mat(0xbcbcbc, 1.0))

        # flange as unit cylinder-like
        uses_box_f = self.viz.add_unit_cylinder_like("robot/ur/flange", color=0x8f8f8f, opacity=1.0)
        self._flange_xz_factor = 2.0 if uses_box_f else 1.0

    def _end_overlap(self, endpoint_index: int) -> float:
        """
        endpoint_index refers to point idx in pts list:
        pts: 0 base, 1..6 joint0..5, 7 ee
        overlap applied for endpoints 1..6 (joint centers)
        """
        if 1 <= endpoint_index <= 6:
            jr = self.joint_rad[endpoint_index - 1]
            return float(self.overlap_frac * jr)
        return 0.0

    def update(self, pts: List[np.ndarray], ee_T: np.ndarray):
        # segments: k connects pts[k] -> pts[k+1]
        for k in range(7):
            p0 = np.asarray(pts[k], dtype=float)
            p1 = np.asarray(pts[k + 1], dtype=float)
            v = p1 - p0
            dist = float(np.linalg.norm(v))
            if dist < 1e-9:
                continue
            d = v / dist

            # overlap into joint spheres
            ol = self._end_overlap(k)
            or_ = self._end_overlap(k + 1)
            L = dist + ol + or_

            # asym overlap => shift center
            center = (p0 + p1) * 0.5 + d * (or_ - ol) * 0.5

            R = align_y_to_dir(d)
            r = self.seg_rad[k]

            # unit cylinder scaled to real: radius=r, length=L
            S = np.diag([self._seg_xz_factor * r, L, self._seg_xz_factor * r])

            T = np.eye(4)
            T[:3, :3] = R @ S
            T[:3, 3] = center
            self.viz.set_transform(f"robot/ur/seg{k}", T)

        # joint spheres
        for k in range(6):
            pj = np.asarray(pts[1 + k], dtype=float)
            jr = float(self.joint_rad[k])
            Tj = np.eye(4)
            Tj[:3, :3] = np.diag([jr, jr, jr])
            Tj[:3, 3] = pj
            self.viz.set_transform(f"robot/ur/joint{k}", Tj)

        # flange
        flange_L = 0.038
        flange_r = 0.040
        Rx90 = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]], dtype=float)
        Tf = np.eye(4)
        Tf[:3, :3] = (ee_T[:3, :3] @ Rx90) @ np.diag([self._flange_xz_factor * flange_r, flange_L, self._flange_xz_factor * flange_r])
        Tf[:3, 3] = ee_T[:3, 3]
        self.viz.set_transform("robot/ur/flange", Tf)


# =========================
# Demo
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
        p.setPhysicsEngineParameter(numSolverIterations=300, deterministicOverlappingPairs=1)

        p.loadURDF("plane.urdf")

        self.viz = MeshcatBridge()
        self.arm_vis = URArmVisual(self.viz)

        self.phone_constraint: Optional[int] = None
        self.gripper_close_frac: float = 0.0

        self._build_scene()

        self.ur_scale = self._choose_ur_scale()
        self.robot_base_pos, self.robot_base_quat, self.robot_base_yaw = self._choose_base_pose_xy_align()

        self._build_robot_and_mount()
        self._setup_control_and_visual()
        self.init_robot_state()

        for _ in range(60):
            p.stepSimulation()
            self._update_meshcat()

    # -----------------------
    # Scene build
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

        shelf_width = W - 2 * w
        shelf_depth = D - w
        self.shelf_center = self.cab_center + np.array([0, 0, 0], dtype=float)
        self.cabinet_bodies.append(self._add_box("env/cab/shelf_mid",
                                                 self.shelf_center,
                                                 np.array([shelf_width, shelf_depth, shelf_t], dtype=float),
                                                 color=0x8f8f8f, opacity=0.98, friction=0.95))

        # front frame (visual only)
        frame_t = 0.01
        frame_y = self.cab_center[1] - (D / 2 - frame_t / 2)
        frame_center = np.array([self.cab_center[0], frame_y, self.cab_center[2]], dtype=float)
        self.viz.add_box("env/cab/front_frame", (W, frame_t, H), color=0xdddddd, opacity=0.10)
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

        # phone on mid shelf
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

        # placeholders
        self.ped_size = np.array(sc.pedestal_size, dtype=float)
        self.viz.add_box("robot/pedestal", tuple(self.ped_size.tolist()), color=0x555555, opacity=1.0)
        self.viz.add_frame("robot/base", length=0.18, radius=0.01)

        self.obstacles = [self.table_top_id] + self.cabinet_bodies + self.workstation_bodies

    # -----------------------
    # Base selection: STRICT XY align
    # -----------------------
    def _table_xy_bounds(self):
        sc = self.scene
        tx0 = self.table_center_xy[0] - sc.table_w / 2 + sc.table_margin
        tx1 = self.table_center_xy[0] + sc.table_w / 2 - sc.table_margin
        ty0 = self.table_center_xy[1] - sc.table_d / 2 + sc.table_margin
        ty1 = self.table_center_xy[1] + sc.table_d / 2 - sc.table_margin
        return tx0, tx1, ty0, ty1

    def _base_yaw_to_face_cab(self, base_xy: np.ndarray) -> float:
        v = np.array([self.cab_center[0] - base_xy[0],
                      self.cab_center[1] - base_xy[1]], dtype=float)
        return float(math.atan2(v[1], v[0]))

    def _choose_base_pose_xy_align(self):
        """
        Requirement:
        base_x = phone_x
        base_y = workstation_y
        yaw faces cabinet center.
        """
        tx0, tx1, ty0, ty1 = self._table_xy_bounds()

        phone = self.phone_init_pos.copy()
        ws = self.ws_target.copy()
        base_xy = np.array([phone[0], ws[1]], dtype=float)

        base_xy[0] = clamp(base_xy[0], tx0, tx1)
        base_xy[1] = clamp(base_xy[1], ty0, ty1)

        cab_W, cab_D, _ = self.scene.cab_outer
        if (abs(base_xy[0] - self.cab_center[0]) < cab_W / 2 + 0.10) and \
           (abs(base_xy[1] - self.cab_center[1]) < cab_D / 2 + 0.10):
            base_xy[1] = clamp(self.cab_center[1] - (cab_D / 2 + 0.40), ty0, ty1)

        yaw = self._base_yaw_to_face_cab(base_xy)
        q = p.getQuaternionFromEuler([0, 0, yaw])

        base = np.array([base_xy[0], base_xy[1], self.table_surface_z + self.scene.pedestal_size[2]], dtype=float)

        print(f"[BaseSelect] base_xy={base_xy.tolist()} yaw(deg)={math.degrees(yaw):.1f}")
        return base, q, yaw

    # -----------------------
    # Reach-aware scaling
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
        ur_urdf = airo_models.get_urdf_path("ur3e")
        flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_IGNORE_VISUAL_SHAPES

        scales = [1.10, 1.20, 1.30, 1.40, 1.48]
        best_s, best_err = scales[0], 1e9

        phone = self.phone_init_pos.copy()
        probe = phone + np.array([0.0, -0.18, 0.10], dtype=float)

        base_xy = np.array([phone[0], self.ws_target[1]], dtype=float)
        yaw = self._base_yaw_to_face_cab(base_xy)
        qbase = p.getQuaternionFromEuler([0, 0, yaw])
        base_pos = np.array([base_xy[0], base_xy[1], self.table_surface_z + self.scene.pedestal_size[2]], dtype=float)

        for s in scales:
            rid = p.loadURDF(ur_urdf, basePosition=base_pos.tolist(),
                             baseOrientation=qbase, useFixedBase=True,
                             flags=flags, globalScaling=s)
            ee = self._guess_ee_link_tmp(rid)

            q = p.calculateInverseKinematics(rid, ee, probe.tolist(),
                                             maxNumIterations=260, residualThreshold=1e-4)
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

    # -----------------------
    # Robot load + mount
    # -----------------------
    def _build_robot_and_mount(self):
        ur_urdf = airo_models.get_urdf_path("ur3e")
        rq_urdf = airo_models.get_urdf_path("robotiq_2f_85")

        flags = (p.URDF_USE_INERTIA_FROM_FILE |
                 p.URDF_IGNORE_VISUAL_SHAPES |
                 p.URDF_USE_SELF_COLLISION)

        self.robot_id = p.loadURDF(
            ur_urdf,
            basePosition=self.robot_base_pos.tolist(),
            baseOrientation=self.robot_base_quat,
            useFixedBase=True, flags=flags, globalScaling=self.ur_scale
        )
        self.gripper_id = p.loadURDF(
            rq_urdf,
            basePosition=self.robot_base_pos.tolist(),
            baseOrientation=self.robot_base_quat,
            useFixedBase=False, flags=flags, globalScaling=self.ur_scale
        )

        self.ee_link = self._guess_ee_link_tmp(self.robot_id)

        p.createConstraint(
            self.robot_id, self.ee_link,
            self.gripper_id, -1,
            p.JOINT_FIXED, [0, 0, 0],
            [0, 0, 0], [0, 0, 0],
            [0, 0, 0, 1], [0, 0, 0, 1]
        )

        ped_center = np.array([self.robot_base_pos[0], self.robot_base_pos[1],
                               self.table_surface_z + self.scene.pedestal_size[2] / 2.0], dtype=float)
        self.viz.set_transform("robot/pedestal", pose_to_T(ped_center, self.robot_base_quat))
        self.viz.set_transform("robot/base", pose_to_T(self.robot_base_pos.tolist(), self.robot_base_quat))

    # -----------------------
    # Joint selection (strict)
    # -----------------------
    def _select_arm_joints_strict(self) -> List[int]:
        preferred = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        num = p.getNumJoints(self.robot_id)
        name_to_index = {}
        for j in range(num):
            info = p.getJointInfo(self.robot_id, j)
            if info[2] != p.JOINT_REVOLUTE:
                continue
            jname = info[1].decode("utf-8")
            name_to_index[jname] = j

        missing = [nm for nm in preferred if nm not in name_to_index]
        if missing:
            raise RuntimeError(
                f"[FATAL] UR joints missing={missing}. Found revolute={list(name_to_index.keys())}"
            )
        return [name_to_index[nm] for nm in preferred]

    def _setup_control_and_visual(self):
        self.arm_joints = self._select_arm_joints_strict()
        print("[Debug] arm joints indices:", self.arm_joints)

        self.movable_joints = [j for j in range(p.getNumJoints(self.robot_id))
                               if p.getJointInfo(self.robot_id, j)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        self.movable_index = {j: i for i, j in enumerate(self.movable_joints)}

        cont = math.radians(359.0)
        self.joint_limit_by_index: Dict[int, Tuple[float, float]] = {}
        self.lower, self.upper, self.ranges = [], [], []
        for j in self.movable_joints:
            info = p.getJointInfo(self.robot_id, j)
            lo, hi = float(info[8]), float(info[9])
            if abs(hi - lo) < 1e-9:
                lo, hi = -cont, +cont
            if (hi - lo) > math.radians(720.0):
                lo, hi = -cont, +cont
            self.joint_limit_by_index[j] = (lo, hi)
            self.lower.append(lo)
            self.upper.append(hi)
            self.ranges.append(max(hi - lo, 1e-6))

        gnum = p.getNumJoints(self.gripper_id)
        self.gripper_joints = [j for j in range(gnum)
                               if p.getJointInfo(self.gripper_id, j)[2] in (p.JOINT_PRISMATIC, p.JOINT_REVOLUTE)]
        self.gripper_limits = []
        for j in self.gripper_joints:
            lo, hi = float(p.getJointInfo(self.gripper_id, j)[8]), float(p.getJointInfo(self.gripper_id, j)[9])
            if abs(hi - lo) < 1e-9:
                lo, hi = 0.0, 0.8
            self.gripper_limits.append((lo, hi))

        # UR-like taper (thin, industrial)
        seg_rad = [0.029, 0.028, 0.026, 0.025, 0.023, 0.0215, 0.020]
        self.arm_vis.create(seg_rad)

        # gripper visuals
        self.viz.add_box("robot/gripper/base", (0.085, 0.045, 0.045), color=0x222222, opacity=1.0)
        self.viz.add_box("robot/gripper/fingerL", (0.018, 0.095, 0.02), color=0x333333, opacity=1.0)
        self.viz.add_box("robot/gripper/fingerR", (0.018, 0.095, 0.02), color=0x333333, opacity=1.0)

        # wrist RGBD frame
        self.viz.add_frame("robot/wrist_rgbd", length=0.10, radius=0.006)

        # ee->gripper transform
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

        self.obstacles = [self.table_top_id] + self.cabinet_bodies + self.workstation_bodies

    def get_joint_limit(self, joint_index: int) -> Tuple[float, float]:
        return self.joint_limit_by_index.get(joint_index, (-math.radians(359), math.radians(359)))

    def get_arm_q(self) -> List[float]:
        return [p.getJointState(self.robot_id, j)[0] for j in self.arm_joints]

    # -----------------------
    # Control
    # -----------------------
    def lock_arm_targets(self, q_arm: Optional[List[float]] = None,
                         kp: Optional[float] = None, kd: Optional[float] = None,
                         force: Optional[float] = None):
        if q_arm is None:
            q_arm = self.get_arm_q()
        kp = self.plan.kp if kp is None else kp
        kd = self.plan.kd if kd is None else kd
        force = self.plan.max_force if force is None else force

        q_cmd = []
        for qi, ji in zip(q_arm, self.arm_joints):
            lo, hi = self.get_joint_limit(ji)
            q_cmd.append(clamp(float(qi), lo, hi))

        p.setJointMotorControlArray(
            self.robot_id,
            jointIndices=self.arm_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=q_cmd,
            targetVelocities=[0.0] * 6,
            positionGains=[kp] * 6,
            velocityGains=[kd] * 6,
            forces=[force] * 6
        )

    def _ensure_motion_or_boost(self, q_target: List[float], tag: str):
        cur0 = self.get_arm_q()
        err0 = max(abs(a - b) for a, b in zip(cur0, q_target))
        if err0 < 1e-3:
            return

        best_err = err0
        for _ in range(self.plan.auto_boost_steps):
            self.lock_arm_targets(q_target)
            p.stepSimulation()
            self._update_meshcat()
            if self.plan.realtime:
                time.sleep(self.plan.dt)
            cur = self.get_arm_q()
            err = max(abs(a - b) for a, b in zip(cur, q_target))
            best_err = min(best_err, err)
            if err < 0.6 * err0:
                return

        print(f"[WARN][{tag}] Arm stuck (err {err0:.3f}->{best_err:.3f}). Boosting.")
        for _ in range(int(0.7 / self.plan.dt)):
            self.lock_arm_targets(q_target, kp=self.plan.auto_boost_kp, kd=self.plan.kd, force=self.plan.auto_boost_force)
            p.stepSimulation()
            self._update_meshcat()
            if self.plan.realtime:
                time.sleep(self.plan.dt)

    # -----------------------
    # Gripper
    # -----------------------
    def set_gripper(self, close_fraction: float):
        close_fraction = clamp(close_fraction, 0.0, 1.0)
        self.gripper_close_frac = close_fraction
        for (j, (lo, hi)) in zip(self.gripper_joints, self.gripper_limits):
            tgt = lo + close_fraction * (hi - lo)
            p.setJointMotorControl2(self.gripper_id, j, p.POSITION_CONTROL, targetPosition=float(tgt), force=320)

    # -----------------------
    # TRUE joint pivots (fix scatter)
    # -----------------------
    def _link_world_pose(self, link_index: int):
        if link_index == -1:
            pos, quat = p.getBasePositionAndOrientation(self.robot_id)
            return pos, quat
        st = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)
        return st[4], st[5]

    def _joint_pivot_world(self, joint_index: int) -> np.ndarray:
        info = p.getJointInfo(self.robot_id, joint_index)
        parent_frame_pos = info[14]
        parent_frame_orn = info[15]
        parent_index = info[16]
        parent_pos, parent_quat = self._link_world_pose(parent_index)
        wpos, _ = p.multiplyTransforms(parent_pos, parent_quat, parent_frame_pos, parent_frame_orn)
        return np.array(wpos, dtype=float)

    def _get_visual_chain_points(self) -> List[np.ndarray]:
        base_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        pts = [np.array(base_pos, dtype=float)]
        for j in self.arm_joints:
            pts.append(self._joint_pivot_world(j))
        st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        pts.append(np.array(st[4], dtype=float))
        return pts

    # -----------------------
    # Orientation helpers
    # -----------------------
    def _quat_mul(self, qa, qb):
        return p.multiplyTransforms([0, 0, 0], qa, [0, 0, 0], qb)[1]

    def _quat_inv(self, q):
        return p.invertTransform([0, 0, 0], q)[1]

    def build_ee_quat_from_world_fwd_down(self, fwd_w: np.ndarray, down_w: np.ndarray) -> np.ndarray:
        fwd = np.asarray(fwd_w, dtype=float)
        down = np.asarray(down_w, dtype=float)
        fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
        down = down / max(np.linalg.norm(down), 1e-9)
        left = np.cross(down, fwd)
        left = left / max(np.linalg.norm(left), 1e-9)
        fwd = np.cross(left, down)
        Rg = np.stack([left, fwd, down], axis=1)
        Rg = orthonormalize(Rg)

        tr = Rg[0, 0] + Rg[1, 1] + Rg[2, 2]
        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (Rg[2, 1] - Rg[1, 2]) / S
            y = (Rg[0, 2] - Rg[2, 0]) / S
            z = (Rg[1, 0] - Rg[0, 1]) / S
            qg = (x, y, z, w)
        else:
            i = int(np.argmax([Rg[0, 0], Rg[1, 1], Rg[2, 2]]))
            if i == 0:
                S = math.sqrt(1.0 + Rg[0, 0] - Rg[1, 1] - Rg[2, 2]) * 2
                w = (Rg[2, 1] - Rg[1, 2]) / S
                x = 0.25 * S
                y = (Rg[0, 1] + Rg[1, 0]) / S
                z = (Rg[0, 2] + Rg[2, 0]) / S
            elif i == 1:
                S = math.sqrt(1.0 + Rg[1, 1] - Rg[0, 0] - Rg[2, 2]) * 2
                w = (Rg[0, 2] - Rg[2, 0]) / S
                x = (Rg[0, 1] + Rg[1, 0]) / S
                y = 0.25 * S
                z = (Rg[1, 2] + Rg[2, 1]) / S
            else:
                S = math.sqrt(1.0 + Rg[2, 2] - Rg[0, 0] - Rg[1, 1]) * 2
                w = (Rg[1, 0] - Rg[0, 1]) / S
                x = (Rg[0, 2] + Rg[2, 0]) / S
                y = (Rg[1, 2] + Rg[2, 1]) / S
                z = 0.25 * S
            qg = (x, y, z, w)

        qg = np.array(qg, dtype=float)
        q_ee = self._quat_mul(qg.tolist(), self._quat_inv(self.ee_to_gripper_quat.tolist()))
        return np.array(q_ee, dtype=float)

    # -----------------------
    # IK (continuous)
    # -----------------------
    def _wrap_near(self, q: float, ref: float) -> float:
        two_pi = 2.0 * math.pi
        dq = q - ref
        dq = (dq + math.pi) % two_pi - math.pi
        return ref + dq

    def ik_limited(self, target_pos, target_quat_xyzw, seed_arm_q: List[float]) -> List[float]:
        rest_m = [0.0] * len(self.movable_joints)
        for k, j in enumerate(self.arm_joints):
            rest_m[self.movable_index[j]] = float(seed_arm_q[k])

        q_sol = p.calculateInverseKinematics(
            self.robot_id, self.ee_link,
            target_pos, targetOrientation=target_quat_xyzw,
            lowerLimits=self.lower, upperLimits=self.upper,
            jointRanges=self.ranges, restPoses=rest_m,
            jointDamping=[self.plan.joint_damping] * len(self.lower),
            solver=p.IK_DLS, maxNumIterations=420, residualThreshold=1e-4
        )
        q_sol = list(q_sol)

        if len(q_sol) == len(self.movable_joints):
            q_map = {j: q_sol[i] for i, j in enumerate(self.movable_joints)}
            q_arm = [float(q_map[j]) for j in self.arm_joints]
        elif len(q_sol) == p.getNumJoints(self.robot_id):
            q_arm = [float(q_sol[j]) for j in self.arm_joints]
        else:
            q_arm = list(seed_arm_q)

        out = []
        for qi, ri in zip(q_arm, seed_arm_q):
            qi = self._wrap_near(float(qi), float(ri))
            d = qi - float(ri)
            if abs(d) > self.plan.max_step_delta_rad:
                qi = float(ri) + math.copysign(self.plan.max_step_delta_rad, d)
            out.append(float(qi))

        for i, j in enumerate(self.arm_joints):
            lo, hi = self.get_joint_limit(j)
            out[i] = clamp(out[i], lo, hi)
        return out

    # -----------------------
    # Collision checks
    # -----------------------
    def _compute_gripper_world_from_ee(self):
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
    # Interpolation
    # -----------------------
    def interpolate_pose_list(self, p0, q0, p1, q1):
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
                out.append(((1 - t) * qa + t * qb).tolist())
        return out

    # -----------------------
    # Shortest-path planning among candidates
    # -----------------------
    def plan_cartesian_raw(self, waypoints: List[Waypoint]):
        q_seed = self.get_arm_q()
        q_traj = []
        ee_path = []
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

    def _traj_cost(self, q_traj: List[List[float]], ee_path: np.ndarray) -> float:
        if len(q_traj) < 2:
            return 1e9
        ee_len = 0.0
        if ee_path is not None and len(ee_path) >= 2:
            dif = ee_path[1:] - ee_path[:-1]
            ee_len = float(np.sum(np.linalg.norm(dif, axis=1)))
        q = np.array(q_traj, dtype=float)
        dq = np.abs(q[1:] - q[:-1])
        joint_cost = float(np.sum(dq))
        return self.plan.cost_w_ee * ee_len + self.plan.cost_w_joint * joint_cost

    def _viz_plan(self, name: str, wps: List[Waypoint], path_xyz: np.ndarray):
        for i, wp in enumerate(wps):
            key = f"plan/{name}/wp{i}_{wp.name}"
            self.viz.add_sphere(key, 0.012, color=0xffaa00, opacity=1.0)
            self.viz.set_transform(key, pose_to_T(wp.pos, (0, 0, 0, 1)))
        if path_xyz is not None and len(path_xyz) > 2:
            self.viz.add_polyline(f"plan/{name}/path", path_xyz, color=0xffaa00, opacity=0.85)

    def plan_cartesian_avoid_shortest(self, name: str, builder: Callable[[int], List[Waypoint]]) -> List[List[float]]:
        state_id = p.saveState()
        best = None  # (cost, traj, wps, path)
        last_any = None

        result_traj: List[List[float]] = []
        result_wps: Optional[List[Waypoint]] = None
        result_path: Optional[np.ndarray] = None

        try:
            for attempt in range(self.plan.max_replan_tries):
                p.restoreState(state_id)
                self.lock_arm_targets(self.get_arm_q())

                wps = builder(attempt)
                traj, path = self.plan_cartesian_raw(wps)

                if len(traj) == 0:
                    continue

                collision = False
                for q in traj:
                    if self._in_collision(q, self.plan.obstacle_margin):
                        collision = True
                        break

                last_any = (traj, wps, path)
                if collision:
                    continue

                cost = self._traj_cost(traj, path)
                if (best is None) or (cost < best[0]):
                    best = (cost, traj, wps, path)

            if best is None:
                if last_any is not None:
                    result_traj, result_wps, result_path = last_any
                else:
                    result_traj, result_wps, result_path = [], None, None
            else:
                _, result_traj, result_wps, result_path = best

            # IMPORTANT: restore sim state so planning doesn't "leave" robot at a sampled pose
            p.restoreState(state_id)

            if result_wps is not None and result_path is not None:
                self._viz_plan(name, result_wps, result_path)

            return result_traj
        finally:
            try:
                p.removeState(state_id)
            except Exception:
                pass

    # -----------------------
    # Execution
    # -----------------------
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

        self._ensure_motion_or_boost(q_traj[0], tag=f"{name}/first")

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
            print(f"[WARN][{name}] final convergence not perfect; locking anyway.")
        self.lock_arm_targets(q_traj[-1])

    def hold(self, seconds: float):
        self.lock_arm_targets(self.get_arm_q())
        steps = max(1, int(seconds * 60))
        for _ in range(steps):
            p.stepSimulation()
            self._update_meshcat()
            if self.plan.realtime:
                time.sleep(1.0 / 60)

    # -----------------------
    # Attach / Detach phone
    # -----------------------
    def try_attach_phone(self) -> bool:
        if self.phone_constraint is not None:
            return True
        if self.gripper_close_frac < 0.86:
            return False

        gpos, _ = p.getBasePositionAndOrientation(self.gripper_id)
        gpos = np.array(gpos, dtype=float)
        opos, _ = p.getBasePositionAndOrientation(self.phone_id)
        opos = np.array(opos, dtype=float)

        if float(np.linalg.norm(gpos - opos)) > 0.085:
            return False

        inv_gpos, inv_gquat = p.invertTransform(gpos.tolist(), p.getBasePositionAndOrientation(self.gripper_id)[1])
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

    # -----------------------
    # Visual update
    # -----------------------
    def _update_meshcat(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.robot_id)
        self.viz.set_transform("robot/base", pose_to_T(base_pos, base_quat))

        pts = self._get_visual_chain_points()
        st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        ee_T = pose_to_T(st[4], st[5])
        self.arm_vis.update(pts, ee_T)

        gpos, gquat = p.getBasePositionAndOrientation(self.gripper_id)
        T_g = pose_to_T(gpos, gquat)
        self.viz.set_transform("robot/gripper/base", T_g)

        max_gap = 0.085
        min_gap = 0.006
        gap = (1.0 - self.gripper_close_frac) * (max_gap - min_gap) + min_gap
        dx = gap / 2.0 + 0.018 / 2.0
        T_L = np.eye(4); T_L[:3, 3] = np.array([+dx, 0.075, 0.0], dtype=float)
        T_R = np.eye(4); T_R[:3, 3] = np.array([-dx, 0.075, 0.0], dtype=float)
        self.viz.set_transform("robot/gripper/fingerL", T_g @ T_L)
        self.viz.set_transform("robot/gripper/fingerR", T_g @ T_R)

        Tw = np.eye(4)
        Tw[:3, 3] = np.array([0.0, 0.06, 0.02], dtype=float)
        self.viz.set_transform("robot/wrist_rgbd", ee_T @ Tw)

        phone_pos, phone_quat = p.getBasePositionAndOrientation(self.phone_id)
        self.viz.set_transform("objects/phone", pose_to_T(phone_pos, phone_quat))

    # -----------------------
    # Init pose
    # -----------------------
    def init_robot_state(self):
        # UR-like init facing cabinet, minimal weird twisting
        q_init = [0.0, -1.05, 1.55, -1.70, -1.57, 0.0]
        for i, ji in enumerate(self.arm_joints):
            lo, hi = self.get_joint_limit(ji)
            q_init[i] = clamp(float(q_init[i]), lo, hi)
            p.resetJointState(self.robot_id, ji, float(q_init[i]))

        self.detach_phone()
        self.set_gripper(0.0)
        self.lock_arm_targets(q_init)

        for _ in range(int(0.7 / self.plan.dt)):
            p.stepSimulation()
            self._update_meshcat()
            if self.plan.realtime:
                time.sleep(self.plan.dt)

        print("[Init] Arm locked & stable, gripper opened. (No motion before commands.)")

    # -----------------------
    # Planning: Pick / Retract / Place (shortest among collision-free)
    # -----------------------
    def ee_world_pos(self) -> np.ndarray:
        st = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        return np.array(st[4], dtype=float)

    def _height_policy_low(self, phone_pos: np.ndarray, attempt: int):
        z_inside = float(min(self.cab_top_z - 0.10, phone_pos[2] + 0.095 + 0.003 * attempt))
        z_outside = float(max(self.table_surface_z + 0.26, self.ws_top_z + 0.10))
        return z_inside, z_outside

    def plan_pick(self) -> List[List[float]]:
        phone_pos = np.array(p.getBasePositionAndOrientation(self.phone_id)[0], dtype=float)

        ee_quat = self.build_ee_quat_from_world_fwd_down(
            fwd_w=np.array([0.0, 1.0, 0.0]),
            down_w=np.array([0.0, 0.0, -1.0])
        )
        cur_pos = self.ee_world_pos()

        x_min = self.cab_center[0] - self.cab_inner_w / 2.0 + 0.08
        x_max = self.cab_center[0] + self.cab_inner_w / 2.0 - 0.08

        def builder(attempt: int):
            detours = [0.0, +0.02, -0.02, +0.04, -0.04, +0.06, -0.06]
            dx = detours[min(attempt, len(detours) - 1)]
            x = float(clamp(phone_pos[0] + dx, x_min, x_max))

            y_edge = float(self.cab_front_y - (0.20 + 0.01 * attempt))
            y_in = float(phone_pos[1])

            z_inside, _ = self._height_policy_low(phone_pos, attempt)
            z_down = float(phone_pos[2] + 0.001)

            return [
                Waypoint("start", cur_pos, ee_quat),
                Waypoint("edge", np.array([x, y_edge, z_inside]), ee_quat),
                Waypoint("in",   np.array([x, y_in,   z_inside]), ee_quat),
                Waypoint("down", np.array([x, y_in,   z_down]), ee_quat),
            ]

        return self.plan_cartesian_avoid_shortest("PICK", builder)

    def plan_retract(self) -> List[List[float]]:
        phone_pos = np.array(p.getBasePositionAndOrientation(self.phone_id)[0], dtype=float)
        ee_quat = self.build_ee_quat_from_world_fwd_down(
            fwd_w=np.array([0.0, 1.0, 0.0]),
            down_w=np.array([0.0, 0.0, -1.0])
        )
        cur_pos = self.ee_world_pos()

        x_min = self.cab_center[0] - self.cab_inner_w / 2.0 + 0.08
        x_max = self.cab_center[0] + self.cab_inner_w / 2.0 - 0.08
        x = float(clamp(phone_pos[0], x_min, x_max))

        def builder(attempt: int):
            y_in = float(phone_pos[1])
            y_edge = float(self.cab_front_y - (0.22 + 0.01 * attempt))
            y_far = float(self.cab_front_y - (0.36 + 0.01 * attempt))
            z_inside, z_out = self._height_policy_low(phone_pos, attempt)

            return [
                Waypoint("start", cur_pos, ee_quat),
                Waypoint("lift", np.array([x, y_in,   z_inside]), ee_quat),
                Waypoint("edge", np.array([x, y_edge, z_inside]), ee_quat),
                Waypoint("far",  np.array([x, y_far,  z_out]),    ee_quat),
            ]

        return self.plan_cartesian_avoid_shortest("RETRACT", builder)

    def plan_place(self) -> List[List[float]]:
        ws = self.ws_target.copy()
        phone_pos = np.array(p.getBasePositionAndOrientation(self.phone_id)[0], dtype=float)

        ee_quat = self.build_ee_quat_from_world_fwd_down(
            fwd_w=np.array([0.0, 1.0, 0.0]),
            down_w=np.array([0.0, 0.0, -1.0])
        )
        cur_pos = self.ee_world_pos()

        def builder(attempt: int):
            _, z_out = self._height_policy_low(phone_pos, attempt)
            z_pre = float(z_out + 0.05 + 0.005 * attempt)
            z_touch = float(self.ws_top_z + self.scene.phone_size[2] / 2.0 + 0.002)

            return [
                Waypoint("start", cur_pos, ee_quat),
                Waypoint("ws_pre",   np.array([ws[0], ws[1], z_pre]),   ee_quat),
                Waypoint("ws_touch", np.array([ws[0], ws[1], z_touch]), ee_quat),
            ]

        return self.plan_cartesian_avoid_shortest("PLACE", builder)

    # -----------------------
    # Snap to target (optional)
    # -----------------------
    def snap_phone_to_workstation_if_close(self):
        if not self.scene.snap_place:
            return
        opos, _ = p.getBasePositionAndOrientation(self.phone_id)
        opos = np.array(opos, dtype=float)
        dxy = float(np.linalg.norm(opos[:2] - self.ws_target[:2]))
        dz = float(abs(opos[2] - (self.ws_top_z + self.scene.phone_size[2] / 2.0)))
        if dxy <= self.scene.snap_xy_tol and dz <= self.scene.snap_z_tol:
            new_pos = np.array([self.ws_target[0], self.ws_target[1],
                                self.ws_top_z + self.scene.phone_size[2] / 2.0], dtype=float)
            p.resetBasePositionAndOrientation(self.phone_id, new_pos.tolist(), [0, 0, 0, 1])
            self._zero_phone_velocity()

    # -----------------------
    # Run flow
    # -----------------------
    def run(self):
        print("\n[Flow] 1) Hold init (stable)...")
        self.hold(0.9)

        print("\n[Flow] 2) Plan pick (shortest collision-free)...")
        pick_traj = self.plan_pick()

        print("[Flow] 3) Execute pick...")
        self.set_gripper(0.0)
        self.hold(0.2)
        self.exec_q_traj("PICK", pick_traj)

        print("[Flow] 4) Close gripper and attach...")
        attached = False
        steps = int(1.0 / self.plan.dt)
        for i in range(steps):
            frac = (i + 1) / max(1, steps)
            self.set_gripper(frac)
            self.lock_arm_targets(self.get_arm_q())
            p.stepSimulation()
            self._update_meshcat()
            if (not attached) and frac > 0.86:
                attached = self.try_attach_phone()
            if self.plan.realtime:
                time.sleep(self.plan.dt)

        print("[Pick] grasp_success =", attached)
        if not attached:
            raise RuntimeError("Pick failed: phone too far / alignment off. (No attach)")

        print("\n[Flow] 5) Retract (shortest collision-free)...")
        ret_traj = self.plan_retract()
        self.exec_q_traj("RETRACT", ret_traj)
        self.hold(0.2)

        print("\n[Flow] 6) Plan & execute place (shortest collision-free)...")
        place_traj = self.plan_place()
        self.exec_q_traj("PLACE", place_traj)

        print("\n[Flow] 7) Release...")
        self.detach_phone()
        self._zero_phone_velocity()

        steps = int(0.9 / self.plan.dt)
        for i in range(steps):
            frac = 1.0 - (i + 1) / max(1, steps)
            self.set_gripper(frac)
            self.lock_arm_targets(self.get_arm_q())
            p.stepSimulation()
            self._update_meshcat()
            if self.plan.realtime:
                time.sleep(self.plan.dt)

        self.set_gripper(0.0)
        self._zero_phone_velocity()
        self.snap_phone_to_workstation_if_close()

        print("\n[Flow] Done (phone placed). Ctrl+C to exit.")
        while True:
            self.lock_arm_targets(self.get_arm_q())
            p.stepSimulation()
            self._update_meshcat()
            time.sleep(1 / 60)


# =========================
# Entry
# =========================
if __name__ == "__main__":
    UR3PickPlaceDemo().run()
