import pybullet as p
import pybullet_data
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time
import numpy as np
import cv2
import math
import os

# --- 1. Initialization ---
vis = meshcat.Visualizer()
vis.delete() # Clear cached stale models
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)
p.setPhysicsEngineParameter(numSolverIterations=200)

TABLE_POS = [0.4, -0.35, 0.3]
TABLE_SIZE = [0.5, 0.5, 0.04]
TABLE_HALF = [s / 2.0 for s in TABLE_SIZE]
SHELF_SIZE = [0.24, 0.36, 0.02]
SHELF_POS = [TABLE_POS[0] - TABLE_HALF[0] + SHELF_SIZE[0] / 2.0, 0.35, 0]
TABLE_TOP_Z = TABLE_POS[2] + TABLE_SIZE[2] / 2.0
SHELF_LAYER_HEIGHT = TABLE_TOP_Z
SHELF1_Z = SHELF_LAYER_HEIGHT - SHELF_SIZE[2] / 2.0
SHELF2_Z = SHELF_LAYER_HEIGHT * 2.0 - SHELF_SIZE[2] / 2.0
SHELF_TOP_Z = SHELF2_Z + SHELF_SIZE[2] / 2.0
SHELF2_BOTTOM_Z = SHELF2_Z - SHELF_SIZE[2] / 2.0
CAB_LEG_HEIGHT = SHELF_TOP_Z
CAB_LEG_HALF_Z = CAB_LEG_HEIGHT / 2.0
SHELF_FRONT_Y = SHELF_POS[1] - SHELF_SIZE[1] / 2.0
CAB_CENTER_X = SHELF_POS[0]
CAB_CENTER_Y = SHELF_POS[1]
CAB_FRONT_CLEAR_Y = SHELF_FRONT_Y - 0.12
CAB_SIDE_CLEAR_X = 0.12
CAB_MIN_X = CAB_CENTER_X - SHELF_SIZE[0] / 2.0
CAB_MAX_X = CAB_CENTER_X + SHELF_SIZE[0] / 2.0
WORKSTATION_SIZE = [0.16, 0.12, 0.02]
WORKSTATION_HALF = [s / 2.0 for s in WORKSTATION_SIZE]
TABLE_CORNERS = [
    [TABLE_POS[0] - TABLE_HALF[0], TABLE_POS[1] - TABLE_HALF[1]],
    [TABLE_POS[0] - TABLE_HALF[0], TABLE_POS[1] + TABLE_HALF[1]],
    [TABLE_POS[0] + TABLE_HALF[0], TABLE_POS[1] - TABLE_HALF[1]],
    [TABLE_POS[0] + TABLE_HALF[0], TABLE_POS[1] + TABLE_HALF[1]],
]
NEAREST_TABLE_CORNER = min(TABLE_CORNERS, key=lambda c: c[0] ** 2 + c[1] ** 2)
WORKSTATION_POS = [
    NEAREST_TABLE_CORNER[0] + (WORKSTATION_HALF[0] if NEAREST_TABLE_CORNER[0] < TABLE_POS[0] else -WORKSTATION_HALF[0]),
    NEAREST_TABLE_CORNER[1] + (WORKSTATION_HALF[1] if NEAREST_TABLE_CORNER[1] < TABLE_POS[1] else -WORKSTATION_HALF[1]),
    TABLE_TOP_Z + WORKSTATION_HALF[2],
]
WORKSTATION_TOP_Z = WORKSTATION_POS[2] + WORKSTATION_HALF[2]
PHONE_SIZE = [0.06, 0.12, 0.01]
PHONE_HALF_Z = PHONE_SIZE[2] / 2.0
# Place the phone near the cabinet side edge for easier side grasping.
# Keep Y fixed and move X to the cabinet edge.
PHONE_POS = [CAB_MIN_X + 0.06, SHELF_POS[1], SHELF1_Z + SHELF_SIZE[2] / 2.0 + PHONE_HALF_Z]

# Arm joints you can tune (order matches IK and target arrays).
# You can adjust these in PREF_APPROACH_BY_NAME / PREF_GRASP_BY_NAME / APPROACH_LIMITS_BY_NAME.
ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
# Gripper joints (prismatic): f1/f2 control finger opening.
GRIPPER_JOINT_NAMES = ["f1", "f2"]
# End-effector link (tool tip).
EE_LINK_NAME = "tool_tip_joint"
# Default arm pose (same order as ARM_JOINT_NAMES).
ARM_REST_Q = [0.0, -1.1, 1.6, -0.8, 0.0, 0.0]
SIM_DT = 1.0 / 240.0
# Phone size: 0.06 (X) x 0.12 (Y); grasp the long side (0.12m / 120mm).
GRIPPER_GAP_OPEN = 0.050    # 130mm, open wider than phone long side for clearance.
GRIPPER_GAP_CLOSED = 0.045  # 115mm, slightly smaller than phone long side to clamp.
LIFT_Z_SCALE = 2.0 / 3.0
MOTION_SPEED = 1.5*6
PATH_STEP = 0.01 * MOTION_SPEED
PATH_STEP_FINE = 0.008 * MOTION_SPEED
PATH_STEP_GRASP = 0.004 * MOTION_SPEED
PATH_STEP_PLACE = 0.005 * MOTION_SPEED
HOLD_STEPS = max(2, int(round(6 / MOTION_SPEED)))
HOLD_STEPS_LIFT = max(3, int(round(8 / MOTION_SPEED)))
HOLD_STEPS_GRASP = max(4, int(round(10 / MOTION_SPEED)))
HOLD_STEPS_PLACE = max(3, int(round(8 / MOTION_SPEED)))
INIT_SETTLE_STEPS = max(60, int(round(120 / MOTION_SPEED)))
GRIPPER_OPEN_STEPS = max(12, int(round(20 / MOTION_SPEED)))
GRIPPER_SETTLE_STEPS = max(40, int(round(80 / MOTION_SPEED)))
PLACE_RELEASE_STEPS = max(40, int(round(60 / MOTION_SPEED)))
PLACE_SETTLE_STEPS = max(30, int(round(50 / MOTION_SPEED)))
SAFE_Z_MARGIN = 0.08
SAFE_Z = max(TABLE_TOP_Z, SHELF_TOP_Z) + SAFE_Z_MARGIN
CAB_TOP_CLEAR_MARGIN = 0.04
PATH_SAFE_Z = min(SAFE_Z, SHELF_TOP_Z - CAB_TOP_CLEAR_MARGIN) * LIFT_Z_SCALE
PATH_POS_TOL = 0.003
PATH_SETTLE_STEPS = max(40, int(round(80 / MOTION_SPEED)))
PATH_STALL_STEPS = max(6, int(round(12 / MOTION_SPEED)))
PATH_STALL_EPS = 1e-4
PLACE_RELEASE_TOL = 0.03
ATTACH_MAX_DIST = 0.08
ATTACH_RETRY_STEPS = 80
ATTACH_MAX_FORCE = 800
GRASP_CLEAR_Z = 0.003
GRASP_NUDGE_Z = 0.004
GRASP_MAX_RETRIES = 4
GRASP_YAW_RETRY = [0.0, 0.12, -0.12, 0.24, -0.24]
GRASP_XY_RETRY = [(0.0, 0.0), (0.004, 0.0), (-0.004, 0.0), (0.0, 0.004), (0.0, -0.004)]
# Per-joint preferred angles during approach (radians).
PREF_APPROACH_BY_NAME = {
    "shoulder_lift_joint": -1.2,
    "elbow_joint": math.pi,
    "wrist_1_joint": math.pi/2,
    "wrist_2_joint": math.pi/2,
    "wrist_3_joint": math.pi/2,
}
# Per-joint preferred angles during final grasp (radians).
PREF_GRASP_BY_NAME = {
    "shoulder_lift_joint": -0.9,
    "elbow_joint": 1.1,
    "wrist_1_joint": -1.2,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
}
# Per-joint hard limits during approach (radians).
APPROACH_LIMITS_BY_NAME = {
    "shoulder_lift_joint": (-2.4, -0.6),
    "elbow_joint": (math.pi - 0.4, math.pi),
}
# Per-joint hard locks during approach (radians).
APPROACH_LOCK_BY_NAME = {
    "wrist_1_joint": 0,
    "wrist_2_joint": 0,
    "wrist_3_joint": 0,
}
ARM_FORCE = 400
ARM_POSITION_GAIN = 0.08
ARM_VELOCITY_GAIN = 1.0
CAB_ENTRY_MARGIN = 0.03
CAB_ENTRY_Z = max(0.18, SHELF2_BOTTOM_Z - CAB_ENTRY_MARGIN)
ARM_MAX_REACH = 0.65
REACH_Z_MARGIN = 0.03
ROBOT_BASE_POS = [0.0, 0.0, 0.0]
EE_DOWN_EULER = [0.0, math.pi, 0.0]
GRASP_YAW_OFFSET = math.pi / 2.0  # Align finger closing axis to phone width.
GRASP_FINAL_YAW_OFFSET = -math.pi / 2.0  # Extra 90deg rotation right before grasp.
GRASP_YAW_FLIP = math.pi  # Rotate gripper 180deg about world Z (flip approach).
APPROACH_Z_OFFSET = 0.08 * LIFT_Z_SCALE
SIDE_ENTRY_Z_OFFSET = 0.05 * LIFT_Z_SCALE
PLACE_HOVER_Z_OFFSET = 0.12 * LIFT_Z_SCALE
PLACE_CLEAR_MARGIN = 0.08
PLACE_RETREAT_Y_OFFSET = 0.25
PLACE_CONTACT_Z_OFFSET = 0.003

# --- 2. Physical Scene Setup ---
p.loadURDF("plane.urdf")

def create_scene():
    ids = {}
    # Cabinet shelves
    shelf_half = [s / 2.0 for s in SHELF_SIZE]
    ids['shelf1'] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=shelf_half), 
                                     p.createVisualShape(p.GEOM_BOX, halfExtents=shelf_half, rgbaColor=[0.4, 0.2, 0.1, 1]), [SHELF_POS[0], SHELF_POS[1], SHELF1_Z])
    ids['shelf2'] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=shelf_half), 
                                     p.createVisualShape(p.GEOM_BOX, halfExtents=shelf_half, rgbaColor=[0.4, 0.2, 0.1, 1]), [SHELF_POS[0], SHELF_POS[1], SHELF2_Z])
    # Cabinet legs (simplified as 4 long legs)
    for i, off in enumerate([[-0.1, -0.15], [0.1, -0.15], [-0.1, 0.15], [0.1, 0.15]]):
        ids[f's_leg{i}'] = p.createMultiBody(
            0,
            p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, CAB_LEG_HALF_Z]),
            p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, CAB_LEG_HALF_Z], rgbaColor=[0.2, 0.2, 0.2, 1]),
            [SHELF_POS[0] + off[0], SHELF_POS[1] + off[1], CAB_LEG_HALF_Z],
        )
    # Table
    table_half = [s / 2.0 for s in TABLE_SIZE]
    ids['table'] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half), 
                                    p.createVisualShape(p.GEOM_BOX, halfExtents=table_half, rgbaColor=[0.8, 0.8, 0.8, 1]), [TABLE_POS[0], TABLE_POS[1], TABLE_POS[2]])
    for i, off in enumerate([[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]]):
        ids[f't_leg{i}'] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.15]), 
                                            p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.15], rgbaColor=[0.3, 0.3, 0.3, 1]), [TABLE_POS[0]+off[0], TABLE_POS[1]+off[1], TABLE_POS[2] / 2.0])
    # Phone
    ids['phone'] = p.createMultiBody(0.2, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.06, 0.005]), 
                                    p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.06, 0.005], rgbaColor=[0.1, 0.1, 0.1, 1]), PHONE_POS)
    # Target workstation (place the phone here)
    ws_half = [s / 2.0 for s in WORKSTATION_SIZE]
    ids['workstation'] = p.createMultiBody(
        0,
        p.createCollisionShape(p.GEOM_BOX, halfExtents=ws_half),
        p.createVisualShape(p.GEOM_BOX, halfExtents=ws_half, rgbaColor=[0.25, 0.25, 0.25, 1]),
        WORKSTATION_POS,
    )
    return ids

scene_ids = create_scene()
URDF_PATH = os.path.join(os.path.dirname(__file__), "arm6dof.urdf")
robot_id = p.loadURDF(
    URDF_PATH,
    ROBOT_BASE_POS,
    useFixedBase=True,
    flags=(
        p.URDF_USE_SELF_COLLISION
        | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    ),
)

JOINT_NAME_TO_INDEX = {p.getJointInfo(robot_id, i)[1].decode("utf-8"): i for i in range(p.getNumJoints(robot_id))}
ARM_JOINTS = [JOINT_NAME_TO_INDEX[n] for n in ARM_JOINT_NAMES]
GRIPPER_JOINTS = [JOINT_NAME_TO_INDEX[n] for n in GRIPPER_JOINT_NAMES]
EE_LINK = JOINT_NAME_TO_INDEX[EE_LINK_NAME]
ARM_INDEX_BY_JOINT = {ji: idx for idx, ji in enumerate(ARM_JOINTS)}

def overrides_from_names(name_map):
    return {JOINT_NAME_TO_INDEX[name]: val for name, val in name_map.items() if name in JOINT_NAME_TO_INDEX}

def limit_overrides_from_names(limit_map):
    return {
        JOINT_NAME_TO_INDEX[name]: (float(lo), float(hi))
        for name, (lo, hi) in limit_map.items()
        if name in JOINT_NAME_TO_INDEX
    }

SHOULDER_LIFT_INDEX = JOINT_NAME_TO_INDEX.get("shoulder_lift_joint")
ELBOW_JOINT_INDEX = JOINT_NAME_TO_INDEX.get("elbow_joint")
WRIST_JOINT_INDICES = [
    JOINT_NAME_TO_INDEX[name]
    for name in ("wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    if name in JOINT_NAME_TO_INDEX
]
PREF_APPROACH_OVERRIDES = overrides_from_names(PREF_APPROACH_BY_NAME)
PREF_GRASP_OVERRIDES = overrides_from_names(PREF_GRASP_BY_NAME)
APPROACH_LIMIT_OVERRIDES = limit_overrides_from_names(APPROACH_LIMITS_BY_NAME)
APPROACH_LOCK_OVERRIDES = overrides_from_names(APPROACH_LOCK_BY_NAME)
PREF_CONFIGS = {
    "approach": {
        "rest_overrides": PREF_APPROACH_OVERRIDES,
        "elbow_index": ELBOW_JOINT_INDEX,
        "elbow_target": PREF_APPROACH_OVERRIDES.get(ELBOW_JOINT_INDEX),
        "shoulder_index": SHOULDER_LIFT_INDEX,
        "shoulder_target": PREF_APPROACH_OVERRIDES.get(SHOULDER_LIFT_INDEX),
        "wrist_indices": WRIST_JOINT_INDICES,
        "pos_weight": 120.0,
        "ang_weight": 10.0,
        "elbow_weight": 2.4,
        "shoulder_weight": 0.8,
        "wrist_weight": 0.8,
        "smooth_weight": 0.2,
        "elbow_offsets": [-0.5, 0.0, 0.5],
    },
    "grasp": {
        "rest_overrides": PREF_GRASP_OVERRIDES,
        "elbow_index": ELBOW_JOINT_INDEX,
        "elbow_target": PREF_GRASP_OVERRIDES.get(ELBOW_JOINT_INDEX),
        "shoulder_index": SHOULDER_LIFT_INDEX,
        "shoulder_target": PREF_GRASP_OVERRIDES.get(SHOULDER_LIFT_INDEX),
        "wrist_indices": WRIST_JOINT_INDICES,
        "pos_weight": 140.0,
        "ang_weight": 12.0,
        "elbow_weight": 1.2,
        "shoulder_weight": 0.6,
        "wrist_weight": 0.4,
        "smooth_weight": 0.2,
        "elbow_offsets": [-0.4, 0.0, 0.4],
    },
}
GRIPPER_LIMITS = {j: (p.getJointInfo(robot_id, j)[8], p.getJointInfo(robot_id, j)[9]) for j in GRIPPER_JOINTS}
ARM_REST_BY_INDEX = {ji: ARM_REST_Q[i] for i, ji in enumerate(ARM_JOINTS)}
NUM_JOINTS = p.getNumJoints(robot_id)
IK_LOWER = []
IK_UPPER = []
IK_RANGE = []
IK_REST = []
for ji in range(NUM_JOINTS):
    info = p.getJointInfo(robot_id, ji)
    joint_type = info[2]
    if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
        lower, upper = info[8], info[9]
        if lower > upper:
            lower, upper = -math.pi, math.pi
        rng = upper - lower
        if rng <= 0.0:
            rng = 2.0 * math.pi
    else:
        lower, upper, rng = 0.0, 0.0, 0.0
    IK_LOWER.append(lower)
    IK_UPPER.append(upper)
    IK_RANGE.append(rng)
    if ji in ARM_REST_BY_INDEX:
        IK_REST.append(ARM_REST_BY_INDEX[ji])
        p.resetJointState(robot_id, ji, ARM_REST_BY_INDEX[ji])
    else:
        IK_REST.append(0.0)

gripper_open_ratio = 1.0
phone_constraint = None

# Arm parameters
JOINT_RADII = {0: 0.052, 1: 0.048, 2: 0.044, 3: 0.040, 4: 0.035, 5: 0.030}
BONE_STRUCTURE = [
    (-1, 0, 0.055, 0x777777),
    (0, 1, 0.048, 0x1a5fb4),
    (1, 2, 0.044, 0x777777),
    (2, 3, 0.040, 0x1a5fb4),
    (3, 4, 0.036, 0x777777),
    (4, 5, 0.032, 0x1a5fb4),
    (5, 6, 0.030, 0x777777),
]

# --- 3. Visual Initialization (set_object only here) ---
def setup_visuals():
    # Scene
    vis["shelf1"].set_object(g.Box(SHELF_SIZE), g.MeshLambertMaterial(color=0x8b4513))
    vis["shelf2"].set_object(g.Box(SHELF_SIZE), g.MeshLambertMaterial(color=0x8b4513))
    for i in range(4): vis[f"s_leg{i}"].set_object(g.Box([0.02, 0.02, CAB_LEG_HEIGHT]), g.MeshLambertMaterial(color=0x333333))
    vis["table"].set_object(g.Box(TABLE_SIZE), g.MeshLambertMaterial(color=0xcccccc))
    for i in range(4): vis[f"t_leg{i}"].set_object(g.Box([0.03, 0.03, 0.3]), g.MeshLambertMaterial(color=0x555555))
    vis["phone"].set_object(g.Box([0.06, 0.12, 0.01]), g.MeshLambertMaterial(color=0x111111))
    vis["workstation"].set_object(g.Box(WORKSTATION_SIZE), g.MeshLambertMaterial(color=0x444444))
    vis["workstation/target"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x00ff00))
    # Robot
    for i, r in JOINT_RADII.items():
        vis[f"robot/joints/j{i}"].set_object(g.Sphere(r), g.MeshLambertMaterial(color=0x333333))
    vis["robot/base"].set_object(g.Cylinder(0.06, 0.05), g.MeshLambertMaterial(color=0x333333))
    vis["robot/gripper/base"].set_object(g.Box([0.08, 0.04, 0.04]), g.MeshLambertMaterial(color=0x222222))
    vis["robot/gripper/l"].set_object(g.Box([0.01, 0.02, 0.08]), g.MeshLambertMaterial(color=0x555555))
    vis["robot/gripper/r"].set_object(g.Box([0.01, 0.02, 0.08]), g.MeshLambertMaterial(color=0x555555))

setup_visuals()
vis["workstation/target"].set_transform(
    tf.translation_matrix([0.0, 0.0, 0.0])
)

# --- 4. RGB-D 3D Vision Capture ---
def get_rgbd():
    state = p.getLinkState(robot_id, EE_LINK)
    pos, orn = state[0], state[1]
    rot = np.reshape(p.getMatrixFromQuaternion(orn), (3, 3))
    near, far = 0.01, 2.0
    eye = pos + rot.dot([0, 0, 0.02])
    target = pos + rot.dot([0, 0, 0.5])
    view_m = p.computeViewMatrix(eye, target, rot.dot([0, -1, 0]))
    proj_m = p.computeProjectionMatrixFOV(75, 1.0, near, far)
    w, h, rgb, depth, _ = p.getCameraImage(320, 320, view_m, proj_m, renderer=p.ER_TINY_RENDERER)
    
    rgb_img = cv2.cvtColor(np.reshape(rgb, (h, w, 4))[:,:,:3], cv2.COLOR_RGB2BGR)
    depth_m = far * near / (far - (far - near) * np.reshape(depth, (h, w)))
    depth_viz = cv2.applyColorMap(np.clip((depth_m - near) / (far - near) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    return rgb_img, depth_viz

# --- 5. Sync Function (only set_transform here) ---
def sync(open_ratio=1.0):
    gw = GRIPPER_GAP_CLOSED + open_ratio * (GRIPPER_GAP_OPEN - GRIPPER_GAP_CLOSED)
    # Sync environment
    for name, p_id in scene_ids.items():
        pos, orn = p.getBasePositionAndOrientation(p_id)
        vis[name].set_transform(tf.translation_matrix(pos) @ tf.quaternion_matrix([orn[3], orn[0], orn[1], orn[2]]))
    
    # Fetch robot link states
    states = {
        i: (p.getBasePositionAndOrientation(robot_id) if i == -1 else p.getLinkState(robot_id, i))
        for i in range(-1, 7)
    }
    
    # Sync joints and links
    vis["robot/base"].set_transform(tf.translation_matrix(states[-1][0]) @ tf.quaternion_matrix([states[-1][1][3], states[-1][1][0], states[-1][1][1], states[-1][1][2]]) @ tf.rotation_matrix(np.pi/2, [1,0,0]))
    for i in range(6):
        vis[f"robot/joints/j{i}"].set_transform(tf.translation_matrix(states[i][0]) @ tf.quaternion_matrix([states[i][1][3], states[i][1][0], states[i][1][1], states[i][1][2]]))
    
    # Dynamic bone links
    for i, (s, e, r, c) in enumerate(BONE_STRUCTURE):
        p1, p2 = np.array(states[s][0]), np.array(states[e][0])
        dist = np.linalg.norm(p2-p1)
        if dist < 0.001: continue
        vis[f"robot/bones/b{i}"].set_object(g.Cylinder(dist, r), g.MeshLambertMaterial(color=c))
        vt = (p2-p1)/dist
        rot = np.eye(4); va = np.cross([0,1,0], vt); vs = np.linalg.norm(va); vc = np.dot([0,1,0], vt)
        if vs > 0.0001: rot = tf.rotation_matrix(np.arctan2(vs, vc), va/vs)
        vis[f"robot/bones/b{i}"].set_transform(tf.translation_matrix((p1+p2)/2.0) @ rot)

    # Gripper
    m4 = tf.translation_matrix(states[6][0]) @ tf.quaternion_matrix([states[6][1][3], states[6][1][0], states[6][1][1], states[6][1][2]])
    vis["robot/gripper/base"].set_transform(m4 @ tf.translation_matrix([0, 0, 0.02]))
    vis["robot/gripper/l"].set_transform(m4 @ tf.translation_matrix([gw, 0, 0.04]))
    vis["robot/gripper/r"].set_transform(m4 @ tf.translation_matrix([-gw, 0, 0.04]))

    # OpenCV RGB-D display
    rgb, depth = get_rgbd()
    cv2.imshow("RGB-D Vision", np.hstack((rgb, depth)))
    cv2.waitKey(1)

# --- 6. Motion Planning + Control ---
def step_sim(open_ratio=1.0, steps=1):
    for _ in range(steps):
        p.stepSimulation()
        sync(open_ratio)
        time.sleep(SIM_DT)

def set_arm_targets(q_arm):
    for ji, qv in zip(ARM_JOINTS, q_arm):
        p.setJointMotorControl2(
            robot_id,
            ji,
            p.POSITION_CONTROL,
            targetPosition=float(qv),
            positionGain=ARM_POSITION_GAIN,
            velocityGain=ARM_VELOCITY_GAIN,
            force=ARM_FORCE,
        )

def set_gripper(open_ratio):
    global gripper_open_ratio
    open_ratio = float(np.clip(open_ratio, 0.0, 1.0))
    gripper_open_ratio = open_ratio
    drive_ratio = 1.0 - open_ratio
    for ji in GRIPPER_JOINTS:
        lo, hi = GRIPPER_LIMITS[ji]
        if lo < 0.0 and hi >= 0.0:
            tgt = hi - drive_ratio * (hi - lo)
        else:
            tgt = lo + drive_ratio * (hi - lo)
        p.setJointMotorControl2(robot_id, ji, p.POSITION_CONTROL, targetPosition=float(tgt), force=60)

def init_robot_pose():
    base_pos, _ = p.getBasePositionAndOrientation(robot_id)
    print(f"[DEBUG init] robot base_pos: {base_pos}")
    print(f"[DEBUG init] CAB_CENTER_X: {CAB_CENTER_X}, CAB_CENTER_Y: {CAB_CENTER_Y}")

    # Initial pose: face the phone and keep gripper pointing down.
    phone_pos, _ = get_phone_pose()
    yaw = math.atan2(phone_pos[1] - base_pos[1], phone_pos[0] - base_pos[0])
    home_radius = 0.22
    home_z = 0.32
    home_pos = np.array(
        [
            base_pos[0] + home_radius * math.cos(yaw),
            base_pos[1] + home_radius * math.sin(yaw),
            home_z,
        ],
        dtype=float,
    )
    print(f"[DEBUG init] home_pos (toward phone): {home_pos.tolist()}")
    home_quat = down_quat_for_xy(home_pos[0], home_pos[1], base_pos=base_pos)
    q_home = solve_ik_preferred(
        home_pos.tolist(),
        home_quat,
        PREF_CONFIGS["approach"],
        rest_overrides=None,
        lock_joints=None,
        use_current_rest=False,
        joint_limit_overrides=APPROACH_LIMIT_OVERRIDES,
    )
    set_arm_targets(q_home)
    set_gripper(1.0)
    step_sim(gripper_open_ratio, steps=INIT_SETTLE_STEPS)

    reached = wait_ee_at(home_pos, tol=PATH_POS_TOL, max_steps=PATH_SETTLE_STEPS)
    if not reached:
        print(f"[WARN] Initial pose not fully reached, current pose: {get_ee_pose()[0].tolist()}")
    return home_pos, list(q_home), home_quat

def get_ee_pose():
    st = p.getLinkState(robot_id, EE_LINK, computeForwardKinematics=True)
    return np.array(st[4], dtype=float), np.array(st[5], dtype=float)

def get_arm_joint_positions():
    return [p.getJointState(robot_id, ji)[0] for ji in ARM_JOINTS]

def build_ik_rest(arm_q=None, overrides=None):
    rest = list(IK_REST)
    if arm_q is None:
        if overrides:
            for ji, qv in overrides.items():
                rest[ji] = float(qv)
        return rest
    for ji, qv in zip(ARM_JOINTS, arm_q):
        rest[ji] = float(qv)
    if overrides:
        for ji, qv in overrides.items():
            rest[ji] = float(qv)
    return rest

def wait_ee_at(target_pos, tol=PATH_POS_TOL, max_steps=PATH_SETTLE_STEPS):
    target_pos = np.asarray(target_pos, dtype=float)
    prev_err = None
    stagnant = 0
    for _ in range(max_steps):
        cur_pos, _ = get_ee_pose()
        err = float(np.linalg.norm(cur_pos - target_pos))
        if err <= tol:
            return True
        if prev_err is not None:
            if err >= prev_err - PATH_STALL_EPS:
                stagnant += 1
                if stagnant >= PATH_STALL_STEPS:
                    return False
            else:
                stagnant = 0
        prev_err = err
        step_sim(gripper_open_ratio, steps=1)
    return False

def get_base_pose():
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    return np.array(pos, dtype=float), np.array(orn, dtype=float)

def down_quat_for_xy(x, y, base_pos=None):
    if base_pos is None:
        base_pos, _ = get_base_pose()
    yaw = math.atan2(y - base_pos[1], x - base_pos[0])
    return p.getQuaternionFromEuler([EE_DOWN_EULER[0], EE_DOWN_EULER[1], yaw])

def safe_z_for_xy(x, y, default_z=PATH_SAFE_Z, margin=REACH_Z_MARGIN):
    base_pos, _ = get_base_pose()
    dx = float(x - base_pos[0])
    dy = float(y - base_pos[1])
    reach_sq = ARM_MAX_REACH * ARM_MAX_REACH - dx * dx - dy * dy
    if reach_sq <= 0.0:
        return default_z
    max_z = math.sqrt(reach_sq) - margin
    if max_z <= 0.0:
        return default_z
    return min(default_z, max_z)

def get_phone_pose():
    pos, orn = p.getBasePositionAndOrientation(scene_ids["phone"])
    return np.array(pos, dtype=float), np.array(orn, dtype=float)

def grasp_pan_for_phone(phone_quat=None, yaw_offset=GRASP_YAW_OFFSET):
    if phone_quat is None:
        _, phone_quat = get_phone_pose()
    _, _, phone_yaw = p.getEulerFromQuaternion(phone_quat)
    pan = phone_yaw + yaw_offset
    return math.atan2(math.sin(pan), math.cos(pan))

def is_phone_contacted():
    phone_id = scene_ids["phone"]
    return len(p.getContactPoints(bodyA=robot_id, bodyB=phone_id)) > 0

def solve_ik(target_pos, target_quat=None, rest_poses=None, lock_joints=None, joint_limit_overrides=None):
    if rest_poses is None:
        rest_poses = IK_REST
    lower_limits = list(IK_LOWER)
    upper_limits = list(IK_UPPER)
    joint_ranges = list(IK_RANGE)
    if joint_limit_overrides:
        for ji, (lo, hi) in joint_limit_overrides.items():
            lower_limits[ji] = float(lo)
            upper_limits[ji] = float(hi)
            joint_ranges[ji] = max(float(hi) - float(lo), 1e-6)
    if lock_joints:
        for ji, qv in lock_joints.items():
            lower_limits[ji] = float(qv)
            upper_limits[ji] = float(qv)
            joint_ranges[ji] = 0.0
    if target_quat is None:
        q = p.calculateInverseKinematics(
            robot_id, EE_LINK, target_pos,
            lowerLimits=lower_limits, upperLimits=upper_limits, jointRanges=joint_ranges, restPoses=rest_poses,
            maxNumIterations=240, residualThreshold=1e-4,
        )
    else:
        q = p.calculateInverseKinematics(
            robot_id, EE_LINK, target_pos,
            targetOrientation=target_quat,
            lowerLimits=lower_limits, upperLimits=upper_limits, jointRanges=joint_ranges, restPoses=rest_poses,
            maxNumIterations=240, residualThreshold=1e-4,
        )
    return [q[j] for j in ARM_JOINTS]

def compute_pose_error(q_arm, target_pos, target_quat=None):
    saved_q = [p.getJointState(robot_id, ji)[0] for ji in ARM_JOINTS]
    for ji, qv in zip(ARM_JOINTS, q_arm):
        p.resetJointState(robot_id, ji, float(qv))
    st = p.getLinkState(robot_id, EE_LINK, computeForwardKinematics=True)
    cur_pos = np.array(st[4], dtype=float)
    cur_quat = np.array(st[5], dtype=float)
    for ji, qv in zip(ARM_JOINTS, saved_q):
        p.resetJointState(robot_id, ji, float(qv))
    pos_err = float(np.linalg.norm(cur_pos - target_pos))
    ang_err = 0.0
    if target_quat is not None:
        dot = float(np.dot(cur_quat, target_quat))
        dot = max(-1.0, min(1.0, abs(dot)))
        ang_err = 2.0 * math.acos(dot)
    return pos_err, ang_err

def score_ik_solution(q_arm, target_pos, target_quat, cur_arm_q, prefer_config):
    pos_err, ang_err = compute_pose_error(q_arm, target_pos, target_quat)
    score = pos_err * prefer_config.get("pos_weight", 100.0) + ang_err * prefer_config.get("ang_weight", 10.0)
    elbow_index = prefer_config.get("elbow_index")
    elbow_target = prefer_config.get("elbow_target")
    if elbow_index is not None and elbow_target is not None:
        arm_idx = ARM_INDEX_BY_JOINT.get(elbow_index)
        if arm_idx is not None:
            score += prefer_config.get("elbow_weight", 1.0) * abs(q_arm[arm_idx] - elbow_target)
    shoulder_index = prefer_config.get("shoulder_index")
    shoulder_target = prefer_config.get("shoulder_target")
    if shoulder_index is not None and shoulder_target is not None:
        arm_idx = ARM_INDEX_BY_JOINT.get(shoulder_index)
        if arm_idx is not None:
            score += prefer_config.get("shoulder_weight", 0.5) * abs(q_arm[arm_idx] - shoulder_target)
    wrist_weight = prefer_config.get("wrist_weight", 0.0)
    wrist_indices = prefer_config.get("wrist_indices") or []
    if wrist_weight and wrist_indices:
        wrist_sum = 0.0
        for wi in wrist_indices:
            arm_idx = ARM_INDEX_BY_JOINT.get(wi)
            if arm_idx is not None:
                wrist_sum += abs(q_arm[arm_idx])
        score += wrist_weight * wrist_sum
    smooth_weight = prefer_config.get("smooth_weight", 0.0)
    if smooth_weight and cur_arm_q is not None:
        smooth = sum(abs(q - c) for q, c in zip(q_arm, cur_arm_q))
        score += smooth_weight * smooth
    return score

def build_rest_candidates(cur_arm_q, prefer_config, rest_overrides=None):
    rest_candidates = []
    base_overrides = {}
    if rest_overrides:
        base_overrides.update(rest_overrides)
    pref_overrides = prefer_config.get("rest_overrides") if prefer_config else None
    if pref_overrides:
        base_overrides.update(pref_overrides)
    rest_candidates.append(build_ik_rest(cur_arm_q, overrides=base_overrides) if cur_arm_q is not None else build_ik_rest(None, overrides=base_overrides))
    rest_candidates.append(build_ik_rest(None, overrides=base_overrides))
    elbow_index = prefer_config.get("elbow_index") if prefer_config else None
    elbow_target = prefer_config.get("elbow_target") if prefer_config else None
    for off in prefer_config.get("elbow_offsets", []) if prefer_config else []:
        if elbow_index is None or elbow_target is None:
            break
        overrides = dict(base_overrides)
        overrides[elbow_index] = elbow_target + off
        rest_candidates.append(build_ik_rest(None, overrides=overrides))
    return rest_candidates

def solve_ik_preferred(
    target_pos,
    target_quat,
    prefer_config,
    rest_overrides=None,
    lock_joints=None,
    use_current_rest=False,
    joint_limit_overrides=None,
):
    if prefer_config is None:
        return solve_ik(
            target_pos,
            target_quat,
            rest_poses=None,
            lock_joints=lock_joints,
            joint_limit_overrides=joint_limit_overrides,
        )
    cur_arm_q = get_arm_joint_positions() if use_current_rest else None
    rest_candidates = build_rest_candidates(cur_arm_q, prefer_config, rest_overrides=rest_overrides)
    best_score = None
    best_q = None
    for rest in rest_candidates:
        q_arm = solve_ik(
            target_pos,
            target_quat,
            rest_poses=rest,
            lock_joints=lock_joints,
            joint_limit_overrides=joint_limit_overrides,
        )
        score = score_ik_solution(q_arm, target_pos, target_quat, cur_arm_q, prefer_config)
        if best_score is None or score < best_score:
            best_score = score
            best_q = q_arm
    return best_q if best_q is not None else solve_ik(
        target_pos,
        target_quat,
        rest_poses=None,
        lock_joints=lock_joints,
        joint_limit_overrides=joint_limit_overrides,
    )

def move_cartesian(
    target_pos,
    target_quat=None,
    step=PATH_STEP,
    hold_steps=HOLD_STEPS,
    align_yaw=False,
    use_current_rest=False,
    pan_target=None,
    rest_overrides=None,
    lock_overrides=None,
    prefer_config=None,
    limit_overrides=None,
):
    cur_pos, _ = get_ee_pose()
    target_pos = np.asarray(target_pos, dtype=float)
    dist = float(np.linalg.norm(target_pos - cur_pos))
    n = max(1, int(math.ceil(dist / max(step, 1e-6))))
    base_pos = None
    if align_yaw:
        base_pos, _ = get_base_pose()
    for i in range(1, n + 1):
        t = i / n
        pos = (1.0 - t) * cur_pos + t * target_pos
        quat = target_quat
        if align_yaw:
            quat = down_quat_for_xy(pos[0], pos[1], base_pos=base_pos)
        rest_poses = None
        lock_joints = None
        if use_current_rest or pan_target is not None or rest_overrides:
            arm_q = get_arm_joint_positions() if use_current_rest else None
            overrides = {}
            if rest_overrides:
                overrides.update(rest_overrides)
            if pan_target is not None:
                overrides[ARM_JOINTS[0]] = pan_target
                lock_joints = {ARM_JOINTS[0]: pan_target}
            rest_poses = build_ik_rest(arm_q, overrides=overrides)
        if lock_overrides:
            if lock_joints is None:
                lock_joints = dict(lock_overrides)
            else:
                lock_joints.update(lock_overrides)
        if prefer_config is not None:
            q_arm = solve_ik_preferred(
                pos.tolist(),
                quat,
                prefer_config,
                rest_overrides=rest_overrides,
                lock_joints=lock_joints,
                use_current_rest=use_current_rest,
                joint_limit_overrides=limit_overrides,
            )
        else:
            q_arm = solve_ik(
                pos.tolist(),
                quat,
                rest_poses=rest_poses,
                lock_joints=lock_joints,
                joint_limit_overrides=limit_overrides,
            )
        set_arm_targets(q_arm)
        step_sim(gripper_open_ratio, steps=hold_steps)
        reached = wait_ee_at(pos, tol=PATH_POS_TOL, max_steps=PATH_SETTLE_STEPS)
        if not reached and i == n:
            print("[WARN] End-effector did not fully reach the waypoint; continuing.")

def orient_pan_in_place(
    target_pos,
    pan_target,
    steps=HOLD_STEPS,
    prefer_config=None,
    limit_overrides=None,
    lock_overrides=None,
):
    target_pos = np.asarray(target_pos, dtype=float)
    cur_q = get_arm_joint_positions()
    overrides = {ARM_JOINTS[0]: pan_target}
    if prefer_config and prefer_config.get("rest_overrides"):
        overrides.update(prefer_config["rest_overrides"])
    rest_poses = build_ik_rest(cur_q, overrides=overrides)
    lock_joints = {ARM_JOINTS[0]: pan_target}
    if lock_overrides:
        lock_joints.update(lock_overrides)
    q_target = solve_ik_preferred(
        target_pos.tolist(),
        None,
        prefer_config,
        rest_overrides=prefer_config.get("rest_overrides") if prefer_config else None,
        lock_joints=lock_joints,
        use_current_rest=False,
        joint_limit_overrides=limit_overrides,
    )
    n = max(3, int(steps))
    for i in range(1, n + 1):
        t = i / n
        q_arm = [(1.0 - t) * cq + t * tq for cq, tq in zip(cur_q, q_target)]
        set_arm_targets(q_arm)
        step_sim(gripper_open_ratio, steps=1)
    wait_ee_at(target_pos, tol=PATH_POS_TOL, max_steps=PATH_SETTLE_STEPS)

def interpolate_path(p0, p1, step=0.01):
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    dist = float(np.linalg.norm(p1 - p0))
    n = max(1, int(math.ceil(dist / max(step, 1e-6))))
    return [p0 + (p1 - p0) * (i / n) for i in range(n + 1)]

def build_path(waypoints, step=0.01):
    if len(waypoints) < 2:
        return waypoints
    out = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        seg = interpolate_path(a, b, step=step)
        if out:
            seg = seg[1:]
        out.extend(seg)
    return out

def draw_path(points, path="plan/trajectory", color=0xff8800):
    if not points:
        return
    pts = np.asarray(points, dtype=float)
    try:
        geom = g.Line(g.PointsGeometry(pts.T), g.LineBasicMaterial(color=color))
        vis[path].set_object(geom)
    except Exception:
        pass

def attach_phone_if_close(max_dist=0.06):
    global phone_constraint
    if phone_constraint is not None:
        return True
    phone_id = scene_ids["phone"]
    gpos, gquat = get_ee_pose()
    ppos, pquat = p.getBasePositionAndOrientation(phone_id)
    if float(np.linalg.norm(np.array(ppos) - gpos)) > max_dist:
        return False
    inv_gpos, inv_gquat = p.invertTransform(gpos.tolist(), gquat.tolist())
    rel_pos, rel_quat = p.multiplyTransforms(inv_gpos, inv_gquat, ppos, pquat)
    phone_constraint = p.createConstraint(
        parentBodyUniqueId=robot_id, parentLinkIndex=EE_LINK,
        childBodyUniqueId=phone_id, childLinkIndex=-1,
        jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos, childFramePosition=[0, 0, 0],
        parentFrameOrientation=rel_quat, childFrameOrientation=[0, 0, 0, 1],
    )
    p.changeConstraint(phone_constraint, maxForce=ATTACH_MAX_FORCE)
    return True

def try_attach_phone(max_dist=ATTACH_MAX_DIST, steps=ATTACH_RETRY_STEPS):
    for _ in range(steps):
        if is_phone_contacted() and attach_phone_if_close(max_dist=max_dist):
            return True
        if attach_phone_if_close(max_dist=max_dist):
            return True
        step_sim(gripper_open_ratio, steps=1)
    return False

def attempt_grasp_phone(
    grasp_pos,
    min_z,
    target_quat=None,
    align_yaw=False,
    use_current_rest=False,
    pan_target=None,
    rest_overrides=None,
    prefer_config=None,
    limit_overrides=None,
):
    grasp_pos = np.asarray(grasp_pos, dtype=float)
    base_xy = grasp_pos[:2].copy()
    min_z = float(min_z)
    set_gripper(1.0)
    step_sim(gripper_open_ratio, steps=GRIPPER_OPEN_STEPS)
    for attempt_idx in range(GRASP_MAX_RETRIES + 1):
        yaw_idx = min(attempt_idx, len(GRASP_YAW_RETRY) - 1)
        xy_idx = min(attempt_idx, len(GRASP_XY_RETRY) - 1)
        yaw_offset = GRASP_YAW_RETRY[yaw_idx]
        xy_offset = GRASP_XY_RETRY[xy_idx]
        trial_pos = grasp_pos.copy()
        trial_pos[0] = base_xy[0] + xy_offset[0]
        trial_pos[1] = base_xy[1] + xy_offset[1]
        trial_quat = target_quat
        if target_quat is not None and abs(yaw_offset) > 1e-6:
            r, pch, yaw = p.getEulerFromQuaternion(target_quat)
            trial_quat = p.getQuaternionFromEuler([r, pch, yaw + yaw_offset])
        local_use_current_rest = use_current_rest if attempt_idx == 0 else False
        local_pan_target = pan_target if attempt_idx == 0 else None
        local_rest_overrides = rest_overrides if attempt_idx == 0 else rest_overrides
        move_cartesian(
            trial_pos,
            trial_quat,
            step=PATH_STEP_GRASP,
            hold_steps=HOLD_STEPS_GRASP,
            align_yaw=align_yaw,
            use_current_rest=local_use_current_rest,
            pan_target=local_pan_target,
            rest_overrides=local_rest_overrides,
            prefer_config=prefer_config,
            limit_overrides=limit_overrides,
        )
        set_gripper(0.0)
        step_sim(gripper_open_ratio, steps=GRIPPER_SETTLE_STEPS)
        if try_attach_phone():
            return True
        grasp_pos[2] = max(min_z, grasp_pos[2] - GRASP_NUDGE_Z)
        set_gripper(1.0)
        step_sim(gripper_open_ratio, steps=GRIPPER_OPEN_STEPS)
    return False

def detach_phone():
    global phone_constraint
    if phone_constraint is None:
        return
    p.removeConstraint(phone_constraint)
    phone_constraint = None

# try:
#     # Adjust initial camera viewpoint
#     vis["/Cameras/default/rotated/<object>"].set_transform(tf.translation_matrix([1.2, 0, 0.5]) @ tf.rotation_matrix(np.pi/4, [0, 1, 0]))
#     move([0.4, 0.35, 0.35]) 
#     move([0.4, 0.35, 0.31]) 
#     move([0.4, 0.35, 0.31], 0.005, steps=40) 
#     move([0.4, 0.35, 0.5]) 
#     move([0.4, -0.35, 0.35])
# finally:
#     cv2.destroyAllWindows()
def task_loop():
    try:
        time.sleep(10)
        # 0. Initial view setup
        vis["/Cameras/default/rotated/<object>"].set_transform(
            tf.translation_matrix([1.2, 0, 0.5]) @ tf.rotation_matrix(np.pi/4, [0, 1, 0]))
        
        print(">>> Starting grasp task")

        home_pos, home_q, home_quat = init_robot_pose()

        phone_pos, phone_quat = get_phone_pose()
        grasp_pan = grasp_pan_for_phone(phone_quat) + GRASP_YAW_FLIP
        approach_quat = p.getQuaternionFromEuler([EE_DOWN_EULER[0], EE_DOWN_EULER[1], grasp_pan])
        grasp_pan = grasp_pan + GRASP_FINAL_YAW_OFFSET
        grasp_quat = p.getQuaternionFromEuler([EE_DOWN_EULER[0], EE_DOWN_EULER[1], grasp_pan])
        base_pos, _ = get_base_pose()
        xy_dist = float(np.linalg.norm(phone_pos[:2] - base_pos[:2]))
        print(f">>> Phone position confirmed: {phone_pos.tolist()}")
        if xy_dist > ARM_MAX_REACH:
            print(f"[WARN] Target XY distance {xy_dist:.3f} > arm max reach {ARM_MAX_REACH:.3f}; may be unreachable.")
        entry_clear = CAB_SIDE_CLEAR_X
        if base_pos[0] <= CAB_CENTER_X:
            entry_out_x = CAB_MIN_X - entry_clear
            entry_in_x = CAB_MIN_X + CAB_ENTRY_MARGIN
        else:
            entry_out_x = CAB_MAX_X + entry_clear
            entry_in_x = CAB_MAX_X - CAB_ENTRY_MARGIN
        entry_z_min = SHELF1_Z + SHELF_SIZE[2] / 2.0 + 0.02
        entry_z_max = SHELF2_BOTTOM_Z - CAB_ENTRY_MARGIN
        entry_z = float(np.clip(phone_pos[2] + 0.06, entry_z_min, entry_z_max))
        entry_z = min(
            entry_z,
            safe_z_for_xy(entry_out_x, phone_pos[1], entry_z),
            safe_z_for_xy(entry_in_x, phone_pos[1], entry_z),
        )
        start_pos, _ = get_ee_pose()
        print(f"[DEBUG] start_pos: {start_pos.tolist()}")
        print(f"[DEBUG] base_pos: {base_pos.tolist()}")
        print(f"[DEBUG] phone_pos: {phone_pos.tolist()}")

        # Replan the path from the new folded (-X) initial pose.
        # Path strategy:
        # 1. Move from the -X area to a safe position near the base
        # 2. Move along Y to align with the cabinet
        # 3. Move along X to the cabinet side
        # 4. Enter the cabinet and grasp
        align_yaw = True

        # Compute key Z heights
        grasp_z = phone_pos[2] + PHONE_HALF_Z + GRASP_CLEAR_Z
        approach_z = grasp_z + APPROACH_Z_OFFSET  # Scaled approach height

        # === Stage 1: move from folded start to above the base ===
        # Transition 1: lift to safe height, keep in the -X region
        safe_height = PATH_SAFE_Z
        lift_pos = np.array([start_pos[0], start_pos[1], safe_height], dtype=float)

        # Transition 2: move near above the base (X ~ 0, avoid singularity)
        base_overhead_x = 0.05  # Slightly offset from base center
        base_overhead_y = start_pos[1]
        base_overhead_z = safe_height
        base_overhead = np.array([base_overhead_x, base_overhead_y, base_overhead_z], dtype=float)

        # === Stage 2: move along Y to cabinet Y ===
        # Transition 3: keep X, adjust Y to cabinet
        y_aligned_z = safe_z_for_xy(base_overhead_x, phone_pos[1], PATH_SAFE_Z)
        y_aligned = np.array([base_overhead_x, phone_pos[1], y_aligned_z], dtype=float)

        # === Stage 3: move along X to cabinet side ===
        # Determine entry direction
        if base_pos[0] <= CAB_CENTER_X:
            entry_out_x = CAB_MIN_X - CAB_SIDE_CLEAR_X
            entry_in_x = CAB_MIN_X + CAB_ENTRY_MARGIN
        else:
            entry_out_x = CAB_MAX_X + CAB_SIDE_CLEAR_X
            entry_in_x = CAB_MAX_X - CAB_ENTRY_MARGIN

        # Transition 4: move to outside the cabinet side
        side_safe_z = safe_z_for_xy(entry_out_x, phone_pos[1], PATH_SAFE_Z)
        side_high = np.array([entry_out_x, phone_pos[1], side_safe_z], dtype=float)

        # === Stage 4: enter cabinet and grasp ===
        # Strategy: keep high Z until above target XY, then descend.

        # Side entry: keep at side_safe_z (do not descend early)
        side_entry_z = side_safe_z
        side_entry = np.array([entry_out_x, phone_pos[1], side_entry_z], dtype=float)

        # Inside entry: keep high Z
        inside_entry = np.array([entry_in_x, phone_pos[1], side_entry_z], dtype=float)

        # Above target: keep high Z to reach target XY
        above_target_z = side_entry_z  # Keep same height
        above_target = np.array([phone_pos[0], phone_pos[1], above_target_z], dtype=float)

        # Start descending to approach height
        approach = np.array([phone_pos[0], phone_pos[1], approach_z], dtype=float)

        # Final grasp pose
        grasp = np.array([phone_pos[0], phone_pos[1], grasp_z], dtype=float)

        # Debug printout
        print(f"[DEBUG] Trajectory plan:")
        print(f"  1. lift_pos: {lift_pos.tolist()}")
        print(f"  2. base_overhead: {base_overhead.tolist()}")
        print(f"  3. y_aligned: {y_aligned.tolist()}")
        print(f"  4. side_high: {side_high.tolist()}")
        print(f"  5. side_entry: {side_entry.tolist()}")
        print(f"  6. inside_entry: {inside_entry.tolist()}")
        print(f"  7. above_target (above target): {above_target.tolist()}")
        print(f"  8. approach (start descent): {approach.tolist()}")
        print(f"  9. grasp (grasp pose): {grasp.tolist()}")

        # Build full path
        waypoints = [start_pos, lift_pos, base_overhead, y_aligned, side_high,
                     side_entry, inside_entry, above_target, approach, grasp]

        pick_path = build_path(waypoints, step=PATH_STEP)
        draw_path(pick_path, path="plan/pick", color=0xff8800)

        print("-> Begin moving to grasp pose")
        print("  [Stage 1] Lift from folded pose")
        move_cartesian(
            lift_pos,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            align_yaw=align_yaw,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
            limit_overrides=APPROACH_LIMIT_OVERRIDES,
            lock_overrides=APPROACH_LOCK_OVERRIDES,
        )

        print("  [Stage 2] Move above base")
        move_cartesian(
            base_overhead,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            align_yaw=align_yaw,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
            limit_overrides=APPROACH_LIMIT_OVERRIDES,
            lock_overrides=APPROACH_LOCK_OVERRIDES,
        )

        print("  [Stage 3] Align Y with cabinet")
        move_cartesian(
            y_aligned,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            align_yaw=align_yaw,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
            limit_overrides=APPROACH_LIMIT_OVERRIDES,
            lock_overrides=APPROACH_LOCK_OVERRIDES,
        )

        print("  [Stage 4] Move along X to cabinet side")
        move_cartesian(
            side_high,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            align_yaw=align_yaw,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
            limit_overrides=APPROACH_LIMIT_OVERRIDES,
            lock_overrides=APPROACH_LOCK_OVERRIDES,
        )
        move_cartesian(
            side_entry,
            step=PATH_STEP_FINE,
            hold_steps=HOLD_STEPS,
            align_yaw=align_yaw,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
            limit_overrides=APPROACH_LIMIT_OVERRIDES,
            lock_overrides=APPROACH_LOCK_OVERRIDES,
        )

        print("  [Stage 5] Enter cabinet, move above target XY")
        move_cartesian(
            inside_entry,
            step=PATH_STEP_FINE,
            hold_steps=HOLD_STEPS,
            align_yaw=align_yaw,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
            limit_overrides=APPROACH_LIMIT_OVERRIDES,
            lock_overrides=APPROACH_LOCK_OVERRIDES,
        )
        move_cartesian(
            above_target,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            align_yaw=align_yaw,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
            limit_overrides=APPROACH_LIMIT_OVERRIDES,
            lock_overrides=APPROACH_LOCK_OVERRIDES,
        )

        print("  [Stage 6] Confirm above target, then align gripper yaw")
        if not wait_ee_at(above_target, tol=PATH_POS_TOL, max_steps=PATH_SETTLE_STEPS):
            print("[WARN] Above-target pose not fully reached; aligning anyway.")
        orient_pan_in_place(
            above_target,
            grasp_pan,
            steps=HOLD_STEPS_GRASP,
            prefer_config=PREF_CONFIGS["approach"],
            limit_overrides=APPROACH_LIMIT_OVERRIDES,
            lock_overrides=APPROACH_LOCK_OVERRIDES,
        )
        print(f"    Currently above the phone, height: {above_target_z:.3f}m")

        print("  [Stage 7] Descend to approach height")
        move_cartesian(
            approach,
            target_quat=approach_quat,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            align_yaw=False,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        )

        print("-> Descend for grasp")

        print("-> Close gripper")
        min_grasp_z = phone_pos[2] + PHONE_HALF_Z + 0.001
        if not attempt_grasp_phone(
            grasp,
            min_grasp_z,
            target_quat=grasp_quat,
            align_yaw=False,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        ):
            print("[ERROR] Grasp failed: no phone contact; stopped at grasp pose.")
            return

        # Exit path after grasp (monotonic Z up)
        print("-> Lift phone and exit cabinet")
        # Lift to the previous above_target height first
        lift = np.array([phone_pos[0], phone_pos[1], above_target_z], dtype=float)
        exit_inside = np.array([entry_in_x, phone_pos[1], side_entry_z], dtype=float)

        # Exit side should be high enough to clear everything
        exit_safe_z = safe_z_for_xy(entry_out_x, phone_pos[1], default_z=max(PATH_SAFE_Z, TABLE_TOP_Z + 0.25))
        exit_side = np.array([entry_out_x, phone_pos[1], exit_safe_z], dtype=float)

        print(f"[DEBUG] Exit heights - lift: {above_target_z:.3f}, exit_inside: {side_entry_z:.3f}, exit_side: {exit_safe_z:.3f}")

        move_cartesian(lift, step=PATH_STEP, hold_steps=HOLD_STEPS_LIFT, align_yaw=align_yaw)
        move_cartesian(exit_inside, step=PATH_STEP, hold_steps=HOLD_STEPS, align_yaw=align_yaw)
        move_cartesian(exit_side, step=PATH_STEP, hold_steps=HOLD_STEPS, align_yaw=align_yaw)

        print(">>> Grasp complete, exiting cabinet")

        # After exiting cabinet, transition to a forward-facing pose
        # This prevents the arm from staying in backward-leaning configuration
        print("-> Reorient to forward pose")
        base_pos, _ = get_base_pose()
        forward_yaw = math.atan2(WORKSTATION_POS[1] - base_pos[1], WORKSTATION_POS[0] - base_pos[0])
        forward_quat = p.getQuaternionFromEuler([EE_DOWN_EULER[0], EE_DOWN_EULER[1], forward_yaw])
        # Stay at exit_side position, just rotate to face forward
        move_cartesian(
            exit_side,
            target_quat=forward_quat,
            step=PATH_STEP_FINE,
            hold_steps=HOLD_STEPS,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        )
        # Place stage: move from cabinet (y=0.35) to table (y ~= -0.35)
        # Key: add Y transitions to avoid a direct jump
        ws_pos = np.array(WORKSTATION_POS, dtype=float)
        print(f"[DEBUG] workstation_pos: {ws_pos.tolist()}")

        front_y = CAB_FRONT_CLEAR_Y
        front_clear_z = max(
            safe_z_for_xy(entry_out_x, front_y, PATH_SAFE_Z),
            safe_z_for_xy(ws_pos[0], front_y, PATH_SAFE_Z),
            safe_z_for_xy(ws_pos[0], ws_pos[1], PATH_SAFE_Z),
            TABLE_TOP_Z + 0.25,
        )
        exit_retreat = np.array([entry_out_x, front_y, front_clear_z], dtype=float)
        front_x_align = np.array([ws_pos[0], front_y, front_clear_z], dtype=float)
        front_y_target = np.array([ws_pos[0], ws_pos[1], front_clear_z], dtype=float)
        print(f"[DEBUG] exit_retreat: {exit_retreat.tolist()}")
        print(f"[DEBUG] front_x_align: {front_x_align.tolist()}")
        print(f"[DEBUG] front_y_target: {front_y_target.tolist()}")
        move_cartesian(
            exit_retreat,
            target_quat=forward_quat,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        )
        move_cartesian(
            front_x_align,
            target_quat=forward_quat,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        )
        move_cartesian(
            front_y_target,
            target_quat=forward_quat,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        )

        # Ensure safe height: should be above table
        min_safe_height = TABLE_TOP_Z + 0.23  # 15cm above table for safety

        # Finally move above the workstation
        place_safe_z = max(safe_z_for_xy(ws_pos[0], ws_pos[1], PATH_SAFE_Z), min_safe_height)
        place_high = np.array([ws_pos[0], ws_pos[1], place_safe_z], dtype=float)

        # Hover height: should be well above workstation/table
        place_hover_z = max(WORKSTATION_TOP_Z + PLACE_HOVER_Z_OFFSET, TABLE_TOP_Z + 0.15)
        place_hover = np.array([ws_pos[0], ws_pos[1], place_hover_z], dtype=float)

        # Final place height: phone bottom should be just above workstation top
        place = np.array(
            [ws_pos[0], ws_pos[1], WORKSTATION_TOP_Z + PHONE_HALF_Z + PLACE_CONTACT_Z_OFFSET],
            dtype=float,
        )

        print(f"[DEBUG] TABLE_TOP_Z: {TABLE_TOP_Z:.3f}")
        print(f"[DEBUG] WORKSTATION_TOP_Z: {WORKSTATION_TOP_Z:.3f}")
        print(f"[DEBUG] min_safe_height: {min_safe_height:.3f}")
        print(f"[DEBUG] place_high: {place_high.tolist()}")
        print(f"[DEBUG] place_hover: {place_hover.tolist()}")
        print(f"[DEBUG] place (final): {place.tolist()}")

        place_path = build_path(
            [exit_side, exit_retreat, front_x_align, front_y_target, place_high, place_hover, place],
            step=PATH_STEP
        )
        draw_path(place_path, path="plan/place", color=0x00cc66)

        print("-> Move along front corridor to workstation")
        # Continue using forward-facing quaternion for all placement movements
        # This maintains the forward-leaning posture established after exiting cabinet
        move_cartesian(
            place_high,
            target_quat=forward_quat,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        )
        move_cartesian(
            place_hover,
            target_quat=forward_quat,
            step=PATH_STEP,
            hold_steps=HOLD_STEPS,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        )
        print("-> Align vertical for place")
        place_quat = p.getQuaternionFromEuler([math.pi, 0.0, forward_yaw])
        move_cartesian(
            place_hover,
            target_quat=place_quat,
            step=PATH_STEP_PLACE,
            hold_steps=HOLD_STEPS_PLACE,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        )
        print("-> Place phone")
        move_cartesian(
            place,
            target_quat=place_quat,
            step=PATH_STEP_PLACE,
            hold_steps=HOLD_STEPS_PLACE,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["grasp"],
        )
        placed = False
        if not wait_ee_at(place, tol=PATH_POS_TOL, max_steps=PATH_SETTLE_STEPS):
            cur_pos, _ = get_ee_pose()
            place_err = float(np.linalg.norm(cur_pos - place))
            if place_err <= PLACE_RELEASE_TOL:
                print(f"[WARN] Place pose not reached; releasing with relaxed tol {PLACE_RELEASE_TOL:.3f}m (err={place_err:.3f}).")
                step_sim(gripper_open_ratio, steps=PLACE_SETTLE_STEPS)
                set_gripper(1.0)
                step_sim(gripper_open_ratio, steps=PLACE_RELEASE_STEPS)
                detach_phone()
                step_sim(gripper_open_ratio, steps=GRIPPER_OPEN_STEPS)
                placed = True
            else:
                print(f"[WARN] Place pose not reached; skipping release (err={place_err:.3f}).")
        else:
            step_sim(gripper_open_ratio, steps=PLACE_SETTLE_STEPS)
            set_gripper(1.0)
            step_sim(gripper_open_ratio, steps=PLACE_RELEASE_STEPS)
            detach_phone()
            step_sim(gripper_open_ratio, steps=GRIPPER_OPEN_STEPS)
            placed = True

        if placed:
            print(">>> Placement complete")
        else:
            print(">>> Placement skipped; returning home")

        print("-> Return to initial position")
        cur_pos, _ = get_ee_pose()
        base_pos, _ = get_base_pose()
        return_lift_z = max(PATH_SAFE_Z, place_hover[2])
        return_lift = np.array([cur_pos[0], cur_pos[1], return_lift_z], dtype=float)
        return_xy = np.array([base_pos[0], base_pos[1], return_lift_z], dtype=float)
        return_home = np.array([base_pos[0], base_pos[1], home_pos[2]], dtype=float)
        return_path = build_path([cur_pos, return_lift, return_xy, return_home], step=PATH_STEP)
        draw_path(return_path, path="plan/return", color=0xff0000)

        move_cartesian(
            return_lift,
            step=PATH_STEP_FINE,
            hold_steps=HOLD_STEPS,
            align_yaw=True,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
        )
        move_cartesian(
            return_xy,
            step=PATH_STEP_FINE,
            hold_steps=HOLD_STEPS,
            align_yaw=True,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
        )
        move_cartesian(
            return_home,
            step=PATH_STEP_FINE,
            hold_steps=HOLD_STEPS,
            align_yaw=True,
            use_current_rest=True,
            prefer_config=PREF_CONFIGS["approach"],
        )
        wait_ee_at(return_home, tol=PATH_POS_TOL, max_steps=PATH_SETTLE_STEPS * 2)
        print("-> Settle to initial pose")
        set_arm_targets(home_q)
        step_sim(gripper_open_ratio, steps=HOLD_STEPS_LIFT)

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

task_loop()
