import pybullet as p
import pybullet_data
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time
import numpy as np
import cv2

# --- 1. 初始化 ---
vis = meshcat.Visualizer()
vis.delete() # 清空之前缓存的残余模型
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

SHELF_POS = [0.4, 0.35, 0]
TABLE_POS = [0.4, -0.35, 0]

# --- 2. 物理场景构建 ---
p.loadURDF("plane.urdf")

def create_scene():
    ids = {}
    # 柜子
    ids['shelf1'] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.12, 0.18, 0.01]), 
                                     p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.18, 0.01], rgbaColor=[0.4, 0.2, 0.1, 1]), [SHELF_POS[0], SHELF_POS[1], 0.2])
    ids['shelf2'] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.12, 0.18, 0.01]), 
                                     p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.18, 0.01], rgbaColor=[0.4, 0.2, 0.1, 1]), [SHELF_POS[0], SHELF_POS[1], 0.4])
    # 柜腿 (简化为4根长腿)
    for i, off in enumerate([[-0.1, -0.15], [0.1, -0.15], [-0.1, 0.15], [0.1, 0.15]]):
        ids[f's_leg{i}'] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.2]), 
                                            p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, 0.01, 0.2], rgbaColor=[0.2, 0.2, 0.2, 1]), [SHELF_POS[0]+off[0], SHELF_POS[1]+off[1], 0.2])
    # 桌子
    ids['table'] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.02]), 
                                    p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.02], rgbaColor=[0.8, 0.8, 0.8, 1]), [TABLE_POS[0], TABLE_POS[1], 0.3])
    for i, off in enumerate([[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]]):
        ids[f't_leg{i}'] = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.15]), 
                                            p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.15], rgbaColor=[0.3, 0.3, 0.3, 1]), [TABLE_POS[0]+off[0], TABLE_POS[1]+off[1], 0.15])
    # 手机
    ids['phone'] = p.createMultiBody(0.2, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.06, 0.005]), 
                                    p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.06, 0.005], rgbaColor=[0.1, 0.1, 0.1, 1]), [SHELF_POS[0], SHELF_POS[1], 0.21])
    return ids

scene_ids = create_scene()
robot_id = p.loadURDF("ur3_robotiq.urdf", [0, 0, 0], useFixedBase=True)

# 机械臂参数
JOINT_RADII = {0: 0.052, 1: 0.048, 2: 0.042, 3: 0.032}
BONE_STRUCTURE = [(-1, 0, 0.055, 0x777777), (0, 1, 0.048, 0x1a5fb4), 
                  (1, 2, 0.042, 0x777777), (2, 3, 0.035, 0x1a5fb4), (3, 4, 0.030, 0x777777)]

# --- 3. 视觉初始化 (只在这里 set_object) ---
def setup_visuals():
    # 场景
    vis["shelf1"].set_object(g.Box([0.24, 0.36, 0.02]), g.MeshLambertMaterial(color=0x8b4513))
    vis["shelf2"].set_object(g.Box([0.24, 0.36, 0.02]), g.MeshLambertMaterial(color=0x8b4513))
    for i in range(4): vis[f"s_leg{i}"].set_object(g.Box([0.02, 0.02, 0.4]), g.MeshLambertMaterial(color=0x333333))
    vis["table"].set_object(g.Box([0.5, 0.5, 0.04]), g.MeshLambertMaterial(color=0xcccccc))
    for i in range(4): vis[f"t_leg{i}"].set_object(g.Box([0.03, 0.03, 0.3]), g.MeshLambertMaterial(color=0x555555))
    vis["phone"].set_object(g.Box([0.06, 0.12, 0.01]), g.MeshLambertMaterial(color=0x111111))
    # 机器人
    for i, r in JOINT_RADII.items(): vis[f"robot/joints/j{i}"].set_object(g.Sphere(r), g.MeshLambertMaterial(color=0x333333))
    vis["robot/base"].set_object(g.Cylinder(0.06, 0.05), g.MeshLambertMaterial(color=0x333333))
    vis["robot/gripper/base"].set_object(g.Box([0.08, 0.04, 0.04]), g.MeshLambertMaterial(color=0x222222))
    vis["robot/gripper/l"].set_object(g.Box([0.01, 0.02, 0.08]), g.MeshLambertMaterial(color=0x555555))
    vis["robot/gripper/r"].set_object(g.Box([0.01, 0.02, 0.08]), g.MeshLambertMaterial(color=0x555555))

setup_visuals()

# --- 4. RGB-D 3D 视觉获取 ---
def get_rgbd():
    state = p.getLinkState(robot_id, 4)
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

# --- 5. 同步函数 (只做 set_transform) ---
def sync(gw=0.035):
    # 同步环境
    for name, p_id in scene_ids.items():
        pos, orn = p.getBasePositionAndOrientation(p_id)
        vis[name].set_transform(tf.translation_matrix(pos) @ tf.quaternion_matrix([orn[3], orn[0], orn[1], orn[2]]))
    
    # 获取机器人 Link 状态
    states = {i: (p.getBasePositionAndOrientation(robot_id) if i == -1 else p.getLinkState(robot_id, i)) for i in range(-1, 5)}
    
    # 同步关节和连杆
    vis["robot/base"].set_transform(tf.translation_matrix(states[-1][0]) @ tf.quaternion_matrix([states[-1][1][3], states[-1][1][0], states[-1][1][1], states[-1][1][2]]) @ tf.rotation_matrix(np.pi/2, [1,0,0]))
    for i in range(4):
        vis[f"robot/joints/j{i}"].set_transform(tf.translation_matrix(states[i][0]) @ tf.quaternion_matrix([states[i][1][3], states[i][1][0], states[i][1][1], states[i][1][2]]))
    
    # 动态骨架连杆
    for i, (s, e, r, c) in enumerate(BONE_STRUCTURE):
        p1, p2 = np.array(states[s][0]), np.array(states[e][0])
        dist = np.linalg.norm(p2-p1)
        if dist < 0.001: continue
        vis[f"robot/bones/b{i}"].set_object(g.Cylinder(dist, r), g.MeshLambertMaterial(color=c))
        vt = (p2-p1)/dist
        rot = np.eye(4); va = np.cross([0,1,0], vt); vs = np.linalg.norm(va); vc = np.dot([0,1,0], vt)
        if vs > 0.0001: rot = tf.rotation_matrix(np.arctan2(vs, vc), va/vs)
        vis[f"robot/bones/b{i}"].set_transform(tf.translation_matrix((p1+p2)/2.0) @ rot)

    # 抓夹
    m4 = tf.translation_matrix(states[4][0]) @ tf.quaternion_matrix([states[4][1][3], states[4][1][0], states[4][1][1], states[4][1][2]])
    vis["robot/gripper/base"].set_transform(m4 @ tf.translation_matrix([0, 0, 0.02]))
    vis["robot/gripper/l"].set_transform(m4 @ tf.translation_matrix([gw, 0, 0.04]))
    vis["robot/gripper/r"].set_transform(m4 @ tf.translation_matrix([-gw, 0, 0.04]))

    # OpenCV RGB-D 显示
    rgb, depth = get_rgbd()
    cv2.imshow("RGB-D Vision", np.hstack((rgb, depth)))
    cv2.waitKey(1)

# --- 6. 运动函数 ---
def move(target, gw=0.035, steps=80):
    for _ in range(steps):
        ik = p.calculateInverseKinematics(robot_id, 4, target, p.getQuaternionFromEuler([0, 1.57, 0]))
        for i in range(4): p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, ik[i])
        p.stepSimulation(); sync(gw); time.sleep(1/120.)

# try:
#     # 调整初始相机视角
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
        # 0. 初始视角设置
        vis["/Cameras/default/rotated/<object>"].set_transform(
            tf.translation_matrix([1.2, 0, 0.5]) @ tf.rotation_matrix(np.pi/4, [0, 1, 0]))
        
        print(">>> 开始闭环任务：从柜子搬运到桌面")
        
        # --- 第一阶段：抓取 ---
        # 1. 移到手机上方 (预抓取)
        print("-> 移动到预抓取位")
        move([0.4, 0.35, 0.3], gw=0.035, steps=60)
        
        # 2. 下降并接触手机
        print("-> 下降抓取")
        move([0.4, 0.35, 0.215], gw=0.035, steps=40)
        
        # 3. 闭合夹爪 (注意：这里需要多步 sync 来显示闭合动作)
        print("-> 闭合夹爪")
        for _ in range(30):
            p.stepSimulation()
            sync(gw=0.005) # 夹爪间距设为 0.005
            
        # --- 第二阶段：搬运 ---
        # 4. 垂直抬起 (避开柜子)
        print("-> 抬起物体")
        move([0.4, 0.35, 0.45], gw=0.005, steps=60)
        
        # 5. 横向移动到桌子上方 (中继点)
        print("-> 移动到桌面上方")
        move([0.4, -0.35, 0.45], gw=0.005, steps=80)
        
        # --- 第三阶段：放置 ---
        # 6. 下降到桌面高度 (预定放置高度 Z=0.31)
        print("-> 准备放置")
        move([0.4, -0.35, 0.315], gw=0.005, steps=40)
        
        # 7. 松开夹爪
        print("-> 松开夹爪")
        for _ in range(30):
            p.stepSimulation()
            sync(gw=0.035) 
            
        # 8. 安全撤离 (抬起机械臂防止碰撞)
        print("-> 任务完成，机械臂撤离")
        move([0.4, -0.35, 0.5], gw=0.035, steps=50)
        
        print(">>> 任务闭环测试成功！")

    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()

task_loop()