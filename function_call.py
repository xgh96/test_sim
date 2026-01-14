import ast
import os
import types
from typing import Any, Callable, Dict, List, Optional, Type

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from schemas import FUNCTION_DECLARATIONS


_RUN6DOF_MODULE = None
_RUN6DOF_STATE = {
    "stage": 0,
    "failed": False,
    "completed": False,
    "values": {},
    "last_message": "",
}

_STAGE_NAMES = {
    0: "init",
    1: "lift_from_folded",
    2: "move_above_base",
    3: "align_y",
    4: "move_to_cabinet_side",
    5: "enter_cabinet",
    6: "align_gripper_yaw",
    7: "descend_to_approach",
    8: "grasp",
    9: "exit_cabinet",
    10: "reorient_forward",
    11: "front_corridor",
    12: "move_to_place_hover",
    13: "align_for_place",
    14: "place_and_release",
    15: "return_home",
}


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def _load_run6dof_module() -> Any:
    path = os.path.join(os.path.dirname(__file__), "run_6dof.py")
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=path)
    new_body = []
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "task_loop"
        ):
            continue
        new_body.append(node)
    tree.body = new_body

    module = types.ModuleType("run_6dof_runtime")
    module.__file__ = path
    exec(compile(tree, path, "exec"), module.__dict__)
    return module


def _ensure_loaded() -> Any:
    global _RUN6DOF_MODULE
    if _RUN6DOF_MODULE is None:
        _RUN6DOF_MODULE = _load_run6dof_module()
    return _RUN6DOF_MODULE


def _guard_stage(expected_stage: int) -> Optional[Dict[str, Any]]:
    if _RUN6DOF_STATE["failed"]:
        return {
            "ok": False,
            "error": "Task already failed; reset before continuing.",
            "stage": _RUN6DOF_STATE["stage"],
        }
    if _RUN6DOF_STATE["completed"]:
        return {
            "ok": False,
            "error": "Task already completed; reset before continuing.",
            "stage": _RUN6DOF_STATE["stage"],
        }
    if _RUN6DOF_STATE["stage"] != expected_stage:
        return {
            "ok": False,
            "error": "Stage order mismatch.",
            "expected_stage": expected_stage,
            "expected_name": _STAGE_NAMES.get(expected_stage, "unknown"),
            "current_stage": _RUN6DOF_STATE["stage"],
            "current_name": _STAGE_NAMES.get(_RUN6DOF_STATE["stage"], "unknown"),
        }
    return None


def run6dof_status() -> Dict[str, Any]:
    return {
        "ok": True,
        "stage": _RUN6DOF_STATE["stage"],
        "stage_name": _STAGE_NAMES.get(_RUN6DOF_STATE["stage"], "unknown"),
        "failed": _RUN6DOF_STATE["failed"],
        "completed": _RUN6DOF_STATE["completed"],
        "last_message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_reset() -> Dict[str, Any]:
    global _RUN6DOF_MODULE
    if _RUN6DOF_MODULE is not None:
        try:
            _RUN6DOF_MODULE.cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            _RUN6DOF_MODULE.p.disconnect()
        except Exception:
            pass
    _RUN6DOF_MODULE = None
    _RUN6DOF_STATE.update(
        {
            "stage": 0,
            "failed": False,
            "completed": False,
            "values": {},
            "last_message": "",
        }
    )
    return {"ok": True, "message": "Reset done. Call stage 00 to initialize."}


def run6dof_stage_00_init() -> Dict[str, Any]:
    err = _guard_stage(0)
    if err:
        return err

    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]
    ctx.clear()

    m.time.sleep(10)
    m.vis["/Cameras/default/rotated/<object>"].set_transform(
        m.tf.translation_matrix([1.2, 0, 0.5])
        @ m.tf.rotation_matrix(m.np.pi / 4, [0, 1, 0])
    )

    home_pos, home_q, home_quat = m.init_robot_pose()

    phone_pos, phone_quat = m.get_phone_pose()
    grasp_pan = m.grasp_pan_for_phone(phone_quat) + m.GRASP_YAW_FLIP
    approach_quat = m.p.getQuaternionFromEuler(
        [m.EE_DOWN_EULER[0], m.EE_DOWN_EULER[1], grasp_pan]
    )
    grasp_pan = grasp_pan + m.GRASP_FINAL_YAW_OFFSET
    grasp_quat = m.p.getQuaternionFromEuler(
        [m.EE_DOWN_EULER[0], m.EE_DOWN_EULER[1], grasp_pan]
    )

    base_pos, _ = m.get_base_pose()
    xy_dist = float(m.np.linalg.norm(phone_pos[:2] - base_pos[:2]))

    entry_clear = m.CAB_SIDE_CLEAR_X
    if base_pos[0] <= m.CAB_CENTER_X:
        entry_out_x = m.CAB_MIN_X - entry_clear
        entry_in_x = m.CAB_MIN_X + m.CAB_ENTRY_MARGIN
    else:
        entry_out_x = m.CAB_MAX_X + entry_clear
        entry_in_x = m.CAB_MAX_X - m.CAB_ENTRY_MARGIN

    entry_z_min = m.SHELF1_Z + m.SHELF_SIZE[2] / 2.0 + 0.02
    entry_z_max = m.SHELF2_BOTTOM_Z - m.CAB_ENTRY_MARGIN
    entry_z = float(m.np.clip(phone_pos[2] + 0.06, entry_z_min, entry_z_max))
    entry_z = min(
        entry_z,
        m.safe_z_for_xy(entry_out_x, phone_pos[1], entry_z),
        m.safe_z_for_xy(entry_in_x, phone_pos[1], entry_z),
    )

    start_pos, _ = m.get_ee_pose()

    align_yaw = True

    grasp_z = phone_pos[2] + m.PHONE_HALF_Z + m.GRASP_CLEAR_Z
    approach_z = grasp_z + m.APPROACH_Z_OFFSET

    safe_height = m.PATH_SAFE_Z
    lift_pos = m.np.array([start_pos[0], start_pos[1], safe_height], dtype=float)

    base_overhead_x = 0.05
    base_overhead_y = start_pos[1]
    base_overhead_z = safe_height
    base_overhead = m.np.array(
        [base_overhead_x, base_overhead_y, base_overhead_z], dtype=float
    )

    y_aligned_z = m.safe_z_for_xy(base_overhead_x, phone_pos[1], m.PATH_SAFE_Z)
    y_aligned = m.np.array([base_overhead_x, phone_pos[1], y_aligned_z], dtype=float)

    side_safe_z = m.safe_z_for_xy(entry_out_x, phone_pos[1], m.PATH_SAFE_Z)
    side_high = m.np.array([entry_out_x, phone_pos[1], side_safe_z], dtype=float)

    side_entry_z = side_safe_z
    side_entry = m.np.array([entry_out_x, phone_pos[1], side_entry_z], dtype=float)
    inside_entry = m.np.array([entry_in_x, phone_pos[1], side_entry_z], dtype=float)

    above_target_z = side_entry_z
    above_target = m.np.array([phone_pos[0], phone_pos[1], above_target_z], dtype=float)

    approach = m.np.array([phone_pos[0], phone_pos[1], approach_z], dtype=float)
    grasp = m.np.array([phone_pos[0], phone_pos[1], grasp_z], dtype=float)

    waypoints = [
        start_pos,
        lift_pos,
        base_overhead,
        y_aligned,
        side_high,
        side_entry,
        inside_entry,
        above_target,
        approach,
        grasp,
    ]
    pick_path = m.build_path(waypoints, step=m.PATH_STEP)
    m.draw_path(pick_path, path="plan/pick", color=0xFF8800)

    ctx.update(
        {
            "home_pos": home_pos,
            "home_q": list(home_q),
            "home_quat": home_quat,
            "phone_pos": phone_pos,
            "phone_quat": phone_quat,
            "grasp_pan": grasp_pan,
            "approach_quat": approach_quat,
            "grasp_quat": grasp_quat,
            "base_pos": base_pos,
            "xy_dist": xy_dist,
            "entry_out_x": entry_out_x,
            "entry_in_x": entry_in_x,
            "entry_z": entry_z,
            "start_pos": start_pos,
            "align_yaw": align_yaw,
            "grasp_z": grasp_z,
            "approach_z": approach_z,
            "lift_pos": lift_pos,
            "base_overhead": base_overhead,
            "y_aligned": y_aligned,
            "side_high": side_high,
            "side_entry": side_entry,
            "inside_entry": inside_entry,
            "above_target": above_target,
            "approach": approach,
            "grasp": grasp,
            "side_entry_z": side_entry_z,
            "above_target_z": above_target_z,
        }
    )

    _RUN6DOF_STATE["stage"] = 1
    _RUN6DOF_STATE["last_message"] = "Init done; pick path planned."

    return {
        "ok": True,
        "stage": 0,
        "stage_name": _STAGE_NAMES[0],
        "next_stage": 1,
        "phone_pos": _to_jsonable(phone_pos),
        "start_pos": _to_jsonable(start_pos),
        "xy_dist": xy_dist,
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_01_lift() -> Dict[str, Any]:
    err = _guard_stage(1)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    m.move_cartesian(
        ctx["lift_pos"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        align_yaw=ctx["align_yaw"],
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
        limit_overrides=m.APPROACH_LIMIT_OVERRIDES,
        lock_overrides=m.APPROACH_LOCK_OVERRIDES,
    )

    _RUN6DOF_STATE["stage"] = 2
    _RUN6DOF_STATE["last_message"] = "Stage 1 done: lift from folded pose."
    return {
        "ok": True,
        "stage": 1,
        "stage_name": _STAGE_NAMES[1],
        "next_stage": 2,
        "lift_pos": _to_jsonable(ctx["lift_pos"]),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_02_base_overhead() -> Dict[str, Any]:
    err = _guard_stage(2)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    m.move_cartesian(
        ctx["base_overhead"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        align_yaw=ctx["align_yaw"],
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
        limit_overrides=m.APPROACH_LIMIT_OVERRIDES,
        lock_overrides=m.APPROACH_LOCK_OVERRIDES,
    )

    _RUN6DOF_STATE["stage"] = 3
    _RUN6DOF_STATE["last_message"] = "Stage 2 done: move above base."
    return {
        "ok": True,
        "stage": 2,
        "stage_name": _STAGE_NAMES[2],
        "next_stage": 3,
        "base_overhead": _to_jsonable(ctx["base_overhead"]),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_03_align_y() -> Dict[str, Any]:
    err = _guard_stage(3)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    m.move_cartesian(
        ctx["y_aligned"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        align_yaw=ctx["align_yaw"],
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
        limit_overrides=m.APPROACH_LIMIT_OVERRIDES,
        lock_overrides=m.APPROACH_LOCK_OVERRIDES,
    )

    _RUN6DOF_STATE["stage"] = 4
    _RUN6DOF_STATE["last_message"] = "Stage 3 done: align Y with cabinet."
    return {
        "ok": True,
        "stage": 3,
        "stage_name": _STAGE_NAMES[3],
        "next_stage": 4,
        "y_aligned": _to_jsonable(ctx["y_aligned"]),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_04_to_cabinet_side() -> Dict[str, Any]:
    err = _guard_stage(4)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    m.move_cartesian(
        ctx["side_high"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        align_yaw=ctx["align_yaw"],
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
        limit_overrides=m.APPROACH_LIMIT_OVERRIDES,
        lock_overrides=m.APPROACH_LOCK_OVERRIDES,
    )
    m.move_cartesian(
        ctx["side_entry"],
        step=m.PATH_STEP_FINE,
        hold_steps=m.HOLD_STEPS,
        align_yaw=ctx["align_yaw"],
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
        limit_overrides=m.APPROACH_LIMIT_OVERRIDES,
        lock_overrides=m.APPROACH_LOCK_OVERRIDES,
    )

    _RUN6DOF_STATE["stage"] = 5
    _RUN6DOF_STATE["last_message"] = "Stage 4 done: move to cabinet side."
    return {
        "ok": True,
        "stage": 4,
        "stage_name": _STAGE_NAMES[4],
        "next_stage": 5,
        "side_high": _to_jsonable(ctx["side_high"]),
        "side_entry": _to_jsonable(ctx["side_entry"]),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_05_enter_cabinet() -> Dict[str, Any]:
    err = _guard_stage(5)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    m.move_cartesian(
        ctx["inside_entry"],
        step=m.PATH_STEP_FINE,
        hold_steps=m.HOLD_STEPS,
        align_yaw=ctx["align_yaw"],
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
        limit_overrides=m.APPROACH_LIMIT_OVERRIDES,
        lock_overrides=m.APPROACH_LOCK_OVERRIDES,
    )
    m.move_cartesian(
        ctx["above_target"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        align_yaw=ctx["align_yaw"],
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
        limit_overrides=m.APPROACH_LIMIT_OVERRIDES,
        lock_overrides=m.APPROACH_LOCK_OVERRIDES,
    )

    _RUN6DOF_STATE["stage"] = 6
    _RUN6DOF_STATE["last_message"] = "Stage 5 done: enter cabinet and move above target."
    return {
        "ok": True,
        "stage": 5,
        "stage_name": _STAGE_NAMES[5],
        "next_stage": 6,
        "inside_entry": _to_jsonable(ctx["inside_entry"]),
        "above_target": _to_jsonable(ctx["above_target"]),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_06_align_gripper_yaw() -> Dict[str, Any]:
    err = _guard_stage(6)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    reached = m.wait_ee_at(
        ctx["above_target"],
        tol=m.PATH_POS_TOL,
        max_steps=m.PATH_SETTLE_STEPS,
    )
    if not reached:
        print("[WARN] Above-target pose not fully reached; aligning anyway.")

    m.orient_pan_in_place(
        ctx["above_target"],
        ctx["grasp_pan"],
        steps=m.HOLD_STEPS_GRASP,
        prefer_config=m.PREF_CONFIGS["approach"],
        limit_overrides=m.APPROACH_LIMIT_OVERRIDES,
        lock_overrides=m.APPROACH_LOCK_OVERRIDES,
    )

    _RUN6DOF_STATE["stage"] = 7
    _RUN6DOF_STATE["last_message"] = "Stage 6 done: align gripper yaw."
    return {
        "ok": True,
        "stage": 6,
        "stage_name": _STAGE_NAMES[6],
        "next_stage": 7,
        "above_target": _to_jsonable(ctx["above_target"]),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_07_descend_approach() -> Dict[str, Any]:
    err = _guard_stage(7)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    m.move_cartesian(
        ctx["approach"],
        target_quat=ctx["approach_quat"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        align_yaw=False,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )

    _RUN6DOF_STATE["stage"] = 8
    _RUN6DOF_STATE["last_message"] = "Stage 7 done: descend to approach height."
    return {
        "ok": True,
        "stage": 7,
        "stage_name": _STAGE_NAMES[7],
        "next_stage": 8,
        "approach": _to_jsonable(ctx["approach"]),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_08_grasp() -> Dict[str, Any]:
    err = _guard_stage(8)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    min_grasp_z = ctx["phone_pos"][2] + m.PHONE_HALF_Z + 0.001
    ok = m.attempt_grasp_phone(
        ctx["grasp"],
        min_grasp_z,
        target_quat=ctx["grasp_quat"],
        align_yaw=False,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )
    if not ok:
        _RUN6DOF_STATE["failed"] = True
        _RUN6DOF_STATE["last_message"] = "Grasp failed; stopping."
        return {
            "ok": False,
            "stage": 8,
            "stage_name": _STAGE_NAMES[8],
            "error": _RUN6DOF_STATE["last_message"],
        }

    _RUN6DOF_STATE["stage"] = 9
    _RUN6DOF_STATE["last_message"] = "Stage 8 done: grasp success."
    return {
        "ok": True,
        "stage": 8,
        "stage_name": _STAGE_NAMES[8],
        "next_stage": 9,
        "grasp": _to_jsonable(ctx["grasp"]),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_09_exit_cabinet() -> Dict[str, Any]:
    err = _guard_stage(9)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    lift = m.np.array(
        [ctx["phone_pos"][0], ctx["phone_pos"][1], ctx["above_target_z"]],
        dtype=float,
    )
    exit_inside = m.np.array(
        [ctx["entry_in_x"], ctx["phone_pos"][1], ctx["side_entry_z"]],
        dtype=float,
    )
    exit_safe_z = m.safe_z_for_xy(
        ctx["entry_out_x"],
        ctx["phone_pos"][1],
        default_z=max(m.PATH_SAFE_Z, m.TABLE_TOP_Z + 0.25),
    )
    exit_side = m.np.array(
        [ctx["entry_out_x"], ctx["phone_pos"][1], exit_safe_z],
        dtype=float,
    )

    ctx["lift"] = lift
    ctx["exit_inside"] = exit_inside
    ctx["exit_safe_z"] = exit_safe_z
    ctx["exit_side"] = exit_side

    m.move_cartesian(lift, step=m.PATH_STEP, hold_steps=m.HOLD_STEPS_LIFT, align_yaw=ctx["align_yaw"])
    m.move_cartesian(exit_inside, step=m.PATH_STEP, hold_steps=m.HOLD_STEPS, align_yaw=ctx["align_yaw"])
    m.move_cartesian(exit_side, step=m.PATH_STEP, hold_steps=m.HOLD_STEPS, align_yaw=ctx["align_yaw"])

    _RUN6DOF_STATE["stage"] = 10
    _RUN6DOF_STATE["last_message"] = "Stage 9 done: exit cabinet."
    return {
        "ok": True,
        "stage": 9,
        "stage_name": _STAGE_NAMES[9],
        "next_stage": 10,
        "exit_side": _to_jsonable(exit_side),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_10_forward_pose() -> Dict[str, Any]:
    err = _guard_stage(10)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    base_pos, _ = m.get_base_pose()
    forward_yaw = m.math.atan2(
        m.WORKSTATION_POS[1] - base_pos[1],
        m.WORKSTATION_POS[0] - base_pos[0],
    )
    forward_quat = m.p.getQuaternionFromEuler(
        [m.EE_DOWN_EULER[0], m.EE_DOWN_EULER[1], forward_yaw]
    )

    ctx["forward_yaw"] = forward_yaw
    ctx["forward_quat"] = forward_quat

    m.move_cartesian(
        ctx["exit_side"],
        target_quat=forward_quat,
        step=m.PATH_STEP_FINE,
        hold_steps=m.HOLD_STEPS,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )

    _RUN6DOF_STATE["stage"] = 11
    _RUN6DOF_STATE["last_message"] = "Stage 10 done: reorient to forward pose."
    return {
        "ok": True,
        "stage": 10,
        "stage_name": _STAGE_NAMES[10],
        "next_stage": 11,
        "forward_yaw": forward_yaw,
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_11_front_corridor() -> Dict[str, Any]:
    err = _guard_stage(11)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    ws_pos = m.np.array(m.WORKSTATION_POS, dtype=float)
    front_y = m.CAB_FRONT_CLEAR_Y
    front_clear_z = max(
        m.safe_z_for_xy(ctx["entry_out_x"], front_y, m.PATH_SAFE_Z),
        m.safe_z_for_xy(ws_pos[0], front_y, m.PATH_SAFE_Z),
        m.safe_z_for_xy(ws_pos[0], ws_pos[1], m.PATH_SAFE_Z),
        m.TABLE_TOP_Z + 0.25,
    )
    exit_retreat = m.np.array([ctx["entry_out_x"], front_y, front_clear_z], dtype=float)
    front_x_align = m.np.array([ws_pos[0], front_y, front_clear_z], dtype=float)
    front_y_target = m.np.array([ws_pos[0], ws_pos[1], front_clear_z], dtype=float)

    ctx["ws_pos"] = ws_pos
    ctx["exit_retreat"] = exit_retreat
    ctx["front_x_align"] = front_x_align
    ctx["front_y_target"] = front_y_target

    m.move_cartesian(
        exit_retreat,
        target_quat=ctx["forward_quat"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )
    m.move_cartesian(
        front_x_align,
        target_quat=ctx["forward_quat"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )
    m.move_cartesian(
        front_y_target,
        target_quat=ctx["forward_quat"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )

    _RUN6DOF_STATE["stage"] = 12
    _RUN6DOF_STATE["last_message"] = "Stage 11 done: move along front corridor."
    return {
        "ok": True,
        "stage": 11,
        "stage_name": _STAGE_NAMES[11],
        "next_stage": 12,
        "front_y_target": _to_jsonable(front_y_target),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_12_place_hover() -> Dict[str, Any]:
    err = _guard_stage(12)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    ws_pos = ctx["ws_pos"]
    min_safe_height = m.TABLE_TOP_Z + 0.23
    place_safe_z = max(
        m.safe_z_for_xy(ws_pos[0], ws_pos[1], m.PATH_SAFE_Z),
        min_safe_height,
    )
    place_high = m.np.array([ws_pos[0], ws_pos[1], place_safe_z], dtype=float)

    place_hover_z = max(m.WORKSTATION_TOP_Z + m.PLACE_HOVER_Z_OFFSET, m.TABLE_TOP_Z + 0.15)
    place_hover = m.np.array([ws_pos[0], ws_pos[1], place_hover_z], dtype=float)

    place = m.np.array(
        [ws_pos[0], ws_pos[1], m.WORKSTATION_TOP_Z + m.PHONE_HALF_Z + m.PLACE_CONTACT_Z_OFFSET],
        dtype=float,
    )

    ctx["place_high"] = place_high
    ctx["place_hover"] = place_hover
    ctx["place"] = place

    place_path = m.build_path(
        [
            ctx["exit_side"],
            ctx["exit_retreat"],
            ctx["front_x_align"],
            ctx["front_y_target"],
            place_high,
            place_hover,
            place,
        ],
        step=m.PATH_STEP,
    )
    m.draw_path(place_path, path="plan/place", color=0x00CC66)

    m.move_cartesian(
        place_high,
        target_quat=ctx["forward_quat"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )
    m.move_cartesian(
        place_hover,
        target_quat=ctx["forward_quat"],
        step=m.PATH_STEP,
        hold_steps=m.HOLD_STEPS,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )

    _RUN6DOF_STATE["stage"] = 13
    _RUN6DOF_STATE["last_message"] = "Stage 12 done: move to place hover."
    return {
        "ok": True,
        "stage": 12,
        "stage_name": _STAGE_NAMES[12],
        "next_stage": 13,
        "place_hover": _to_jsonable(place_hover),
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_13_align_place() -> Dict[str, Any]:
    err = _guard_stage(13)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    place_quat = m.p.getQuaternionFromEuler([m.math.pi, 0.0, ctx["forward_yaw"]])
    ctx["place_quat"] = place_quat

    m.move_cartesian(
        ctx["place_hover"],
        target_quat=place_quat,
        step=m.PATH_STEP_PLACE,
        hold_steps=m.HOLD_STEPS_PLACE,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )

    _RUN6DOF_STATE["stage"] = 14
    _RUN6DOF_STATE["last_message"] = "Stage 13 done: align vertical for place."
    return {
        "ok": True,
        "stage": 13,
        "stage_name": _STAGE_NAMES[13],
        "next_stage": 14,
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_14_place_release() -> Dict[str, Any]:
    err = _guard_stage(14)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    m.move_cartesian(
        ctx["place"],
        target_quat=ctx["place_quat"],
        step=m.PATH_STEP_PLACE,
        hold_steps=m.HOLD_STEPS_PLACE,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["grasp"],
    )

    placed = False
    if not m.wait_ee_at(ctx["place"], tol=m.PATH_POS_TOL, max_steps=m.PATH_SETTLE_STEPS):
        cur_pos, _ = m.get_ee_pose()
        place_err = float(m.np.linalg.norm(cur_pos - ctx["place"]))
        if place_err <= m.PLACE_RELEASE_TOL:
            m.step_sim(m.gripper_open_ratio, steps=m.PLACE_SETTLE_STEPS)
            m.set_gripper(1.0)
            m.step_sim(m.gripper_open_ratio, steps=m.PLACE_RELEASE_STEPS)
            m.detach_phone()
            m.step_sim(m.gripper_open_ratio, steps=m.GRIPPER_OPEN_STEPS)
            placed = True
    else:
        m.step_sim(m.gripper_open_ratio, steps=m.PLACE_SETTLE_STEPS)
        m.set_gripper(1.0)
        m.step_sim(m.gripper_open_ratio, steps=m.PLACE_RELEASE_STEPS)
        m.detach_phone()
        m.step_sim(m.gripper_open_ratio, steps=m.GRIPPER_OPEN_STEPS)
        placed = True

    ctx["placed"] = placed

    _RUN6DOF_STATE["stage"] = 15
    _RUN6DOF_STATE["last_message"] = "Stage 14 done: place and release."
    return {
        "ok": True,
        "stage": 14,
        "stage_name": _STAGE_NAMES[14],
        "next_stage": 15,
        "placed": placed,
        "message": _RUN6DOF_STATE["last_message"],
    }


def run6dof_stage_15_return_home() -> Dict[str, Any]:
    err = _guard_stage(15)
    if err:
        return err
    m = _ensure_loaded()
    ctx = _RUN6DOF_STATE["values"]

    cur_pos, _ = m.get_ee_pose()
    base_pos, _ = m.get_base_pose()
    return_lift_z = max(m.PATH_SAFE_Z, ctx["place_hover"][2])
    return_lift = m.np.array([cur_pos[0], cur_pos[1], return_lift_z], dtype=float)
    return_xy = m.np.array([base_pos[0], base_pos[1], return_lift_z], dtype=float)
    return_home = m.np.array([base_pos[0], base_pos[1], ctx["home_pos"][2]], dtype=float)

    return_path = m.build_path([cur_pos, return_lift, return_xy, return_home], step=m.PATH_STEP)
    m.draw_path(return_path, path="plan/return", color=0xFF0000)

    m.move_cartesian(
        return_lift,
        step=m.PATH_STEP_FINE,
        hold_steps=m.HOLD_STEPS,
        align_yaw=True,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
    )
    m.move_cartesian(
        return_xy,
        step=m.PATH_STEP_FINE,
        hold_steps=m.HOLD_STEPS,
        align_yaw=True,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
    )
    m.move_cartesian(
        return_home,
        step=m.PATH_STEP_FINE,
        hold_steps=m.HOLD_STEPS,
        align_yaw=True,
        use_current_rest=True,
        prefer_config=m.PREF_CONFIGS["approach"],
    )
    m.wait_ee_at(return_home, tol=m.PATH_POS_TOL, max_steps=m.PATH_SETTLE_STEPS * 2)
    m.set_arm_targets(ctx["home_q"])
    m.step_sim(m.gripper_open_ratio, steps=m.HOLD_STEPS_LIFT)

    try:
        m.cv2.destroyAllWindows()
    except Exception:
        pass

    _RUN6DOF_STATE["stage"] = 16
    _RUN6DOF_STATE["completed"] = True
    _RUN6DOF_STATE["last_message"] = "Stage 15 done: return home."

    return {
        "ok": True,
        "stage": 15,
        "stage_name": _STAGE_NAMES[15],
        "next_stage": None,
        "return_home": _to_jsonable(return_home),
        "message": _RUN6DOF_STATE["last_message"],
        "completed": True,
    }


TOOL_REGISTRY: Dict[str, Callable[..., Any]] = {
    "run6dof_status": run6dof_status,
    "run6dof_reset": run6dof_reset,
    "run6dof_stage_00_init": run6dof_stage_00_init,
    "run6dof_stage_01_lift": run6dof_stage_01_lift,
    "run6dof_stage_02_base_overhead": run6dof_stage_02_base_overhead,
    "run6dof_stage_03_align_y": run6dof_stage_03_align_y,
    "run6dof_stage_04_to_cabinet_side": run6dof_stage_04_to_cabinet_side,
    "run6dof_stage_05_enter_cabinet": run6dof_stage_05_enter_cabinet,
    "run6dof_stage_06_align_gripper_yaw": run6dof_stage_06_align_gripper_yaw,
    "run6dof_stage_07_descend_approach": run6dof_stage_07_descend_approach,
    "run6dof_stage_08_grasp": run6dof_stage_08_grasp,
    "run6dof_stage_09_exit_cabinet": run6dof_stage_09_exit_cabinet,
    "run6dof_stage_10_forward_pose": run6dof_stage_10_forward_pose,
    "run6dof_stage_11_front_corridor": run6dof_stage_11_front_corridor,
    "run6dof_stage_12_place_hover": run6dof_stage_12_place_hover,
    "run6dof_stage_13_align_place": run6dof_stage_13_align_place,
    "run6dof_stage_14_place_release": run6dof_stage_14_place_release,
    "run6dof_stage_15_return_home": run6dof_stage_15_return_home,
}


_JSON_TYPE_MAP = {
    "number": float,
    "integer": int,
    "string": str,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _json_type_to_py_type(schema: Dict[str, Any]) -> Any:
    if "enum" in schema:
        enum_vals = schema.get("enum") or []
        if enum_vals:
            return type(enum_vals[0])

    schema_type = schema.get("type", "string")
    if schema_type == "array":
        items = schema.get("items", {"type": "string"})
        return List[_json_type_to_py_type(items)]

    return _JSON_TYPE_MAP.get(schema_type, Any)


def _create_args_schema(name: str, parameters: Dict[str, Any]) -> Type[BaseModel]:
    props = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    fields: Dict[str, Any] = {}

    for prop_name, prop_schema in props.items():
        py_type = _json_type_to_py_type(prop_schema)
        desc = prop_schema.get("description", "")
        if prop_name in required:
            fields[prop_name] = (py_type, Field(..., description=desc))
        else:
            fields[prop_name] = (Optional[py_type], Field(None, description=desc))

    return create_model(f"{name}_Args", **fields)


def build_tools(strict: bool = True) -> List[StructuredTool]:
    tools: List[StructuredTool] = []
    missing: List[str] = []

    for decl in FUNCTION_DECLARATIONS:
        name = decl["name"]
        func = TOOL_REGISTRY.get(name)
        if func is None:
            missing.append(name)
            continue
        args_schema = _create_args_schema(name, decl.get("parameters", {}))
        tools.append(
            StructuredTool.from_function(
                func=func,
                name=name,
                description=decl.get("description", ""),
                args_schema=args_schema,
            )
        )

    if missing and strict:
        raise RuntimeError(f"Missing tool implementations: {', '.join(missing)}")

    return tools
