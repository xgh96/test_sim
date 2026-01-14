# Tool schema declarations for run_6dof staged execution.

FUNCTION_DECLARATIONS = [
    {
        "name": "run6dof_status",
        "category": "run6dof_control",
        "description": "Get current run_6dof stage status.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_reset",
        "category": "run6dof_control",
        "description": "Reset run_6dof runtime and stage state.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_00_init",
        "category": "run6dof_stage",
        "description": "Stage 00: initialize runtime and plan pick path.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_01_lift",
        "category": "run6dof_stage",
        "description": "Stage 01: lift from folded pose to safe height.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_02_base_overhead",
        "category": "run6dof_stage",
        "description": "Stage 02: move above base with safe height.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_03_align_y",
        "category": "run6dof_stage",
        "description": "Stage 03: align Y with cabinet.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_04_to_cabinet_side",
        "category": "run6dof_stage",
        "description": "Stage 04: move along X to cabinet side.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_05_enter_cabinet",
        "category": "run6dof_stage",
        "description": "Stage 05: enter cabinet and move above target.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_06_align_gripper_yaw",
        "category": "run6dof_stage",
        "description": "Stage 06: align gripper yaw above target.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_07_descend_approach",
        "category": "run6dof_stage",
        "description": "Stage 07: descend to approach height.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_08_grasp",
        "category": "run6dof_stage",
        "description": "Stage 08: attempt grasp sequence.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_09_exit_cabinet",
        "category": "run6dof_stage",
        "description": "Stage 09: lift and exit cabinet.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_10_forward_pose",
        "category": "run6dof_stage",
        "description": "Stage 10: reorient to forward pose.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_11_front_corridor",
        "category": "run6dof_stage",
        "description": "Stage 11: move along front corridor toward workstation.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_12_place_hover",
        "category": "run6dof_stage",
        "description": "Stage 12: move to workstation hover positions.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_13_align_place",
        "category": "run6dof_stage",
        "description": "Stage 13: align tool for placing.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_14_place_release",
        "category": "run6dof_stage",
        "description": "Stage 14: place phone and release.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "run6dof_stage_15_return_home",
        "category": "run6dof_stage",
        "description": "Stage 15: return to home pose.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]
