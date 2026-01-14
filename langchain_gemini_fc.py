import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from function_call import build_tools

def build_llm() -> ChatGoogleGenerativeAI:
    api_key = "AIzaSyDuTLZIgtHaH6DrqI-fzHCLa_J0LgewnZ0"
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.6,
    )


def _write_trace(trace: Dict[str, object], path: str = "tool_trace.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2, default=str)


def run_once(user_text: str, max_steps: int = 5) -> str:
    tools = build_tools()
    tool_map: Dict[str, object] = {t.name: t for t in tools}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use tools when needed to answer the user.",
            ),
            ("human", "{input}"),
        ]
    )

    llm = build_llm().bind_tools(tools)
    messages = prompt.format_messages(input=user_text)

    trace: Dict[str, object] = {
        "input": user_text,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    for step_idx in range(max_steps):
        reply = llm.invoke(messages)
        step_record = {
            "step": step_idx + 1,
            "model_reply": reply.content,
            "tool_calls": reply.tool_calls,
            "tool_results": [],
        }
        if not reply.tool_calls:
            trace["steps"].append(step_record)
            trace["final"] = reply.content
            _write_trace(trace)
            return reply.content

        tool_messages = []
        for call in reply.tool_calls:
            tool_fn = tool_map.get(call["name"])
            if tool_fn is None:
                tool_messages.append(
                    ToolMessage(
                        content=f"Tool not found: {call['name']}",
                        tool_call_id=call["id"],
                    )
                )
                step_record["tool_results"].append(
                    {
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "args": call.get("args", {}),
                        "result": "Tool not found",
                    }
                )
                continue

            result = tool_fn.invoke(call.get("args", {}))
            tool_messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )
            step_record["tool_results"].append(
                {
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "args": call.get("args", {}),
                    "result": result,
                }
            )

        trace["steps"].append(step_record)
        messages = messages + [reply] + tool_messages

    trace["final"] = "达到最大工具调用步数，未完成。"
    _write_trace(trace)
    return trace["final"]


def _run_tool(
    tool_map: Dict[str, object],
    name: str,
    args: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    tool = tool_map.get(name)
    if tool is None:
        raise RuntimeError(f"工具不存在: {name}")
    return tool.invoke(args or {})


def run_run6dof_sequence() -> int:
    tools = build_tools()
    tool_map: Dict[str, object] = {t.name: t for t in tools}

    stage_tools: List[str] = [
        "run6dof_reset",
        "run6dof_stage_00_init",
        "run6dof_stage_01_lift",
        "run6dof_stage_02_base_overhead",
        "run6dof_stage_03_align_y",
        "run6dof_stage_04_to_cabinet_side",
        "run6dof_stage_05_enter_cabinet",
        "run6dof_stage_06_align_gripper_yaw",
        "run6dof_stage_07_descend_approach",
        "run6dof_stage_08_grasp",
        "run6dof_stage_09_exit_cabinet",
        "run6dof_stage_10_forward_pose",
        "run6dof_stage_11_front_corridor",
        "run6dof_stage_12_place_hover",
        "run6dof_stage_13_align_place",
        "run6dof_stage_14_place_release",
        "run6dof_stage_15_return_home",
    ]

    trace: Dict[str, object] = {
        "mode": "run6dof_sequence",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "steps": [],
    }

    for name in stage_tools:
        result = _run_tool(tool_map, name)
        trace["steps"].append({"tool": name, "result": result})
        print(f"[阶段执行] {name} -> {result.get('message', '完成')}")
        if not result.get("ok", True):
            trace["final"] = f"阶段失败: {name}"
            _write_trace(trace)
            print(f"[终止] 阶段失败: {name}")
            return 1
        if result.get("completed"):
            trace["final"] = "任务完成"
            _write_trace(trace)
            print("[完成] run_6dof 全流程已执行完毕")
            return 0

    trace["final"] = "任务完成"
    _write_trace(trace)
    print("[完成] run_6dof 全流程已执行完毕")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        return run_run6dof_sequence()
    user_text = " ".join(sys.argv[1:])
    answer = run_once(user_text)
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
