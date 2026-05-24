"""
FastAPI server
- Serves the web UI
- POST /run  → streams agent steps as Server-Sent Events (SSE)
- Saves each run as logs/run_<timestamp>_<scenario>.jsonl
"""
import asyncio
import json
import os
import sys
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from world.grid import build_scenario
from world.multi_agent import build_collab_scenario
from agent.harness import Agent, MultiAgent

app = FastAPI(title="LLM Agent World")

# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class RunRequest(BaseModel):
    scenario: str = "key_door"
    api_key: str
    max_steps: int = 40


@app.post("/run")
async def run_agent(req: RunRequest):
    """Stream agent steps as SSE, and write a JSONL log file."""

    if req.scenario == "collab_delivery":
        return StreamingResponse(
            _multi_agent_stream(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        _single_agent_stream(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _single_agent_stream(req: RunRequest):
    """Original single-agent SSE stream."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_{timestamp}_{req.scenario}.jsonl")

    def write_line(record: dict):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        world, mission = build_scenario(req.scenario)
        agent = Agent(api_key=req.api_key)

        write_line({
            "event": "init", "timestamp": timestamp,
            "scenario": req.scenario, "mission": mission,
            "world_size": {"width": world.width, "height": world.height},
        })

        yield _sse("init", {
            "mission": mission,
            "scenario": req.scenario,
            "world": _world_state(world),
            "log_file": os.path.basename(log_path),
        })

        last_action = ""
        last_result = ""

        for step in range(req.max_steps):
            if world.done:
                break

            obs = world.get_observation()
            yield _sse("thinking", {"step": step + 1})

            try:
                action, reasoning, raw, reflected, llm_stuck, llm_stuck_reason = \
                    await agent.decide(obs, mission, last_action, last_result)
            except Exception as e:
                write_line({"event": "error", "step": step + 1, "message": str(e)})
                yield _sse("error", {"message": str(e)})
                break

            if reflected:
                write_line({"event": "reflection", "step": step + 1, "reflection": agent.last_reflection})
                yield _sse("reflection", {"step": step + 1, "reflection": agent.last_reflection})

            result_msg = world.step(action)
            last_action = action
            last_result = result_msg

            write_line({
                "event": "step", "step": step + 1,
                "action": action, "reasoning": reasoning, "result": result_msg,
                "pos": {"x": world.agent_pos[0], "y": world.agent_pos[1]},
                "inventory": list(world.inventory),
                "done": world.done, "goal_reached": world.goal_reached,
                "reflected": reflected, "llm_stuck": llm_stuck,
                "llm_stuck_reason": llm_stuck_reason if llm_stuck else "",
            })

            yield _sse("step", {
                "step": step + 1, "action": action,
                "reasoning": reasoning, "result": result_msg,
                "world": _world_state(world),
                "done": world.done, "goal_reached": world.goal_reached,
            })

            await asyncio.sleep(0.1)

        write_line({"event": "done", "total_steps": world.steps, "goal_reached": world.goal_reached})
        yield _sse("done", {
            "steps": world.steps,
            "goal_reached": world.goal_reached,
            "log_file": os.path.basename(log_path),
        })

    except Exception as e:
        yield _sse("error", {"message": str(e)})


async def _multi_agent_stream(req: RunRequest):
    """Multi-agent SSE stream for collab_delivery."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_{timestamp}_collab_delivery.jsonl")

    def write_line(record: dict):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        ma_world, mission_a, mission_b = build_collab_scenario()
        multi_agent = MultiAgent(api_key=req.api_key)
        missions = {"A": mission_a, "B": mission_b}

        write_line({
            "event": "init", "timestamp": timestamp,
            "scenario": "collab_delivery",
            "mission_a": mission_a, "mission_b": mission_b,
            "world_size": {"width": ma_world.world.width, "height": ma_world.world.height},
        })

        yield _sse("init", {
            "mission": f"[Robot A] {mission_a[:80]}… | [Robot B] {mission_b[:80]}…",
            "scenario": "collab_delivery",
            "world": _multi_world_state(ma_world, "A"),
            "log_file": os.path.basename(log_path),
        })

        last_actions = {"A": "", "B": ""}
        last_results = {"A": "", "B": ""}

        for step in range(req.max_steps):
            if ma_world.done:
                break

            robot_id = ma_world.turn
            yield _sse("thinking", {"step": step + 1, "robot": robot_id})

            try:
                obs = ma_world.get_observation(robot_id)
                action, reasoning, _, reflected = await multi_agent.decide(
                    robot_id, obs, missions[robot_id],
                    last_actions[robot_id], last_results[robot_id],
                )
            except Exception as e:
                write_line({"event": "error", "step": step + 1, "robot": robot_id, "message": str(e)})
                yield _sse("error", {"message": str(e)})
                break

            if reflected:
                agent = multi_agent.agents[robot_id]
                write_line({"event": "reflection", "step": step + 1, "robot": robot_id, "reflection": agent.last_reflection})
                yield _sse("reflection", {"step": step + 1, "robot": robot_id, "reflection": agent.last_reflection})

            result_msg = ma_world.step(robot_id, action)
            last_actions[robot_id] = action
            last_results[robot_id] = result_msg

            robot = ma_world.robots[robot_id]

            write_line({
                "event": "step", "step": step + 1, "robot": robot_id,
                "action": action, "reasoning": reasoning, "result": result_msg,
                "pos": {"x": robot.pos[0], "y": robot.pos[1]},
                "inventory": list(robot.inventory),
                "done": ma_world.done, "goal_reached": ma_world.goal_reached,
                "reflected": reflected,
            })

            yield _sse("step", {
                "step": step + 1, "robot": robot_id,
                "action": action, "reasoning": reasoning, "result": result_msg,
                "world": _multi_world_state(ma_world, robot_id),
                "done": ma_world.done, "goal_reached": ma_world.goal_reached,
            })

            await asyncio.sleep(0.1)

        write_line({"event": "done", "total_steps": ma_world.total_steps, "goal_reached": ma_world.goal_reached})
        yield _sse("done", {
            "steps": ma_world.total_steps,
            "goal_reached": ma_world.goal_reached,
            "log_file": os.path.basename(log_path),
        })

    except Exception as e:
        yield _sse("error", {"message": str(e)})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _world_state(world) -> dict:
    """Single-agent world state for Web UI."""
    world._update_seen()
    return {
        "width": world.width,
        "height": world.height,
        "agent": {"x": world.agent_pos[0], "y": world.agent_pos[1]},
        "walls": [{"x": x, "y": y} for x, y in world.walls],
        "objects": [{"x": x, "y": y, "type": t} for (x, y), t in world.objects.items()
                    if world.vision_radius is None or (x, y) in world.seen],
        "inventory": world.inventory,
        "steps": world.steps,
        "done": world.done,
        "goal_reached": world.goal_reached,
        "ascii": world.render_ascii(),
        "vision_radius": world.vision_radius,
        "seen": [{"x": x, "y": y} for x, y in world.seen] if world.vision_radius is not None else None,
        "visited": [{"x": x, "y": y} for x, y in world.visited],
    }


def _multi_world_state(ma_world, pov_robot_id: str) -> dict:
    """Multi-agent world state for Web UI. A is always 'agent', B is always 'agent_b'."""
    w = ma_world.world
    robot_a = ma_world.robots["A"]
    robot_b = ma_world.robots["B"]
    pov = ma_world.robots[pov_robot_id]
    ma_world._update_seen(pov)

    return {
        "width": w.width,
        "height": w.height,
        "agent":   {"x": robot_a.pos[0], "y": robot_a.pos[1]},   # always Robot A
        "agent_b": {"x": robot_b.pos[0], "y": robot_b.pos[1]},   # always Robot B
        "walls": [{"x": x, "y": y} for x, y in w.walls],
        "objects": [{"x": x, "y": y, "type": t} for (x, y), t in w.objects.items()
                    if w.vision_radius is None or (x, y) in pov.seen],
        "inventory": pov.inventory,
        "steps": ma_world.total_steps,
        "done": ma_world.done,
        "goal_reached": ma_world.goal_reached,
        "ascii": ma_world._render_ascii(pov),
        "vision_radius": w.vision_radius,
        "seen": [{"x": x, "y": y} for x, y in pov.seen] if w.vision_radius is not None else None,
        "visited": [{"x": x, "y": y} for x, y in pov.visited],
        "multi_agent": True,
        "turn": ma_world.turn,
    }


HTML = open(os.path.join(os.path.dirname(__file__), "static", "index.html")).read()

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML