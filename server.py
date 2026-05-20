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
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(__file__))
from world.grid import build_scenario
from agent.harness import Agent

app = FastAPI(title="LLM Agent World")

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class RunRequest(BaseModel):
    scenario: str = "key_door"
    api_key: str
    max_steps: int = 30


@app.post("/run")
async def run_agent(req: RunRequest):
    """Stream agent steps as SSE, and write a JSONL log file."""

    async def event_stream():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"run_{timestamp}_{req.scenario}.jsonl")

        def write_line(record: dict):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        try:
            world, mission = build_scenario(req.scenario)
            agent = Agent(api_key=req.api_key)

            # Log run header
            write_line({
                "event": "init",
                "timestamp": timestamp,
                "scenario": req.scenario,
                "mission": mission,
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
                    action, reasoning, raw, reflected = await agent.decide(
                        obs, mission, last_action, last_result
                    )
                except Exception as e:
                    write_line({"event": "error", "step": step + 1, "message": str(e)})
                    yield _sse("error", {"message": str(e)})
                    break

                if reflected:
                    write_line({
                        "event": "reflection",
                        "step": step + 1,
                        "reflection": agent.last_reflection,
                    })
                    yield _sse("reflection", {
                        "step": step + 1,
                        "reflection": agent.last_reflection,
                    })

                result_msg = world.step(action)
                last_action = action
                last_result = result_msg

                # Write one JSONL line per step
                write_line({
                    "event": "step",
                    "step": step + 1,
                    "action": action,
                    "reasoning": reasoning,
                    "result": result_msg,
                    "pos": {"x": world.agent_pos[0], "y": world.agent_pos[1]},
                    "inventory": list(world.inventory),
                    "done": world.done,
                    "goal_reached": world.goal_reached,
                    "reflected": reflected,
                })

                yield _sse("step", {
                    "step": step + 1,
                    "action": action,
                    "reasoning": reasoning,
                    "result": result_msg,
                    "world": _world_state(world),
                    "done": world.done,
                    "goal_reached": world.goal_reached,
                })

                await asyncio.sleep(0.1)

            # Log summary
            write_line({
                "event": "done",
                "total_steps": world.steps,
                "goal_reached": world.goal_reached,
            })

            yield _sse("done", {
                "steps": world.steps,
                "goal_reached": world.goal_reached,
                "log_file": os.path.basename(log_path),
            })

        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _world_state(world) -> dict:
    return {
        "width": world.width,
        "height": world.height,
        "agent": {"x": world.agent_pos[0], "y": world.agent_pos[1]},
        "walls": [{"x": x, "y": y} for x, y in world.walls],
        "objects": [{"x": x, "y": y, "type": t} for (x, y), t in world.objects.items()],
        "inventory": world.inventory,
        "steps": world.steps,
        "done": world.done,
        "goal_reached": world.goal_reached,
        "ascii": world.render_ascii(),
    }


HTML = open(os.path.join(os.path.dirname(__file__), "static", "index.html")).read()

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML