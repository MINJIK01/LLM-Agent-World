#!/usr/bin/env python3
"""
CLI runner — run the agent in your terminal (no web UI needed).
Usage: python run_cli.py --scenario key_door --api-key sk-ant-...
Logs saved to: logs/run_<timestamp>_<scenario>.jsonl
"""
import asyncio
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from world.grid import build_scenario
from agent.harness import Agent

RESET = "\033[0m"
BOLD  = "\033[1m"
GREEN = "\033[92m"
AMBER = "\033[93m"
BLUE  = "\033[94m"
RED   = "\033[91m"
GRAY  = "\033[90m"
CYAN  = "\033[96m"
MAGENTA = "\033[95m"

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def render_pretty(world) -> str:
    TILE_MAP = {
        "@": f"{BOLD}{BLUE}@{RESET}",
        "#": f"{GRAY}#{RESET}",
        "·": f"{GRAY}·{RESET}",
        "G": f"{GREEN}G{RESET}",
        "K": f"{AMBER}K{RESET}",
        "D": f"{RED}D{RESET}",
        "T": f"{RED}T{RESET}",
        "X": f"{BOLD}{RED}X{RESET}",
        "C": f"{CYAN}C{RESET}",
        "P": f"{MAGENTA}P{RESET}",
        "O": f"{MAGENTA}O{RESET}",
        "A": f"{GREEN}A{RESET}",
        "B": f"{GREEN}B{RESET}",
        ".": f"{GRAY}.{RESET}",
    }
    lines = world.render_ascii().split("\n")
    return "\n".join(" ".join(TILE_MAP.get(ch, ch) for ch in line) for line in lines)


async def main():
    parser = argparse.ArgumentParser(description="LLM Agent World — CLI")
    parser.add_argument("--scenario", default="key_door",
                        choices=["reach_goal", "key_door", "exploration", "factory_delivery", "warehouse_sort", "hazard_navigate"])
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if not args.api_key:
        print(f"{RED}Error: provide --api-key or set ANTHROPIC_API_KEY{RESET}")
        sys.exit(1)

    world, mission = build_scenario(args.scenario)
    agent = Agent(api_key=args.api_key)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_{timestamp}_{args.scenario}.jsonl")

    def write_line(record: dict):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{BOLD}=== LLM Agent World ==={RESET}")
    print(f"Scenario : {args.scenario}")
    print(f"Mission  : {mission}")
    print(f"Log      : {log_path}\n")
    print(render_pretty(world))
    print()

    write_line({
        "event": "init",
        "timestamp": timestamp,
        "scenario": args.scenario,
        "mission": mission,
        "world_size": {"width": world.width, "height": world.height},
    })

    last_action = ""
    last_result = ""

    for step in range(1, args.max_steps + 1):
        if world.done:
            break

        print(f"{GRAY}[step {step}] thinking…{RESET}", end="\r", flush=True)
        try:
            obs = world.get_observation()
            action, reasoning, _, reflected = await agent.decide(
                obs, mission, last_action, last_result
            )
        except Exception as e:
            print(f"\n{RED}LLM error: {e}{RESET}")
            write_line({"event": "error", "step": step, "message": str(e)})
            break

        if reflected:
            print(f"\r{BOLD}{RED}[reflection]{RESET} {agent.last_reflection}")
            print()
            write_line({
                "event": "reflection",
                "step": step,
                "reflection": agent.last_reflection,
            })

        result = world.step(action)
        last_action = action
        last_result = result

        write_line({
            "event": "step",
            "step": step,
            "action": action,
            "reasoning": reasoning,
            "result": result,
            "pos": {"x": world.agent_pos[0], "y": world.agent_pos[1]},
            "inventory": list(world.inventory),
            "done": world.done,
            "goal_reached": world.goal_reached,
            "reflected": reflected,
        })

        print(f"\r{BOLD}[step {step:02d}]{RESET} {BLUE}{action:<15}{RESET} {GRAY}{reasoning}{RESET}")
        print(f"         → {result}")
        if world.inventory:
            print(f"         inventory: {world.inventory}")
        print()
        print(render_pretty(world))
        print()

        await asyncio.sleep(args.delay)

    write_line({
        "event": "done",
        "total_steps": world.steps,
        "goal_reached": world.goal_reached,
    })

    if world.goal_reached:
        print(f"{GREEN}{BOLD}🎉 Mission complete in {world.steps} steps!{RESET}")
    else:
        print(f"{RED}Agent stopped after {world.steps} steps without reaching the goal.{RESET}")

    print(f"{GRAY}Log saved → {log_path}{RESET}")


if __name__ == "__main__":
    asyncio.run(main())