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
                        choices=["reach_goal", "key_door", "exploration",
                                 "factory_delivery", "warehouse_sort", "hazard_navigate",
                                 "collab_delivery"])
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if not args.api_key:
        print(f"{RED}Error: provide --api-key or set ANTHROPIC_API_KEY{RESET}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Multi-agent mode ──────────────────────────────────────────────────────
    if args.scenario == "collab_delivery":
        from world.multi_agent import build_collab_scenario
        from agent.harness import MultiAgent

        ma_world, mission_a, mission_b = build_collab_scenario()
        multi_agent = MultiAgent(api_key=args.api_key)
        missions = {"A": mission_a, "B": mission_b}

        log_path = os.path.join(LOG_DIR, f"run_{timestamp}_collab_delivery.jsonl")
        def write_line(record: dict):
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"\n{BOLD}=== LLM Agent World — MULTI-AGENT ==={RESET}")
        print(f"Scenario : collab_delivery")
        print(f"Robot A  : {mission_a[:80]}...")
        print(f"Robot B  : {mission_b[:80]}...")
        print(f"Log      : {log_path}\n")

        write_line({
            "event": "init", "timestamp": timestamp,
            "scenario": "collab_delivery",
            "mission_a": mission_a, "mission_b": mission_b,
            "world_size": {"width": ma_world.world.width, "height": ma_world.world.height},
        })

        last_actions = {"A": "", "B": ""}
        last_results = {"A": "", "B": ""}

        for step in range(1, args.max_steps + 1):
            if ma_world.done:
                break

            robot_id = ma_world.turn
            print(f"{GRAY}[step {step}] Robot {robot_id} thinking…{RESET}", end="\r", flush=True)

            try:
                obs = ma_world.get_observation(robot_id)
                action, reasoning, _, reflected = await multi_agent.decide(
                    robot_id, obs, missions[robot_id],
                    last_actions[robot_id], last_results[robot_id],
                )
            except Exception as e:
                print(f"\n{RED}LLM error (Robot {robot_id}): {e}{RESET}")
                write_line({"event": "error", "step": step, "robot": robot_id, "message": str(e)})
                break

            result = ma_world.step(robot_id, action)
            last_actions[robot_id] = action
            last_results[robot_id] = result

            robot = ma_world.robots[robot_id]
            color = BLUE if robot_id == "A" else GREEN
            print(f"\r{BOLD}[step {step:02d}]{RESET} Robot {color}{robot_id}{RESET} "
                  f"{BLUE}{action:<15}{RESET} {GRAY}{reasoning[:60]}{RESET}")
            print(f"         → {result}")
            if robot.inventory:
                print(f"         inventory: {robot.inventory}")
            print()

            write_line({
                "event": "step", "step": step,
                "robot": robot_id,
                "action": action, "reasoning": reasoning, "result": result,
                "pos": {"x": robot.pos[0], "y": robot.pos[1]},
                "inventory": list(robot.inventory),
                "done": ma_world.done, "goal_reached": ma_world.goal_reached,
                "reflected": reflected,
            })

            await asyncio.sleep(args.delay)

        write_line({
            "event": "done",
            "total_steps": ma_world.total_steps,
            "goal_reached": ma_world.goal_reached,
        })
        if ma_world.goal_reached:
            print(f"{GREEN}{BOLD}🎉 Mission complete in {ma_world.total_steps} steps!{RESET}")
        else:
            print(f"{RED}Stopped after {ma_world.total_steps} steps.{RESET}")
        print(f"{GRAY}Log saved → {log_path}{RESET}")
        return

    # ── Single-agent mode (existing) ──────────────────────────────────────────
    world, mission = build_scenario(args.scenario)
    agent = Agent(api_key=args.api_key)

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
            action, reasoning, _, reflected, llm_stuck, llm_stuck_reason = await agent.decide(
                obs, mission, last_action, last_result
            )
        except Exception as e:
            print(f"\n{RED}LLM error: {e}{RESET}")
            write_line({"event": "error", "step": step, "message": str(e)})
            break

        if llm_stuck and not reflected:
            print(f"\r{AMBER}[self: stuck]{RESET} {llm_stuck_reason[:80]}")

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
            "llm_stuck": llm_stuck,
            "llm_stuck_reason": llm_stuck_reason if llm_stuck else "",
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