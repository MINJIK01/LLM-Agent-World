"""
Agent Harness
The critical interface: translates world observations → LLM prompt,
parses LLM response → structured action, feeds result back.
"""
import json
import re
import os
import httpx
from typing import Optional


SYSTEM_PROMPT = """You are an intelligent robot agent navigating a 2D grid world.
Your goal is to accomplish the given mission as efficiently as possible.

## World Rules
- The grid uses (x, y) coordinates. x increases eastward, y increases southward.
- Symbols:
    @  = you (the robot)
    #  = wall
    ·  = visited empty tile
    .  = unvisited empty tile (within sensor range)
    ?  = unknown tile (outside sensor range — never seen yet)
    G  = goal
    K  = key
    D  = door (locked — needs key)
    T  = gate (locked — needs key)
    X  = hazard zone (impassable)
    P  = part (pick up and deliver)
    O  = box  (pick up and deliver)
    A  = assembly line (delivery target)
    B  = depot (delivery target)
    C  = chest (pick up)
- Doors/gates block movement unless you carry a key. Walking into one with a key auto-unlocks it.
- You can only pick up items you are standing ON.
- Hazard zones (X) are permanently impassable — plan a route around them.
- Unknown tiles (?) may contain anything — move toward them to reveal what's there.

## Available Actions
- move_north / move_south / move_east / move_west  — move one tile
- pick_up                                           — pick up item at your position
- look                                              — inspect all 4 neighbours (costs a step, use sparingly)
- wait                                              — do nothing (avoid unless truly stuck)

## Response Format
You MUST reply with ONLY a JSON object — no prose, no markdown fences:
{
  "reasoning": "concise explanation of why you chose this action",
  "action": "<one of the action names above>"
}

## Strategy
- Read the ascii_map carefully: '·' = explored, '.' = unexplored. Prefer unexplored tiles.
- Follow the goal_hint — it always tells you the next priority target and direction.
- For delivery tasks: pick up the item FIRST, then navigate to the target.
- Plan routes around hazards (X) — you cannot pass through them.
- Avoid revisiting '·' cells unless backtracking is truly necessary.
- Never waste steps with 'look' if the map already shows what's nearby.
"""


def build_prompt(observation: dict, mission: str, history: list[dict]) -> list[dict]:
    """
    Construct the full message list for the LLM.
    history: list of {role, content} dicts (last N turns).
    """
    # Delivery progress line (only shown for delivery scenarios)
    delivery_line = ""
    if "deliveries_remaining" in observation:
        done  = observation["deliveries_done"]
        total = done + observation["deliveries_remaining"]
        delivery_line = f"- Deliveries: {done}/{total} complete\n"

    obs_text = f"""
## Mission
{mission}

## Current State
- Position: ({observation['position']['x']}, {observation['position']['y']})
- World size: {observation['world_size']['width']}×{observation['world_size']['height']}
- Inventory: {observation['inventory'] if observation['inventory'] else 'empty'}
- Steps taken: {observation['steps_taken']}
{delivery_line}- {observation['goal_hint']}

## Adjacent tiles
{json.dumps(observation['neighbors'], indent=2)}

## Recent events
{chr(10).join(observation['recent_messages']) if observation['recent_messages'] else 'None yet'}

## Map  (@ = you | # = wall | X = hazard | · = visited | . = unvisited | ? = unknown)
```
{observation['ascii_map']}
```

What is your next action?
""".strip()

    messages = list(history)
    messages.append({"role": "user", "content": obs_text})
    return messages


def parse_action(raw: str) -> tuple[str, str]:
    """
    Extract action and reasoning from LLM response.
    Returns (action, reasoning). Falls back gracefully.
    """
    VALID_ACTIONS = {
        "move_north", "move_south", "move_east", "move_west",
        "north", "south", "east", "west",
        "pick_up", "look", "wait"
    }
    # Strip markdown fences if model wraps in ```
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    try:
        data = json.loads(raw)
        action = data.get("action", "wait").strip().lower()
        reasoning = data.get("reasoning", "")
        if action not in VALID_ACTIONS:
            return "wait", f"[parse fallback] invalid action '{action}'"
        return action, reasoning
    except json.JSONDecodeError:
        # Last-ditch: longest-first keyword match to avoid 'east' matching inside 'move_east'
        for a in sorted(VALID_ACTIONS, key=len, reverse=True):
            if a in raw.lower():
                return a, "[extracted from malformed response]"
        return "wait", "[could not parse response]"


async def call_llm(messages: list[dict], api_key: str) -> str:
    """Call Claude claude-sonnet-4-5 and return raw text response."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 512,
                "system": SYSTEM_PROMPT,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]


class Agent:
    def __init__(self, api_key: str, max_history: int = 10):
        self.api_key = api_key
        self.history: list[dict] = []
        self.max_history = max_history

    async def decide(self, observation: dict, mission: str) -> tuple[str, str, str]:
        """
        Given current observation and mission, ask LLM for next action.
        Returns (action, reasoning, raw_llm_response).
        """
        messages = build_prompt(observation, mission, self.history[-self.max_history:])
        raw = await call_llm(messages, self.api_key)
        action, reasoning = parse_action(raw)

        self.history.append({"role": "user", "content": messages[-1]["content"]})
        self.history.append({"role": "assistant", "content": raw})

        return action, reasoning, raw