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


SYSTEM_PROMPT = """You are an intelligent agent navigating a 2D grid world.
Your goal is to accomplish the given mission as efficiently as possible.

## World Rules
- The grid uses (x, y) coordinates. x increases eastward, y increases southward.
- Symbols: @ = you, # = wall, G = goal, K = key, D = door (locked), C = chest
- Empty tile symbols: '.' = unvisited,  '·' = already visited
- Doors block movement unless you carry a key. Walking into a door with a key auto-unlocks it.
- You can only pick up items you are standing ON (use pick_up when on the same tile).

## Available Actions
- move_north / move_south / move_east / move_west  — move one tile
- pick_up                                           — pick up item at your position
- look                                              — inspect all 4 neighbours (costs a step, use sparingly)
- wait                                              — do nothing (avoid unless stuck)

## Response Format
You MUST reply with ONLY a JSON object — no prose, no markdown fences:
{
  "reasoning": "concise explanation of why you chose this action",
  "action": "<one of the action names above>"
}

## Strategy
- Read the ascii_map carefully: '·' cells are already explored, prefer '.' cells.
- Follow the goal_hint — it tells you what to find next and which direction.
- Pick up items before you need them (key before door).
- Avoid revisiting '·' cells unless backtracking is truly necessary.
- Never waste steps with 'look' if the map already shows what's nearby.
"""


def build_prompt(observation: dict, mission: str, history: list[dict]) -> list[dict]:
    """
    Construct the full message list for the LLM.
    history: list of {role, content} dicts (last N turns).

    Note: visited_cells list is intentionally omitted — the ascii_map already
    encodes visited state via '·' vs '.', saving significant tokens.
    """
    obs_text = f"""
## Mission
{mission}

## Current State
- Position: ({observation['position']['x']}, {observation['position']['y']})
- World size: {observation['world_size']['width']}×{observation['world_size']['height']}
- Inventory: {observation['inventory'] if observation['inventory'] else 'empty'}
- Steps taken: {observation['steps_taken']}
- {observation['goal_hint']}

## Adjacent tiles
{json.dumps(observation['neighbors'], indent=2)}

## Recent events
{chr(10).join(observation['recent_messages']) if observation['recent_messages'] else 'None yet'}

## Map  (@ = you | # = wall | · = visited | . = unvisited)
```
{observation['ascii_map']}
```

What is your next action?
""".strip()

    messages = list(history)  # copy
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
        # Last-ditch: look for any valid action keyword in the text
        for a in sorted(VALID_ACTIONS, key=len, reverse=True):  # longest first to avoid 'east' in 'move_east'
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
                "max_tokens": 512,   # FIX: increased from 256 — reasoning can be verbose
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

        # Append to history so the LLM sees conversational context
        self.history.append({"role": "user", "content": messages[-1]["content"]})
        self.history.append({"role": "assistant", "content": raw})

        return action, reasoning, raw