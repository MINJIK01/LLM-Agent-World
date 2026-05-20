"""
Agent Harness
The critical interface: translates world observations → LLM prompt,
parses LLM response → structured action, feeds result back.

Key feature: Reflection loop — when the agent gets stuck (repeated
positions, looping actions, or wall-bumping), a separate LLM call
diagnoses the failure and injects a corrective strategy into history.
"""
import json
import re
import httpx
from collections import Counter


# ── System prompts ───────────────────────────────────────────────────────────

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

REFLECTION_SYSTEM_PROMPT = """You are a self-diagnostic module for a robot agent.
The robot has become stuck. Your job is to analyse what went wrong and produce
a concrete corrective strategy — one or two sentences the robot can act on immediately.

Be specific: name directions to try, objects to seek, or patterns to avoid.
Do NOT suggest actions the robot has already tried repeatedly without success.
Reply with ONLY a JSON object:
{
  "diagnosis": "one sentence explaining why the robot is stuck",
  "strategy": "one or two sentences of concrete corrective action"
}
"""


# ── Stuck detection ──────────────────────────────────────────────────────────

class StuckDetector:
    """
    Tracks recent agent behaviour and fires when the agent is stuck.

    Stuck conditions (any one triggers reflection):
      1. Position loop   — same position visited ≥ POSITION_REPEAT times in last N steps
      2. Action loop     — same action repeated ≥ ACTION_REPEAT times consecutively
      3. Wall bumping    — "Blocked" result ≥ BUMP_THRESHOLD times in last N steps
    """

    WINDOW        = 8   # steps to look back
    POSITION_REPEAT = 3 # same pos this many times in window → stuck
    ACTION_REPEAT   = 4 # same action this many times in a row → stuck
    BUMP_THRESHOLD  = 3 # this many "Blocked" results in window → stuck

    def __init__(self):
        self.recent_positions: list[tuple] = []
        self.recent_actions:   list[str]   = []
        self.recent_results:   list[str]   = []
        self.reflection_count: int         = 0

    def record(self, pos: tuple, action: str, result: str):
        self.recent_positions.append(pos)
        self.recent_actions.append(action)
        self.recent_results.append(result)
        # Keep only the last WINDOW entries
        self.recent_positions = self.recent_positions[-self.WINDOW:]
        self.recent_actions   = self.recent_actions[-self.WINDOW:]
        self.recent_results   = self.recent_results[-self.WINDOW:]

    def is_stuck(self) -> tuple[bool, str]:
        """Returns (stuck, reason_string)."""
        if len(self.recent_actions) < self.WINDOW // 2:
            return False, ""  # not enough data yet

        # 1. Position loop
        pos_counts = Counter(self.recent_positions)
        most_common_pos, count = pos_counts.most_common(1)[0]
        if count >= self.POSITION_REPEAT:
            return True, (
                f"Position loop detected: visited {most_common_pos} "
                f"{count} times in the last {len(self.recent_positions)} steps."
            )

        # 2. Action loop — same action AND not making progress (position not changing)
        if len(self.recent_actions) >= self.ACTION_REPEAT:
            tail_actions = self.recent_actions[-self.ACTION_REPEAT:]
            tail_positions = self.recent_positions[-self.ACTION_REPEAT:]
            if len(set(tail_actions)) == 1 and len(set(tail_positions)) <= 2:
                return True, (
                    f"Action loop detected: '{tail_actions[0]}' repeated "
                    f"{self.ACTION_REPEAT} times without meaningful progress."
                )

        # 3. Wall bumping
        bump_count = sum(1 for r in self.recent_results if "Blocked" in r)
        if bump_count >= self.BUMP_THRESHOLD:
            return True, (
                f"Wall-bumping detected: {bump_count} blocked moves "
                f"in the last {len(self.recent_results)} steps."
            )

        return False, ""

    def reset_window(self):
        """Call after a successful reflection to give the agent a fresh start."""
        self.recent_positions.clear()
        self.recent_actions.clear()
        self.recent_results.clear()
        self.reflection_count += 1


# ── LLM calls ────────────────────────────────────────────────────────────────

async def call_llm(messages: list[dict], api_key: str,
                   system: str = SYSTEM_PROMPT, max_tokens: int = 512) -> str:
    """Call Claude and return raw text response."""
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
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
            },
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]


# ── Prompt builders ───────────────────────────────────────────────────────────

def build_prompt(observation: dict, mission: str, history: list[dict]) -> list[dict]:
    """Construct the full message list for the action LLM."""
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


def build_reflection_prompt(
    observation: dict,
    mission: str,
    history: list[dict],
    stuck_reason: str,
) -> list[dict]:
    """
    Build a prompt for the reflection LLM call.
    Summarises recent history and the stuck condition for diagnosis.
    """
    # Extract last few (action, result) pairs from history for the reflection context
    action_summary = []
    for i in range(0, len(history) - 1, 2):
        if i + 1 < len(history):
            # history alternates user/assistant; extract action from assistant messages
            assistant_msg = history[i + 1].get("content", "")
            try:
                parsed = json.loads(re.sub(r"```[a-z]*\n?", "", assistant_msg).strip("`").strip())
                action_summary.append(f"  • {parsed.get('action', '?')}: {parsed.get('reasoning', '')[:80]}")
            except Exception:
                pass

    recent_actions_text = "\n".join(action_summary[-6:]) if action_summary else "  (none recorded)"

    prompt_text = f"""
## Mission
{mission}

## Stuck condition
{stuck_reason}

## Agent's recent decisions
{recent_actions_text}

## Current position
({observation['position']['x']}, {observation['position']['y']})

## Current map
```
{observation['ascii_map']}
```

## Inventory
{observation['inventory'] if observation['inventory'] else 'empty'}

Diagnose why the agent is stuck and suggest a concrete corrective strategy.
""".strip()

    return [{"role": "user", "content": prompt_text}]


# ── Action parser ─────────────────────────────────────────────────────────────

def parse_action(raw: str) -> tuple[str, str]:
    """Extract (action, reasoning) from LLM response. Falls back gracefully."""
    VALID_ACTIONS = {
        "move_north", "move_south", "move_east", "move_west",
        "north", "south", "east", "west",
        "pick_up", "look", "wait"
    }
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    try:
        data = json.loads(raw)
        action = data.get("action", "wait").strip().lower()
        reasoning = data.get("reasoning", "")
        if action not in VALID_ACTIONS:
            return "wait", f"[parse fallback] invalid action '{action}'"
        return action, reasoning
    except json.JSONDecodeError:
        for a in sorted(VALID_ACTIONS, key=len, reverse=True):
            if a in raw.lower():
                return a, "[extracted from malformed response]"
        return "wait", "[could not parse response]"


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    LLM-driven agent with a reflection loop.

    Normal flow:     Observe → Decide → Act
    Reflection flow: Stuck detected → Reflect → inject strategy → Decide

    The reflection is a separate LLM call that diagnoses the failure and
    injects a corrective strategy message into the conversation history,
    so the action LLM immediately benefits from the self-critique.
    """

    MAX_REFLECTIONS = 3   # cap total reflections per run to avoid infinite loops

    def __init__(self, api_key: str, max_history: int = 10):
        self.api_key = api_key
        self.history: list[dict] = []
        self.max_history = max_history
        self.stuck_detector = StuckDetector()
        self.last_reflection: str = ""   # stored for logging

    async def reflect(self, observation: dict, mission: str, stuck_reason: str) -> str:
        """
        Run a separate LLM call to diagnose why the agent is stuck
        and produce a corrective strategy.
        Returns the strategy string.
        """
        messages = build_reflection_prompt(
            observation, mission,
            self.history[-self.max_history:],
            stuck_reason,
        )
        raw = await call_llm(
            messages,
            self.api_key,
            system=REFLECTION_SYSTEM_PROMPT,
            max_tokens=256,
        )
        raw_clean = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
        try:
            data = json.loads(raw_clean)
            diagnosis = data.get("diagnosis", "")
            strategy  = data.get("strategy", "")
            return f"[SELF-REFLECTION #{self.stuck_detector.reflection_count + 1}] " \
                   f"Diagnosis: {diagnosis} Strategy: {strategy}"
        except Exception:
            return f"[SELF-REFLECTION] {raw[:200]}"

    async def decide(
        self,
        observation: dict,
        mission: str,
        last_action: str = "",
        last_result: str = "",
    ) -> tuple[str, str, str, bool]:
        """
        Given current observation and mission, ask LLM for next action.

        Returns (action, reasoning, raw_llm_response, reflected).
        `reflected` is True if a reflection was triggered this step.
        """
        reflected = False
        self.last_reflection = ""

        # ── Stuck detection ───────────────────────────────────────────────────
        if last_action and last_result:
            self.stuck_detector.record(
                tuple(observation["position"].values()),
                last_action,
                last_result,
            )

        stuck, stuck_reason = self.stuck_detector.is_stuck()
        if stuck and self.stuck_detector.reflection_count < self.MAX_REFLECTIONS:
            # Run reflection LLM call
            strategy = await self.reflect(observation, mission, stuck_reason)
            self.last_reflection = strategy

            # Inject the strategy as a system-level hint into history
            # Using a fake assistant turn so the action LLM sees it as prior context
            self.history.append({
                "role": "user",
                "content": f"[System notice]: {stuck_reason}\n\nPlease reconsider your approach.",
            })
            self.history.append({
                "role": "assistant",
                "content": json.dumps({
                    "reasoning": strategy,
                    "action": "wait"   # placeholder — real action decided next
                })
            })
            self.stuck_detector.reset_window()
            reflected = True

        # ── Normal action decision ─────────────────────────────────────────────
        messages = build_prompt(observation, mission, self.history[-self.max_history:])
        raw = await call_llm(messages, self.api_key)
        action, reasoning = parse_action(raw)

        self.history.append({"role": "user",      "content": messages[-1]["content"]})
        self.history.append({"role": "assistant",  "content": raw})

        return action, reasoning, raw, reflected