"""
Agent Harness
The critical interface: translates world observations → LLM prompt,
parses LLM response → structured action, feeds result back.

Reflection loop (two-layer):
  1. LLM self-assessment  — the action LLM flags stuck=true in its own response
  2. Rule-based fallback  — StuckDetector catches cases the LLM misses

When either layer fires, a separate reflection LLM call diagnoses the
failure and injects a corrective strategy into conversation history.
"""
import json
import re
import httpx
from collections import Counter


# ── System prompts ────────────────────────────────────────────────────────────

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
You MUST reply with ONLY a valid JSON object — no prose, no markdown fences, no extra text before or after:
{
  "reasoning": "concise explanation of why you chose this action",
  "action": "<one of the action names above>",
  "stuck": <true if you believe you are stuck or looping, false otherwise>,
  "stuck_reason": "<if stuck=true: one sentence describing why, else empty string>"
}

IMPORTANT: Your entire response must be parseable as JSON. Do not write anything outside the JSON object.

Set stuck=true when you notice any of these:
- You have visited the same tile multiple times without progress
- You keep hitting walls in the same direction
- You cannot see a path to your current target and need a new strategy

## Strategy
- Read the ascii_map carefully: '·' = explored, '.' = unexplored. Prefer unexplored tiles.
- Check adjacent tiles for exit counts — "empty (1 exit ⚠ DEAD-END)" means that tile leads nowhere new. AVOID moving into dead-ends unless it contains your target.
- Follow the goal_hint — it always tells you the next priority target and direction.
- Follow the explore_hint — it tells you the nearest unseen tile and exactly which direction to step first. Use it when you don't know where to go next.
- For delivery tasks: pick up the item FIRST, then navigate to the target.
- Plan routes around hazards (X) — you cannot pass through them.
- Avoid revisiting '·' cells unless backtracking is truly necessary.
- Never waste steps with 'look' if the map already shows what's nearby.

## Wall-following rule (use when stuck or oscillating)
When surrounded by visited tiles or unable to find a new path, use the LEFT-HAND RULE.
Given your current facing direction, priorities are:
- facing north → try: west first, then north, then east, then south
- facing south → try: east first, then south, then west, then north
- facing east  → try: north first, then east, then south, then west
- facing west  → try: south first, then west, then north, then east
Always pick the first unblocked direction in that priority order.
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


# ── Rule-based fallback stuck detector ───────────────────────────────────────

class StuckDetector:
    """
    Safety-net stuck detector — catches cases the LLM fails to self-report.

    Stuck conditions (any one triggers):
      1. Position loop — same position ≥ POSITION_REPEAT times in last WINDOW steps
      2. Action loop   — same action repeated ≥ ACTION_REPEAT times with no positional progress
      3. Wall bumping  — "Blocked" result ≥ BUMP_THRESHOLD times in last WINDOW steps
    """

    WINDOW          = 8
    POSITION_REPEAT = 3
    ACTION_REPEAT   = 4
    BUMP_THRESHOLD  = 3

    def __init__(self):
        self.recent_positions: list[tuple] = []
        self.recent_actions:   list[str]   = []
        self.recent_results:   list[str]   = []
        self.reflection_count: int         = 0

    def record(self, pos: tuple, action: str, result: str):
        self.recent_positions.append(pos)
        self.recent_actions.append(action)
        self.recent_results.append(result)
        self.recent_positions = self.recent_positions[-self.WINDOW:]
        self.recent_actions   = self.recent_actions[-self.WINDOW:]
        self.recent_results   = self.recent_results[-self.WINDOW:]

    def is_stuck(self) -> tuple[bool, str]:
        """Returns (stuck, reason). Only fires if LLM did not self-report."""
        if len(self.recent_actions) < self.WINDOW // 2:
            return False, ""

        # 1. Position loop
        pos_counts = Counter(self.recent_positions)
        most_common_pos, count = pos_counts.most_common(1)[0]
        if count >= self.POSITION_REPEAT:
            return True, (
                f"[fallback] Position loop: visited {most_common_pos} "
                f"{count}× in last {len(self.recent_positions)} steps."
            )

        # 2. Action loop with no progress
        if len(self.recent_actions) >= self.ACTION_REPEAT:
            tail_actions    = self.recent_actions[-self.ACTION_REPEAT:]
            tail_positions  = self.recent_positions[-self.ACTION_REPEAT:]
            if len(set(tail_actions)) == 1 and len(set(tail_positions)) <= 2:
                return True, (
                    f"[fallback] Action loop: '{tail_actions[0]}' repeated "
                    f"{self.ACTION_REPEAT}× without positional progress."
                )

        # 3. Wall bumping
        bump_count = sum(1 for r in self.recent_results if "Blocked" in r)
        if bump_count >= self.BUMP_THRESHOLD:
            return True, (
                f"[fallback] Wall-bumping: {bump_count} blocked moves "
                f"in last {len(self.recent_results)} steps."
            )

        return False, ""

    def reset_window(self):
        self.recent_positions.clear()
        self.recent_actions.clear()
        self.recent_results.clear()
        self.reflection_count += 1


# ── LLM calls ────────────────────────────────────────────────────────────────

async def call_llm(messages: list[dict], api_key: str,
                   system: str = SYSTEM_PROMPT, max_tokens: int = 512) -> str:
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
- Facing: {observation.get('facing', 'unknown')} (determined by your last move)
- World size: {observation['world_size']['width']}×{observation['world_size']['height']}
- Inventory: {observation['inventory'] if observation['inventory'] else 'empty'}
- Steps taken: {observation['steps_taken']}
{delivery_line}- {observation['goal_hint']}
{f"- Explore hint: {observation['explore_hint']}" if observation.get('explore_hint') else ""}

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
    """Build the prompt for the dedicated reflection LLM call."""
    action_summary = []
    for i in range(0, len(history) - 1, 2):
        if i + 1 < len(history):
            assistant_msg = history[i + 1].get("content", "")
            try:
                parsed = json.loads(
                    re.sub(r"```[a-z]*\n?", "", assistant_msg).strip("`").strip()
                )
                action_summary.append(
                    f"  • {parsed.get('action', '?')}: {parsed.get('reasoning', '')[:80]}"
                )
            except Exception:
                pass

    recent_actions_text = "\n".join(action_summary[-6:]) or "  (none recorded)"

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


# ── Response parser ───────────────────────────────────────────────────────────

def parse_response(raw: str) -> tuple[str, str, bool, str]:
    """
    Parse the LLM response into (action, reasoning, stuck, stuck_reason).
    Falls back gracefully on malformed output.
    """
    VALID_ACTIONS = {
        "move_north", "move_south", "move_east", "move_west",
        "north", "south", "east", "west",
        "pick_up", "look", "wait",
    }
    cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
    try:
        data = json.loads(cleaned)
        action      = data.get("action", "wait").strip().lower()
        reasoning   = data.get("reasoning", "")
        stuck       = bool(data.get("stuck", False))
        stuck_reason = data.get("stuck_reason", "")
        if action not in VALID_ACTIONS:
            return "wait", f"[parse fallback] invalid action '{action}'", stuck, stuck_reason
        return action, reasoning, stuck, stuck_reason
    except json.JSONDecodeError:
        # Last-ditch keyword extraction
        for a in sorted(VALID_ACTIONS, key=len, reverse=True):
            if a in cleaned.lower():
                return a, "[extracted from malformed response]", False, ""
        return "wait", "[could not parse response]", False, ""


# ── Agent ─────────────────────────────────────────────────────────────────────

class Agent:
    """
    LLM-driven agent with a two-layer reflection loop.

    Layer 1 (LLM self-assessment):
      The action LLM includes stuck=true/false in every response.
      If stuck=true, reflection triggers immediately — the LLM noticed
      before any rule had to catch it.

    Layer 2 (rule-based fallback):
      StuckDetector watches position loops, action loops, and wall-bumping.
      Fires only when the LLM failed to self-report being stuck.

    Either layer triggers the same reflection flow:
      → separate LLM call diagnoses the failure
      → strategy injected into conversation history
      → action LLM benefits immediately on the next step
    """

    MAX_REFLECTIONS = 3

    def __init__(self, api_key: str, max_history: int = 10):
        self.api_key      = api_key
        self.history:     list[dict] = []
        self.max_history  = max_history
        self.stuck_detector = StuckDetector()
        self.last_reflection: str = ""

    async def reflect(self, observation: dict, mission: str, stuck_reason: str) -> str:
        """Dedicated LLM call to diagnose failure and produce corrective strategy."""
        messages = build_reflection_prompt(
            observation, mission,
            self.history[-self.max_history:],
            stuck_reason,
        )
        raw = await call_llm(
            messages, self.api_key,
            system=REFLECTION_SYSTEM_PROMPT,
            max_tokens=256,
        )
        cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("`").strip()
        try:
            data     = json.loads(cleaned)
            diagnosis = data.get("diagnosis", "")
            strategy  = data.get("strategy", "")
            n = self.stuck_detector.reflection_count + 1
            return f"[REFLECTION #{n}] Diagnosis: {diagnosis} → Strategy: {strategy}"
        except Exception:
            return f"[REFLECTION] {raw[:200]}"

    async def _trigger_reflection(self, observation: dict, mission: str, reason: str):
        """Run reflection and inject result into history."""
        strategy = await self.reflect(observation, mission, reason)
        self.last_reflection = strategy
        self.history.append({
            "role": "user",
            "content": f"[System notice] {reason}\n\nPlease reconsider your approach.",
        })
        self.history.append({
            "role": "assistant",
            "content": json.dumps({"reasoning": strategy, "action": "wait"}),
        })
        self.stuck_detector.reset_window()

    async def decide(
        self,
        observation: dict,
        mission: str,
        last_action: str = "",
        last_result: str = "",
    ) -> tuple[str, str, str, bool]:
        """
        Ask the LLM for the next action.
        Returns (action, reasoning, raw_response, reflected).

        Reflection trigger priority:
          1. LLM self-reports stuck=true  → reflect immediately
          2. StuckDetector fires          → reflect as fallback
          3. Neither                      → act normally
        """
        reflected = False
        self.last_reflection = ""

        # Feed last step into the rule-based detector regardless
        if last_action and last_result:
            self.stuck_detector.record(
                tuple(observation["position"].values()),
                last_action,
                last_result,
            )

        # ── Get action from LLM ───────────────────────────────────────────────
        messages = build_prompt(observation, mission, self.history[-self.max_history:])
        raw = await call_llm(messages, self.api_key)
        action, reasoning, llm_stuck, llm_stuck_reason = parse_response(raw)

        self.history.append({"role": "user",      "content": messages[-1]["content"]})
        self.history.append({"role": "assistant",  "content": raw})

        total_reflections = self.stuck_detector.reflection_count

        # ── Layer 1: LLM self-assessment ──────────────────────────────────────
        if llm_stuck and total_reflections < self.MAX_REFLECTIONS:
            reason = f"[LLM self-report] {llm_stuck_reason or 'Agent flagged itself as stuck.'}"
            await self._trigger_reflection(observation, mission, reason)
            reflected = True

        # ── Layer 2: Rule-based fallback (only if LLM didn't catch it) ───────
        elif not llm_stuck and total_reflections < self.MAX_REFLECTIONS:
            rule_stuck, rule_reason = self.stuck_detector.is_stuck()
            if rule_stuck:
                await self._trigger_reflection(observation, mission, rule_reason)
                reflected = True

        return action, reasoning, raw, reflected, llm_stuck, llm_stuck_reason