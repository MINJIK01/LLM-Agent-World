# LLM Agent World

An LLM agent placed into a 2D grid world — it perceives its environment through a limited sensor range, reasons about what to do next, and acts to accomplish goals.

## Quick Start

```bash
# 1. Install dependencies
pip install fastapi uvicorn httpx

# 2a. Web UI
uvicorn server:app --reload
# Open http://localhost:8000 — enter your Anthropic API key and pick a scenario

# 2b. Terminal
python run_cli.py --scenario key_door --api-key sk-ant-...
# (or export ANTHROPIC_API_KEY=sk-ant-... first)
```

---

## Scenarios

| Scenario | Description | Vision | Key mechanic |
|---|---|---|---|
| `reach_goal` | Navigate a walled maze to reach the goal tile | radius 2 | Pathfinding |
| `key_door` | Pick up a key, unlock a door, reach the goal | radius 3 | Sequencing |
| `exploration` | Explore a grid with multiple objects to collect | radius 2 | Exploration |
| `factory_delivery` | Carry a part through a locked gate to the assembly line | radius 3 | Delivery + gate |
| `warehouse_sort` | Sort two boxes to their respective depots | radius 3 | Multi-delivery |
| `hazard_navigate` | Deliver a part while avoiding impassable hazard zones | radius 2 | Obstacle avoidance |

The **robotics scenarios** (`factory_delivery`, `warehouse_sort`, `hazard_navigate`) are inspired by real industrial robot tasks — navigating factory floors, handling parts, and responding to hazards.

---

## Architecture

```
llm-agent-world/
├── world/
│   └── grid.py       # GridWorld engine — tiles, objects, fog of war, observations
├── agent/
│   └── harness.py    # Agent harness — prompt builder, LLM caller, action parser
├── static/
│   └── index.html    # Web UI — live grid visualiser (SSE consumer)
├── logs/             # One .jsonl file per run (auto-created)
├── server.py         # FastAPI server — /run endpoint with SSE streaming
├── run_cli.py        # Terminal runner — coloured ASCII output + JSONL log
└── requirements.txt
```

### Data flow

```
GridWorld                  Harness                     Claude
    │                         │                           │
    │── get_observation() ───▶│                           │
    │   · position            │── build_prompt() ────────▶│
    │   · fog-of-war map      │   · mission               │
    │   · visible neighbors   │   · observation           │
    │   · inventory           │   · history (last 10)     │
    │   · goal hint           │                           │
    │   · recent messages     │◀── { reasoning, action } ─│
    │                         │                           │
    │◀── step(action) ────────│                           │
    │   · move / pick_up      │                           │
    │   · update visited      │                           │
    │   · return result       │                           │
    │                         │                           │
    └──────── repeat until done or max_steps ─────────────┘
```

### The agent loop (per step)

1. **Observe** — `GridWorld.get_observation()` serialises position, fog-of-war ASCII map, visible neighbours, inventory, and a goal hint into a structured dict.
2. **Decide** — `Agent.decide()` formats the observation into a prompt, calls Claude, and parses the JSON response `{ "reasoning": "...", "action": "..." }`.
3. **Act** — `GridWorld.step(action)` executes the action and returns a result message, which is fed back into the next prompt as recent history.

---

## Design choices

### Fog of war (limited sensor range)

Each agent has a `vision_radius` — a Chebyshev-distance sensor cone. Tiles outside this range render as `?` until the agent physically moves close enough to reveal them. Once seen, tiles remain revealed (persistent memory), matching how a real robot would build a local map incrementally.

```
# Step 1 — agent at top-left, vision_radius=2
@..????
...????
...????
???????

# Step 5 — after exploring south-east
·····??
·····??
·@···??
·····??
```

This makes the challenge meaningful: the agent cannot simply read the goal position off the initial map — it has to plan under uncertainty and update as it explores.

### Observation design

The agent receives:

- **Position** `(x, y)` — for self-reference
- **Fog-of-war ASCII map** — visited cells `·`, unvisited seen cells `.`, unseen cells `?`. Encodes spatial memory without an explicit coordinate list (cheaper tokens, easier for the LLM to parse visually)
- **Directional neighbours** — what is immediately adjacent in each compass direction
- **Goal hint** — context-aware next-target hint. Respects fog: only reveals coordinates of objects the agent has already seen; otherwise says "explore to find it"
- **Inventory** — what the agent is carrying
- **Recent messages** — last 3 action results (short-term memory without growing the context unboundedly)

### Action space

```
move_north / move_south / move_east / move_west
pick_up   — pick up item at current tile
look      — inspect all 4 neighbours (costs a step)
wait      — do nothing
```

Small and unambiguous. The LLM names one action per turn; no argument parsing needed.

### Structured output

The agent responds in JSON:

```json
{ "reasoning": "Key is one tile east — pick it up before approaching the door.", "action": "move_east" }
```

The `reasoning` field makes the agent's intent observable and debuggable. The harness falls back gracefully (keyword extraction, then `wait`) if the model produces malformed output.

### Conversation history

A rolling window of the last 10 turns (observation + response pairs) gives the LLM short-term memory of what it has tried — preventing it from repeating the same failed action in a loop.

### Multi-delivery sequencing

Factory scenarios use a `deliveries` queue: a list of `(item, target)` pairs processed in order. The hint system always points at the current priority target. On delivery, the queue advances and the agent is told what comes next.

---

## Example run

```
=== LLM Agent World ===
Scenario : key_door
Mission  : Find the key (K), pick it up, unlock the door (D), and reach the goal (G).

@ . . # . . . .
. . . # . . . .
. . . D . . . .
. . . # . . G .
. K . # . . . .
. . . # . . . .

[step 01] move_south      Heading south to explore and find the key
          → Moved south.

[step 05] move_east       Key is directly east of me
          → Moved east. There is a key here.

[step 06] pick_up         Standing on the key — pick it up now
          → Picked up key.
          inventory: ['key']

[step 10] move_east       Walking into the door with the key to unlock it
          → Used key to unlock and open the door. Moved through.

[step 14] move_south      Goal is directly south of me
          → 🎉 Goal reached! Mission complete!

🎉 Mission complete in 14 steps!
Log saved → logs/run_20260520_153012_key_door.jsonl
```

Full run logs are saved as JSONL (one line per step) in `logs/`. See `logs/example_key_door.jsonl` for a complete example.

---

## What worked

- **Fog of war** makes exploration meaningful — the agent has to navigate under uncertainty rather than reading the answer off the initial map
- **Context-aware goal hint** dramatically reduces aimless wandering while still respecting what the agent has and hasn't seen
- **ASCII map with `·` / `.` / `?`** encodes visited, unvisited, and unknown state in a single visual token the LLM can reason about spatially
- **JSON output with `reasoning`** makes debugging easy and lets you see exactly why the agent made each decision
- **Rolling history window** prevents context blowup while keeping enough memory to avoid repeating mistakes

## What could be improved

- **No path planning** — the agent reasons step-by-step rather than computing a full route; a planning step before acting would reduce backtracking
- **No long-term spatial map** — the `seen` set persists per run but the LLM only receives the current ASCII snapshot; an explicit map structure passed in context could help on larger grids
- **`look` action underused** — with fog of war, `look` has real value (extends effective vision by one tile) but the LLM still tends to favour moving over looking
- **Single-agent only** — a natural extension for multi-robot factory scenarios