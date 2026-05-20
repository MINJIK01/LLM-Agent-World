# LLM Agent World

An LLM agent placed into a 2D grid world — it perceives its environment, reasons, and acts to accomplish goals.

## Quick Start

```bash
# 1. Install dependencies
pip install fastapi uvicorn httpx

# 2. Run the web UI
uvicorn server:app --reload
# Open http://localhost:8000 → enter your Anthropic API key → click Run

# OR run in the terminal
python run_cli.py --scenario key_door --api-key sk-ant-...
# (or export ANTHROPIC_API_KEY=sk-ant-... first)
```

## Scenarios

| Scenario | Description |
|---|---|
| `reach_goal` | Navigate a walled maze to reach the goal tile |
| `key_door` | Pick up a key, unlock a door, reach the goal |
| `exploration` | Explore an open grid with multiple objects |

## Architecture

### Project Structure

```
llm-agent-world/
├── world/
│   └── grid.py          # GridWorld engine — tiles, objects, movement, observation
├── agent/
│   └── harness.py       # Agent harness — prompt builder, LLM caller, action parser
├── static/
│   └── index.html       # Web UI — live grid visualiser (SSE consumer)
├── logs/                # Auto-created — one .jsonl file per run
├── server.py            # FastAPI server — /run endpoint with SSE streaming
├── run_cli.py           # Terminal runner — coloured ASCII output + JSONL log
├── requirements.txt
└── README.md
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    server.py / run_cli.py                   │
│                                                             │
│   ┌──────────────┐   get_observation()   ┌───────────────┐  │
│   │              │ ─────────────────────▶│               │  │
│   │  world/      │                       │  agent/       │  │
│   │  grid.py     │      step(action)     │  harness.py   │  │
│   │  GridWorld   │ ◀─────────────────────│  Agent        │  │
│   │              │                       │               │  │
│   └──────────────┘                       └──────┬────────┘  │
│         │                                       │           │
│         │ SSE stream                     call_llm()         │
│         ▼                                       │           │
│   ┌──────────────┐                       ┌──────▼────────┐  │
│   │  static/     │                       │  Anthropic    │  │
│   │  index.html  │                       │  API          │  │
│   │  Web UI      │                       │  (Claude)     │  │
│   └──────────────┘                       └───────────────┘  │
│         │                                                    │
│         ▼                                                    │
│   logs/*.jsonl  (one line per step)                         │
└─────────────────────────────────────────────────────────────┘
```

### Agent Loop (per step)

```
GridWorld                  Harness                     Claude
    │                         │                           │
    │── get_observation() ───▶│                           │
    │   · position (x, y)     │                           │
    │   · neighbors           │── build_prompt() ────────▶│
    │   · visited cells       │   · mission               │
    │   · inventory           │   · observation           │
    │   · goal hint           │   · history (last 10)     │
    │   · ascii map           │                           │
    │                         │◀── { reasoning, action } ─│
    │◀── step(action) ────────│                           │
    │   · move / pick_up      │                           │
    │   · update visited      │                           │
    │   · return result msg   │                           │
    │                         │                           │
    └──────── repeat until done or max_steps ─────────────┘
```

### The Harness (core idea)

The "harness" is the interface between the LLM and the world. Three steps per tick:

1. **Observe** — `GridWorld.get_observation()` serialises agent position, neighbours, inventory, goal hint, and an ASCII map into a structured dict.
2. **Decide** — `Agent.decide()` formats the observation into a prompt, calls Claude, and parses the JSON response `{ "reasoning": "...", "action": "..." }`.
3. **Act** — `GridWorld.step(action)` executes the action and returns a human-readable result message, which is fed back into the next prompt.

### Observation Design

The agent receives:
- **Exact position** (x, y) — for self-reference
- **Directional neighbours** — what's immediately adjacent in each compass direction
- **Goal hint** — cardinal direction + distance to goal, so the agent can orient without searching
- **Inventory** — what the agent is carrying
- **Recent messages** — last 3 action results, giving short-term memory
- **ASCII map** — full world view for spatial reasoning

The ASCII map is critical: Claude can visually interpret the grid layout and plan multi-step paths.

### Action Space

```
move_north / move_south / move_east / move_west
pick_up   — pick up item at current tile
look      — inspect all 4 neighbours (costs a step)
wait      — do nothing
```

Kept small and unambiguous. The LLM doesn't need to parse complex syntax — it just names one action.

### Why JSON output?

Structured output (`{ "reasoning": ..., "action": ... }`) makes parsing deterministic and lets us log the agent's reasoning separately from its decision. The harness falls back gracefully if the model wraps it in markdown fences.

### Conversation History

The agent maintains a rolling window of the last 10 turns (observation + response pairs). This gives the LLM short-term memory of what it has tried, preventing repeated mistakes.

## What worked

- Goal hint (direction + distance) dramatically reduces aimless wandering
- ASCII map gives the LLM a spatial overview it can "look at" holistically
- JSON output with explicit `reasoning` field makes debugging easy
- Short action names with exact semantics reduce ambiguity

## What could be improved

- No long-term memory — the agent forgets the full map after the context window fills
- No explicit path-planning step before acting
- Goal hint breaks if there are multiple goals
- The agent sometimes wastes steps with `look` when the map already shows everything