"""
Grid World Environment
Defines the physical world: tiles, objects, agent state.
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Tile(str, Enum):
    EMPTY   = "."
    WALL    = "#"
    GOAL    = "G"
    KEY     = "K"
    DOOR    = "D"
    CHEST   = "C"
    AGENT   = "@"


@dataclass
class WorldObject:
    name: str
    symbol: str
    pickable: bool = False
    blocks: bool = False


OBJECTS = {
    # Classic
    "key":      WorldObject("key",      "K", pickable=True,  blocks=False),
    "chest":    WorldObject("chest",    "C", pickable=True,  blocks=False),
    "door":     WorldObject("door",     "D", pickable=False, blocks=True),
    "goal":     WorldObject("goal",     "G", pickable=False, blocks=False),
    # Robotics / factory themed
    "part":     WorldObject("part",     "P", pickable=True,  blocks=False),  # factory part to deliver
    "assembly": WorldObject("assembly", "A", pickable=False, blocks=False),  # assembly line drop-off
    "gate":     WorldObject("gate",     "T", pickable=False, blocks=True),   # locked gate (needs key)
    "hazard":   WorldObject("hazard",   "X", pickable=False, blocks=True),   # impassable hazard zone
    "depot":    WorldObject("depot",    "B", pickable=False, blocks=False),  # box/item depot (goal variant)
    "box":      WorldObject("box",      "O", pickable=True,  blocks=False),  # cargo box to sort
}


@dataclass
class GridWorld:
    width: int
    height: int
    walls: set = field(default_factory=set)
    objects: dict = field(default_factory=dict)   # (x,y) -> object_name
    agent_pos: tuple = (0, 0)
    inventory: list = field(default_factory=list)
    steps: int = 0
    messages: list = field(default_factory=list)
    done: bool = False
    goal_reached: bool = False
    visited: set = field(default_factory=set)
    # Multi-delivery tracking: list of (item, target_object) pairs to complete
    deliveries: list = field(default_factory=list)   # e.g. [("box", "depot")]
    deliveries_done: int = 0
    # Fog of war: how many tiles the agent can see in each direction (Chebyshev distance).
    # None = full map visible (no fog). Seen tiles are remembered even after leaving.
    vision_radius: int = None
    seen: set = field(default_factory=set)   # tiles revealed so far

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x, y):
        if not self.in_bounds(x, y):
            return True
        if (x, y) in self.walls:
            return True
        obj = self.objects.get((x, y))
        if obj and OBJECTS[obj].blocks:
            return True
        return False

    def _update_seen(self):
        """Expand the set of revealed tiles based on current position and vision_radius."""
        if self.vision_radius is None:
            return  # full visibility — seen set unused
        ax, ay = self.agent_pos
        for dy in range(-self.vision_radius, self.vision_radius + 1):
            for dx in range(-self.vision_radius, self.vision_radius + 1):
                nx, ny = ax + dx, ay + dy
                if self.in_bounds(nx, ny):
                    self.seen.add((nx, ny))

    def _in_vision(self, x, y) -> bool:
        """True if (x,y) is currently within the agent's live vision cone."""
        if self.vision_radius is None:
            return True
        ax, ay = self.agent_pos
        return max(abs(x - ax), abs(y - ay)) <= self.vision_radius

    def render_ascii(self):
        """
        Render the grid as ASCII with optional fog of war.

        Tile legend:
          @  = agent
          #  = wall
          ·  = visited empty tile
          .  = unvisited empty tile (within seen area)
          ?  = fog of war (tile never revealed yet)
          Objects shown only within seen area.
        """
        self._update_seen()
        rows = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (x, y) == self.agent_pos:
                    row += "@"
                elif self.vision_radius is not None and (x, y) not in self.seen:
                    row += "?"   # fog — never seen
                elif (x, y) in self.walls:
                    row += "#"
                elif (x, y) in self.objects:
                    row += OBJECTS[self.objects[(x, y)]].symbol
                elif (x, y) in self.visited:
                    row += "·"   # visited empty tile
                else:
                    row += "."   # seen but unvisited
            rows.append(row)
        return "\n".join(rows)

    def get_neighbors(self):
        ax, ay = self.agent_pos
        dirs = {"north": (0,-1), "south": (0,1), "east": (1,0), "west": (-1,0)}
        result = {}
        for name, (dx, dy) in dirs.items():
            nx, ny = ax+dx, ay+dy
            if not self.in_bounds(nx, ny):
                result[name] = "wall (out of bounds)"
            elif (nx, ny) in self.walls:
                result[name] = "wall"
            elif (nx, ny) in self.objects:
                obj = self.objects[(nx, ny)]
                result[name] = obj
            else:
                result[name] = "empty"
        return result

    def _build_goal_hint(self) -> str:
        """
        Build a navigation hint toward the next relevant target.
        Priority order:
          1. gate/door exists + no key → hint at key
          2. gate/door exists + has key + door/gate seen → hint at door/gate
          3. delivery task pending + item not in hand → hint at item
          4. delivery task pending + item in hand + target seen → hint at target
          5. delivery task pending + item in hand + target not seen → explore
          6. classic key_door: key seen but not held → hint at key
          7. classic key_door: key held + door seen → hint at door
          8. default → hint at goal/assembly/depot if seen, else explore
        """
        ax, ay = self.agent_pos

        def direction_str(tx, ty):
            dx, dy = tx - ax, ty - ay
            parts = []
            if dx > 0: parts.append(f"{dx} east")
            elif dx < 0: parts.append(f"{-dx} west")
            if dy > 0: parts.append(f"{dy} south")
            elif dy < 0: parts.append(f"{-dy} north")
            return " and ".join(parts) if parts else "HERE"

        def find_obj(name):
            """Find object position — only if already seen (or no fog)."""
            for pos, o in self.objects.items():
                if o == name:
                    if self.vision_radius is None or pos in self.seen:
                        return pos
            return None

        has_key = "key" in self.inventory

        # ── 1 & 2. Gate/door scenarios ────────────────────────────────────────
        # Only active if a blocker still exists AND hasn't been unlocked yet
        existing_blockers = [o for o in self.objects.values() if o in ("gate", "door")]
        if existing_blockers and not has_key:
            key_pos = find_obj("key")
            if key_pos:
                return (f"Next target: KEY at {key_pos} — {direction_str(*key_pos)} from you."
                        " (Pick it up to open the gate/door.)")
            else:
                return "You need a KEY to open the gate/door — explore to find it."

        if existing_blockers and has_key:
            for blocker in ("gate", "door"):
                blocker_pos = find_obj(blocker)
                if blocker_pos:
                    return (f"You have the key. Head to {blocker.upper()} at {blocker_pos} "
                            f"— {direction_str(*blocker_pos)} from you to unlock it.")
            # blocker exists but not seen yet
            return "You have the key. Find the gate/door — explore to locate it."

        # ── 3–5. Delivery task ────────────────────────────────────────────────
        if self.deliveries:
            item_name, target_name = self.deliveries[0]
            if item_name not in self.inventory:
                item_pos = find_obj(item_name)
                if item_pos:
                    return (f"Next target: {item_name.upper()} at {item_pos} — "
                            f"{direction_str(*item_pos)} from you. Pick it up.")
                else:
                    return f"Find the {item_name.upper()} — explore unvisited areas (? tiles)."
            else:
                target_pos = find_obj(target_name)
                if target_pos:
                    return (f"Carrying {item_name}. Deliver to {target_name.upper()} "
                            f"at {target_pos} — {direction_str(*target_pos)} from you.")
                else:
                    return (f"Carrying {item_name}. Find the {target_name.upper()} "
                            f"— explore unvisited areas.")

        # ── 6 & 7. Classic key/chest pickup (no deliveries queue) ─────────────
        # key visible but not held
        key_pos = find_obj("key")
        if key_pos and not has_key:
            return (f"Next target: KEY at {key_pos} — {direction_str(*key_pos)} from you."
                    " Pick it up.")

        # chest visible but not held
        chest_pos = find_obj("chest")
        if chest_pos and "chest" not in self.inventory:
            return (f"Next target: CHEST at {chest_pos} — {direction_str(*chest_pos)} from you."
                    " Pick it up.")

        # ── 8. Default: goal-type objects ─────────────────────────────────────
        for goal_type in ("goal", "assembly", "depot"):
            pos = find_obj(goal_type)
            if pos:
                return f"Target ({goal_type.upper()}) at {pos} — {direction_str(*pos)} from you."

        return "Explore the map — navigate toward unvisited (?) tiles to find your target."

    def _build_explore_hint(self) -> str:
        """
        Find the nearest tile that has been SEEN (revealed by sensor) but not yet
        VISITED (stepped on). This guides the agent toward the frontier of its
        known map rather than re-exploring already-visited territory.

        Uses BFS from agent position through passable tiles to find:
          1. Nearest seen-but-unvisited tile (frontier of known map)
          2. First step direction to reach it

        Returns a hint string, or empty string if the entire seen area is visited.
        """
        if self.vision_radius is None:
            return ""  # full visibility — no explore hint needed

        # Ensure seen set is current before computing frontier
        self._update_seen()

        ax, ay = self.agent_pos

        # Candidate frontier tiles: seen but not visited, not a wall/hazard
        def is_frontier(x, y):
            if (x, y) in self.walls:
                return False
            obj = self.objects.get((x, y))
            if obj and OBJECTS[obj].blocks:
                return False
            if (x, y) not in self.seen:
                return False
            if (x, y) in self.visited:
                return False
            return True

        # Also consider unseen tiles adjacent to seen tiles as exploration targets
        def is_unseen_border(x, y):
            if not self.in_bounds(x, y):
                return False
            if (x, y) in self.seen:
                return False
            # Is any seen tile adjacent to it?
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                if (nx, ny) in self.seen:
                    return True
            return False

        # BFS through passable seen tiles to find nearest frontier or unseen edge
        from collections import deque
        queue = deque()
        queue.append((ax, ay, None, 0))   # (x, y, first_step_direction, dist)
        visited_bfs = {(ax, ay)}
        DIR_MAP = {"north": (0,-1), "south": (0,1), "east": (1,0), "west": (-1,0)}

        best_frontier = None
        best_dir = None
        best_dist = float("inf")

        while queue:
            cx, cy, first_dir, dist = queue.popleft()

            if dist >= best_dist:
                continue

            # Target 1: seen but unvisited tile
            if is_frontier(cx, cy) and (cx, cy) != (ax, ay):
                best_frontier = (cx, cy)
                best_dir = first_dir
                best_dist = dist
                continue

            for name, (dx, dy) in DIR_MAP.items():
                nx, ny = cx+dx, cy+dy
                if (nx, ny) in visited_bfs:
                    continue
                if not self.in_bounds(nx, ny):
                    continue
                visited_bfs.add((nx, ny))
                step_dir = first_dir if first_dir else name

                # Target 2: unseen tile adjacent to seen area — this is the fog frontier
                if (nx, ny) not in self.seen:
                    # Skip if it's obviously impassable (wall or known hazard)
                    if (nx, ny) in self.walls:
                        continue
                    obj = self.objects.get((nx, ny))
                    if obj and OBJECTS[obj].blocks:
                        continue
                    if dist + 1 < best_dist:
                        best_frontier = (nx, ny)
                        best_dir = step_dir
                        best_dist = dist + 1
                    continue

                # Only traverse seen, passable tiles
                if self.is_blocked(nx, ny):
                    continue
                queue.append((nx, ny, step_dir, dist + 1))

        if best_frontier and best_dir:
            fx, fy = best_frontier
            dx, dy = fx - ax, fy - ay
            parts = []
            if dy < 0: parts.append(f"{-dy} north")
            elif dy > 0: parts.append(f"{dy} south")
            if dx > 0: parts.append(f"{dx} east")
            elif dx < 0: parts.append(f"{-dx} west")
            dist_str = " and ".join(parts)
            return (f"Nearest unexplored tile: ({fx},{fy}) — {dist_str} away. "
                    f"First step: move_{best_dir}.")

        # Fallback: hint toward nearest unseen border tile
        # (agent hasn't seen enough to BFS to a frontier)
        nearest_unseen = None
        nearest_dist = float("inf")
        for y in range(self.height):
            for x in range(self.width):
                if is_unseen_border(x, y):
                    d = abs(x - ax) + abs(y - ay)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_unseen = (x, y)

        if nearest_unseen:
            ux, uy = nearest_unseen
            dx, dy = ux - ax, uy - ay
            parts = []
            if dy < 0: parts.append(f"{-dy} north")
            elif dy > 0: parts.append(f"{dy} south")
            if dx > 0: parts.append(f"{dx} east")
            elif dx < 0: parts.append(f"{-dx} west")
            return f"Nearest unseen area: ({ux},{uy}) — {'and '.join(parts)} away."

        return ""

    def get_observation(self) -> dict:
        ax, ay = self.agent_pos
        explore_hint = self._build_explore_hint()
        obs = {
            "position": {"x": ax, "y": ay},
            "world_size": {"width": self.width, "height": self.height},
            "neighbors": self.get_neighbors(),
            "inventory": self.inventory,
            "steps_taken": self.steps,
            "goal_hint": self._build_goal_hint(),
            "explore_hint": explore_hint,
            "recent_messages": self.messages[-3:],
            "ascii_map": self.render_ascii(),
        }
        if self.deliveries:
            obs["deliveries_remaining"] = len(self.deliveries)
            obs["deliveries_done"] = self.deliveries_done
        return obs

    def step(self, action: str, arg: Optional[str] = None) -> str:
        self.steps += 1
        ax, ay = self.agent_pos
        msg = ""
        if action.startswith("move_"):
            action = action[5:]

        DIR_MAP = {"north": (0,-1), "south": (0,1), "east": (1,0), "west": (-1,0)}

        if action in DIR_MAP:
            dx, dy = DIR_MAP[action]
            nx, ny = ax+dx, ay+dy
            if self.is_blocked(nx, ny):
                obj = self.objects.get((nx, ny))
                if obj in ("door", "gate"):
                    if "key" in self.inventory:
                        del self.objects[(nx, ny)]
                        self.agent_pos = (nx, ny)
                        self.visited.add((nx, ny))
                        label = "door" if obj == "door" else "gate"
                        msg = f"Used key to unlock and open the {label}. Moved through."
                    else:
                        label = "door" if obj == "door" else "gate"
                        msg = f"The {label} is locked. You need a key."
                else:
                    msg = "Blocked. Cannot move that way."
            else:
                self.agent_pos = (nx, ny)
                self.visited.add((nx, ny))
                landed = self.objects.get((nx, ny))

                # Check delivery completion
                if landed in ("goal", "assembly", "depot") and self.deliveries:
                    item_needed, target_needed = self.deliveries[0]
                    if landed == target_needed and item_needed in self.inventory:
                        self.inventory.remove(item_needed)
                        self.deliveries.pop(0)
                        self.deliveries_done += 1
                        if not self.deliveries:
                            msg = f"🎉 Delivered {item_needed} to {landed}! All deliveries complete — mission accomplished!"
                            self.goal_reached = True
                            self.done = True
                        else:
                            next_item, next_target = self.deliveries[0]
                            msg = (f"✅ Delivered {item_needed} to {landed}! "
                                   f"{len(self.deliveries)} delivery remaining. "
                                   f"Next: bring {next_item} to {next_target}.")
                    elif landed == target_needed and item_needed not in self.inventory:
                        msg = f"Reached {landed}, but you're not carrying the {item_needed}. Go pick it up first."
                    else:
                        msg = f"Moved {action}. Standing on {landed}."
                elif landed == "goal" and not self.deliveries:
                    msg = "🎉 Goal reached! Mission complete!"
                    self.goal_reached = True
                    self.done = True
                elif landed:
                    msg = f"Moved {action}. There is a {landed} here."
                else:
                    msg = f"Moved {action}."

        elif action == "pick_up":
            obj = self.objects.get((ax, ay))
            if obj and OBJECTS[obj].pickable:
                self.inventory.append(obj)
                del self.objects[(ax, ay)]
                msg = f"Picked up {obj}."
            elif obj:
                msg = f"Cannot pick up {obj}."
            else:
                msg = "Nothing to pick up here."

        elif action == "look":
            neighbors = self.get_neighbors()
            parts = [f"{d}: {v}" for d, v in neighbors.items()]
            msg = "Surroundings — " + " | ".join(parts)

        elif action == "wait":
            msg = "Waited one step."

        else:
            msg = f"Unknown action: {action}"

        self.messages.append(msg)
        return msg


# ── Scenario Loader ──────────────────────────────────────────────────────────

def build_scenario(name: str) -> tuple[GridWorld, str]:
    """Returns (world, goal_description)"""

    # ── Classic scenarios ────────────────────────────────────────────────────

    if name == "reach_goal":
        w = GridWorld(width=7, height=7)
        w.walls = {(2,0),(2,1),(2,2),(2,3),(2,4),(4,2),(4,3),(4,4),(4,5),(4,6)}
        w.objects = {(6,6): "goal"}
        w.agent_pos = (0,0)
        w.visited = {(0,0)}
        w.vision_radius = 2   # tight corridors — limited lookahead
        return w, "Navigate through the winding corridors and reach the goal (G) at the far corner."

    elif name == "key_door":
        w = GridWorld(width=8, height=6)
        w.walls = {(3,y) for y in range(6) if y != 2}
        w.objects = {(3,2): "door", (1,4): "key", (6,3): "goal"}
        w.agent_pos = (0,0)
        w.visited = {(0,0)}
        w.vision_radius = 3   # moderate — can see across a room but not the whole map
        return w, "Find the key (K), pick it up, unlock the door (D), and reach the goal (G)."

    elif name == "exploration":
        w = GridWorld(width=6, height=6)
        w.walls = {(1,1),(1,2),(3,0),(3,1),(0,4),(1,4),(4,3),(4,4),(4,5)}
        w.objects = {(5,5): "goal", (2,3): "key", (5,0): "chest"}
        w.agent_pos = (0,0)
        w.visited = {(0,0)}
        w.vision_radius = 2   # smallest — must actively explore
        return w, "Explore the grid: find and pick up the chest (C) and the key (K), then reach the goal (G)."

    # ── Robotics / factory scenarios ─────────────────────────────────────────

    elif name == "factory_delivery":
        """
        A factory robot must pick up a part (P) from the supply shelf,
        carry it through a locked gate (T) to the assembly line (A).

        Layout (8×7):
          - Supply shelf (part P) is top-right
          - Gate (T) runs vertically through the middle (col 4), open at row 3
          - Key (K) is bottom-left area
          - Assembly line (A) is bottom-right
        """
        w = GridWorld(width=8, height=7)
        w.walls = {
            # outer frame gaps intentionally left open
            (4,0),(4,1),(4,2),(4,4),(4,5),(4,6),   # vertical wall with gate gap at row 3
        }
        w.objects = {
            (4,3): "gate",    # locked gate in the wall gap
            (1,1): "key",     # key near start
            (6,1): "part",    # factory part to pick up
            (6,5): "assembly",# assembly line drop-off point
        }
        w.agent_pos = (0,3)
        w.visited = {(0,3)}
        w.deliveries = [("part", "assembly")]
        w.vision_radius = 3   # industrial sensor range
        return w, (
            "You are a factory robot. Pick up the PART (P), pass through the locked GATE (T) "
            "using the KEY (K), and deliver the part to the ASSEMBLY line (A)."
        )

    elif name == "warehouse_sort":
        """
        Sort two boxes (O) to their labelled depots (B).
        Box1 → Depot1 (top-right), Box2 → Depot2 (bottom-right).
        No locks — pure navigation and sequencing challenge.

        Layout (9×7):
          Boxes at left side, depots at right side, maze-like shelving in between.
        """
        w = GridWorld(width=9, height=7)
        w.walls = {
            (2,1),(2,2),(2,3),
            (4,3),(4,4),(4,5),
            (6,1),(6,2),(6,3),
        }
        w.objects = {
            (0,2): "box",    # box 1
            (0,5): "box",    # box 2
            (8,1): "depot",  # depot 1
            (8,5): "depot",  # depot 2
        }
        w.agent_pos = (0,0)
        w.visited = {(0,0)}
        w.deliveries = [("box", "depot"), ("box", "depot")]
        w.vision_radius = 3   # warehouse sensor range
        return w, (
            "You are a warehouse robot. Pick up each BOX (O) and carry it to a DEPOT (B). "
            "There are 2 boxes and 2 depots — deliver both to complete the mission."
        )

    elif name == "hazard_navigate":
        """
        Navigate a hazardous factory floor to deliver an emergency part.
        Hazard zones (X) are impassable — the robot must find a safe path.

        Layout (8×8):
          Hazards form a broken diagonal forcing a non-obvious detour.
          Part (P) is mid-map, assembly (A) is far corner.
        """
        w = GridWorld(width=8, height=8)
        w.walls = {
            (0,3),(1,3),(2,3),       # top barrier
            (5,4),(6,4),(7,4),       # bottom barrier
        }
        w.objects = {
            (1,5): "hazard",
            (2,5): "hazard",
            (2,6): "hazard",
            (3,6): "hazard",
            (3,7): "hazard",
            (4,2): "hazard",
            (4,3): "hazard",
            (5,2): "hazard",
            (3,1): "part",       # emergency part
            (7,7): "assembly",   # destination
        }
        w.agent_pos = (0,0)
        w.visited = {(0,0)}
        w.deliveries = [("part", "assembly")]
        w.vision_radius = 2   # limited sensors — hazards only visible up close
        return w, (
            "EMERGENCY: A production line is stalled. Navigate the hazardous factory floor "
            "(avoid HAZARD zones X — they are impassable), pick up the PART (P), "
            "and deliver it to the ASSEMBLY line (A)."
        )

    else:
        raise ValueError(f"Unknown scenario: {name}")