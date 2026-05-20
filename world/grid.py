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
    "key":   WorldObject("key",   "K", pickable=True,  blocks=False),
    "chest": WorldObject("chest", "C", pickable=True,  blocks=False),  # FIX: chest is now pickable
    "door":  WorldObject("door",  "D", pickable=False, blocks=True),
    "goal":  WorldObject("goal",  "G", pickable=False, blocks=False),
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

    def render_ascii(self):
        """
        Render the grid as ASCII.
        Visited empty cells are shown as '·' (middle dot) so the agent
        can visually distinguish explored vs unexplored territory.
        """
        rows = []
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if (x, y) == self.agent_pos:
                    row += "@"
                elif (x, y) in self.walls:
                    row += "#"
                elif (x, y) in self.objects:
                    obj_name = self.objects[(x, y)]
                    row += OBJECTS[obj_name].symbol
                elif (x, y) in self.visited:
                    row += "·"   # visited empty tile
                else:
                    row += "."   # unvisited empty tile
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
        Priority: unpicked key (if door exists) > goal.
        Returns a human-readable direction + distance string.
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

        # If there's a door and we don't have the key yet, hint toward the key first
        has_door = any(obj == "door" for obj in self.objects.values())
        has_key_in_hand = "key" in self.inventory

        if has_door and not has_key_in_hand:
            key_pos = next((pos for pos, obj in self.objects.items() if obj == "key"), None)
            if key_pos:
                return f"Next target: KEY at {key_pos} — {direction_str(*key_pos)} from you. (Pick it up to unlock the door.)"

        # Default: hint toward goal
        goal_pos = next((pos for pos, obj in self.objects.items() if obj == "goal"), None)
        if goal_pos:
            return f"Goal (G) is {direction_str(*goal_pos)} from you."

        return ""

    def get_observation(self) -> dict:
        ax, ay = self.agent_pos
        neighbors = self.get_neighbors()
        return {
            "position": {"x": ax, "y": ay},
            "world_size": {"width": self.width, "height": self.height},
            "neighbors": neighbors,
            "inventory": self.inventory,
            "steps_taken": self.steps,
            "goal_hint": self._build_goal_hint(),
            "recent_messages": self.messages[-3:],
            "ascii_map": self.render_ascii(),
        }

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
                if obj == "door":
                    if "key" in self.inventory:
                        del self.objects[(nx, ny)]
                        self.agent_pos = (nx, ny)
                        self.visited.add((nx, ny))
                        msg = "Used key to unlock and open the door. Moved through."
                    else:
                        msg = "The door is locked. You need a key."
                else:
                    msg = "Blocked. Cannot move that way."
            else:
                self.agent_pos = (nx, ny)
                self.visited.add((nx, ny))
                landed = self.objects.get((nx, ny))
                if landed == "goal":
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


# ── Scenario Loader ─────────────────────────────────────────────────────────

def build_scenario(name: str) -> tuple[GridWorld, str]:
    """Returns (world, goal_description)"""

    if name == "reach_goal":
        w = GridWorld(width=7, height=7)
        w.walls = {(2,0),(2,1),(2,2),(2,3),(2,4),(4,2),(4,3),(4,4),(4,5),(4,6)}
        w.objects = {(6,6): "goal"}
        w.agent_pos = (0,0)
        w.visited = {(0,0)}
        return w, "Navigate through the winding corridors and reach the goal (G) at the far corner."

    elif name == "key_door":
        w = GridWorld(width=8, height=6)
        w.walls = {(3,y) for y in range(6) if y != 2}
        w.objects = {(3,2): "door", (1,4): "key", (6,3): "goal"}
        w.agent_pos = (0,0)
        w.visited = {(0,0)}
        return w, "Find the key (K), pick it up, unlock the door (D), and reach the goal (G)."

    elif name == "exploration":
        w = GridWorld(width=6, height=6)
        w.walls = {(1,1),(1,2),(3,0),(3,1),(0,4),(1,4),(4,3),(4,4),(4,5)}
        w.objects = {(5,5): "goal", (2,3): "key", (5,0): "chest"}
        w.agent_pos = (0,0)
        w.visited = {(0,0)}
        # FIX: chest is now pickable, so this task is completable
        return w, "Explore the grid: find and pick up the chest (C) and the key (K), then reach the goal (G)."

    else:
        raise ValueError(f"Unknown scenario: {name}")