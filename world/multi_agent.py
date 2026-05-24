"""
Multi-Agent World
Two robots share the same grid and must cooperate to complete a mission.
Each robot has its own position, inventory, vision, and LLM agent instance.
Turns alternate: Robot A acts, then Robot B acts.
"""
from dataclasses import dataclass, field
from typing import Optional
from world.grid import GridWorld, OBJECTS, WorldObject


# ── Robot state ───────────────────────────────────────────────────────────────

@dataclass
class Robot:
    id: str                          # "A" or "B"
    pos: tuple                       # (x, y)
    inventory: list = field(default_factory=list)
    visited: set   = field(default_factory=set)
    seen: set      = field(default_factory=set)
    facing: str    = "south"
    messages: list = field(default_factory=list)   # last N results
    explore_target: Optional[tuple] = None
    steps: int     = 0


# ── Multi-agent world ─────────────────────────────────────────────────────────

class MultiAgentWorld:
    """
    Wraps a GridWorld and manages two robots.

    The shared grid holds walls, objects, and delivery state.
    Each Robot holds its own position, inventory, vision, and message log.
    Collision rule: robots cannot occupy the same tile.
    """

    def __init__(self, world: GridWorld, robot_a: Robot, robot_b: Robot):
        self.world   = world
        self.robots  = {"A": robot_a, "B": robot_b}
        self.turn    = "A"     # whose turn it is
        self.done    = False
        self.goal_reached = False
        self.total_steps  = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def other(self, robot_id: str) -> Robot:
        return self.robots["B" if robot_id == "A" else "A"]

    def _update_seen(self, robot: Robot):
        if self.world.vision_radius is None:
            return
        rx, ry = robot.pos
        r = self.world.vision_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                nx, ny = rx + dx, ry + dy
                if self.world.in_bounds(nx, ny):
                    robot.seen.add((nx, ny))

        # Robots share sensor data — merge both seen sets
        other = self.other(robot.id)
        shared = robot.seen | other.seen
        robot.seen = shared
        other.seen = shared

    def _count_exits(self, x, y) -> int:
        count = 0
        for dx, dy in [(0,-1),(0,1),(1,0),(-1,0)]:
            nx, ny = x+dx, y+dy
            if self.world.in_bounds(nx, ny) and not self._is_blocked_for(nx, ny, exclude=None):
                count += 1
        return count

    def _is_blocked_for(self, x, y, exclude=None) -> bool:
        """Is (x,y) blocked? Robots also block tiles (except self)."""
        if not self.world.in_bounds(x, y): return True
        if (x, y) in self.world.walls: return True
        obj = self.world.objects.get((x, y))
        if obj and OBJECTS[obj].blocks: return True
        # Other robot blocks movement
        for rid, r in self.robots.items():
            if rid != exclude and r.pos == (x, y):
                return True
        return False

    def _get_neighbors(self, robot: Robot) -> dict:
        ax, ay = robot.pos
        dirs = {"north":(0,-1),"south":(0,1),"east":(1,0),"west":(-1,0)}
        result = {}
        for name, (dx, dy) in dirs.items():
            nx, ny = ax+dx, ay+dy
            if not self.world.in_bounds(nx, ny):
                result[name] = "wall (out of bounds)"
            elif (nx, ny) in self.world.walls:
                result[name] = "wall"
            elif (nx, ny) == self.other(robot.id).pos:
                other = self.other(robot.id)
                result[name] = f"Robot {other.id} (ally — carrying {other.inventory or 'nothing'})"
            elif (nx, ny) in self.world.objects:
                obj = self.world.objects[(nx, ny)]
                if OBJECTS[obj].blocks:
                    result[name] = obj
                else:
                    exits = self._count_exits(nx, ny)
                    warn = " ⚠ DEAD-END" if exits <= 1 else ""
                    result[name] = f"{obj} ({exits} exit{'s' if exits!=1 else ''}{warn})"
            else:
                exits = self._count_exits(nx, ny)
                warn = " ⚠ DEAD-END" if exits <= 1 else ""
                result[name] = f"empty ({exits} exit{'s' if exits!=1 else ''}{warn})"
        return result

    def _render_ascii(self, robot: Robot) -> str:
        """Render the map from this robot's perspective (fog of war)."""
        self._update_seen(robot)
        other = self.other(robot.id)
        rows = []
        for y in range(self.world.height):
            row = ""
            for x in range(self.world.width):
                if (x, y) == robot.pos:
                    row += robot.id   # "A" or "B"
                elif (x, y) == other.pos:
                    if self.world.vision_radius is None or (x,y) in robot.seen:
                        row += other.id
                    else:
                        row += "?"
                elif self.world.vision_radius is not None and (x,y) not in robot.seen:
                    row += "?"
                elif (x, y) in self.world.walls:
                    row += "#"
                elif (x, y) in self.world.objects:
                    row += OBJECTS[self.world.objects[(x,y)]].symbol
                elif (x, y) in robot.visited:
                    row += "·"
                else:
                    row += "."
            rows.append(row)
        return "\n".join(rows)

    def _build_goal_hint(self, robot: Robot) -> str:
        """Role-specific goal hint for each robot."""
        ax, ay = robot.pos

        def direction_str(tx, ty):
            dx, dy = tx-ax, ty-ay
            parts = []
            if dx>0: parts.append(f"{dx} east")
            elif dx<0: parts.append(f"{-dx} west")
            if dy>0: parts.append(f"{dy} south")
            elif dy<0: parts.append(f"{-dy} north")
            return " and ".join(parts) if parts else "HERE"

        def find_obj(name):
            for pos, o in self.world.objects.items():
                if o == name:
                    if self.world.vision_radius is None or pos in robot.seen:
                        return pos
            return None

        role = getattr(robot, "role", None)

        # Gate opener role: get key → open gate
        if role == "opener":
            has_key = "key" in robot.inventory
            gate_exists = any(o in ("gate","door") for o in self.world.objects.values())
            if not has_key:
                key_pos = find_obj("key")
                if key_pos:
                    return f"Your role: OPENER. Get the KEY at {key_pos} — {direction_str(*key_pos)} from you."
                return "Your role: OPENER. Find the KEY — explore to locate it."
            if gate_exists:
                for blocker in ("gate","door"):
                    pos = find_obj(blocker)
                    if pos:
                        return (f"You have the key! Head to {blocker.upper()} at {pos} "
                                f"— {direction_str(*pos)} — unlock it for Robot B.")
                return "You have the key. Find the gate/door — explore to locate it."
            return "Gate is open. Help Robot B if needed, or explore."

        # Deliverer role: get part → deliver to assembly
        if role == "deliverer":
            if self.world.deliveries:
                item_name, target_name = self.world.deliveries[0]
                if item_name not in robot.inventory:
                    item_pos = find_obj(item_name)
                    if item_pos:
                        return (f"Your role: DELIVERER. Pick up {item_name.upper()} "
                                f"at {item_pos} — {direction_str(*item_pos)}.")
                    return f"Your role: DELIVERER. Find the {item_name.upper()} — explore to locate it."
                else:
                    target_pos = find_obj(target_name)
                    if target_pos:
                        return (f"Carrying {item_name}. Deliver to {target_name.upper()} "
                                f"at {target_pos} — {direction_str(*target_pos)}.")
                    # Gate still locked?
                    gate_exists = any(o in ("gate","door") for o in self.world.objects.values())
                    if gate_exists:
                        return (f"Carrying {item_name}. Gate is still locked — "
                                f"wait for Robot A to open it, then head south.")
                    return f"Carrying {item_name}. Find the {target_name.upper()} — explore."

        # Fallback
        for goal_type in ("goal","assembly","depot"):
            pos = find_obj(goal_type)
            if pos:
                return f"Target ({goal_type.upper()}) at {pos} — {direction_str(*pos)}."
        return "Explore the map to find your target."

    def get_observation(self, robot_id: str) -> dict:
        """Full observation for one robot, including partner state."""
        robot = self.robots[robot_id]
        other = self.other(robot_id)
        self._update_seen(robot)

        obs = {
            "robot_id":      robot.id,
            "position":      {"x": robot.pos[0], "y": robot.pos[1]},
            "world_size":    {"width": self.world.width, "height": self.world.height},
            "inventory":     robot.inventory,
            "steps_taken":   robot.steps,
            "facing":        robot.facing,
            "neighbors":     self._get_neighbors(robot),
            "goal_hint":     self._build_goal_hint(robot),
            "recent_messages": robot.messages[-3:],
            "ascii_map":     self._render_ascii(robot),
            "partner": {
                "id":          other.id,
                "pos":         {"x": other.pos[0], "y": other.pos[1]},
                "inventory":   other.inventory,
                "last_action": getattr(other, "last_action", "unknown"),
            },
        }
        if self.world.deliveries:
            obs["deliveries_remaining"] = len(self.world.deliveries)
            obs["deliveries_done"]      = self.world.deliveries_done
        return obs

    def step(self, robot_id: str, action: str) -> str:
        """Execute one action for the given robot. Returns result message."""
        robot  = self.robots[robot_id]
        robot.steps += 1
        self.total_steps += 1
        robot.last_action = action

        ax, ay = robot.pos
        msg = ""

        if action.startswith("move_"):
            action = action[5:]

        DIR_MAP = {"north":(0,-1),"south":(0,1),"east":(1,0),"west":(-1,0)}

        if action in DIR_MAP:
            dx, dy = DIR_MAP[action]
            nx, ny = ax+dx, ay+dy
            if self._is_blocked_for(nx, ny, exclude=robot_id):
                obj = self.world.objects.get((nx, ny))
                if obj in ("door","gate"):
                    if "key" in robot.inventory:
                        del self.world.objects[(nx, ny)]
                        robot.pos = (nx, ny)
                        robot.visited.add((nx, ny))
                        self._update_seen(robot)
                        robot.facing = action
                        msg = f"Robot {robot_id} used key to unlock {obj}. Moved through."
                    else:
                        msg = f"The {obj} is locked. Robot A needs to open it first."
                elif (nx, ny) == self.other(robot_id).pos:
                    msg = f"Robot {self.other(robot_id).id} is blocking that tile. Wait or go another way."
                else:
                    msg = "Blocked. Cannot move that way."
            else:
                robot.pos = (nx, ny)
                robot.visited.add((nx, ny))
                self._update_seen(robot)
                robot.facing = action
                landed = self.world.objects.get((nx, ny))

                # Delivery check
                if landed in ("goal","assembly","depot") and self.world.deliveries:
                    item_needed, target_needed = self.world.deliveries[0]
                    if landed == target_needed and item_needed in robot.inventory:
                        robot.inventory.remove(item_needed)
                        self.world.deliveries.pop(0)
                        self.world.deliveries_done += 1
                        if not self.world.deliveries:
                            msg = f"🎉 Robot {robot_id} delivered {item_needed} to {landed}! Mission accomplished!"
                            self.done = True
                            self.goal_reached = True
                        else:
                            msg = (f"✅ Robot {robot_id} delivered {item_needed}! "
                                   f"{len(self.world.deliveries)} delivery remaining.")
                    else:
                        msg = f"Robot {robot_id} moved {action}. Standing on {landed}."
                elif landed == "goal" and not self.world.deliveries:
                    msg = f"🎉 Robot {robot_id} reached the goal! Mission complete!"
                    self.done = True
                    self.goal_reached = True
                elif landed:
                    msg = f"Robot {robot_id} moved {action}. There is a {landed} here."
                else:
                    msg = f"Robot {robot_id} moved {action}."

        elif action == "pick_up":
            obj = self.world.objects.get((ax, ay))
            if obj and OBJECTS[obj].pickable:
                robot.inventory.append(obj)
                del self.world.objects[(ax, ay)]
                msg = f"Robot {robot_id} picked up {obj}."
            elif obj:
                msg = f"Cannot pick up {obj}."
            else:
                msg = "Nothing to pick up here."

        elif action == "wait":
            msg = f"Robot {robot_id} waited."

        else:
            msg = f"Unknown action: {action}"

        robot.messages.append(msg)
        # Advance turn
        self.turn = "B" if robot_id == "A" else "A"
        return msg


# ── Scenario ──────────────────────────────────────────────────────────────────

def build_collab_scenario() -> tuple["MultiAgentWorld", str, str]:
    """
    collab_delivery: Two robots must cooperate.

    Layout (10×8):
      - Left half: Robot A starts here with key nearby
      - Vertical wall in the middle with a locked gate
      - Right half: part to pick up, assembly line at far corner
      - Robot B starts right half, must deliver part through gate

    Robot A role: OPENER  — find key, unlock gate
    Robot B role: DELIVERER — find part, pass through gate, deliver to assembly
    """
    from world.grid import GridWorld

    w = GridWorld(width=10, height=8)
    w.walls = {
        # Vertical dividing wall, gate gap at row 4
        (4,y) for y in range(8) if y != 4
    }
    w.objects = {
        (4,4): "gate",      # locked gate in the wall
        (1,6): "key",       # key for Robot A
        (7,1): "part",      # part for Robot B to pick up
        (1,7): "assembly",  # delivery target — LEFT side, forces B to cross gate
    }
    w.vision_radius = 3
    w.deliveries = [("part", "assembly")]

    robot_a = Robot(id="A", pos=(0,0), visited={(0,0)})
    robot_a.role = "opener"

    robot_b = Robot(id="B", pos=(9,0), visited={(9,0)})
    robot_b.role = "deliverer"

    world = MultiAgentWorld(w, robot_a, robot_b)

    mission_a = (
        "You are Robot A (OPENER). Find the KEY (K) on the left side of the map, "
        "pick it up, and unlock the GATE (T) in the middle wall. "
        "Robot B cannot pass until you open it."
    )
    mission_b = (
        "You are Robot B (DELIVERER). Find the PART (P) on the RIGHT side of the map, pick it up, "
        "then pass through the GATE (T) to the LEFT side — Robot A will unlock it — "
        "and deliver the part to the ASSEMBLY line (A) on the left side."
    )
    return world, mission_a, mission_b