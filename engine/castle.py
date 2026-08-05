"""Siege data: walls, moat, towers, bridge, catapult.

Pure data + geometry — no rendering or battle-state mutation.
All coordinates use our (col, row) system on the 11x9 board.

Board layout (team 0 = attacker outside, team 1 = defender inside):

   col:  0  1  2  3  4  5  6  7  8  9 10
row 0:   .  .  .  .  .  .  . [M] [W] I  I
row 1:   .  .  .  .  .  .  . [M] [T] I  I
row 2:   .  .  .  .  .  . [M] [W]  I  I  I
row 3:   .  .  .  .  .  . [M] [G]  I  I  I   G = gate tower (non-shooting)
row 4:   .  .  .  .  . [M] [==] [G]  I  I  I   == = gate/bridge
row 5:   .  .  .  .  .  . [M] [G]  I  I  I
row 6:   .  .  .  .  .  . [M] [W]  I  I  I
row 7:   .  .  .  .  .  .  . [M] [T] I  I   T = archer tower (shooting)
row 8:   .  .  .  .  .  .  . [M] [W] I  I

Index mapping: our (col, row) = fheroes2 flat index  row*11 + col.

Simplifications vs fheroes2 (no hero skills / artifacts / castle buildings):
  - Wall HP fixed at 2 (no fortification 3-HP variant)
  - Shooting penalty: fixed 50% (Archery exemption handled in battle_state)
  - 3 towers always present (no build prerequisites)
"""

import math
import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ── Board geometry constants ─────────────────────────────────────

# 4 wall segments: HP 2 (intact) → 1 (damaged) → 0 (destroyed, passable).
WALL_POSITIONS: List[Tuple[int, int]] = [(8, 0), (7, 2), (7, 6), (8, 8)]

# 9 moat cells (outside the wall line, attacker side).
MOAT_CELLS: FrozenSet[Tuple[int, int]] = frozenset({
    (7, 0), (7, 1), (6, 2), (6, 3), (5, 4),
    (6, 5), (6, 6), (7, 7), (7, 8),
})

# Gate / drawbridge position.
GATE_POS: Tuple[int, int] = (6, 4)

# Cells the attacker stands on to be in position to damage each wall /
# gate (fheroes2 cellsUnderWallsIndexes = {7, 28, 49, 72, 95}). The first
# four are the cells directly under each of the four wall segments; the
# fifth is the moat cell just outside the gate (the attacker must cross it
# to attack the bridge / gate tower).
CELLS_UNDER_WALLS: Tuple[Tuple[int, int], ...] = (
    (7, 0),  # under wall at (8, 0)
    (6, 2),  # under wall at (7, 2)
    (5, 4),  # moat cell just outside the gate
    (6, 6),  # under wall at (7, 6)
    (7, 8),  # under wall at (8, 8)
)

# Gate towers (non-shooting, impassable, damageable by earthquake only).
GATE_TOWER_POSITIONS: List[Tuple[int, int]] = [(7, 3), (7, 5)]

# Archer tower positions (shooting, damageable by catapult).
ARCHER_TOWER_POSITIONS: List[Tuple[int, int]] = [(8, 1), (8, 7)]

# First "inside-walls" column per row (0-indexed).
# A cell (c, r) is inside the castle if c >= _INSIDE_COL[r].
_INSIDE_COL = [9, 9, 8, 8, 7, 8, 8, 9, 9]

# Catapult placement (far left, attacker side).
CATAPULT_POS: Tuple[int, int] = (0, 7)


# ── Tower ────────────────────────────────────────────────────────

class Tower:
    """Virtual archer tower — shoots once per round.

    Modelled after fheroes2's Battle::Tower (inherits from Unit as a
    pseudo-Archer).  Simplified: fixed count, no mage-guild attack bonus.
    """

    # Base Archer stats (from fheroes2 monster_info.cpp)
    _ARCHER_ATTACK = 5
    _ARCHER_DAMAGE_MIN = 2
    _ARCHER_DAMAGE_MAX = 3
    _ARCHER_HP = 10

    def __init__(self, kind: str):
        """kind: "center", "left", or "right"."""
        assert kind in ("center", "left", "right")
        self.kind = kind
        self.destroyed = False
        # Center tower (Ballista): count = 10.  Side turrets: count = 5.
        self.count = 10 if kind == "center" else 5
        self.attack = self._ARCHER_ATTACK

    @property
    def is_valid(self) -> bool:
        return not self.destroyed

    @property
    def damage_avg(self) -> float:
        return (self._ARCHER_DAMAGE_MIN + self._ARCHER_DAMAGE_MAX) / 2

    @property
    def strength(self) -> float:
        """Tower combat strength for AI evaluation.

        Mirrors the base_strength formula: sqrt(dmg_avg * hp) * count.
        """
        if self.destroyed:
            return 0.0
        base = math.sqrt(self.damage_avg * self._ARCHER_HP)
        return base * self.count

    def select_target(self, enemies):
        """Pick highest-strength enemy (fheroes2: highest evaluateThreatForUnit)."""
        if self.destroyed or not enemies:
            return None
        return max(enemies, key=lambda e: e.strength)

    def expected_damage(self) -> float:
        """Average damage per shot (AI evaluation)."""
        if self.destroyed:
            return 0.0
        return self.count * self.damage_avg

    def roll_damage(self) -> int:
        """Roll actual damage (combat execution)."""
        if self.destroyed:
            return 0
        return sum(random.randint(self._ARCHER_DAMAGE_MIN, self._ARCHER_DAMAGE_MAX)
                   for _ in range(self.count))


# ── Castle ───────────────────────────────────────────────────────

class Castle:
    """All siege structures for a single castle battle.

    Created once per siege battle and passed to BattleState.
    Non-siege battles simply have ``castle=None``.
    """

    def __init__(self, color: int = 1):
        # The team that owns (defends) this castle.  0 = attacker, 1 = defender.
        # fheroes2: Castle::GetColor() is used by BattlePlanner to set
        #   _defendingCastle = (_myColor == castle->GetColor()).
        self.color: int = color
        # Wall HP: position → {2=intact, 1=damaged, 0=destroyed}
        self.walls: Dict[Tuple[int, int], int] = {p: 2 for p in WALL_POSITIONS}
        # Gate-tower condition is 2 → 1 and can never reach zero. These
        # structures remain impassable after Earthquake damage.
        self.gate_tower_hp: Dict[Tuple[int, int], int] = {
            p: 2 for p in GATE_TOWER_POSITIONS
        }

        # Bridge / gate state
        self.bridge_down: bool = False
        self.bridge_destroyed: bool = False

        # 3 towers: [left, center, right]
        self.towers: List[Tower] = [
            Tower("left"), Tower("center"), Tower("right"),
        ]

    # ── geometry queries ─────────────────────────────────────

    @staticmethod
    def is_moat(col: int, row: int) -> bool:
        return (col, row) in MOAT_CELLS

    @property
    def has_moat(self) -> bool:
        """True if the castle has a moat (always True in our board layout).

        fheroes2 ``Battle::Board::isMoatIndex`` is index-based; here we expose
        a boolean matching the question the AI asks: "does movement through
        this cell terminate for non-flying units?".
        """
        return len(MOAT_CELLS) > 0

    @staticmethod
    def is_inside_walls(col: int, row: int) -> bool:
        """True if the cell is on the defender (castle interior) side."""
        if row < 0 or row >= len(_INSIDE_COL):
            return False
        return col >= _INSIDE_COL[row]

    @staticmethod
    def is_outside_walls(col: int, row: int) -> bool:
        return not Castle.is_inside_walls(col, row)

    # ── wall state ───────────────────────────────────────────

    def wall_intact_cells(self) -> Set[Tuple[int, int]]:
        """Wall cells that still block movement (HP > 0)."""
        return {p for p, hp in self.walls.items() if hp > 0}

    def wall_destroyed_cells(self) -> Set[Tuple[int, int]]:
        """Wall cells that no longer block movement (HP == 0)."""
        return {p for p, hp in self.walls.items() if hp == 0}

    def damage_wall(self, pos: Tuple[int, int], amount: int = 1) -> int:
        """Deal damage to a known wall segment. Returns remaining HP."""
        if pos not in self.walls:
            raise ValueError(f"unknown wall position: {pos}")
        hp = max(0, self.walls[pos] - amount)
        self.walls[pos] = hp
        return hp

    def any_wall_standing(self) -> bool:
        return any(hp > 0 for hp in self.walls.values())

    # ── bridge / gate ────────────────────────────────────────

    def is_gate_passable(self, team: int) -> bool:
        """Can *team* walk through the gate cell?

        ``Bridge::isPassable`` allows everyone through a lowered or destroyed
        bridge. With the bridge raised, only the castle owner may pass when
        occupancy permits; bridge occupancy is not modelled here.
        """
        return self.bridge_down or team == self.color

    def lower_bridge(self):
        """Defender lowers the drawbridge."""
        if not self.bridge_destroyed and not self.bridge_down:
            self.bridge_down = True

    def destroy_bridge(self):
        """Catapult destroys the bridge — permanently passable for everyone."""
        self.bridge_destroyed = True
        self.bridge_down = True  # destroyed implies permanently down

    @property
    def gate_block_cells(self) -> Set[Tuple[int, int]]:
        """Static structure cells that are always impassable.

        The gate itself is team/state dependent and is handled by
        ``is_gate_passable``. Physical side and gate towers remain blocking
        even after their shooting/condition state is damaged.
        """
        return (self.wall_intact_cells()
                | set(ARCHER_TOWER_POSITIONS)
                | set(GATE_TOWER_POSITIONS))

    # ── tower helpers ────────────────────────────────────────

    def towers_active(self) -> bool:
        return any(t.is_valid for t in self.towers)

    def tower_strength(self, tower_type: Optional[str] = None) -> float:
        """Sum of strengths.

        C++ counts towers individually (``TWR_CENTER`` / ``TWR_LEFT`` /
        ``TWR_RIGHT``); when ``tower_type`` is given, only the matching
        valid tower contributes.
        """
        if tower_type is None:
            return sum(t.strength for t in self.towers)
        idx = {"LEFT": 0, "CENTER": 1, "RIGHT": 2}.get(tower_type.upper())
        if idx is None or idx >= len(self.towers):
            return 0.0
        t = self.towers[idx]
        return t.strength if t.is_valid else 0.0

    def tower_count(self) -> int:
        """Number of alive towers — fheroes2 counts by Castle::isBuild + alive."""
        return sum(1 for t in self.towers if t.is_valid)

    def damage_tower(self, index: int):
        """Disable tower fire; its physical cell remains impassable."""
        if 0 <= index < len(self.towers):
            self.towers[index].destroyed = True

    # fheroes2 battle_catapult.cpp target enum is duplicated in
    # engine/actions.py CatapultAction:
    #   NONE=0  WALL1=1  WALL2=2  WALL3=3  WALL4=4
    #   TOWER1=5  TOWER2=6  BRIDGE=7  CENTRAL_TOWER=8
    # The AI expresses its planned shots as ``CatapultAction.shots`` in
    # that exact shape; the engine converts to the legacy wall-pos
    # string / tower-index APIs in this method.
    def apply_catapult_damage(self, target_id: int, damage: int = 1) -> int:
        """Apply one CatapultAction shot to a *target_id* slot.

        Returns the amount of wall-HP actually deducted (towers /
        bridge report 0 because they are one-shot destroyed and have
        no separate HP counter).
        """
        if target_id == 0:                                  # NONE — no-op
            return 0
        if target_id == 7:                                  # BRIDGE
            self.destroy_bridge()
            return 0
        if target_id in (5, 6, 8):                          # side/center tower
            # TOWER1 = side-left (index 0), TOWER2 = side-right (index 2),
            # CENTRAL_TOWER = index 1.
            tower_idx = {5: 0, 6: 2, 8: 1}[target_id]
            self.damage_tower(tower_idx)
            return 0
        if 1 <= target_id <= 4:                             # WALL1..WALL4
            # WALL1..WALL4 align with the four wall positions defined at
            # top-of-file; lookup by index keeps the mapping explicit.
            wall_positions = list(self.walls.keys())
            if target_id - 1 < len(wall_positions):
                return self.damage_wall(wall_positions[target_id - 1], damage)
        return 0

    def earthquake(self, power: int,
                   rng: Optional[random.Random] = None) -> List[dict]:
        """Apply Earthquake independently to every C++ spell target.

        Targets are four walls, two side towers, bridge, and two gate towers.
        The central tower is intentionally excluded. The bridge has an extra
        50% miss chance. Gate towers can degrade from condition 2 to 1 only.
        """
        if rng is None:
            rng = random
        if 1 <= power <= 2:
            min_damage, max_damage = 0, 1
        elif 3 <= power <= 5:
            min_damage, max_damage = 0, 2
        elif 6 <= power <= 9:
            min_damage, max_damage = 0, 3
        elif power >= 10:
            min_damage, max_damage = 1, 3
        else:
            min_damage = max_damage = 0

        results: List[dict] = []
        targets = ([str(pos) for pos in WALL_POSITIONS]
                   + ["tower_0", "tower_2", "bridge"]
                   + [f"gate_tower_{i}" for i in range(2)])
        for target in targets:
            condition = self._earthquake_target_condition(target)
            if condition <= 0:
                continue
            if target == "bridge" and rng.randint(0, 1) == 0:
                damage = 0
            else:
                damage = rng.randint(min_damage, max_damage)
            damage = min(damage, condition)
            if damage <= 0:
                continue
            remaining = self._apply_earthquake_hit(target, damage)
            results.append({
                "target": target,
                "damage": damage,
                "remaining_hp": remaining,
            })
        return results

    def _earthquake_target_condition(self, target: str) -> int:
        if target == "bridge":
            return 0 if self.bridge_destroyed else 1
        if target.startswith("gate_tower_"):
            idx = int(target.rsplit("_", 1)[1])
            return max(0, self.gate_tower_hp[GATE_TOWER_POSITIONS[idx]] - 1)
        if target.startswith("tower_"):
            idx = int(target.rsplit("_", 1)[1])
            return 1 if self.towers[idx].is_valid else 0
        return self.walls[self._parse_wall_target(target)]

    def _apply_earthquake_hit(self, target: str, damage: int) -> int:
        if target == "bridge":
            self.destroy_bridge()
            return 0
        if target.startswith("gate_tower_"):
            idx = int(target.rsplit("_", 1)[1])
            pos = GATE_TOWER_POSITIONS[idx]
            self.gate_tower_hp[pos] = max(1, self.gate_tower_hp[pos] - damage)
            return self.gate_tower_hp[pos]
        if target.startswith("tower_"):
            idx = int(target.rsplit("_", 1)[1])
            self.damage_tower(idx)
            return 0
        return self.damage_wall(self._parse_wall_target(target), damage)

    # ── catapult round ───────────────────────────────────────

    def catapult_round(self, ballistics: int = 0,
                       rng: Optional[random.Random] = None) -> List[dict]:
        """Execute one catapult firing round.

        Returns list of shot dicts: {target, hit, damage, remaining_hp}.
        Target priority (fheroes2): random intact wall → tower → bridge → center.

        M7d: Ballistics skill modifies behaviour (battle_catapult.cpp:44-62):
          - Default (0): 1 shot, 75% hit, 1 damage
          - Basic (1):   1 shot, always hit, 50% chance double damage
          - Advanced (2): 2 shots, always hit, 50% chance double damage
          - Expert (3):   2 shots, always hit, always double damage
        """
        if rng is None:
            rng = random.Random()

        # Determine catapult parameters from Ballistics skill level.
        can_miss = True
        double_damage_chance = 25  # percent
        shots_count = 1
        if ballistics >= 1:  # Basic
            can_miss = False
            double_damage_chance = 50
        if ballistics >= 2:  # Advanced
            shots_count = 2
        if ballistics >= 3:  # Expert
            double_damage_chance = 100

        shots: List[dict] = []
        for _ in range(shots_count):
            target = self._catapult_pick_target(rng)
            if target is None:
                break

            hit = not can_miss or rng.randint(1, 20) >= 6
            if hit:
                # Determine damage: chance for double (2) vs normal (1).
                damage = 2 if rng.randint(1, 100) <= double_damage_chance else 1
                remaining = self._apply_catapult_hit(target, damage)
            else:
                damage = 0
                remaining = self._target_hp(target)

            shots.append({
                "target": target,
                "hit": hit,
                "damage": damage,
                "remaining_hp": remaining,
            })
        return shots

    # ── catapult internals ───────────────────────────────────

    def _apply_catapult_hit(self, target: str, damage: int) -> int:
        """Apply catapult damage to *target*, return remaining HP."""
        if target == "bridge":
            self.destroy_bridge()
            return 0
        if target.startswith("tower_"):
            idx = int(target.split("_")[1])
            self.damage_tower(idx)
            return 0  # towers are one-shot destroy
        # target is a wall position string like "(8, 0)"
        pos = self._parse_wall_target(target)
        return self.damage_wall(pos, damage)

    def _catapult_pick_target(self, rng: random.Random) -> Optional[str]:
        """Pick target: walls → side towers → bridge → center tower."""
        # 1. Intact walls (random among remaining)
        intact_walls = [str(p) for p, hp in self.walls.items() if hp > 0]
        if intact_walls:
            return rng.choice(intact_walls)

        # 2. Active left/right towers. The central tower is a separate final
        # stage in battle_catapult.cpp and must never enter this random pool.
        active_side_towers = [f"tower_{i}" for i in (0, 2)
                              if self.towers[i].is_valid]
        if active_side_towers:
            return rng.choice(active_side_towers)

        # 3. Bridge
        if not self.bridge_destroyed:
            return "bridge"

        # 4. Center tower (last resort)
        if self.towers[1].is_valid:
            return "tower_1"

        return None

    @staticmethod
    def _parse_wall_target(target: str) -> Tuple[int, int]:
        """Parse wall target string '(c, r)' back to tuple."""
        # target is like "(8, 0)"
        parts = target.strip("()").split(",")
        return (int(parts[0]), int(parts[1]))

    def _target_hp(self, target: str) -> int:
        if target == "bridge":
            return 0 if self.bridge_destroyed else 1
        if target.startswith("tower_"):
            idx = int(target.split("_")[1])
            return 1 if self.towers[idx].is_valid else 0
        pos = self._parse_wall_target(target)
        return self.walls.get(pos, 0)
