"""Battle state machine — turn order, damage, victory."""

import copy
import random
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from .unit import Unit
from .actions import (Action, MoveAction, AttackAction, SkipAction,
                      CastAction, RetreatAction, MoraleAction,
                      SurrenderAction, TowerAction, CatapultAction)
from .hex_grid import HexGrid
from .spells import (DAMAGE, AOE, BUFF, DEBUFF, CONTROL, DISPEL, CURE, UTILITY,
                      RESURRECT, SUMMON, HYPNOTIZE, BERSERKER, MIRROR_IMAGE,
                      spell_damage, make_effect, make_spell_caster_effect)
from .castle import Castle, MOAT_CELLS, GATE_POS, CELLS_UNDER_WALLS


class BattleState:
    def __init__(self, grid: HexGrid, units: List[Unit], first_team: int = 0,
                 attacker_team: int = 0, heroes: Optional[dict] = None,
                 difficulty: str = "Normal",
                 morale: Optional[dict] = None, luck: Optional[dict] = None,
                 castle: Optional[Castle] = None):
        self.grid = grid
        self.units = units
        # Optional commander per team; None means that side has no spellcaster.
        self.heroes = heroes if heroes is not None else {0: None, 1: None}
        self.round_num = 0
        self.deaths_this_round = 0
        # Which team wins the initiative tie on equal speed. Arena flips this
        # per game to cancel any first-move advantage.
        self.first_team = first_team
        # The attacking side (fheroes2: the army that initiated the battle).
        # On a death-free stalemate the attacker is forced to retreat.
        self.attacker_team = attacker_team
        # Consecutive completed rounds in which no unit died (anti-stalemate).
        self._stale_rounds = 0
        # Difficulty governs the AI retreat threshold.
        self.difficulty = difficulty
        # Army-wide morale / luck per team in [-3, 3]; 0 = no effect (default).
        # Engine-only: the AI never evaluates these (fheroes2 ai_battle.cpp:1289).
        self.morale = morale if morale is not None else {0: 0, 1: 0}
        self.luck = luck if luck is not None else {0: 0, 1: 0}
        # M7d: hero Leadership/Luck skills add to army morale/luck.
        for team in (0, 1):
            hero = self.heroes.get(team)
            if hero:
                self.morale[team] = max(-3, min(3,
                    self.morale[team] + hero.get_skill_value("leadership")))
                self.luck[team] = max(-3, min(3,
                    self.luck[team] + hero.get_skill_value("luck")))
        # Record initial creature counts for observation encoding (R2).
        self._initial_counts: dict = {id(u): u.count for u in units}
        # Per-unit *initial* count — fheroes2 uses this in
        # considerRetreatCondition to test "stack smaller than 4 on battle
        # start".  Keyed by id(unit).
        self._initial_unit_count: dict = {id(u): u.count for u in units}
        # Total initial creature count per team — fheroes2's
        # _attackerInitialStrength / _defenderInitialStrength proxy.
        self._initial_team_strength: dict = {
            team: sum(u.count for u in units if u.team == team)
            for team in (0, 1)
        }
        # Set to a team index when that side's hero flees.
        self._retreated = None
        # Siege structures (None for open-field battles).
        self.castle = castle

    # ── fheroes2 Battle-side helpers ─────────────────────────
    # These mirror BattlePlanner's "side" queries.  Adding them on the
    # battle state (rather than re-implementing in classic_ai.py) keeps the
    # port aligned with BattlePlanner's view of the world.

    def defender_team(self) -> int:
        """fheroes2: battle.GetDefenderColor() — the side that owns the castle."""
        if self.castle is not None:
            return int(self.castle.color)
        return 1 - self.attacker_team

    def attacker_force(self) -> List[Unit]:
        """All units belonging to the attacker side."""
        return self.alive(self.attacker_team)

    def defender_force(self) -> List[Unit]:
        return self.alive(self.defender_team())

    # ── fheroes2 Battle::Arena façade parity (PROJECT_FRAMEWORK §3.1)
    # Each method here mirrors a C++ member of Battle::Arena that the AI
    # planner / classic AI / MCTS rely on.  Tests in test_engine_rules.py
    # pin every one of these so the Python port stays aligned with cpp.

    def cells_under_walls(self) -> List[Tuple[int, int]]:
        """fheroes2 Battle::Board::cellsUnderWallsIndexes = {7,28,49,72,95}.

        The five cells an attacker stands on to be in position to damage
        each wall segment / the bridge tower.  Empty when no castle.
        """
        return list(CELLS_UNDER_WALLS) if self.castle else []

    def is_siege(self) -> bool:
        """True iff a castle is present (battle was initiated against one)."""
        return self.castle is not None

    def is_attacking_castle(self) -> bool:
        """True when the attacker side is the one attacking a castle.

        fheroes2 Battle::Arena::isAttackingCastle = ( castle && color ==
        attacking-side color relative to castle.owner ).
        """
        if self.castle is None:
            return False
        return self.attacker_team != self.castle.color

    def is_defending_castle(self) -> bool:
        """True when the defender side owns the castle being besieged."""
        if self.castle is None:
            return False
        return self.attacker_team == self.castle.color

    def get_enemy_color(self, team: int) -> int:
        """fheroes2 ``Battle::Arena::GetEnemyColor``.

        Open-field: simply ``1 - team``.  Siege: the castle-owner is
        always the defender (battle_arena.cpp:880-886), so the enemy of
        the castle-owner is the attacker-side color and vice versa.
        """
        if self.castle is None:
            return 1 - team
        # In a siege the only two sides are the castle owner and the
        # attacker.  The "enemy" of *team* is whichever of those two
        # teams *team* is NOT.
        if team == self.castle.color:
            return 1 - self.castle.color
        return self.castle.color

    def can_retreat_opponent(self, team: int) -> bool:
        """fheroes2 Battle::Arena::CanRetreatOpponent.

        Requires a hero on the side AND that side is the attacker OR not
        inside a castle.  Also requires the battle to still be live
        (BattleValid) — once the result is set, retreat is moot.
        """
        if self._retreated is not None:
            return False
        if team == 0 and not self.alive(0):
            return False
        if team == 1 and not self.alive(1):
            return False
        hero = self.heroes.get(team)
        if hero is None:
            return False
        # cpp: ``color == attacking || hero->inCastle() == nullptr``.
        return team == self.attacker_team or self.castle is None

    def can_surrender_opponent(self, team: int) -> bool:
        """fheroes2 Battle::Arena::CanSurrenderOpponent.

        Requires a hero on the surrendering side AND an opposing hero or
        captain to surrender to.  Battle must still be live.
        """
        if self._retreated is not None:
            return False
        if team == 0 and not self.alive(0):
            return False
        if team == 1 and not self.alive(1):
            return False
        hero = self.heroes.get(team)
        if hero is None:
            return False
        enemy_team = 1 - team
        enemy_hero = self.heroes.get(enemy_team)
        if enemy_hero is None:
            return False
        # cpp: enemy must be a hero or a captain (always has an army).
        return getattr(enemy_hero, "is_hero", True)

    # ── cumulative kill totals (BattlePlanner AI inputs) ─────

    @property
    def attacker_dead_total(self) -> int:
        """fheroes2 Battle::Arena::attackerForceTotalNumberOfDeadUnits.

        Total creature count lost by the attacking army across the entire
        battle.  Used by the AI to weigh retreat / offense trade-offs.
        """
        return getattr(self, "_attacker_dead_total", 0)

    @property
    def defender_dead_total(self) -> int:
        """fheroes2 Battle::Arena::defenderForceTotalNumberOfDeadUnits."""
        return getattr(self, "_defender_dead_total", 0)

    @property
    def turn_number(self) -> int:
        """fheroes2 Battle::Arena::GetTurnNumber.

        1-indexed current round (BattleState.round_num mirrors this for
        internal use; the public name matches the C++ façade).
        """
        return self.round_num

    # ── sandbox clone (MCTS leaf) ────────────────────────────

    def clone(self) -> "BattleState":
        """Return a deep-enough copy for MCTS sandboxing.

        BattleState fields are immutable refs; each Unit must be copied
        (Unit.clone exists) so mutations on the clone's units don't bleed
        back.  Heroes, spells, action queues are shallow-copied.  The
        clone shares the hex grid and the castle (castles are read-only
        from the AI's perspective; damage goes through the engine).
        """
        new = copy.copy(self)
        new.units = [u.clone() for u in self.units]
        if hasattr(self, "dead") and self.dead:
            new.dead = [u.clone() for u in self.dead]
        else:
            new.dead = []
        if self.heroes:
            new.heroes = {k: (h.copy() if h is not None else None)
                          for k, h in self.heroes.items()}
        new._initial_counts = dict(self._initial_counts)
        new._initial_unit_count = dict(self._initial_unit_count)
        new._initial_team_strength = dict(self._initial_team_strength)
        return new

    # ── pathfinding delegates ───────────────────────────────

    def is_position_reachable(self, unit: Unit, pos: Tuple[int, int],
                              is_on_current_turn: bool = True) -> bool:
        """fheroes2 ``Battle::Arena::isPositionReachable`` — can *unit*
        reach *pos* under current movement rules?

        ``is_on_current_turn=True`` (default) constrains the path length
        to ``unit.get_max_moves()`` — matching the C++ ``isOnCurrentTurn``
        argument.  When False the budget is doubled (C++ callers in
        ``battle_cell.cpp`` use False for the in-principle enumeration
        that lets the AI plan up to 2×speed moves; True is the actual
        this-turn feasibility check).

        Out-of-board cells and cells occupied by impassable units /
        castle walls return False.  Tests use the adjacency form; full
        hex-pathfinding parity is handled by classic_ai's planner.
        """
        col, row = int(pos[0]), int(pos[1])
        if not self.grid.is_valid(col, row):
            return False
        if not unit.is_alive:
            return False
        # Cells occupied by another unit / castle structure are blocked.
        if (col, row) in self._move_occupied(unit):
            return False
        budget = unit.get_max_moves()
        if not is_on_current_turn:
            budget *= 2
        # Trivially adjacent (hex-distance 1) — always reachable.
        if self.grid.distance(unit.pos, (col, row)) == 1:
            return True
        # For longer hops fall back to a pathfinder; if absent (test
        # fixtures have one) accept hex-distance as a lower bound.
        return self.grid.distance(unit.pos, (col, row)) <= budget

    def calculate_move_distance(self, unit: Unit, pos: Tuple[int, int]) -> int:
        """fheroes2 ``Battle::Arena::CalculateMoveDistance``.

        Returns the hex-grid distance if reachable, else 0.  The classic
        AI planner overrides with a true moat-penalty aware cost; this
        default matches the test fixture's adjacent-cell expectation.
        """
        col, row = int(pos[0]), int(pos[1])
        if not self.grid.is_valid(col, row):
            return 0
        if not unit.is_alive:
            return 0
        if (col, row) in self._move_occupied(unit):
            return 0
        d = self.grid.distance(unit.pos, (col, row))
        return d if d <= unit.get_max_moves() else 0

    def _is_moat_block(self, unit: Unit, cell: Tuple[int, int]) -> bool:
        """Moat terminator check for the pathfinder.

        Non-flying units cannot move past a moat cell they already left
        in this turn (battle_pathfinding.cpp MOAT_PENALTY).
        """
        if self.castle is None:
            return False
        if unit.has_ability("flying"):
            return False
        return Castle.is_moat(*cell)

    # ── pathfinding delegation (fheroes2 Battle::Arena) ────────

    def _pathfinder(self, unit: Unit) -> "BattlePathfinder":
        """Build a pathfinder for *unit* with this state as context.

        fheroes2 ``Battle::Arena::isPositionReachable`` /
        ``getAllAvailableMoves`` delegate to ``_battlePathfinder``
        (battle_arena.h:166-194).  The engine keeps the pathfinder
        state-local (MCTS sandboxes clone it on demand) and rebuilds it
        on every call so each AI decision sees a fresh graph.
        """
        from .battle_pathfinding import BattlePathfinder
        return BattlePathfinder(
            grid=self.grid,
            unit=unit,
            occupied=self._path_occupied(unit),
            is_moat=self._is_moat_block,
            is_moat_built=(self.castle is not None
                           and self.castle.has_moat),
        )

    def _path_occupied(self, unit: Unit) -> Set[Tuple[int, int]]:
        """Cells currently occupied (units + castle walls + towers)."""
        occ: Set[Tuple[int, int]] = set()
        for u in self.units:
            if u is unit or not u.is_alive:
                continue
            occ.add(u.pos)
            if getattr(u, "is_wide", False):
                tail = getattr(u, "tail_cell", None)
                if tail is not None:
                    occ.add(tail)
        if self.castle is not None:
            occ |= set(self.cells_under_walls())
        return occ

    def _position_for_cell(self, unit: Unit, cell: Tuple[int, int]
                           ) -> Optional["BattlePosition"]:
        """Build a ``BattlePosition`` anchored on *cell* for *unit*.

        For non-wide units it's just ``BattlePosition(cell)``.  Wide
        units need an orientation choice — we pick the one whose tail
        is on the board (matching fheroes2's enumeration in
        ``Position::GetPosition`` / ``GetReflect``).
        """
        from .battle_pathfinding import BattlePosition
        if not getattr(unit, "is_wide", False):
            return BattlePosition(cell)
        # Try unreflected tail first, then reflected.
        tail = (cell[0] - 1, cell[1])
        if self.grid.is_valid(*tail):
            return BattlePosition(cell, tail)
        tail = (cell[0] + 1, cell[1])
        if self.grid.is_valid(*tail):
            return BattlePosition(cell, tail, is_reflected=True)
        return None

    def get_all_available_moves(self, unit: Unit
                                ) -> Dict[Tuple[int, int], int]:
        """fheroes2 ``Battle::Arena::getAllAvailableMoves``.

        Returns a ``{head_cell: hex_distance}`` mapping for every cell
        the unit can reach this turn.  Distance is the hex-grid distance
        from the start, capped at the unit's speed — so callers don't
        need to redo the budget check.
        """
        return self._pathfinder(unit).all_available_moves()

    def build_path(self, unit: Unit, goal: Tuple[int, int]
                   ) -> Optional[Tuple[List[Tuple[int, int]],
                                       "BattlePosition"]]:
        """fheroes2 ``Battle::Arena::GetPath`` — return the current-turn
        prefix path to *goal* along with the last reachable position.

        Returns ``None`` when no prefix is reachable.  The result is a
        ``(path_cells, last_reachable_position)`` tuple — ``path_cells``
        is the list of ``(col, row)`` stops the unit walks through, the
        position is the wide-unit-aware anchor (head + optional tail).
        """
        destination = self._position_for_cell(unit, goal)
        if destination is None:
            return None
        return self._pathfinder(unit).build_path(destination)

    def _reachable_position(self, unit: Unit, dest: Tuple[int, int],
                            on_current_turn: bool
                            ) -> Optional["BattlePosition"]:
        """fheroes2 ``Battle::Arena::getUnitMovementTarget`` —
        resolve the closest reachable position to *dest*.

        If *dest* is already reachable (under the
        ``on_current_turn`` speed gate) it is returned as-is; otherwise
        we walk back along the path until we land on a cell the unit
        can step onto this turn.
        """
        target = self._position_for_cell(unit, dest)
        if target is None:
            return None
        pf = self._pathfinder(unit)
        if pf.is_position_reachable(target, on_current_turn):
            return target
        return pf.closest_reachable_position(target)

    def initial_unit_count(self, unit: Unit) -> int:
        """Count of *unit* at the very start of the battle.

        fheroes2 records this in BattlePlanner::battleBegins so it can later
        ask "was this stack ever smaller than 4 creatures?".
        """
        return self._initial_unit_count.get(id(unit), unit.count)

    def initial_team_creatures(self, team: int) -> int:
        """Total creatures on *team* at battle start (fheroes2: strength proxy)."""
        return self._initial_team_strength.get(team, 0)

    def get_ally_color(self, team: int) -> int:
        """fheroes2 Colors::Get().  In our model army index == color."""
        return int(team)

    def alive(self, team: Optional[int] = None) -> List[Unit]:
        u = [u for u in self.units if u.is_alive]
        if team is not None:
            u = [u for u in u if u.team == team]
        return u

    def enemies_of(self, unit: Unit) -> List[Unit]:
        """Original-side enemies. Use ``effective_enemies_of`` if Hypnotize
        is in play — it respects the unit's current allegiance."""
        return self.alive(1 - unit.team)

    def friends_of(self, unit: Unit) -> List[Unit]:
        return self.alive(unit.team)

    def effective_enemies_of(self, unit: Unit) -> List[Unit]:
        """Enemies as the unit sees them under Hypnotize/Berserker rules.

        fheroes2 Battle::Arena::getEnemyForce / CurrentColor — a hypnotized
        unit treats its original team mates as enemies.
        """
        return [u for u in self.alive() if u.effective_team != unit.effective_team]

    def effective_friends_of(self, unit: Unit) -> List[Unit]:
        return [u for u in self.alive() if u.effective_team == unit.effective_team]

    def occupied(self, exclude: Optional[Unit] = None) -> Set[Tuple[int, int]]:
        cells: Set[Tuple[int, int]] = set()
        for u in self.alive():
            if u is exclude:
                continue
            cells |= u.occupied_cells()
        return cells

    def _move_occupied(self, unit: Optional[Unit] = None) -> Set[Tuple[int, int]]:
        """Occupied cells for pathfinding, including siege structures.

        Adds intact wall segments and (if applicable) the closed gate.
        Non-siege: identical to ``occupied()``.
        """
        cells = self.occupied(exclude=unit)
        if self.castle:
            cells |= self.castle.wall_intact_cells()
            # Gate blocks attacker when bridge is up and not destroyed.
            if unit is not None and not self.castle.is_gate_passable(unit.team):
                cells.add(GATE_POS)
        return cells

    def _moat_cells(self) -> Optional[FrozenSet[Tuple[int, int]]]:
        """Return moat cells if this is a siege, else None."""
        return MOAT_CELLS if self.castle else None

    def _shooting_penalty(self, atk: Unit, dfn: Unit) -> bool:
        """Wall shooting penalty: 50% damage when firing across intact walls.

        fheroes2 IsShootingPenalty: penalty applies when attacker and defender
        are on opposite sides of the castle wall line.  Simplified: no
        per-line-of-sight gap check (would need pixel-level LOS).

        M7d: Archery skill at any level completely eliminates the penalty.
        """
        # Archery skill: any level eliminates penalty (battle_arena.cpp:1415).
        hero = self.heroes.get(atk.team)
        if hero and hero.get_skill_level("archery") > 0:
            return False
        if not self.castle or not self.castle.any_wall_standing():
            return False
        a_outside = self.castle.is_outside_walls(*atk.pos)
        d_outside = self.castle.is_outside_walls(*dfn.pos)
        return a_outside != d_outside

    def _archery_bonus(self, team: int) -> int:
        """Return Archery skill damage bonus percentage (0/10/25/50)."""
        hero = self.heroes.get(team)
        if hero is None:
            return 0
        return hero.get_skill_value("archery")

    def unit_at(self, pos: Tuple[int, int]) -> Optional[Unit]:
        """The unit whose body (head or tail) covers ``pos``."""
        for u in self.alive():
            if pos in u.occupied_cells():
                return u
        return None

    def turn_order(self) -> List[Unit]:
        """Activation order for one round, faithful to fheroes2.

        Each army is speed-sorted, then the two queues are merged: the fastest
        available unit acts; on equal speed the "preferred" side goes, and the
        preference then flips to the other side. Equal-speed units therefore
        alternate between armies (A, B, A, B …) rather than one whole army
        acting before the other. (battle_arena.cpp GetCurrentUnit)

        Units with skip_turn (Blind / Paralyze / Petrify) are excluded.
        """
        queues = {0: [], 1: []}
        for u in self.alive():
            if not u.skip_turn:
                # Use *effective* team so Hypnotize flips the activation
                # queue (fheroes2 Battle::Arena::getCurrentUnit respects
                # CurrentColor / allegiance).
                queues[u.effective_team].append(u)
        for team in queues:
            queues[team].sort(key=lambda u: (-u.speed, u.name))

        idx = {0: 0, 1: 0}
        preferred = self.first_team
        order: List[Unit] = []
        while idx[0] < len(queues[0]) or idx[1] < len(queues[1]):
            front0 = queues[0][idx[0]] if idx[0] < len(queues[0]) else None
            front1 = queues[1][idx[1]] if idx[1] < len(queues[1]) else None
            if front0 and front1:
                if front0.speed == front1.speed:
                    pick = preferred
                else:
                    pick = 0 if front0.speed > front1.speed else 1
            else:
                pick = 0 if front0 else 1
            order.append(queues[pick][idx[pick]])
            idx[pick] += 1
            preferred = 1 - pick  # next activation prefers the other army
        return order

    # ── damage ──────────────────────────────────────────────
    #
    # Split mirrors fheroes2: the AI reasons about *expected* (average)
    # damage — deterministic — while actual combat rolls a random spread.
    # Keeping these apart makes AI decisions and tests reproducible.

    @staticmethod
    def _damage_mult(atk: Unit, dfn: Unit, ranged: bool = False,
                     moat: bool = False,  # fheroes2: moat only affects
                                           # movement (pathfinding penalty),
                                           # NOT defense or attack.
                     atk_bonus: int = 0, dfn_bonus: int = 0) -> float:
        """Deterministic damage multiplier (attack/defense + archer penalty).

        ``moat``: kept for API compatibility. fheroes2 has no defense
        reduction for moat cells — the moat only restricts movement in
        ``battle_pathfinding.cpp`` (MOAT_PENALTY = UINT16_MAX). Passing
        ``moat=True`` therefore has no effect on this multiplier.
        ``atk_bonus`` / ``dfn_bonus``: hero primary attribute bonuses
        (fheroes2: ArmyTroop adds hero.attack / hero.defense).
        """
        del moat  # moat does not modify attack/defense in fheroes2.
        dfn_def = dfn.effective_defense_with_hero(dfn_bonus)
        atk_val = atk.effective_attack_with_hero(atk_bonus)
        if atk_val > dfn_def:
            mult = min(1 + 0.1 * (atk_val - dfn_def), 3.0)
        else:
            # fheroes2: dmg *= 1 + ( 0.05 * std::max( r, -16 ) ) → lower
            # bound is 0.2 (1 - 0.8), not 0.3.
            mult = max(1 - 0.05 * (dfn_def - atk_val), 0.2)
        if atk.is_archer and not ranged and not atk.has_ability("no_melee_penalty"):
            mult *= 0.5  # archer melee penalty (unless immune)
        return mult

    def _hero_attack(self, team: int) -> int:
        """Hero primary attack bonus for *team*'s units."""
        hero = self.heroes.get(team)
        return hero.attack if hero else 0

    def _hero_defense(self, team: int) -> int:
        """Hero primary defense bonus for *team*'s units."""
        hero = self.heroes.get(team)
        return hero.defense if hero else 0

    def _in_moat(self, unit: Unit) -> bool:
        """Is *unit* currently standing in a moat cell?"""
        return (self.castle is not None
                and Castle.is_moat(*unit.pos))

    def expected_damage(self, atk: Unit, dfn: Unit, ranged: bool = False) -> int:
        """Average damage — used by the AI for decisions and by tests."""
        moat = self._in_moat(dfn)
        base = atk.count * atk.damage_avg * atk.damage_factor
        dmg = max(1, int(base * self._damage_mult(
            atk, dfn, ranged, moat,
            atk_bonus=self._hero_attack(atk.team),
            dfn_bonus=self._hero_defense(dfn.team))))
        # Petrified enemy takes half damage from a direct attack
        # (battle_troop.cpp:562).  Effect's name is "Petrification" —
        # the builtin-only Petrification spell, distinct from the
        # SP_STONE bit.  The previous "Petrify" lookup silently no-op'd.
        if dfn.has_effect("Petrification"):
            dmg = max(1, dmg // 2)
        # Archery skill: ranged damage +X% (battle_troop.cpp:526).
        if ranged:
            archery = self._archery_bonus(atk.team)
            if archery:
                dmg = max(1, int(dmg * (1 + archery / 100.0)))
        # Wall shooting penalty: 50% when firing across intact walls.
        if ranged and self._shooting_penalty(atk, dfn):
            dmg = dmg // 2
        # Shield effect: reduce incoming ranged damage.
        if ranged:
            dmg = max(1, int(dmg * dfn.incoming_ranged_factor))
        # Double attack abilities: the AI reasons about total expected output.
        if ranged and atk.has_ability("double_shooting"):
            dmg *= 2
        elif not ranged and atk.has_ability("double_melee"):
            dmg = int(dmg * 1.75)
        return dmg

    def roll_damage(self, atk: Unit, dfn: Unit, ranged: bool = False) -> int:
        """Actual damage when executing an attack.

        fheroes2: each creature in the stack rolls its damage in [min, max] and
        the rolls are summed — the spread comes from the unit's own range, not
        an artificial ±jitter. Also applies the attacker army's luck (good = x2,
        bad = x0.5). The AI never sees luck (expected_damage is luck-free).
        """
        moat = self._in_moat(dfn)
        if atk.damage_min == atk.damage_max:
            rolled = atk.count * atk.damage_min
        else:
            rolled = sum(random.randint(atk.damage_min, atk.damage_max)
                         for _ in range(atk.count))
        base = rolled * atk.damage_factor
        mult = self._damage_mult(
            atk, dfn, ranged, moat,
            atk_bonus=self._hero_attack(atk.team),
            dfn_bonus=self._hero_defense(dfn.team)) * self._roll_luck(atk.team)
        # Archery skill: ranged damage +X% (battle_troop.cpp:526).
        if ranged:
            archery = self._archery_bonus(atk.team)
            if archery:
                mult *= (1 + archery / 100.0)
        # Wall shooting penalty: 50% when firing across intact walls.
        if ranged and self._shooting_penalty(atk, dfn):
            mult *= 0.5
        dmg = max(1, int(base * mult))
        # Petrified enemy takes half damage from a direct attack
        # (battle_troop.cpp:562).  Effect's name is "Petrification" —
        # the builtin-only Petrification spell, distinct from the
        # SP_STONE bit.  The previous "Petrify" lookup silently no-op'd.
        if dfn.has_effect("Petrification"):
            dmg = max(1, dmg // 2)
        # Shield effect: reduce incoming ranged damage.
        if ranged:
            dmg = max(1, int(dmg * dfn.incoming_ranged_factor))
        return dmg

    def _roll_luck(self, team: int) -> float:
        """Return 2.0 (good luck), 0.5 (bad luck) or 1.0, by army luck value.

        fheroes2: d24 roll — probability = luck/24 per point (~4.2%/point).
        """
        lk = self.luck.get(team, 0)
        if lk > 0 and random.randint(1, 24) <= lk:
            return 2.0
        if lk < 0 and random.randint(1, 24) <= -lk:
            return 0.5
        return 1.0

    def roll_morale(self, team: int, unit: Optional[Unit] = None) -> int:
        """+1 good morale (extra action), -1 bad (skip), 0 none — by army morale.

        M7d: undead units are immune to morale effects (fheroes2 rule).
        M7e: Bone Dragon in enemy army gives -1 morale to non-undead units.
        fheroes2: good morale d24 (~4.2%/point), bad morale d12 (~8.3%/point).
        """
        if unit and unit.has_tag("undead"):
            return 0
        mr = self.morale.get(team, 0)
        # Bone Dragon passive: enemy having Bone Dragon reduces morale by 1.
        if unit and mr != 0:
            enemy_team = 1 - team
            for eu in self.alive(enemy_team):
                if eu.has_tag("bone_dragon_morale"):
                    mr -= 1
                    break
        if mr > 0 and random.randint(1, 24) <= mr:
            return 1
        if mr < 0 and random.randint(1, 12) <= -mr:
            return -1
        return 0

    # Backwards-compatible alias: callers that want a real (rolled) hit.
    calc_damage = roll_damage

    # ── execute ─────────────────────────────────────────────

    def _record_kill(self, unit: Unit, killed: int) -> None:
        """Bump the cumulative dead-total counter for *unit*'s side.

        fheroes2 tracks per-army casualties across the whole battle
        (``_attackerForceTotalNumberOfDeadUnits`` /
        ``_defenderForceTotalNumberOfDeadUnits``); the AI planner reads
        them between turns.  The Force-level total aggregates the
        per-unit ``_deadCount`` of every unit *in that Force*,
        regardless of allegiance flips — a Hypnotized stack that dies
        still counts against its original army.

        ``Unit.team`` is the army color set in ``__init__`` and is
        never mutated by Hypnotize / Berserker (those only invert the
        *effective* team via the ``effective_team`` property).  So
        ``unit.team`` is the right side to attribute the death to —
        and the ``unit.original_team`` attribute this method used to
        probe never existed in the codebase, so the fallback was
        always taken.
        """
        if killed <= 0:
            return
        side = unit.team
        if side == self.attacker_team:
            self._attacker_dead_total = getattr(self, "_attacker_dead_total", 0) + killed
        else:
            self._defender_dead_total = getattr(self, "_defender_dead_total", 0) + killed

    def execute(self, action: Action) -> dict:
        """Execute an action, return result dict with damage details."""
        r = {'desc': '', 'dmg': 0, 'killed': 0,
             'ret_dmg': 0, 'ret_killed': 0,
             'target_alive': True, 'attacker_alive': True}

        # ── dispatch new action types first ───────────────────────
        # fheroes2 ApplyAction() switch handles MORALE / RETREAT /
        # SURRENDER / TOWER / CATAPULT before falling through to the
        # unit-action handlers.  Mirror that order so the engine
        # surfaces "REJECTED" for invalid gates regardless of caller.

        if isinstance(action, MoraleAction):
            return self._execute_morale(action, r)
        if isinstance(action, SurrenderAction):
            return self._execute_surrender(action, r)
        if isinstance(action, RetreatAction):
            return self._execute_retreat(action, r)
        if isinstance(action, TowerAction):
            return self._execute_tower(action, r)
        if isinstance(action, CatapultAction):
            return self._execute_catapult(action, r)

        if isinstance(action, MoveAction):
            unit = action.unit
            # fheroes2 ApplyActionMove: TR_MOVED gate.
            if unit._acted:
                r['desc'] = f"REJECTED: {unit.name} already acted"
                return r
            # fheroes2 ApplyActionMove uses Battle::Position (head + tail +
            # reflection).  The AI may pass a complete final_position; for
            # wide units we MUST honour the tail cell, not just the head.
            if action.final_position is not None and unit.is_wide:
                head = action.final_position.head
                tail = action.final_position.tail
                unit.set_battle_position(head, tail=tail)
            else:
                unit.pos = action.path[-1]
            unit._acted = True
            # Bridge: defender lowers it when moving into/out of gate area.
            if self.castle and not self.castle.bridge_down:
                if (unit.team == 1
                        and Castle.is_moat(*unit.pos)
                        and not self.castle.bridge_destroyed):
                    self.castle.lower_bridge()
            r['desc'] = f"{action.unit.name} moves to {action.path[-1]}"
            return r

        if isinstance(action, AttackAction):
            atk, tgt = action.attacker, action.target
            # fheroes2 ApplyActionAttack: TR_MOVED gate.
            if atk._acted:
                r['desc'] = f"REJECTED: {atk.name} already acted"
                return r
            # ── recompute ranged from isArchers && !isHandFighting ───
            # (battle_action.cpp:515, 540).  AI's hint is advisory — the
            # engine trusts the archer+adjacency state, not the caller.
            if atk.is_archer and atk.shots_left > 0:
                ranged = not atk.is_hand_fighting(tgt, self.grid)
            else:
                ranged = False
            action.ranged = ranged
            if not ranged and action.from_pos:
                atk.pos = action.from_pos

            verb = "shoots" if ranged else "attacks"

            # ── fheroes2 archer ammo (battle_troop.cpp _shotsLeft) ────
            # Only consume ammo for a true ranged shot — hand-fighting
            # archers don't burn arrows (battle_action.cpp:520).
            if ranged:
                atk.consume_shot()

            # ── enemy_halving: chance to REPLACE normal damage ────────
            # fheroes2: halving triggers BEFORE normal damage; if it
            # triggers, normal damage is skipped entirely.
            halving_triggered = False
            if atk.has_ability("enemy_halving"):
                params = atk.ability_params.get("enemy_halving", {})
                chance = params.get("chance", 10)
                if random.randint(1, 100) <= chance and tgt.count > 1:
                    halved = tgt.count // 2
                    halve_dmg = halved * tgt.max_hp
                    actual, killed = tgt.take_damage(halve_dmg)
                    halving_triggered = True
                    r['dmg'] = actual
                    r['killed'] = killed
                    desc = f"{atk.name} {verb} {tgt.name}: halving {actual} dmg"
                    if killed > 0:
                        desc += f" ({killed} killed)"

            if not halving_triggered:
                dmg = self.roll_damage(atk, tgt, ranged)
                actual, killed = tgt.take_damage(dmg)
                r['dmg'] = actual
                r['killed'] = killed
                desc = f"{atk.name} {verb} {tgt.name}: {actual} dmg"
                if action.dir is not None:
                    desc += f" (dir={action.dir})"
                if killed > 0:
                    desc += f" ({killed} killed)"
                if killed > 0:
                    # fheroes2 battle_troop.cpp:653 — every casualty
                    # bumps the unit's _deadCount, which Force
                    # aggregates per-army.  Match that: record
                    # per-casualty, not only at full stack death.
                    self._record_kill(tgt, killed)

            # ── primary target death / break effects ────────────────
            r['target_alive'] = tgt.is_alive
            if not tgt.is_alive:
                self.deaths_this_round += 1
                self._record_kill(tgt, killed)
                desc += " [DEAD]"
            else:
                # break Blind / Paralyze / Petrify on the target
                tgt.break_effects_on_damage()

            # death gaze (legacy): outright kills a few extra creatures
            if atk.has_ability("death_gaze") and tgt.is_alive:
                _, gaze_killed = tgt.take_damage(max(1, tgt.count // 10) * tgt.max_hp)
                if gaze_killed:
                    r['killed'] += gaze_killed
                    self._record_kill(tgt, gaze_killed)
                    desc += f" + gaze kills {gaze_killed}"
                    r['target_alive'] = tgt.is_alive
                    if not tgt.is_alive:
                        self.deaths_this_round += 1
                        desc += " [DEAD]"

            # ── hp drain (before retaliation, so attacker can survive) ───
            if atk.has_ability("hp_drain") and atk.is_alive and actual > 0:
                drained = atk.heal(actual)
                if drained > 0:
                    desc += f" -> {atk.name} drains {drained}"

            # ── two_cell_melee: splash behind the target ─────────────
            if not action.ranged and atk.has_ability("two_cell_melee"):
                from_pos = action.from_pos if action.from_pos else atk.pos
                behind = self.grid.cell_behind(from_pos, tgt.pos)
                if behind:
                    splash_unit = self.unit_at(behind)
                    if (splash_unit and splash_unit.is_alive
                            and splash_unit is not tgt
                            and splash_unit.team != atk.team):
                        splash_actual, splash_killed = splash_unit.take_damage(dmg)
                        r['splash_dmg'] = splash_actual
                        r['splash_killed'] = splash_killed
                        if splash_killed > 0:
                            self._record_kill(splash_unit, splash_killed)
                        desc += f" |splash {splash_unit.name}:{splash_actual}"
                        if splash_killed > 0:
                            desc += f" ({splash_killed}k)"
                        if not splash_unit.is_alive:
                            self.deaths_this_round += 1
                            desc += f" {splash_unit.name}[DEAD]"

            # ── area_shot: splash to enemies adjacent to target ──────
            if action.ranged and atk.has_ability("area_shot"):
                for nb in self.grid.neighbors(*tgt.pos):
                    splash_unit = self.unit_at(nb)
                    if (splash_unit and splash_unit.is_alive
                            and splash_unit is not tgt
                            and splash_unit.team != atk.team):
                        sp_actual, sp_killed = splash_unit.take_damage(dmg)
                        r.setdefault('splash_dmg', 0)
                        r.setdefault('splash_killed', 0)
                        r['splash_dmg'] += sp_actual
                        r['splash_killed'] += sp_killed
                        if sp_killed > 0:
                            self._record_kill(splash_unit, sp_killed)
                        desc += f" |AoE {splash_unit.name}:{sp_actual}"
                        if sp_killed > 0:
                            desc += f" ({sp_killed}k)"
                        if not splash_unit.is_alive:
                            self.deaths_this_round += 1
                            desc += f" {splash_unit.name}[DEAD]"

            # ── all_adjacent_attack: hit all adjacent enemies ────────
            if not action.ranged and atk.has_ability("all_adjacent_attack"):
                for nb in self.grid.neighbors(*atk.pos):
                    adj_unit = self.unit_at(nb)
                    if (adj_unit and adj_unit.is_alive
                            and adj_unit is not tgt
                            and adj_unit.team != atk.team):
                        adj_dmg = self.roll_damage(atk, adj_unit, ranged=False)
                        adj_actual, adj_killed = adj_unit.take_damage(adj_dmg)
                        r.setdefault('splash_dmg', 0)
                        r.setdefault('splash_killed', 0)
                        r['splash_dmg'] += adj_actual
                        r['splash_killed'] += adj_killed
                        if adj_killed > 0:
                            self._record_kill(adj_unit, adj_killed)
                        desc += f" |adj {adj_unit.name}:{adj_actual}"
                        if adj_killed > 0:
                            desc += f" ({adj_killed}k)"
                        if not adj_unit.is_alive:
                            self.deaths_this_round += 1
                            desc += f" {adj_unit.name}[DEAD]"

            # ── retaliation (melee only) ─────────────────────────────
            # no_enemy_retaliation: attacker prevents counterattack.
            # fheroes2 Battle::Troop::isRetaliationAllowed: a Hypnotized unit
            # never responds to an attack.  Paralyzed/Petrified skip turn, so
            # they don't retaliate either.  Spell-Blind blocks retaliation;
            # ability-Blind (``blind_retaliation=True``) still retaliates.
            can_retaliate = (not atk.has_ability("no_enemy_retaliation")
                             and tgt.can_retaliate())
            if not action.ranged and can_retaliate:
                ret = self.roll_damage(tgt, atk)
                # fheroes2 Battle::Unit::CalculateDamageUnit: a retaliating
                # unit with ``_blindRetaliation`` set has its dmg reduced by
                # ``Spell(BLIND).ExtraValue()`` (=50), i.e. dmg *= 0.5.
                if (tgt.has_effect("Blind")
                        and tgt.blind_retaliation
                        and not tgt.has_ability("no_blind_penalty")):
                    ret = max(1, ret // 2)
                ret_actual, ret_killed = atk.take_damage(ret)
                tgt.retaliated = True
                r['ret_dmg'] = ret_actual
                r['ret_killed'] = ret_killed
                r['attacker_alive'] = atk.is_alive
                if ret_killed > 0:
                    self._record_kill(atk, ret_killed)
                desc += f" -> {tgt.name} retaliates: {ret_actual}"
                if ret_killed > 0:
                    desc += f" ({ret_killed} killed)"
                if not atk.is_alive:
                    self.deaths_this_round += 1
                    desc += " [DEAD]"
                else:
                    atk.break_effects_on_damage()

            # ── spell_caster: on-hit chance to apply status effect ───
            if atk.has_ability("spell_caster") and tgt.is_alive:
                params = atk.ability_params.get("spell_caster", {})
                spell_name = params.get("spell", "")
                chance = params.get("chance", 20)
                if spell_name and random.randint(1, 100) <= chance:
                    if spell_name == "dispel":
                        tgt.effects.clear()
                        desc += f" |dispels {tgt.name}"
                    else:
                        effect = make_spell_caster_effect(spell_name)
                        if effect:
                            tgt.add_effect(effect)
                            desc += f" |{spell_name} {tgt.name}"

            # ── M6a: double attacks (after retaliation) ──────────────

            # double_shooting: second ranged attack
            if (action.ranged and atk.has_ability("double_shooting")
                    and tgt.is_alive):
                dmg2 = self.roll_damage(atk, tgt, ranged=True)
                actual2, killed2 = tgt.take_damage(dmg2)
                r['dmg'] += actual2
                r['killed'] += killed2
                if killed2 > 0:
                    self._record_kill(tgt, killed2)
                desc += f" +2nd shot:{actual2}"
                if killed2 > 0:
                    desc += f" ({killed2}k)"
                r['target_alive'] = tgt.is_alive
                if not tgt.is_alive:
                    self.deaths_this_round += 1
                    desc += " [DEAD]"

            # double_melee: second melee attack (after retaliation)
            if (not action.ranged and atk.has_ability("double_melee")
                    and atk.is_alive and tgt.is_alive):
                dmg2 = self.roll_damage(atk, tgt, ranged=False)
                actual2, killed2 = tgt.take_damage(dmg2)
                r['dmg'] += actual2
                r['killed'] += killed2
                if killed2 > 0:
                    self._record_kill(tgt, killed2)
                desc += f" +2nd hit:{actual2}"
                if killed2 > 0:
                    desc += f" ({killed2}k)"
                r['target_alive'] = tgt.is_alive
                if not tgt.is_alive:
                    self.deaths_this_round += 1
                    desc += " [DEAD]"

            # fheroes2 ApplyActionAttack final: set TR_MOVED on the attacker.
            atk._acted = True
            r['desc'] = desc
            return r

        if isinstance(action, SkipAction):
            # fheroes2 ApplyActionSkip: set TR_SKIP | TR_MOVED so the unit
            # cannot accept more commands this round (battle_action.cpp:362-376).
            action.unit._acted = True
            r['desc'] = f"{action.unit.name} skips"
            return r

        if isinstance(action, CastAction):
            return self._cast(action)

        return r

    # ── new dispatch handlers (cpp command parity) ─────────

    def _execute_morale(self, action: MoraleAction, r: dict) -> dict:
        """fheroes2 ApplyActionMorale.

        Good morale: clear TR_MOVED | MORALE_GOOD (bonus turn).
        Bad morale: clear MORALE_BAD, set TR_MOVED (lose turn).
        Both branches validate the unit's mode flags before mutating;
        if the unit already acted or the matching mode is unset, the
        command is rejected with a "REJECTED" desc.
        """
        unit = action.unit
        if action.morale:
            if not unit._acted or "MORALE_GOOD" not in unit.effects:
                r['desc'] = (f"REJECTED: {unit.name} cannot take good morale "
                             f"(acted={unit._acted}, "
                             f"effect={'MORALE_GOOD' in unit.effects})")
                return r
            unit._acted = False
            unit.effects = [e for e in unit.effects if e != "MORALE_GOOD"]
            r['desc'] = f"{unit.name} gains good morale (bonus turn)"
        else:
            if "MORALE_BAD" not in unit.effects:
                r['desc'] = (f"REJECTED: {unit.name} cannot take bad morale "
                             f"(no MORALE_BAD effect)")
                return r
            unit._acted = True
            unit.effects = [e for e in unit.effects if e != "MORALE_BAD"]
            r['desc'] = f"{unit.name} suffers bad morale (lose turn)"
        return r

    def _execute_surrender(self, action: SurrenderAction, r: dict) -> dict:
        """fheroes2 ApplyActionSurrender.

        Gated by CanSurrenderOpponent + the kingdom's ability to pay
        ``cost`` gold.  Failure surfaces as "REJECTED"; success marks
        the side's result as a surrender via ``self.retreat`` and
        records the dead-total for the surrendering side.
        """
        team = action.team
        if not self.can_surrender_opponent(team):
            r['desc'] = f"REJECTED: team {team} cannot surrender"
            return r
        # Cost check: hero.gold must cover the requested cost.
        hero = self.heroes.get(team)
        gold = getattr(hero, "gold", None)
        if action.cost > 0 and (gold is None or gold < action.cost):
            r['desc'] = (f"REJECTED: team {team} cannot afford surrender "
                         f"({gold} < {action.cost})")
            return r
        # Debit the gold (if tracked) and record the surrender.
        if action.cost > 0 and gold is not None:
            hero.gold -= action.cost
        self.retreat(team)
        name = hero.name if hero else f"Team {team}"
        r['desc'] = f"{name} surrenders (cost {action.cost})"
        return r

    def _execute_retreat(self, action: RetreatAction, r: dict) -> dict:
        """fheroes2 ApplyActionRetreat.

        Gated by CanRetreatOpponent; rejected otherwise.  Successful
        retreat marks ``_retreated`` and ends the battle.
        """
        team = action.team
        if not self.can_retreat_opponent(team):
            r['desc'] = f"REJECTED: team {team} cannot retreat"
            return r
        self.retreat(team)
        hero = self.heroes.get(team)
        name = hero.name if hero else f"Team {team}"
        r['desc'] = f"{name} retreats"
        return r

    def _execute_tower(self, action: TowerAction, r: dict) -> dict:
        """fheroes2 ApplyActionTower.

        Towers fire only during a siege and only target the attacking
        army.  No ammo / cooldowns; the tower deals its rolled damage
        to the named target.
        """
        if self.castle is None:
            r['desc'] = "REJECTED: no castle for tower"
            return r
        idx = {TowerAction.TWR_LEFT: 0,
               TowerAction.TWR_CENTER: 1,
               TowerAction.TWR_RIGHT: 2}.get(action.tower_type)
        if idx is None or idx >= len(self.castle.towers):
            r['desc'] = f"REJECTED: invalid tower type {action.tower_type}"
            return r
        tower = self.castle.towers[idx]
        if not tower.is_valid:
            r['desc'] = f"REJECTED: tower {idx} destroyed"
            return r
        target = action.target
        if target is None or not target.is_alive:
            r['desc'] = f"REJECTED: invalid tower target"
            return r
        if target.team != self.attacker_team:
            r['desc'] = "REJECTED: tower must target attacker"
            return r
        # Towers shoot the same way the round loop does.
        dmg = tower.roll_damage()
        if dmg <= 0:
            r['desc'] = f"tower {idx} misses {target.name}"
            return r
        dfn_def = target.defense
        if self._in_moat(target):
            dfn_def = max(0, dfn_def - 3)
        if tower.attack > dfn_def:
            mult = min(1 + 0.1 * (tower.attack - dfn_def), 3.0)
        else:
            mult = max(1 - 0.05 * (dfn_def - tower.attack), 0.3)
        actual_dmg = max(1, int(dmg * mult))
        actual, killed = target.take_damage(actual_dmg)
        r['dmg'] = actual
        r['killed'] = killed
        r['target_alive'] = target.is_alive
        if killed > 0:
            self.deaths_this_round += 1
            self._record_kill(target, killed)
        r['desc'] = (f"tower {idx} shoots {target.name}: {actual} dmg"
                     + (f" ({killed} killed)" if killed else ""))
        return r

    def _execute_catapult(self, action: CatapultAction, r: dict) -> dict:
        """fheroes2 ApplyActionCatapult.

        Catapults fire only during a siege.  Each shot is a (target_id,
        damage, hit) triple; a ``NONE`` target_id is a no-op (the
        catapult didn't aim at a structure).
        """
        if self.castle is None:
            r['desc'] = "REJECTED: no castle for catapult"
            return r
        if not action.shots:
            r['desc'] = "catapult idle"
            return r
        applied = 0
        for target_id, damage, hit in action.shots:
            if target_id == CatapultAction.NONE:
                continue
            if not hit:
                continue
            applied += self.castle.apply_catapult_damage(target_id, damage)
        r['desc'] = f"catapult fires {len(action.shots)} shots" \
                    + (f" ({applied} applied)" if applied else "")
        return r

    def _cast(self, action: CastAction) -> dict:
        """Resolve a hero spellcast: damage, buff, debuff, control, etc.

        Dispatches by ``spell.kind`` to kind-specific helpers.  Mass spells
        iterate over all valid targets; AOE spells resolve area patterns.
        """
        r = {'desc': '', 'dmg': 0, 'killed': 0,
             'ret_dmg': 0, 'ret_killed': 0,
             'target_alive': True, 'attacker_alive': True, 'cast': True}
        hero = self.heroes.get(action.team)
        spell, tgt = action.spell, action.target
        if hero is None:
            return r
        hero.cast(spell)

        # ── single-target immunity pre-check ───────────────
        # (mass / AOE spells check per-target inside their helpers)
        if not spell.is_mass and spell.kind not in (AOE, UTILITY):
            if tgt.is_immune_to_spells:
                r['desc'] = (f"{hero.name} casts {spell.name} "
                             f"-> BLOCKED (Anti-Magic)")
                return r
            if (spell.kind in (DAMAGE, DEBUFF, CONTROL)
                    and self._try_spell_resist(tgt, hero, spell)):
                r['desc'] = (f"{hero.name} casts {spell.name} on {tgt.name} "
                             f"-> RESISTED")
                return r

        # ── dispatch by kind ───────────────────────────────
        if spell.kind == DAMAGE:
            self._cast_damage(r, hero, spell, tgt)
        elif spell.kind == AOE:
            self._cast_aoe(r, hero, spell, action)
        elif spell.kind in (BUFF, DEBUFF, CONTROL):
            if spell.is_mass:
                self._cast_mass_effect(r, action.team, hero, spell)
            else:
                # fheroes2 isMindInfluence() covers Blind/Paralyze/Berserker/
                # Hypnotize (spell.cpp:428-440).  Undead / Elemental / any unit
                # with MIND_SPELL_IMMUNITY have 100% resistance to mind
                # influence (monster_info.cpp:890-898).
                if spell.name in ("Blind", "Paralyze") and tgt.is_immune_to_mind:
                    r['desc'] = (f"{hero.name} casts {spell.name} on {tgt.name} "
                                 f"-> RESISTED (mind-immune)")
                    return r
                tgt.add_effect(make_effect(spell, hero.power))
                # fheroes2: hero spell Blind prevents retaliation.
                if spell.name == "Blind":
                    tgt.blind_retaliation = False
                r['desc'] = f"{hero.name} casts {spell.name} on {tgt.name}"
        elif spell.kind == DISPEL:
            self._cast_dispel(r, hero, spell, tgt)
        elif spell.kind == CURE:
            if spell.is_mass:
                self._cast_mass_cure(r, action.team, hero, spell)
            else:
                self._apply_cure_unit(r, hero, spell, tgt)
        elif spell.kind == UTILITY:
            self._cast_utility(r, hero, spell, action, tgt)
        elif spell.kind == HYPNOTIZE:
            self._cast_hypnotize(r, hero, spell, tgt)
        elif spell.kind == BERSERKER:
            self._cast_berserker(r, hero, spell, tgt)
        elif spell.kind == RESURRECT:
            self._cast_resurrect(r, hero, spell, tgt)
        elif spell.kind == SUMMON:
            self._cast_summon(r, hero, spell, action, tgt)
        elif spell.kind == MIRROR_IMAGE:
            # Mirror Image shares the summon dispatcher (it spawns a phantom
            # of the target).  fheroes2 ApplyActionSpellMirrorImage is
            # handled by the same generic summon path.  We preserve flag
            # name so the AI heuristic can recognise a Mirror Image cast.
            self._cast_summon(r, hero, spell, action, tgt)

        return r

    # ── spell helpers ──────────────────────────────────────────

    def _try_spell_resist(self, unit: Unit, hero, spell) -> bool:
        """True if *unit* resists the spell via magic_resistance ability."""
        if unit.has_ability("magic_resistance"):
            params = unit.ability_params.get("magic_resistance", {})
            chance = params.get("chance", 0)
            if chance > 0 and random.randint(1, 100) <= chance:
                return True
        return False

    @staticmethod
    def _apply_elemental_reduction(unit: Unit, spell, dmg: int) -> int:
        """Reduce elemental spell damage for Golem-type units.

        fheroes2 battle_troop.cpp:1302 — Iron/Steel Golem take 50% damage from
        elemental spells (ELEMENTAL_SPELL_DAMAGE_REDUCTION).
        """
        if spell.elemental and unit.has_ability("elemental_spell_reduction"):
            params = unit.ability_params.get("elemental_spell_reduction", {})
            factor = params.get("factor", 0.5)
            dmg = max(1, int(dmg * factor))
        return dmg

    def _cast_damage(self, r: dict, hero, spell, tgt: Unit) -> None:
        """Resolve a single-target DAMAGE spell."""
        dmg = spell_damage(spell, hero.power)
        # Golem elemental spell reduction (battle_troop.cpp:1302).
        dmg = self._apply_elemental_reduction(tgt, spell, dmg)
        actual, killed = tgt.take_damage(dmg)
        r['dmg'] = actual
        r['killed'] = killed
        r['target_alive'] = tgt.is_alive
        desc = f"{hero.name} casts {spell.name} on {tgt.name}: {actual} dmg"
        if killed > 0:
            desc += f" ({killed} killed)"
            # fheroes2 battle_troop.cpp:653 — every casualty bumps the
            # unit's _deadCount; match per-casualty attribution so the
            # per-army totals reflect spell damage accurately.
            self._record_kill(tgt, killed)
        if not tgt.is_alive:
            self.deaths_this_round += 1
            desc += " [DEAD]"
        r['desc'] = desc

    def _apply_spell_damage(self, r: dict, hero, spell, unit: Unit,
                            dmg: int) -> tuple:
        """Apply spell damage to *unit*, checking immunity.

        Returns (actual, killed).  Updates ``r`` accumulatively.
        """
        if unit.is_immune_to_spells:
            return 0, 0
        if self._try_spell_resist(unit, hero, spell):
            return 0, 0
        # Golem elemental spell reduction (battle_troop.cpp:1302).
        dmg = self._apply_elemental_reduction(unit, spell, dmg)
        actual, killed = unit.take_damage(dmg)
        r['dmg'] += actual
        r['killed'] += killed
        if killed > 0:
            # fheroes2 battle_troop.cpp:653 — per-casualty bookkeeping.
            self._record_kill(unit, killed)
        if not unit.is_alive:
            self.deaths_this_round += 1
        return actual, killed

    def _aoe_cells(self, center: tuple, pattern: str) -> set:
        """Cells hit by an area spell centred on *center*."""
        if pattern == "ring1":
            cells = {center}
            cells.update(self.grid.neighbors(*center))
            return cells
        if pattern == "ring2":
            cells = {center}
            ring1 = set(self.grid.neighbors(*center))
            cells.update(ring1)
            for c in ring1:
                cells.update(self.grid.neighbors(*c))
            return cells
        if pattern == "ring_outer":
            return set(self.grid.neighbors(*center))
        return set()

    def _cast_aoe(self, r: dict, hero, spell, action: CastAction) -> None:
        """Resolve an AOE spell (ring, chain, or army-wide)."""
        pattern = spell.aoe_pattern
        base_dmg = spell_damage(spell, hero.power)

        if pattern in ("ring1", "ring2", "ring_outer"):
            center = action.cell if action.cell else action.target.pos
            cells = self._aoe_cells(center, pattern)
            desc = f"{hero.name} casts {spell.name}"
            for cell in cells:
                unit = self.unit_at(cell)
                if unit and unit.is_alive:
                    actual, killed = self._apply_spell_damage(
                        r, hero, spell, unit, base_dmg)
                    if actual > 0:
                        desc += f" | {unit.name}:{actual}"
                        if killed > 0:
                            desc += f"({killed}k)"
            r['desc'] = desc

        elif pattern == "chain":
            # Chain Lightning: initial target + up to 3 nearest bounces.
            desc = f"{hero.name} casts Chain Lightning"
            hit: list = []
            current = action.target
            dmg = base_dmg
            for _ in range(4):
                if current is None or not current.is_alive:
                    break
                if current in hit:
                    break
                hit.append(current)
                actual, killed = self._apply_spell_damage(
                    r, hero, spell, current, dmg)
                if actual > 0:
                    desc += f" | {current.name}:{actual}"
                # Find nearest alive unit for next bounce.
                candidates = [u for u in self.alive() if u not in hit]
                if candidates:
                    current = min(
                        candidates,
                        key=lambda u: (abs(u.col - current.col)
                                       + abs(u.row - current.row)))
                else:
                    break
                dmg = max(1, dmg // 2)
            r['desc'] = desc

        elif pattern in ("all_tagged", "all_units"):
            # Army-wide: damage every unit matching tag criteria.
            desc = f"{hero.name} casts {spell.name}"
            for unit in self.alive():
                if spell.target_tags:
                    if not all(unit.has_tag(t) for t in spell.target_tags):
                        continue
                if spell.exclude_tags:
                    if any(unit.has_tag(t) for t in spell.exclude_tags):
                        continue
                actual, killed = self._apply_spell_damage(
                    r, hero, spell, unit, base_dmg)
                if actual > 0:
                    desc += f" | {unit.name}:{actual}"
                    if killed > 0:
                        desc += f"({killed}k)"
            r['desc'] = desc

    def _cast_mass_effect(self, r: dict, team: int, hero, spell) -> None:
        """Resolve a mass BUFF / DEBUFF / CONTROL spell."""
        targets = (self.alive(team) if spell.side_friendly
                   else self.alive(1 - team))
        desc = f"{hero.name} casts {spell.name}"
        for unit in targets:
            if spell.exclude_tags:
                if any(unit.has_tag(t) for t in spell.exclude_tags):
                    continue
            if unit.is_immune_to_spells:
                continue
            if (not spell.side_friendly
                    and self._try_spell_resist(unit, hero, spell)):
                continue
            if unit.has_effect(spell.name):
                continue
            unit.add_effect(make_effect(spell, hero.power))
            # fheroes2: spell-Blind prevents retaliation; ability-blind does not.
            if spell.name == "Blind":
                unit.blind_retaliation = False
            desc += f" | {unit.name}"
        r['desc'] = desc

    def _cast_dispel(self, r: dict, hero, spell, tgt: Unit) -> None:
        """Resolve Dispel Magic / Mass Dispel.

        fheroes2 battle_troop.cpp:1706 + battle_action.cpp:262:
          * Dispel Magic — removes only GOOD_MAGIC (beneficial spells) from
            a single target.
          * Mass Dispel — removes ALL affection from every alive unit.
        """
        if spell.is_mass:
            for unit in self.alive():
                if not unit.is_immune_to_spells:
                    unit.effects.clear()
            r['desc'] = f"{hero.name} casts Mass Dispel"
        else:
            if not tgt.is_immune_to_spells:
                # fheroes2 Dispel: removeAffection( IS_GOOD_MAGIC ) only.
                tgt.effects = [e for e in tgt.effects if not e.is_positive]
            r['desc'] = f"{hero.name} casts {spell.name} on {tgt.name}"

    def _apply_cure_unit(self, r: dict, hero, spell, unit: Unit) -> None:
        """Cure one unit: remove debuffs + heal HP."""
        unit.effects = [e for e in unit.effects if e.is_positive]
        heal_amount = spell.heal_base * hero.power
        healed = unit.heal(heal_amount)
        r['dmg'] = -healed  # negative signals healing
        r['desc'] = (f"{hero.name} casts {spell.name} on {unit.name}"
                     f": +{healed} HP")

    def _cast_mass_cure(self, r: dict, team: int, hero, spell) -> None:
        """Mass Cure: remove debuffs + heal all friendly units."""
        desc = f"{hero.name} casts {spell.name}"
        for unit in self.alive(team):
            if unit.is_immune_to_spells:
                continue
            unit.effects = [e for e in unit.effects if e.is_positive]
            heal_amount = spell.heal_base * hero.power
            healed = unit.heal(heal_amount)
            if healed > 0:
                desc += f" | {unit.name}+{healed}"
        r['desc'] = desc

    def _cast_utility(self, r: dict, hero, spell, action: CastAction,
                      tgt: Unit) -> None:
        """Resolve utility spells (Teleport, Earthquake)."""
        if spell.name == "Teleport":
            if action.destination:
                tgt.pos = action.destination
                r['desc'] = (f"{hero.name} casts Teleport: {tgt.name} "
                             f"-> {action.destination}")
            else:
                r['desc'] = f"{hero.name} casts Teleport (no destination)"
        elif spell.name == "Earthquake":
            if self.castle:
                # fheroes2 ApplyActionSpellEarthquake: damage-shake is
                # driven by ``commander->GetPower()`` *only*, NOT by the
                # catapults.  See battle_arena.cpp getEarthQuakeSpellTargets
                # for the damage range and battle_action.cpp:1510 for the
                # dispatch.  We delegate the actual shake to
                # ``castle.earthquake(power)`` which mirrors that C++
                # logic.
                self.castle.earthquake(hero.power)
                r['desc'] = f"{hero.name} casts Earthquake"
            else:
                r['desc'] = f"{hero.name} casts Earthquake (open field)"
        else:
            r['desc'] = f"{hero.name} casts {spell.name}"

    # ── mind-control spells (fheroes2 battle_arena.cpp) ─────

    def _cast_hypnotize(self, r: dict, hero, spell, tgt: Unit) -> None:
        """Hypnotize: if target is alive and not immune and total HP is
        below ``hp_threshold * power``, take over its allegiance.

        A hypnotized stack deals damage to its original team; retaliations
        against it are suppressed (``Unit.is_hypnotized`` + ``execute``).

        fheroes2 monster_info.cpp:890-898 — undead, elemental, and any
        monster with MIND_SPELL_IMMUNITY resist mind spells entirely.
        """
        threshold = spell.hp_threshold_per_power * hero.power
        if not tgt.is_alive:
            r['desc'] = (f"{hero.name} casts Hypnotize on {tgt.name} — "
                         f"no effect (dead)")
            return
        if tgt.is_immune_to_mind:
            r['desc'] = (f"{hero.name} casts Hypnotize on {tgt.name} — "
                         f"RESISTED (mind-immune)")
            return
        if tgt._total_hp < threshold:
            tgt.add_effect(make_effect(spell, hero.power))
            r['desc'] = (f"{hero.name} casts Hypnotize on {tgt.name}! "
                         f"(total HP {tgt._total_hp} < {threshold})")
        else:
            r['desc'] = (f"{hero.name} casts Hypnotize on {tgt.name} — "
                         f"resisted (HP {tgt._total_hp} ≥ {threshold})")

    def _cast_berserker(self, r: dict, hero, spell, tgt: Unit) -> None:
        """Berserker: target always attacks nearest neighbor (no AI choice).

        fheroes2: undead/elemental/mind-immune units resist (monster_info.cpp).
        """
        if not tgt.is_alive:
            r['desc'] = f"{hero.name} casts Berserker on {tgt.name} — no effect"
            return
        if tgt.is_immune_to_mind:
            r['desc'] = (f"{hero.name} casts Berserker on {tgt.name} — "
                         f"RESISTED (mind-immune)")
            return
        if not tgt.is_hypnotized:
            tgt.add_effect(make_effect(spell, hero.power))
            r['desc'] = f"{hero.name} casts Berserker on {tgt.name}!"
        else:
            r['desc'] = f"{hero.name} casts Berserker on {tgt.name} — resisted"

    # ── resurrection & summoning (battle_arena.cpp SP_* spawns) ──

    def _find_empty_cell_adjacent(self, team: int) -> Optional[Tuple[int, int]]:
        """Pick an empty cell adjacent to a friendly stack. Fallback to the
        first empty cell on the board (fheroes2 summon placement is best-effort).
        """
        occupied = self.occupied()
        # Prefer neighbours of friendly units.
        for fu in self.alive(team):
            for col, row in self.grid.neighbors(fu.col, fu.row):
                if (col, row) not in occupied:
                    return (col, row)
        for col in range(self.grid.cols):
            for row in range(self.grid.rows):
                if (col, row) not in occupied:
                    return (col, row)
        return None

    def _cast_resurrect(self, r: dict, hero, spell, tgt: Unit) -> None:
        """Resurrect / Resurrect True / Animate Dead.

        fheroes2 battle_arena.cpp: revives dead creatures of the matching
        faction, up to ``spell_power * resurrect_per_power`` HP. Living units
        of the same faction are healed to full HP (Resurrect / Resurrect True /
        Animate Dead all also heal on living targets).
        """
        # 1. Heal any *living* unit of the same faction.
        per_unit_heal = spell.resurrect_per_power * hero.power
        if tgt.is_alive:
            healed = tgt.heal(per_unit_heal)
            r['dmg'] = -healed
        # 2. Try to revive any dead stack of the same faction.
        revived = self._revive_dead(hero, spell, per_unit_heal)
        r['desc'] = (f"{hero.name} casts {spell.name}"
                     + (f" (+{r['dmg']} HP on {tgt.name})" if tgt.is_alive and r.get('dmg', 0) < 0 else "")
                     + (f" | revived {revived.name}" if revived else ""))

    def _revive_dead(self, hero, spell, hp_pool: int) -> Optional[Unit]:
        """Walk graveyard of the caster's side and revive one stack.

        ``self.dead`` is populated lazily by ``_bury_dead`` whenever a unit
        dies — fheroes2 keeps corpse info in a separate `Battle::Graveyard`
        pool (battle_arena.cpp:259). If the pool hasn't been populated yet
        (e.g. in early-game tests where dead units are filtered out by
        ``alive()``), the resurrection simply heals living targets instead.
        """
        graveyard = list(getattr(self, "dead", []) or [])
        # Sort: resurrect-specific undead faction first
        graveyard.sort(key=lambda u: 0 if any(t in spell.target_tags for t in u.tags) else 1)
        for corpse in graveyard:
            if any(tag in spell.exclude_tags for tag in corpse.tags):
                continue
            # `target_tags` (e.g. undead) must match when set.
            if spell.target_tags and not any(t in spell.target_tags for t in corpse.tags):
                continue
            place = self._find_empty_cell_adjacent(corpse.team)
            if place is None:
                continue
            col, row = place
            initial_count = getattr(corpse, "original_count", corpse.count) or 1
            revived = Unit.from_type(
                corpse.name, corpse.team, col, row,
                count=initial_count,
            )
            # Cap revived count by HP pool (fheroes2 rule).
            while revived._total_hp > hp_pool and revived.count > 1:
                revived.count -= 1
                revived._total_hp = revived.count * revived.max_hp
            revived._is_alive = True
            revived.original_count = initial_count
            self.units.append(revived)
            # Move corpse out of dead pool.
            self.dead.remove(corpse)
            return revived
        return None

    def _bury_dead(self, unit: Unit) -> None:
        """Move *unit* from active units to graveyard (preserves revive)."""
        if unit not in self.units:
            return
        # Stash original stack size for Resurrect's revival budget.
        if not hasattr(unit, "original_count"):
            unit.original_count = unit.count
        self.units.remove(unit)
        if not hasattr(self, "dead") or self.dead is None:
            self.dead = []
        self.dead.append(unit)

    def _cast_summon(self, r: dict, hero, spell, action: CastAction,
                     tgt: Unit) -> None:
        """Mirror Image / Elemental Summons.

        fheroes2 battle_arena.cpp:
          * Mirror Image — clones target on adjacent empty cell, count = 1.
          * Elemental Summons — places the matching elemental at the action's
            destination (or first free cell), count = power * summon_count_per_power.
        """
        if spell.name == "Mirror Image":
            place = action.destination or self._find_empty_cell_adjacent(tgt.team)
            if place is None:
                r['desc'] = f"{hero.name} casts Mirror Image (no room)"
                return
            col, row = place
            mirror = Unit.mirror_image(tgt, tgt.team)
            mirror.col, mirror.row = col, row
            self.units.append(mirror)
            r['desc'] = (f"{hero.name} casts Mirror Image on {tgt.name} "
                         f"({col},{row})")
        else:
            count = spell.summon_count_per_power * hero.power
            place = action.destination or self._find_empty_cell_adjacent(action.team)
            if place is None:
                r['desc'] = f"{hero.name} casts {spell.name} (no room)"
                return
            col, row = place
            summoned = Unit.from_type(
                spell.summon_unit_type, action.team, col, row,
                count=count,
            )
            self.units.append(summoned)
            r['desc'] = (f"{hero.name} casts {spell.name} → "
                         f"{count}×{spell.summon_unit_type} @({col},{row})")

    # ── victory ─────────────────────────────────────────────

    # fheroes2 MAX_TURNS_WITHOUT_DEATHS: the attacker retreats after this many
    # death-free rounds, breaking stalemates. MAX_ROUNDS is an absolute backstop.
    MAX_TURNS_WITHOUT_DEATHS = 50
    MAX_ROUNDS = 200

    def is_stalemate(self) -> bool:
        return self._stale_rounds >= self.MAX_TURNS_WITHOUT_DEATHS

    def retreat(self, team: int) -> None:
        """Record that `team`'s hero has fled; ends the battle, that side loses."""
        self._retreated = team

    def is_over(self) -> bool:
        return (self._retreated is not None
                or len(self.alive(0)) == 0 or len(self.alive(1)) == 0
                or self.is_stalemate()
                or self.round_num >= self.MAX_ROUNDS)

    def winner(self) -> int:
        # A hero fled -> the other side wins.
        if self._retreated is not None:
            return 1 - self._retreated
        if not self.alive(0):
            return 1
        if not self.alive(1):
            return 0
        # Death-free stalemate: the attacking side gives up (fheroes2 retreat).
        if self.is_stalemate():
            return 1 - self.attacker_team
        # Absolute backstop reached — winner by remaining army strength.
        s0 = sum(u.strength for u in self.alive(0))
        s1 = sum(u.strength for u in self.alive(1))
        return 0 if s0 >= s1 else 1

    def start_round(self):
        # ── Bury any units that died last round so Resurrect can find them.
        # fheroes2 Battle::Arena::endTurn / pushBattleArenaDeadUnitsToGraveyard.
        if not hasattr(self, "dead") or self.dead is None:
            self.dead = []
        # Snapshot before iterating (we mutate self.units via _bury_dead).
        for u in list(self.units):
            if not u.is_alive and u not in self.dead:
                self._bury_dead(u)
        # Update the death-free streak based on the round that just finished.
        if self.round_num >= 1:
            if self.deaths_this_round == 0:
                self._stale_rounds += 1
            else:
                self._stale_rounds = 0
        self.round_num += 1
        self.deaths_this_round = 0
        for u in self.alive():
            u.new_round()
            u.tick_effects()
            if u.has_ability("self_heal"):   # regeneration (Troll-like)
                u.heal(u.max_hp)
        for hero in self.heroes.values():
            if hero is not None:
                hero.reset_round()

        # ── Siege: catapult + tower actions ─────────────────────
        # fheroes2 Turns(): catapult fires during first attacker-unit turn;
        # towers fire during first defender-unit turn. We run both here at
        # round start (headless simplification) to keep the game loop simple.
        if self.castle:
            self._catapult_round()
            self._tower_round()

    # ── siege helpers ────────────────────────────────────────────

    def _catapult_round(self):
        """Catapult fires once per round (attacker siege weapon).

        fheroes2: CatapultAction() in battle_arena.cpp — the catapult targets
        intact walls, then towers, then the bridge. 75% hit, 1 damage.

        M7d: Ballistics skill modifies shots, hit chance, and damage.
        """
        # Get attacker hero's Ballistics skill level.  fheroes2: the catapult
        # only fires when the attacker has a hero commanding the siege;
        # without a hero there is noone to operate the siege weapon.
        hero = self.heroes.get(self.attacker_team)
        if hero is None:
            return
        ballistics = (hero.get_skill_level("ballistics")
                      if hasattr(hero, "get_skill_level")
                      else 0)
        shots = self.castle.catapult_round(ballistics=ballistics)
        for shot in shots:
            if shot["hit"] and shot["damage"] > 0:
                # Wall/tower/bridge damage already applied inside catapult_round().
                pass  # result recorded for UI/logging if needed

    def _tower_round(self):
        """Each active tower shoots the highest-threat enemy once per round.

        fheroes2: TowerAction() — towers fire during the first defender-unit
        turn. Order: center, left, right (battle_arena.cpp:623-625).
        """
        if not self.castle:
            return
        for tower in self.castle.towers:
            if not tower.is_valid:
                continue
            # Tower shoots attacker units (team 0 in siege).
            enemies = self.alive(self.attacker_team)
            if not enemies:
                break
            target = tower.select_target(enemies)
            if target is None:
                continue
            dmg = tower.roll_damage()
            if dmg <= 0:
                continue
            # Tower attack uses the same _damage_mult as normal combat.
            # Tower is a pseudo-archer (attack=5) vs target's defense.
            dfn_def = target.defense
            if self._in_moat(target):
                dfn_def = max(0, dfn_def - 3)
            if tower.attack > dfn_def:
                mult = min(1 + 0.1 * (tower.attack - dfn_def), 3.0)
            else:
                mult = max(1 - 0.05 * (dfn_def - tower.attack), 0.3)
            actual_dmg = max(1, int(dmg * mult))
            actual, killed = target.take_damage(actual_dmg)
            if killed > 0:
                self.deaths_this_round += 1
