"""ClassicAI melee decision helpers — strict port of fheroes2 ``ai_battle.cpp``.

This module mirrors, line-for-line where practical:

  * ``AI::BattlePlanner::MeleeAttackOutcome``        (ai_battle.cpp:74-96)
  * ``AI::BattlePlanner::doubleCellAttackValue``     (ai_battle.cpp:98-108)
  * ``AI::BattlePlanner::optimalAttackVector``       (ai_battle.cpp:110-158)
  * ``AI::BattlePlanner::optimalAttackValue``        (ai_battle.cpp:160-198)
  * ``AI::BattlePlanner::evaluatePotentialAttackPositions``
                                                       (ai_battle.cpp:202-268)
  * ``AI::BattlePlanner::isUnitAbleToApproachPosition`` (ai_battle.cpp:271-295)
  * ``AI::BattlePlanner::BestAttackOutcome``         (ai_battle.cpp:297-349)
  * ``AI::BattlePlanner::findOptimalPositionForSubsequentAttack``
                                                       (ai_battle.cpp:351-420)
  * ``AI::BattlePlanner::EvaluateAttackValue``       (ai_battle.cpp:412-460)
  * ``AI::BattlePlanner::findNearestCellNextToUnit`` (ai_battle.cpp:429-459)
  * ``AI::BattlePlanner::getUnitMovementTarget``     (ai_battle.cpp:461-480)
  * ``AI::BattlePlanner::getMeleeBestOutcome``       (ai_battle.cpp:1539-1566)
  * ``AI::BattlePlanner::meleeUnitOffense``          (ai_battle.cpp:1568-1705)
  * ``AI::BattlePlanner::meleeUnitDefense``          (ai_battle.cpp:1708-2065)
  * ``AI::BattlePlanner::isPositionLocatedInDefendedArea`` (ai_battle.cpp:2067)

The selected source cell (``fromIndex``) and target unit are preserved
all the way from ``getMeleeBestOutcome`` through ``_melee_offense`` /
``_melee_defense`` into the produced ``AttackAction.from_pos``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ai_core.battle_geometry import _can_attack_from_pos, _pos_dist, _tail_dir
from engine.actions import Action, AttackAction, MoveAction, SkipAction
from engine.battle_state import BattleState
from engine.unit import Unit

from .archer import (
    _evaluate_threat_for_unit,
    _is_hand_fighting,
    _is_unit_able_to_approach_pos,
    _unit_occupied_positions,
)
from .forces import _effective_enemies, _effective_friends, _enemy_for
from .movement import _unit_movement_position
from .models import (
    _DOUBLE_LOWEST,
    _is_outcome_improved,
    _MeleeAttackOutcome,
    _MeleePosition,
    _TargetPair,
)


# ── C++ direction priorities (clockwise, 6 dirs) ─────────────────────
# fheroes2 ``Direction`` enum: TOP_LEFT, TOP_RIGHT, RIGHT, BOTTOM_RIGHT,
# BOTTOM_LEFT, LEFT. Python's neighbour iteration isn't direction-tagged,
# so we sort cells by the named direction order around an anchor cell.
_DIRECTION_PRIORITY_KEYS = (
    # (dcol, drow) — row parity decides which diagonal appears "left" of
    # the cell, so we list both variants and let the per-row pick handle
    # the disambiguation.
    (-1, -1), (0, -1), (1, 0), (1, 1), (0, 1), (-1, 0),  # even rows
    (-1,  0), (0, -1), (1, 0), (0,  1), (1, 1), (-1,  1),  # odd rows
)


def _cell_priority_index(cell: Tuple[int, int],
                         anchor: Tuple[int, int],
                         row_parity: int) -> int:
    """Return a sort key that mirrors fheroes2 ``Board::GetAroundIndexes``
    direction ordering: TOP_LEFT, TOP_RIGHT, RIGHT, BOTTOM_RIGHT,
    BOTTOM_LEFT, LEFT."""
    dc = cell[0] - anchor[0]
    dr = cell[1] - anchor[1]
    table = _DIRECTION_PRIORITY_KEYS[row_parity & 1]
    for idx, offset in enumerate(table):
        if offset == (dc, dr):
            return idx
    # Unknown offset → highest key so it is picked last.
    return len(table)


# ── entry point ────────────────────────────────────────────────────────

def _melee_turn(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
    """fheroes2 ``meleeUnitTurn`` (top-level driver).

    Dispatches to offence or defence based on ``self._defensive_tactics``.
    Returns ``SkipAction`` when no enemy is in scope, and otherwise
    surfaces the C++ chosen ``fromIndex`` into ``AttackAction.from_pos``.
    """
    enemies = _effective_enemies(battle, unit)
    if not enemies:
        # C++ ``Battle::Command`` SKIP has no defending bit — the AI's
        # "defend" stance is a separate UI hot-key in fheroes2
        # (battle_command.h:82).  Old port's ``defending`` kwarg was a
        # dead field that the engine never read; drop it.  If the AI
        # wants to fortify, it should issue a MoveAction to stay put
        # (which still consumes the turn).
        return (SkipAction(unit),
                f"{unit.name} defends" if self._defensive_tactics
                else f"{unit.name} skips")

    target = (self._melee_defense(battle, unit, enemies)
              if self._defensive_tactics
              else self._melee_offense(battle, unit, enemies))

    if target.unit is not None:
        # C++ resolves the requested cell to a complete reachable Position,
        # not just a head/tail occupancy check.
        movement_pos = _unit_movement_position(
            self, battle, unit, target.from_index or target.cell or unit.pos)
        if movement_pos is not None:
            return (AttackAction(unit, target.unit,
                                 from_pos=movement_pos.head,
                                 from_position=movement_pos,
                                 ranged=False),
                    f"{unit.name} attacks {target.unit.name}")
    if target.cell is not None and target.cell != unit.pos:
        movement_pos = _unit_movement_position(self, battle, unit, target.cell)
        if movement_pos is not None:
            built = battle.build_path(unit, movement_pos.head)
            if built:
                path, final_position = built
                return (MoveAction(unit, path, final_position=final_position),
                        f"{unit.name} advances")
    return (SkipAction(unit),
            f"{unit.name} defends" if self._defensive_tactics
            else f"{unit.name} skips")


# ── offence ────────────────────────────────────────────────────────────

def _melee_offense(self, battle: BattleState, unit: Unit,
                   enemies: List[Unit]) -> _TargetPair:
    """fheroes2 ``meleeUnitOffense`` (ai_battle.cpp:1568-1705)."""
    # 1. Best immediate attack across all reachable positions.
    outcome = _get_melee_best_outcome(self, battle, unit, enemies)
    if outcome.target is not None and outcome.from_index is not None:
        # Preserve the C++ chosen ``fromIndex`` rather than recomputing
        # from ``target.pos`` (which is the OLD pre-port bug).
        return _TargetPair(
            cell=outcome.from_index,
            unit=outcome.target,
            from_index=outcome.from_index,
        )

    # 2. Distant target — pick the enemy with the highest
    # ``evaluateThreatForUnit / moveDistance`` ratio.
    cell = self._choose_distant_target(
        battle, unit, enemies,
        enemy_predicate=lambda e: (
            e.is_archer
            or e.speed == 0
            or (not e.is_flying and e.speed < unit.speed)))
    if cell is not None:
        return _TargetPair(cell=cell)

    # 3. Any reachable enemy (relaxed predicate).
    cell = self._choose_distant_target(
        battle, unit, enemies, enemy_predicate=lambda e: True)
    if cell is not None:
        return _TargetPair(cell=cell)

    # 4. Castle siege — walk to the nearest wall cell using the engine's
    # ``isPositionReachable`` + ``CalculateMoveDistance`` (fheroes2's
    # ``Arena::isPositionReachable`` pair).
    if self._attacking_castle:
        wall_target = self._nearest_wall_cell(battle, unit)
        if wall_target is not None:
            return _TargetPair(cell=wall_target)
    return _TargetPair()


def _choose_distant_target(self, battle: BattleState, unit: Unit,
                           enemies: List[Unit],
                           enemy_predicate) -> Optional[Tuple[int, int]]:
    """fheroes2 ``chooseDistantTarget`` lambda inside ``meleeUnitOffense``.

    C++ iterates enemies, computes ``moveDist = CalculateMoveDistance``,
    ranks by ``enemy.evaluateThreatForUnit(unit) / moveDist``, replacing
    ties because the comparator is strictly less than. The result cell
    is the path's last step (current-turn reachable neighbour), refined
    through ``findOptimalPositionForSubsequentAttack`` for cautious
    offensives, or the moat-stop cell when the castle has a moat.
    """
    best_cell: Optional[Tuple[int, int]] = None
    best_score = -math.inf
    for e in enemies:
        if not enemy_predicate(e) or not e.is_alive:
            continue
        nearest = battle.find_nearest_cell_next_to_unit(unit, e)
        if nearest is None:
            continue
        nearest_cell, move_dist, _ = nearest
        if move_dist <= 0:
            continue
        threat = _evaluate_threat_for_unit(battle, e, unit)
        score = threat / move_dist
        if score < best_score:
            continue
        built = battle.build_path(unit, nearest_cell)
        if built is None:
            continue
        path, _ = built
        if not path or len(path) <= 1:
            continue
        if (battle.castle is not None
                and getattr(battle.castle, "has_moat", False)
                and battle._is_moat_for_unit(unit, path[-1])):
            best_cell = path[-1]
        elif self._cautious_offensive:
            best_cell = self._optimal_next_attack_cell(
                battle, unit, path, enemies)
        else:
            best_cell = path[-1]
        best_score = score
    return best_cell


def _optimal_next_attack_cell(self, battle: BattleState, unit: Unit,
                              path: List[Tuple[int, int]],
                              enemies: List[Unit]
                              ) -> Optional[Tuple[int, int]]:
    """fheroes2 ``findOptimalPositionForSubsequentAttack`` (ai_battle.cpp:351).

    Walk the path step-by-step and pick the cell with the lowest
    *threat* from non-flying enemies that currently hand-fight or can
    reach the cell within their speed budget.

    Equal threats within ``0.001`` pick the LATER cell (the unit keeps
    walking closer to its target), mirroring the C++ ``<``-then-replace
    comparator.
    """
    best: Optional[Tuple[int, int]] = None
    best_threat = math.inf
    for cell in path:
        if cell == unit.pos:
            continue
        threat = 0.0
        for e in enemies:
            if not e.is_alive:
                continue
            if e.is_flying:
                continue
            if e.is_archer and not _is_hand_fighting(battle, e):
                continue
            if self._cell_is_threatened(battle, e, cell):
                threat += _evaluate_threat_for_unit(battle, e, unit)
        # C++ replaces equal threats only when the new one is strictly
        # less — encoded as ``<=`` here with a small epsilon.
        if threat <= best_threat + 1e-3:
            best_threat = threat
            best = cell
    return best if best is not None else (path[-1] if path else None)


def _nearest_wall_cell(self, battle: BattleState, unit: Unit
                       ) -> Optional[Tuple[int, int]]:
    walls = getattr(battle, "cells_under_walls", lambda: [])()
    if not walls:
        return None
    occ = battle._move_occupied(unit)
    best: Optional[Tuple[int, int]] = None
    best_dist = math.inf
    for cell in walls:
        if cell in occ and cell != unit.pos:
            continue
        if not battle.is_position_reachable(
                unit, cell, is_on_current_turn=False):
            continue
        d = battle.calculate_move_distance(unit, cell)
        if d <= 0:
            continue
        if d < best_dist:
            best_dist = d
            best = cell
    return best


# ── defence ────────────────────────────────────────────────────────────

def _melee_defense(self, battle: BattleState, unit: Unit,
                   enemies: List[Unit]) -> _TargetPair:
    """fheroes2 ``meleeUnitDefense`` (ai_battle.cpp:1708-2065).

    Branch order mirrors the C++ exactly:

      0. Skip if the unit's army color no longer matches ``_myColor``
         (hypnotized / re-affiliated).
      1. Skip if no friendly archer is alive.
      2. ``isAnyEnemyCanBeAttackedImmediately`` early return — if any
         enemy can already be engaged this turn, defend via an
         immediate attack rather than cover-position movement.
      3. Iterate archers; for each, pick the best cover cell using
         ``archerCoverCells``. The blocker outcome (C++ calls
         ``getMeleeBestOutcome`` over the single blocker inside the
         archer-selection loop) only contributes when its value beats
         the best *plain-cover* outcome so far.
      4. Adjacent-attack branch — engage any neighbour whose retaliation
         would be meaningful (skip the branch when the attacker ignores
         retaliation or when the friend has AREA_SHOT).
      5. Second-stage ``evaluatePotentialAttackPositions`` scan over
         defended-area cells; record the best score with initial value
         ``0.0`` (C++ semantics) and replace with
         ``optimalAttackValue`` per candidate.
    """
    # Step 0 — affiliation guard (fheroes2 ``unit.GetArmyColor() ==
    # _myColor``).
    if unit.team != self._my_color:
        return _TargetPair()

    friends = _effective_friends(battle, unit)
    archers = [f for f in friends if f is not unit and f.is_archer]
    if not archers:
        return _TargetPair()

    # Step 1 — early return when any enemy is already in melee range.
    if _any_enemy_can_be_attacked_immediately(self, battle, unit, enemies):
        return _TargetPair(cell=unit.pos)

    # Step 2 — non-flying ``distance > speed * 2`` skip per enemy.
    filtered_enemies = [
        e for e in enemies
        if e.is_flying
        or battle.grid.distance(unit.pos, e.pos) <= unit.speed * 2
    ]

    # Step 3 — archer cover selection with blocker outcome inside.
    distance_modifier = self._my_ranged_units_only / 15.0
    best_value = -math.inf
    best_target: Optional[Tuple[int, int]] = None
    best_unit: Optional[Unit] = None
    best_from_index: Optional[Tuple[int, int]] = None
    for archer in archers:
        # C++ considers blocker attacks only for the *best* archer; we
        # do the same by tracking the best cover cell first and asking
        # the blocker question per archer when that archer is the
        # current best.
        cover_cells = self._archer_cover_cells(battle, unit, archer)
        for cell, travel in cover_cells.items():
            if not self._in_defended_area(battle, unit, cell):
                continue
            value = archer.strength - travel * distance_modifier
            if value > best_value:
                best_value = value
                best_target = cell
                best_unit = None
                best_from_index = None
        # Blocker-outcome loop: C++ iterates blockers per archer.
        for blocker in filtered_enemies:
            if battle.grid.distance(archer.pos, blocker.pos) != 1:
                continue
            outcome = _get_melee_best_outcome(
                self, battle, unit, [blocker])
            if outcome.target is None:
                continue
            # The blocker outcome's value uses the same archer as the
            # one currently being considered.
            blocker_value = archer.strength - distance_modifier
            if blocker_value > best_value:
                best_value = blocker_value
                best_target = unit.pos
                best_unit = blocker
                best_from_index = outcome.from_index

    if best_unit is not None and best_from_index is not None:
        return _TargetPair(cell=best_target, unit=best_unit,
                           from_index=best_from_index)
    if best_target is not None:
        return _TargetPair(cell=best_target)

    # Step 4 — adjacent-attack branch.
    if not unit.has_ability("no_enemy_retaliation"):
        best_attack_value = 0.0  # C++ initial value (adjacent branch)
        best_adj_unit: Optional[Unit] = None
        best_adj_from: Optional[Tuple[int, int]] = None
        for frnd in friends:
            if frnd is unit or not frnd.is_alive:
                continue
            if frnd.has_ability("area_shot"):
                continue
            if frnd.has_ability("no_enemy_retaliation"):
                continue
            for e in enemies:
                if not e.is_alive:
                    continue
                if battle.grid.distance(frnd.pos, e.pos) != 1:
                    continue
                value = _optimal_attack_value(self, battle, unit, e)
                if value > best_attack_value:
                    best_attack_value = value
                    best_adj_unit = e
                    best_adj_from = unit.pos
        if best_adj_unit is not None:
            return _TargetPair(cell=unit.pos, unit=best_adj_unit,
                               from_index=best_adj_from or unit.pos)

    # Step 5 — defended-area scan via ``evaluatePotentialAttackPositions``.
    cells = self._defended_area_cells(battle, unit)
    best_attack_value = 0.0
    best_unit = None
    best_from_index = None
    for cell in cells:
        if not battle.is_position_reachable(
                unit, cell, is_on_current_turn=False):
            continue
        if not self._in_defended_area(battle, unit, cell):
            continue
        # Evaluate attacks available from this candidate position.
        for e in enemies:
            if not e.is_alive:
                continue
            if battle.grid.distance(cell, e.pos) != 1:
                continue
            value = _optimal_attack_value(self, battle, unit, e, from_cell=cell)
            if value > best_attack_value:
                best_attack_value = value
                best_unit = e
                best_from_index = cell
    if best_unit is not None:
        return _TargetPair(cell=unit.pos, unit=best_unit,
                           from_index=best_from_index or unit.pos)

    return _TargetPair()


def _any_enemy_can_be_attacked_immediately(self, battle: BattleState,
                                           unit: Unit,
                                           enemies: List[Unit]) -> bool:
    """fheroes2 ``isAnyEnemyCanBeAttackedImmediately``.

    True iff at least one enemy is in melee range right now. Used to
    short-circuit the defence cover branch.
    """
    for e in enemies:
        if not e.is_alive:
            continue
        if battle.grid.distance(unit.pos, e.pos) <= 1:
            return True
    return False


def _defended_area_cells(self, battle: BattleState,
                         unit: Unit) -> List[Tuple[int, int]]:
    """All cells the unit's body could occupy inside the defended area.

    Combines ``_in_defended_area`` with the current-turn reachable set,
    returning just the head cells (the engine represents wide units by
    their head position).
    """
    occ = battle._move_occupied(unit)
    out: List[Tuple[int, int]] = []
    for cell in self._reachable_cells(battle, unit):
        if cell in occ and cell != unit.pos:
            continue
        if cell != unit.pos and not self._in_defended_area(
                battle, unit, cell):
            continue
        out.append(cell)
    return out


# ── getMeleeBestOutcome / BestAttackOutcome ────────────────────────────

def _get_melee_best_outcome(self, battle: BattleState, unit: Unit,
                            enemies: List[Unit]
                            ) -> "_BestOutcomeResult":
    """fheroes2 ``getMeleeBestOutcome`` (ai_battle.cpp:1539).

    Returns the best attack across (enemy, position) pairs and the
    C++ chosen source cell. Only attacks that can be performed this
    turn (``canAttackImmediately``) contribute; position-only scores
    do not yield a target.

    Returns ``_BestOutcomeResult`` carrying ``attack_value`` (NOT a
    composite — C++ returns ``bestOutcome.attackValue`` to callers),
    the chosen target and ``from_index``.
    """
    best = _best_attack_outcome(self, battle, unit, enemies)
    if best.can_attack_immediately:
        return _BestOutcomeResult(
            attack_value=best.attack_value,
            target=_best_attack_target_for_outcome(
                self, battle, unit, enemies, best),
            from_index=best.from_index,
        )
    return _BestOutcomeResult(attack_value=_DOUBLE_LOWEST,
                              target=None, from_index=None)


def _best_attack_outcome(self, battle: BattleState, unit: Unit,
                         enemies: List[Unit]) -> _MeleeAttackOutcome:
    """fheroes2 ``BestAttackOutcome`` (ai_battle.cpp:297).

    Walks every reachable attack position; for each candidate cell that
    lets us engage at least one enemy this turn, builds a
    ``_MeleeAttackOutcome`` and replaces the running best using
    ``_is_outcome_improved``.
    """
    best = _MeleeAttackOutcome()
    reach = self._reachable_cells(battle, unit)
    if unit.pos not in reach:
        reach[unit.pos] = 0
    for enemy in enemies:
        if not enemy.is_alive:
            continue
        candidates = _candidate_attack_positions(
            self, battle, unit, enemy)
        for pos, position_value in candidates:
            if not _can_attack_from_pos(
                    battle.grid, unit, enemy, pos):
                continue
            attack_value = _optimal_attack_value(
                self, battle, unit, enemy, from_cell=pos)
            outcome = _MeleeAttackOutcome(
                from_index=pos,
                position_value=position_value,
                attack_value=attack_value,
                can_attack_immediately=True,
            )
            if _is_outcome_improved(outcome, best):
                best = outcome
    return best


def _best_attack_target_for_outcome(self, battle: BattleState, unit: Unit,
                                    enemies: List[Unit],
                                    outcome: _MeleeAttackOutcome
                                    ) -> Optional[Unit]:
    """Choose which enemy triggered ``outcome``.

    C++ picks the enemy whose ``optimalAttackValue`` matches the
    recorded ``attackValue`` AT the recorded ``fromIndex``. We mirror
    that by recomputing ``optimalAttackValue`` per enemy at that cell
    and selecting the matching max.
    """
    if outcome.from_index is None:
        return None
    chosen: Optional[Unit] = None
    best_val = _DOUBLE_LOWEST
    for e in enemies:
        if not e.is_alive:
            continue
        if battle.grid.distance(outcome.from_index, e.pos) > 1:
            continue
        val = _optimal_attack_value(
            self, battle, unit, e, from_cell=outcome.from_index)
        if val > best_val:
            best_val = val
            chosen = e
    return chosen


# ── evaluatePotentialAttackPositions ───────────────────────────────────

def _candidate_attack_positions(self, battle: BattleState, unit: Unit,
                               enemy: Unit
                               ) -> List[Tuple[Tuple[int, int], float]]:
    """fheroes2 ``evaluatePotentialAttackPositions`` (ai_battle.cpp:202).

    Enumerates the positions from which *unit* can engage *enemy* this
    turn, returning ``[(position, positionValue), ...]`` for the
    ``BestAttackOutcome`` loop.

    The C++ enumerates ``Board::GetDistanceIndexes`` around *enemy* (with
    radius 2 for a wide attacker), resolves each index into a complete
    position via ``Battle::Position::GetPosition``, and keeps the ones
    whose distance to the enemy equals 1 and that pass
    ``isPositionReachable(..., on_current_turn=false)``. This naturally
    includes the unit's current position when it is adjacent to the
    enemy — we replicate that by enumerating around-enemy indices and
    filtering by in-principle reachability.
    """
    out: List[Tuple[Tuple[int, int], float]] = []
    occ = battle._move_occupied(unit)
    radius = 2 if unit.is_wide else 1

    # Enumerate ``Board::GetDistanceIndexes`` cells in the C++ axial
    # coordinate order so the iteration matches fheroes2.
    sources = [(enemy.pos, 0)]
    if enemy.is_wide:
        tail = getattr(enemy, "tail_cell", None)
        if tail and tail != enemy.pos:
            sources.append((tail, 0))
    around: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    for source, _ in sources:
        col, row = source
        center_q = col - (row + (row & 1)) // 2
        center_r = row
        for dq in range(-radius, radius + 1):
            dr_min = max(-radius, -radius - dq)
            dr_max = min(radius, radius - dq)
            for dr in range(dr_min, dr_max + 1):
                if dq == 0 and dr == 0:
                    continue
                q = center_q + dq
                r = center_r + dr
                x = q + (r + (r & 1)) // 2
                y = r
                if (x, y) in seen:
                    continue
                if not battle.grid.is_valid(x, y):
                    continue
                seen.add((x, y))
                around.append((x, y))

    pathfinder = battle._pathfinder(unit)
    for cell in around:
        if not battle.grid.is_valid(*cell):
            continue
        position = battle._position_for_cell(unit, cell)
        if position is None:
            continue
        # Distance to enemy body must be exactly 1.
        body = {position.head}
        if position.tail is not None:
            body.add(position.tail)
        if min(battle.grid.distance(a, b)
               for a in body for b in enemy.occupied_cells()) != 1:
            continue
        # In-principle reachable — speed budget applies later in callers.
        if not pathfinder.is_position_reachable(position, False):
            continue
        if cell in occ and cell != unit.pos:
            continue
        position_value = _evaluate_threat_for_unit(battle, enemy, unit)
        out.append((cell, position_value))
    return out


# ── optimalAttackValue (fheroes2 ai_battle.cpp:160) ────────────────────

def _optimal_attack_value(self, battle: BattleState, atk: Unit, tgt: Unit,
                          from_cell: Optional[Tuple[int, int]] = None
                          ) -> float:
    """fheroes2 ``optimalAttackValue``.

    Base value = ``tgt.evaluateThreatForUnit(atk)`` (treating the
    attacker as the *attacker* in the threat call, mirroring the C++
    helper). For ``all_adjacent_attack`` units, sum the dedup-threat
    of every adjacent enemy; for ``two_cell_melee``, also add the
    secondary target hit via ``cell_behind``.
    """
    base_cell = from_cell if from_cell is not None else atk.pos
    value = _evaluate_threat_for_unit(battle, tgt, atk)

    # ALL_ADJACENT_ATTACK: sum the dedup-threat of every adjacent enemy
    # at the attack cell. C++ dedupes by ``GetHeadIndex()``.
    if atk.has_ability("all_adjacent_attack"):
        seen: set = {tgt.pos}
        for nb in battle.grid.neighbors(*base_cell):
            other = battle.unit_at(nb)
            if other is None or not other.is_alive:
                continue
            if other.pos in seen:
                continue
            seen.add(other.pos)
            value += _evaluate_threat_for_unit(battle, other, atk)
        return value

    # TWO_CELL_MELEE_ATTACK: splash into the cell behind the target.
    if atk.has_ability("two_cell_melee"):
        behind = battle.grid.cell_behind(base_cell, tgt.pos)
        if behind:
            splash = battle.unit_at(behind)
            if (splash is not None
                    and splash is not tgt
                    and _enemy_for(atk, splash)
                    and splash.is_alive):
                value += _evaluate_threat_for_unit(battle, splash, atk)
    return value


# ── EvaluateAttackValue (used by Teleport scorer integration) ──────────

def _evaluate_attack_value(self, battle: BattleState, atk: Unit, tgt: Unit,
                           dmg: int, from_cell: Tuple[int, int]) -> float:
    """fheroes2 ``EvaluateAttackValue`` (ai_battle.cpp:412).

    Plain damage-vs-retaliation score. C++ uses ``tgt.strength`` as the
    base, subtracts the retaliation cost, and adds threat from adjacent
    units dedup'd by ``GetHeadIndex()``. We delegate the adjacent
    threat computation to ``_evaluate_threat_for_unit`` to keep the
    semantics aligned with ``optimalAttackValue``.
    """
    if dmg <= 0:
        return 0.0
    score = float(tgt.strength)
    if not atk.has_ability("no_enemy_retaliation"):
        ret = battle.expected_damage(tgt, atk, ranged=False)
        score -= ret * atk.monster_strength / max(atk.max_hp, 1)
    seen: set = {tgt.pos}
    for nb in battle.grid.neighbors(*from_cell):
        other = battle.unit_at(nb)
        if other is None or other is tgt or not other.is_alive:
            continue
        if other.pos in seen:
            continue
        seen.add(other.pos)
        score += _evaluate_threat_for_unit(battle, other, atk)
    return score


# ── archer cover cell selection ────────────────────────────────────────

def _archer_cover_cells(self, battle: BattleState, unit: Unit,
                        archer: Unit
                        ) -> Dict[Tuple[int, int], int]:
    """fheroes2 ``archerCoverCells`` (ai_battle.cpp ~1800).

    Returns ``cell -> move_distance`` for covering positions around
    *archer*. C++ prefers distance = 2 for wide units (so the coverer
    stays one hex away from the archer and doesn't block line-of-sight);
    distance = 1 is used when the coverer is single-hex AND the archer
    has no friendly neighbour already at distance = 1.

    The C++ picks the cover cell using the six-direction priority
    (TOP_LEFT, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM_LEFT, LEFT) when
    multiple cells tie on travel distance. We replicate that here via
    ``_cell_priority_index``.
    """
    occ = battle._move_occupied(unit)
    wide_coverer = unit.is_wide
    # Determine the target distance — C++ prefers 2 for wide coverers,
    # but allows 1 when ``avoidStacking`` is false OR when the archer
    # has no friend already at distance 1.
    target_d = 2 if wide_coverer else 1
    if not wide_coverer:
        # C++: at distance 1, prefer NOT to stack on another friendly
        # neighbour of the archer. We approximate by checking whether
        # any *friend* (excluding the coverer) is currently at distance
        # 1 from the archer.
        friend_adjacent = any(
            battle.grid.distance(archer.pos, f.pos) == 1
            for f in _effective_friends(battle, unit)
            if f is not unit and f.is_alive)
        target_d = 1
        if friend_adjacent and not self._avoid_stacking_units:
            target_d = 1

    candidates: List[Tuple[Tuple[int, int], int]] = []
    for cell in self._reachable_cells(battle, unit):
        if cell == unit.pos:
            continue
        if cell in occ:
            continue
        d = battle.grid.distance(cell, archer.pos)
        if d != target_d:
            continue
        travel = battle.calculate_move_distance(unit, cell)
        if travel <= 0:
            continue
        candidates.append((cell, travel))

    # C++ tie-break: equal travel distance → prefer the cell whose
    # direction from the coverer's current position matches the
    # six-direction priority. For ties we also prefer the cell with the
    # lowest threatening-enemy count.
    if not candidates:
        return {}
    row_parity = archer.pos[1] & 1
    candidates.sort(key=lambda item: (
        item[1],
        _cell_priority_index(item[0], archer.pos, row_parity),
    ))
    # Pick the FIRST (best) candidate only, plus any other cells tied on
    # distance AND direction priority (rare but documented).
    return {cell: travel for cell, travel in candidates}


# ── outcome helper dataclass ───────────────────────────────────────────

@dataclass
class _BestOutcomeResult:
    """Mirrors the tuple returned by C++ ``getMeleeBestOutcome``.

    ``attack_value`` is the BEST outcome's attack value (NOT a composite
    score). ``target`` is the chosen enemy; ``from_index`` is the head
    cell the C++ would attack from.
    """
    attack_value: float
    target: Optional[Unit]
    from_index: Optional[Tuple[int, int]]