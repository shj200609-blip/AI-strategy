"""ClassicAI archer decision helpers.

Direct port of fheroes2 ``AI::BattlePlanner::archerDecision`` and the helpers
it relies on (``isHandFighting``, ``isUnitAbleToApproachPosition``,
``evaluateThreatForUnit``, ``isPositionUnderThreat``,
``calculateAreaShotAttackPriority``). The branch order, dedup keys,
damage accounting, and the ``isDangerousMove`` threshold all mirror the C++
verbatim — see the per-function docstrings for the corresponding source
locations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from engine.actions import Action, AttackAction, MoveAction, SkipAction
from engine.battle_state import BattleState
from engine.unit import Unit

from .forces import _effective_enemies, _enemy_for
from .models import _PositionCharacteristics
from . import movement


# ── fheroes2 Battle::Unit::isHandFighting (battle_troop.cpp:341) ──────
def _is_hand_fighting(battle: BattleState, unit: Unit) -> bool:
    """Return True iff *unit* is adjacent to an enemy.

    Mirrors ``Battle::Unit::isHandFighting``: for wide units the neighbours
    of BOTH head and tail are checked, so an archer whose tail is in melee
    (but whose head is not) is correctly considered hand-fighting.
    """
    if not unit.is_alive:
        return False
    for pos in _unit_occupied_positions(battle, unit):
        for nb in battle.grid.neighbors(*pos):
            other = battle.unit_at(nb)
            if other is None or other is unit:
                continue
            if _enemy_for(unit, other):
                return True
    return False


# ── fheroes2 isUnitAbleToApproachPosition (ai_battle.cpp:271) ────────
def _is_unit_able_to_approach_pos(battle: BattleState, unit: Unit,
                                  pos: Tuple[int, int]) -> bool:
    """True if *unit* can reach any neighbour of *pos* within its speed.

    C++ enumerates ``Board::GetAroundIndexes(pos)`` and asks
    ``Position::GetReachable(unit, nearbyIdx, speed)`` for each. The Python
    equivalent is ``BattleState.is_position_reachable`` (which already runs
    the BFS over head+tail with the unit's speed budget).
    """
    if unit.speed <= 0:
        return False
    for nb in battle.grid.neighbors(*pos):
        if not battle.grid.is_valid(*nb):
            continue
        if battle.is_position_reachable(unit, nb, is_on_current_turn=True):
            return True
    return False


# ── fheroes2 Battle::Unit::evaluateThreatForUnit (battle_troop.cpp:1007)
def _evaluate_threat_for_unit(battle: BattleState, attacker: Unit,
                              defender: Unit) -> float:
    """fheroes2 ``evaluateThreatForUnit`` — attacker is scored as a threat
    to *defender*.

    Mirrors the C++ priority: damage / distanceModifier, with the same
    double-attack adjustment, ability multipliers (enemy_halving,
    soul_eater, hp_drain, mirror image), and the same-color -2x penalty
    that makes friendly fire self-penalising.
    """
    if not attacker.is_alive:
        return 0.0

    # fheroes2: ranged attackers hitting from outside melee use the
    # ranged damage formula (no archer-melee penalty).
    ranged = attacker.is_archer and not _is_hand_fighting(battle, attacker)
    damage = battle.expected_damage(attacker, defender, ranged=ranged)
    threat = float(damage)

    # Distance modifier (battle_troop.cpp:1015-1037). Only melee walkers
    # suffer it: ranged and flying attackers always get 1.0.
    if not attacker.is_flying and not attacker.is_archer:
        attacker_speed = attacker.speed
        if attacker_speed > 0:
            attack_range = attacker_speed + 1
            dist = battle.grid.distance(attacker.pos, defender.pos)
            if dist > attack_range:
                threat /= 1.5 * dist / attacker_speed

    # Double attack (mutually exclusive with area_shot per the C++ assert).
    is_double = (attacker.has_ability("double_shooting")
                 or attacker.has_ability("double_melee"))
    if is_double and not attacker.has_ability("area_shot"):
        if attacker.is_archer and not _is_hand_fighting(battle, attacker):
            # Ranged double-shot: no retaliation, second shot full damage.
            threat *= 2
        else:
            ret = battle.expected_damage(defender, attacker, ranged=False)
            if ret > 0:
                ratio = min(ret / max(1, attacker._total_hp), 1.0)
                threat += threat * (1.0 - ratio)
            else:
                threat *= 2

    # Ability multipliers.
    if attacker.has_ability("enemy_halving"):
        threat *= 2
    if attacker.has_ability("soul_eater"):
        threat *= 3
    if attacker.has_ability("hp_drain"):
        threat *= 1.3

    # Mirror Image: any hit kills, so the AI prefers targeting them.
    if getattr(attacker, "is_mirror", False):
        threat *= 10

    # Same-color / hypnotize / immovable / already-moved adjustments
    # (battle_troop.cpp:1141-1158). Order matters: same-color outranks the
    # others (a hypnotized attacker on our own side still gets the -2x).
    if not _enemy_for(attacker, defender):
        threat *= -2
    elif getattr(attacker, "is_hypnotized", False):
        threat *= -1
    elif attacker.speed <= 0:
        threat = 0
    elif getattr(attacker, "_acted", False):
        threat /= 1.25

    return threat


# ── Board center helper (fheroes2 Board::sizeInCells / 2) ─────────────
def _board_center_cell(battle: BattleState) -> Tuple[int, int]:
    """Center cell as a (col, row) tuple.

    fheroes2 uses the linear index ``Board::sizeInCells / 2`` (== 49 for
    the 11×9 board). Our previous code used ``(cols // 2, rows // 2)``,
    which agrees for the default grid but drifts on non-standard sizes.
    """
    idx = (battle.grid.cols * battle.grid.rows) // 2
    return (idx % battle.grid.cols, idx // battle.grid.cols)


# ── Wide-unit helpers ─────────────────────────────────────────────────
def _unit_occupied_positions(battle: BattleState, unit: Unit
                              ) -> List[Tuple[int, int]]:
    """All cells *unit*'s body covers.

    fheroes2 always considers a unit's *head* and (for wide units) *tail*
    cells as a single logical position. Engine ``Unit.occupied_cells``
    already returns the set, but we expose an ordered list for iteration.
    """
    pos = getattr(unit, "pos", None)
    if pos is None:
        return []
    positions = [pos]
    tail = getattr(unit, "tail_cell", None)
    if tail and tail != pos:
        positions.append(tail)
    return positions


def _wide_around_cells(battle: BattleState, target: Unit,
                       shot_cell: Tuple[int, int]) -> Set[Tuple[int, int]]:
    """Cells hit by an area-shot aimed at *shot_cell* on *target*.

    C++ ``Board::GetAroundIndexes(Position)`` returns the union of
    head's and tail's neighbours. We replicate that union here so the
    Python dedup key (head position) can match each wide neighbour unit
    exactly once across the combined ring.
    """
    cells: Set[Tuple[int, int]] = set(battle.grid.neighbors(*shot_cell))
    tail = getattr(target, "tail_cell", None)
    if tail and tail != shot_cell:
        cells.update(battle.grid.neighbors(*tail))
    return cells


# ── archer decision (fheroes2 AI::BattlePlanner::archerDecision) ─────
def _archer_decision(self, battle: BattleState, unit: Unit
                      ) -> Tuple[Action, str]:
    enemies = _effective_enemies(battle, unit)

    # 1. Try to retreat to a safer cell.
    retreat_pos = self._archer_retreat_position(battle, unit, enemies)
    if retreat_pos is not None:
        movement_pos = movement._unit_movement_position(
            self, battle, unit, retreat_pos)
        if movement_pos is not None and movement_pos.head != unit.pos:
            built = battle.build_path(unit, movement_pos.head)
            if built:
                path, final_position = built
                return (MoveAction(unit, path, final_position=final_position),
                        f"{unit.name} retreats from melee")

    # 2. Hand-fighting? C++ uses ``currentUnit.isHandFighting()`` — a
    # single check that returns true for ANY adjacent enemy (including
    # archer-vs-archer adjacency). The C++ does NOT special-case ranged
    # vs melee here; the check is purely geometric.
    if _is_hand_fighting(battle, unit):
        target = self._best_melee_target_adjacent(battle, unit, enemies)
        if target is not None:
            return (AttackAction(unit, target, ranged=False),
                    f"{unit.name} melee strikes {target.name}")
        return (SkipAction(unit), f"{unit.name} skips")

    # 3. Ranged attack. C++ emits ``Battle::Command::ATTACK`` with
    # ``target.cell`` (the head or tail index for area-shot, -1 for
    # normal shot). Mirror that in the Python ``AttackAction.cell``.
    shot = self._best_ranged_target(battle, unit, enemies)
    if shot is None:
        return (SkipAction(unit), f"{unit.name} skips")
    target, cell = shot
    return (AttackAction(unit, target, ranged=True, cell=cell),
            f"{unit.name} shoots {target.name}")


def _best_melee_target_adjacent(self, battle: BattleState,
                                 attacker: Unit, enemies: List[Unit]
                                 ) -> Optional[Unit]:
    """Pick the adjacent enemy maximising ``damage - retaliation``.

    Mirrors fheroes2 ``damageDiff = archerMeleeDmg - retaliatoryDmg``.
    C++ iterates with ``dist != 1`` filter; the Python equivalent uses
    ``grid.distance``.
    """
    best: Optional[Unit] = None
    best_score = float("-inf")
    for e in enemies:
        if battle.grid.distance(attacker.pos, e.pos) != 1:
            continue
        dmg = battle.expected_damage(attacker, e, ranged=False)
        ret = battle.expected_damage(e, attacker, ranged=False)
        score = dmg - ret
        if score > best_score:
            best_score = score
            best = e
    return best


def _best_ranged_target(self, battle: BattleState, attacker: Unit,
                         enemies: List[Unit]
                         ) -> Optional[Tuple[Unit, Optional[Tuple[int, int]]]]:
    """Pick the best (target, cell) for the archer's ranged attack.

    For non-area-shot archers this is the single highest-priority enemy
    in range, with ``cell=None`` (engine hits the head). For area-shot
    archers each enemy contributes one or two shot candidates (head,
    and tail if wide); we keep the highest-priority non-dangerous shot.
    The ``cell`` is the cell the shot *landed on* — used by the engine
    to disambiguate which end of a wide stack was hit.
    """
    if not attacker.has_ability("area_shot"):
        best: Optional[Unit] = None
        best_score = float("-inf")
        for e in enemies:
            if not e.is_alive:
                continue
            if battle.grid.distance(attacker.pos, e.pos) > self.ranged_range:
                continue
            score = _evaluate_threat_for_unit(battle, e, attacker)
            if score > best_score:
                best_score = score
                best = e
        if best is None:
            return None
        return (best, None)

    # Area-shot: try every (enemy, head|tail) shot. The C++ iterates
    # wide targets twice (once per body cell); we mirror that.
    best_target: Optional[Unit] = None
    best_cell: Optional[Tuple[int, int]] = None
    best_score = float("-inf")
    for e in enemies:
        if not e.is_alive:
            continue
        for shot_cell in _unit_occupied_positions(battle, e):
            if not battle.grid.is_valid(*shot_cell):
                continue
            if battle.grid.distance(attacker.pos, shot_cell) > self.ranged_range:
                continue
            score, dangerous = _area_shot_priority(
                battle, attacker, e, shot_cell)
            if dangerous:
                continue
            if score > best_score:
                best_score = score
                best_target = e
                best_cell = shot_cell
    if best_target is None:
        return None
    return (best_target, best_cell)


def _area_shot_priority(battle: BattleState, attacker: Unit,
                        target: Unit, shot_cell: Tuple[int, int]
                        ) -> Tuple[float, bool]:
    """Score a single area-shot at *shot_cell* on *target*.

    fheroes2 ``calculateAreaShotAttackPriority`` (ai_battle.cpp:1453):
      * ``affectedUnitsIndexes = {target.head} ∪ {head of every unit in
        around(shot_cell) ∪ around(tail)}``
      * For each unit: ``friendDamageHitPoints`` /
        ``enemyDamageHitPoints`` accumulate HP that would be lost, and
        ``result += unit.evaluateThreatForUnit(currentUnit)``.
      * ``isDangerousMove = (friendDamageHitPoints >= 3 * enemyDamageHitPoints)``
        — when enemy damage is 0, the comparison evaluates as `>= 0` so
        any friendly HP loss makes the move dangerous. The previous
        Python used ``max(enemy, 1)`` which artificially widened the
        safe threshold.
    """
    # Dedup is by HEAD cell — fheroes2 keys the set by ``GetHeadIndex()``.
    # For non-wide units the head is the only body cell; for wide units
    # ``unit_at`` may return the same unit via either head or tail.
    seen: Set[Tuple[int, int]] = {target.pos}
    around = _wide_around_cells(battle, target, shot_cell)
    friend_dmg = 0.0
    enemy_dmg = 0.0
    score = _evaluate_threat_for_unit(battle, target, attacker)
    for cell in around:
        other = battle.unit_at(cell)
        if other is None or not other.is_alive:
            continue
        if other.pos in seen:
            continue
        seen.add(other.pos)
        dmg = min(other._total_hp,
                  battle.expected_damage(attacker, other, ranged=True))
        if _enemy_for(attacker, other):
            enemy_dmg += dmg
        else:
            friend_dmg += dmg
        # Use the full ``evaluateThreatForUnit`` — it already applies the
        # -2x same-color penalty for friendlies, so the C++ priority
        # naturally penalises friendly fire rather than subtracting
        # ``other.strength`` outright.
        score += _evaluate_threat_for_unit(battle, other, attacker)

    # C++ ``isDangerousMove`` — 0 enemy damage is treated as 0, NOT as 1.
    dangerous = friend_dmg >= 3.0 * enemy_dmg
    return score, dangerous


def _archer_retreat_position(self, battle: BattleState, archer: Unit,
                              enemies: List[Unit]
                              ) -> Optional[Tuple[int, int]]:
    """fheroes2 archer retreat (ai_battle.cpp:1180).

    Steps mirror the C++ exactly:

      1. Skip if any enemy is flying (no point retreating from a flier).
      2. Skip if the archer has already acted this turn — C++ uses
         ``currentUnit.GetSpeed()`` which returns STANDING for a unit
         that has already moved, making retreat impossible. Our engine
         stores the post-Haste/Slow speed on the unit unconditionally,
         so we add an explicit ``_acted`` short-circuit.
      3. Skip if no enemy threatens the archer's current cell.
      4. Skip if any threatening enemy is fast enough to catch us
         (``enemySpeed + 2 >= currentUnitSpeed``).
      5. Score every reachable cell (and the archer's own cell) by the
         set of enemies that can threaten it; pick the safest cell that
         maximises ``(distance to nearest enemy, inverse distance to
         board centre)``.
    """
    if any(e.is_flying for e in enemies):
        return None
    if getattr(archer, "_acted", False):
        # fheroes2 GetSpeed() with ``skipMovedCheck=false`` returns 0
        # for a unit that has already moved this turn, blocking retreat.
        return None

    threatened_now = [e for e in enemies
                      if self._cell_is_threatened(battle, e, archer.pos)]
    if not threatened_now:
        return None

    archer_speed = archer.speed
    if archer_speed <= 0:
        return None
    # C++: retreat is worth trying iff for every threatening enemy,
    # ``enemy.GetSpeed(false, true) + 2 < currentUnit.GetSpeed()``. In
    # our engine ``e.speed`` is the post-Haste/Slow speed (matches
    # ``GetSpeed(false, true)`` which skips the moved-check).
    if any(e.speed + 2 >= archer_speed for e in threatened_now):
        return None

    candidates: Dict[Tuple[int, int], _PositionCharacteristics] = {
        archer.pos: _PositionCharacteristics()
    }
    for cell in self._reachable_cells(battle, archer):
        if cell != archer.pos:
            candidates[cell] = _PositionCharacteristics()

    # UnitRemover: hide the archer from threat checks (C++ toggles the
    # board occupancy directly; the engine has no such side-channel so
    # we flip ``is_alive`` and restore it).
    was_alive = archer.is_alive
    archer.is_alive = False
    try:
        for e in enemies:
            for pos in candidates:
                if self._cell_is_threatened(battle, e, pos):
                    candidates[pos].threatening.add(id(e))
                d = battle.grid.distance(pos, e.pos)
                if d < candidates[pos].distance:
                    candidates[pos].distance = d
    finally:
        archer.is_alive = was_alive

    safe = {pos: c for pos, c in candidates.items()
            if not c.threatening and pos != archer.pos}
    if not safe:
        return None
    centre = _board_center_cell(battle)
    return max(safe, key=lambda pos: (
        safe[pos].distance,
        1.0 / max(1, battle.grid.distance(pos, centre))))


def _cell_is_threatened(self, battle: BattleState, enemy: Unit,
                        pos: Tuple[int, int]) -> bool:
    """fheroes2 ``isPositionUnderThreat`` (ai_battle.cpp:1272).

    Adjacent cells (distance=1) are always threatened — an enemy
    standing next to *pos* denies the archer a clean shot even if it
    is otherwise immovable.

    Ranged enemies that are NOT in hand-fighting are skipped — they
    don't threaten positions directly (only in melee). Hand-fighting
    archers (those with adjacent enemies of their own) DO threaten,
    because they'll likely move into melee on the next turn.

    All other enemies threaten iff ``isUnitAbleToApproachPosition``
    (C++ ai_battle.cpp:271) returns true — i.e. they have a free
    neighbour of *pos* they can reach within their speed budget. The
    previous Python used the rough heuristic ``d <= enemy.speed``,
    which ignored obstacles and path-finding constraints.
    """
    head_positions = _unit_occupied_positions(battle, enemy)
    for hpos in head_positions:
        d = battle.grid.distance(hpos, pos)
        if d == 1:
            return True
        # fheroes2: only skip archer-threat when the archer is NOT in
        # hand-fighting. A hand-fighting archer is still a melee threat.
        if enemy.is_archer and not _is_hand_fighting(battle, enemy):
            continue
        if _is_unit_able_to_approach_pos(battle, enemy, pos):
            return True
    return False
