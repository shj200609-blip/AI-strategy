"""ClassicAI battle-state analysis helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

from engine.battle_state import BattleState
from engine.unit import Unit

from .constants import (
    CAUTIOUS_OFFENSE_SHOOTER_RATIO,
    CASTLE_WALL_RANGED_PENALTY,
    DEFENSE_OVERPOWER_RATIO_FLYER,
    DEFENSE_OVERPOWER_RATIO_WALKER,
    MAX_OFFENSIVE_ENEMY_SHOOTER_RATIO,
    MIN_DEFENSIVE_SHOOTER_RATIO,
)
from .forces import (
    _commander_max_spell_damage_value,
    _effective_enemies,
    _effective_friends,
)
from .spells import _attacker_ignores_cover


def _occupied_positions(unit: Unit) -> List[Tuple[int, int]]:
    """Head and (if wide) tail cells for *unit*.

    Mirrors fheroes2 ``Board::GetAroundIndexes(Position)`` semantics:
    positions are always considered by *both* body cells of a wide unit.
    """
    pos = getattr(unit, "pos", None)
    if pos is None:
        return []
    out: List[Tuple[int, int]] = [pos]
    tail = getattr(unit, "tail_cell", None)
    if tail and tail != pos:
        out.append(tail)
    return out


def _analyze_battle_state(self, battle: BattleState, unit: Unit) -> None:
    """Mirror ``BattlePlanner::analyzeBattleState``.

    Direct port of ``AI::BattlePlanner::analyzeBattleState`` in
    ``src/fheroes2/ai/ai_battle.cpp``.
    """
    self._my_color = unit.team
    self._commander = battle.heroes.get(self._my_color)

    enemies = _effective_enemies(battle, unit)
    friends = _effective_friends(battle, unit)

    # Reset strength accumulators (C++ resets every field listed below).
    self._my_army_strength = 0.0
    self._enemy_army_strength = 0.0
    self._my_shooters_strength = 0.0
    self._enemy_shooters_strength = 0.0
    self._my_ranged_units_only = 0.0
    self._enemy_ranged_units_only = 0.0
    self._my_army_average_speed = 0.0
    self._enemy_average_speed = 0.0
    self._enemy_spell_strength = 0.0
    self._attacking_castle = False
    self._defending_castle = False
    self._consider_retreat = False
    self._defensive_tactics = False
    self._cautious_offensive = False
    self._avoid_stacking_units = False

    # Enemy force analysis: skip invalid (dead) units.
    if not enemies:
        return

    sum_enemy_str = 0.0
    area_attack_threat = 0.0

    for e in enemies:
        if not e.is_alive:
            continue
        s = e.strength
        self._enemy_army_strength += s
        # fheroes2 ai_battle.cpp:992 — ``isArchers() && !isImmovable()``.
        # In C++, towers are Units with ``isImmovable()==true``; in the
        # Python engine towers live on ``castle.towers`` and never appear
        # here, so the immovable filter is a no-op for normal Units but is
        # kept as an explicit guard for symmetry.
        if e.is_archer and not getattr(e, "is_immovable", False):
            self._enemy_ranged_units_only += s
            if e.has_ability("area_shot"):
                area_attack_threat += s
        self._enemy_average_speed += e.speed * s
        sum_enemy_str += s

    self._enemy_shooters_strength = self._enemy_ranged_units_only
    if sum_enemy_str > 0.0:
        self._enemy_average_speed /= sum_enemy_str

    # C++ uses the literal 0.1 for area-attack stacking detection.
    self._avoid_stacking_units = (
        sum_enemy_str > 0.0
        and area_attack_threat / sum_enemy_str > 0.1)

    # Friendly force: iterate the *raw* friendly force so we can count
    # both alive and dead troops (``count > 0 or dead > 0``), exactly
    # mirroring the C++ ``if ( count > 0 || dead > 0 ) initialUnitCount++``
    # branch.
    raw_friends = battle.alive(self._my_color)
    initial_unit_count = 0
    sum_my_str = 0.0

    for f in raw_friends:
        count = f.count
        dead = getattr(f, "dead", max(0, f.original_count - f.count))
        if count > 0 or dead > 0:
            initial_unit_count += 1

        self._my_army_average_speed += f.speed * f.strength
        sum_my_str += f.strength

        # Dead unit (count==0, dead>0): trigger retreat, skip strength.
        if count == 0 and dead > 0:
            self._consider_retreat = True
            continue

        self._my_army_strength += f.strength
        # fheroes2 ai_battle.cpp:1046 — ``isArchers() && !isImmovable()``.
        # See enemy branch above for why ``is_immovable`` is duck-typed.
        if f.is_archer and not getattr(f, "is_immovable", False):
            self._my_ranged_units_only += f.strength

    self._my_shooters_strength = self._my_ranged_units_only
    if sum_my_str > 0.0:
        self._my_army_average_speed /= sum_my_str

    self._consider_retreat = self._consider_retreat or initial_unit_count < 4

    # Castle modifiers (C++ Battle::Arena::GetCastle() + isAnyTowerPresent()).
    castle = battle.castle
    if castle is not None and castle.towers_active():
        # fheroes2 ai_battle.cpp:1064 — NO_SHOOTING_PENALTY artifact or any
        # Archery skill level makes the *attacker* ignore castle wall cover.
        # C++ uses ``arena.getAttackingForce().GetCommander()`` regardless of
        # whether ``_myColor`` is the attacker — when we are defending, this
        # still reads the *other* team's commander.
        attacker_commander = battle.heroes.get(battle.attacker_team)
        attacker_ignores_cover = _attacker_ignores_cover(attacker_commander)

        tower_str = (
            castle.tower_strength("CENTER")
            + castle.tower_strength("LEFT")
            + castle.tower_strength("RIGHT"))
        assert tower_str >= 0

        if self._my_color == castle.color:
            self._defending_castle = True
            self._my_shooters_strength += tower_str

            if not attacker_ignores_cover:
                # fheroes2 ai_battle.cpp:1095 — enemy ranged strength
                # is divided by 1 + (wallPenalty/100).
                self._enemy_shooters_strength /= (
                    1.0 + CASTLE_WALL_RANGED_PENALTY / 100.0)
        else:
            self._attacking_castle = True
            self._enemy_shooters_strength += tower_str

            if not attacker_ignores_cover:
                # fheroes2 ai_battle.cpp:1103 — our ranged strength
                # is divided by 1 + (wallPenalty/100).
                self._my_shooters_strength /= (
                    1.0 + CASTLE_WALL_RANGED_PENALTY / 100.0)

    # Spell-damage correction for shooters (commander spell power).
    if self._commander is not None and self._my_shooters_strength > 1.0:
        self._my_shooters_strength += _commander_max_spell_damage_value(
            self._commander)

    enemy_commander = battle.heroes.get(1 - self._my_color)
    if enemy_commander is not None:
        self._enemy_spell_strength = getattr(
            enemy_commander, "get_magic_strategic_value",
            lambda _x: 0.0)(self._my_army_strength)
        self._enemy_shooters_strength += _commander_max_spell_damage_value(
            enemy_commander)

    assert self._my_army_strength > 0.0 and self._enemy_army_strength > 0.0

    my_archer_ratio = (
        self._my_shooters_strength / self._my_army_strength)
    enemy_archer_ratio = (
        self._enemy_shooters_strength / self._enemy_army_strength)

    # Defensive-tactics decision tree (verbatim C++ lambda).
    defensive_tactics = True
    if not self._in_defended_area(battle, unit, unit.pos):
        defensive_tactics = False
    else:
        over_power = (DEFENSE_OVERPOWER_RATIO_FLYER
                      if unit.is_flying else DEFENSE_OVERPOWER_RATIO_WALKER)
        if self._my_army_strength > self._enemy_army_strength * over_power:
            defensive_tactics = False
        elif self._my_shooters_strength < self._enemy_shooters_strength:
            defensive_tactics = False
        elif self._defending_castle:
            defensive_tactics = True
        elif my_archer_ratio < MIN_DEFENSIVE_SHOOTER_RATIO:
            defensive_tactics = False
        elif enemy_archer_ratio > MAX_OFFENSIVE_ENEMY_SHOOTER_RATIO:
            defensive_tactics = False
        else:
            defensive_tactics = True
    self._defensive_tactics = defensive_tactics

    self._cautious_offensive = enemy_archer_ratio < CAUTIOUS_OFFENSE_SHOOTER_RATIO


def _in_defended_area(self, battle: BattleState, unit: Unit,
                      pos: Tuple[int, int]) -> bool:
    """fheroes2 ``isPositionLocatedInDefendedArea``.

    Hypnotize-aware: a unit whose effective team differs from ``_myColor``
    (i.e. it has been mind-controlled) cannot shelter our archers any
    longer — C++ guards the body with
    ``if ( unit.GetArmyColor() != _myColor ) return false;``.

    Wide-unit aware: the C++ checks every cell the unit's body covers,
    so a defender whose tail would be exposed is not "in" the defended
    area.
    """
    castle = battle.castle
    # C++ checks ownership before the castle-walls branch.
    if getattr(unit, "team", self._my_color) != self._my_color:
        return False
    if castle is not None and self._defending_castle:
        # For a wide unit, every body cell must be inside the walls.
        cells = _occupied_positions(unit)
        if not cells:
            return not castle.is_outside_walls(*pos)
        return all(not castle.is_outside_walls(*c) for c in cells)
    cells = _occupied_positions(unit) or [pos]
    if unit.team == 0:
        return all(c[0] <= battle.grid.cols // 2 for c in cells)
    return all(c[0] >= battle.grid.cols // 2 for c in cells)
