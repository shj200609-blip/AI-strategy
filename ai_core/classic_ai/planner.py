"""Stateful orchestration for the classic fheroes2 battle planner."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ai_core.base import AIPlayer
from engine.actions import (
    Action, AttackAction, CastAction, MoveAction, RetreatAction, SkipAction,
)
from engine.battle_state import BattleState
from engine.unit import Unit

from .constants import *
from .forces import (
    _can_retreat_opponent, _can_surrender_opponent, _effective_enemies,
)
from . import analysis, melee, movement, retreat, spells
from . import archer as archer_logic


class ClassicAI(AIPlayer):
    """Greedy heuristic battle AI — direct port of fheroes2's BattlePlanner."""

    def __init__(self, spellbook: Optional[list] = None,
                 difficulty: str = "Normal", randomize: float = 0.0,
                 ranged_range: int = DEFAULT_RANGED_RANGE,
                 turn_limit: int = MAX_TURNS_WITHOUT_DEATHS):
        self.spellbook = spellbook
        self.difficulty = difficulty
        self.randomize = randomize
        self.ranged_range = ranged_range
        self._turn_limit = turn_limit

        # fheroes2 ``BattlePlanner::battleBegins`` initialises these.
        self._current_turn_number = 0
        self._remaining_turns_without_deaths = self._turn_limit
        self._attacker_dead_total = 0
        self._defender_dead_total = 0

        # Per-turn analysis state (mirrors BattlePlanner member vars).
        self._my_color = 0
        self._commander = None
        self._my_army_strength = 0.0
        self._enemy_army_strength = 0.0
        self._my_shooters_strength = 0.0
        self._enemy_shooters_strength = 0.0
        self._my_ranged_units_only = 0.0
        self._enemy_ranged_units_only = 0.0
        self._my_army_average_speed = 0.0
        self._enemy_average_speed = 0.0
        self._attacking_castle = False
        self._defending_castle = False
        self._consider_retreat = False
        self._defensive_tactics = False
        self._cautious_offensive = False
        self._avoid_stacking_units = False
        self._pending_farewell_spell = None

    def battle_begins(self) -> None:
        """Port of ``BattlePlanner::battleBegins``.

        Resets the per-battle turn-limit bookkeeping.
        """
        self._current_turn_number = 0
        self._remaining_turns_without_deaths = self._turn_limit
        self._attacker_dead_total = 0
        self._defender_dead_total = 0

    def decide(self, battle: BattleState, unit: Unit
               ) -> Tuple[Action, str]:
        """Port of ``BattlePlanner::BattleTurn`` / ``planUnitTurn``.

        Order matches the C++ exactly:

          1. ``isLimitOfTurnsExceeded`` guard — early return with RETREAT.
          2. Berserker override.
          3. ``analyzeBattleState`` — fills per-turn member vars.
          4. Outcome lambda — RETREAT_NONE / RETREAT_RETREAT /
             RETREAT_SURRENDER.
          5. Spell heuristics.
          6. Archer / Melee decision tree.
        """
        # Step 0. BattleTurn: turn-limit guard.
        limit = self._turn_limit_exceeded(battle, unit)
        if limit is not None:
            return limit

        # Step 0a. Berserker override.
        # Engine tracks Berserker via ``Unit.is_berserk`` (Effect flag),
        # not a ``modes`` dict — read the property directly.
        if getattr(unit, "is_berserk", False):
            action = self._berserk_turn(battle, unit)
            if action is not None:
                return action

        # Step 1. Analyze current battle state.
        self._analyze_battle_state(battle, unit)

        # Step 2. Retreat/surrender decision.
        retreat = self._retreat_outcome(battle, unit)
        if retreat is not None:
            action, reason = retreat
            # fheroes2 fires the farewell spell *before* the retreat
            # command.  When the spell is available we surface it first
            # so the caller can execute cast → retreat in that order.
            if self._pending_farewell_spell is not None:
                cast_action, cast_reason = self._pending_farewell_spell
                self._pending_farewell_spell = None
                return (cast_action,
                        f"{cast_reason} → {reason}")
            return (action, reason)

        # Step 3. Spell heuristics.
        hero = battle.heroes.get(unit.team)
        if hero is not None and not hero._cast_this_round:
            spell = self.maybe_cast_spell(battle, unit)
            if spell is not None:
                return spell

        # Step 4. Per-unit decision tree.
        if unit.is_archer:
            return self._archer_decision(battle, unit)
        return self._melee_turn(battle, unit)

    def _advance_turn_limit_counter(self, battle: BattleState, unit: Unit
                                     ) -> bool:
        """State half of ``BattlePlanner::isLimitOfTurnsExceeded``.

        Mirrors the C++ logic verbatim:

          * Skip non-attacker turns entirely.
          * On every new turn, observe ``attackerDeadTotal`` /
            ``defenderDeadTotal``.  If either changed since the previous
            turn, reset the counter to ``MAX_TURNS_WITHOUT_DEATHS``;
            otherwise decrement it by one.

        Returns whether the counter is exhausted.

        Like the C++, this is **idempotent within a single turn number**: the
        update block is guarded by ``currentTurnNumber > _currentTurnNumber``
        and stamps ``_currentTurnNumber`` on the way out, so calling it twice
        in one activation (``check_retreat`` then ``decide``) decrements once.

        The engine does not surface turn numbers on every stand-in, so we
        treat ``getattr(battle, "turn_number", 0)`` as the current value with
        backward-compatible default 0; tests that don't increment it
        therefore never trip the guard, matching the C++ behaviour for a
        freshly-started battle.
        """
        if unit.team != getattr(battle, "attacker_team", 0):
            return False

        turn_number = int(getattr(battle, "turn_number", 0))
        atk_dead = int(getattr(battle, "attacker_dead_total",
                               self._attacker_dead_total))
        def_dead = int(getattr(battle, "defender_dead_total",
                               self._defender_dead_total))
        if (turn_number > self._current_turn_number
                and self._remaining_turns_without_deaths > 0):
            prev = (self._attacker_dead_total, self._defender_dead_total)
            curr = (atk_dead, def_dead)
            if (self._current_turn_number == 0
                    or turn_number - self._current_turn_number != 1
                    or prev != curr):
                self._attacker_dead_total = atk_dead
                self._defender_dead_total = def_dead
                self._remaining_turns_without_deaths = self._turn_limit
            else:
                self._remaining_turns_without_deaths -= 1
            self._current_turn_number = turn_number

        return self._remaining_turns_without_deaths <= 0

    def _turn_limit_decision(self, battle: BattleState, unit: Unit) -> int:
        """Verdict half of ``isLimitOfTurnsExceeded``, as a RETREAT_* code.

        The C++ emits ``Command::TOGGLE_AUTO_COMBAT`` when a battle UI is
        present and ``Command::RETREAT`` otherwise; the headless engine has no
        auto-combat UI, so only the retreat branch is reachable and the
        resulting state is identical.

        C++ asserts ``arena.CanRetreatOpponent( currentColor )`` here.  We
        cannot assert in a library, so a pinned commander falls back to
        surrender and then to RETREAT_NONE (the caller makes it a skip).
        """
        if not self._advance_turn_limit_counter(battle, unit):
            return RETREAT_NONE
        self._remaining_turns_without_deaths = 0
        if _can_retreat_opponent(battle, unit.team):
            return RETREAT_RETREAT
        if _can_surrender_opponent(battle, unit.team):
            return RETREAT_SURRENDER
        return RETREAT_NONE

    def _turn_limit_exceeded(self, battle: BattleState, unit: Unit
                              ) -> Optional[Tuple[Action, str]]:
        """``isLimitOfTurnsExceeded`` as an action for ``BattleTurn``."""
        if not self._advance_turn_limit_counter(battle, unit):
            return None
        decision = self._turn_limit_decision(battle, unit)
        hero = battle.heroes.get(unit.team)
        name = getattr(hero, "name", "Hero")
        if decision == RETREAT_RETREAT:
            return (RetreatAction(unit.team),
                    f"{name} retreats (turn limit exceeded)")
        if decision == RETREAT_SURRENDER:
            return (RetreatAction(unit.team),
                    f"{name} surrenders (turn limit exceeded)")
        # Limit exhausted with no exit open. The C++ returns true regardless,
        # so the unit must not fall through to planUnitTurn.
        return (SkipAction(unit),
                f"{name} skips (turn limit, no exit)")

    def check_retreat(self, battle: BattleState, unit: Unit
                      ) -> Tuple[int, Optional[Tuple[Optional[Tuple[CastAction, str]],
                                                     RetreatAction]]]:
        """Three-state retreat wrapper around ``BattlePlanner::BattleTurn``.

        Follows the C++ call order: ``isLimitOfTurnsExceeded`` runs *before*
        ``planUnitTurn``'s Outcome lambda (ai_battle.cpp:620-624), so a battle
        that has burned through ``MAX_TURNS_WITHOUT_DEATHS`` forces the
        attacker out regardless of what the Outcome lambda would have said.
        The turn-limit path emits a bare ``Command::RETREAT`` with no
        ``farewellSpellcast()``, hence the ``None`` farewell slot.
        """
        # Step 0. BattleTurn: turn-limit guard.
        limit = self._turn_limit_decision(battle, unit)
        if limit != RETREAT_NONE:
            return (limit, (None, RetreatAction(unit.team)))

        self._analyze_battle_state(battle, unit)
        outcome = self._retreat_decision(battle, unit)
        if outcome == RETREAT_NONE:
            return (RETREAT_NONE, None)
        hero = battle.heroes.get(unit.team)
        farewell: Optional[Tuple[CastAction, str]] = None
        if hero is not None and not hero._cast_this_round:
            farewell = self._maybe_farewell_spell(
                battle, hero, unit, ignore_threshold=True)
        retreat_action = RetreatAction(unit.team)
        if outcome == RETREAT_SURRENDER:
            return (RETREAT_SURRENDER, (None, retreat_action))
        return (RETREAT_RETREAT, (farewell, retreat_action))

    def _berserk_turn(self, battle: BattleState, unit: Unit
                        ) -> Optional[Tuple[Action, str]]:
        enemies = _effective_enemies(battle, unit)
        if not enemies:
            return (SkipAction(unit), f"{unit.name} rages alone")

        def body_distance(target: Unit) -> int:
            return min(battle.grid.distance(a, b)
                       for a in unit.occupied_cells()
                       for b in target.occupied_cells())

        nearest_units = sorted(enemies, key=body_distance)
        if unit.is_archer and not archer_logic._is_hand_fighting(battle, unit):
            target = nearest_units[0]
            return (AttackAction(unit, target, ranged=True),
                    f"{unit.name} berserks shot at {target.name}")

        target = None
        target_cell = None
        target_position = None
        pathfinder = battle._pathfinder(unit)

        # Prefer the nearest unit that can actually be attacked this turn.
        for nearby in nearest_units:
            info = battle.find_nearest_cell_next_to_unit(unit, nearby)
            if info is None:
                continue
            cell, _, position = info
            if not pathfinder.is_position_reachable(position, True):
                continue
            target = nearby
            target_cell = cell
            target_position = position
            break

        # Otherwise move toward the first nearest unit reachable in principle.
        if target_cell is None:
            for nearby in nearest_units:
                info = battle.find_nearest_cell_next_to_unit(unit, nearby)
                if info is None:
                    continue
                target_cell, _, target_position = info
                break

        if target_cell is None:
            return (SkipAction(unit), f"{unit.name} berserks stuck")

        movement_pos = movement._unit_movement_position(
            self, battle, unit, target_cell)
        if movement_pos is None:
            return (SkipAction(unit), f"{unit.name} berserks stuck")

        if target is not None:
            return (AttackAction(unit, target,
                                 from_pos=movement_pos.head,
                                 from_position=movement_pos,
                                 ranged=False),
                    f"{unit.name} berserks into {target.name}")

        built = battle.build_path(unit, movement_pos.head)
        if built is None:
            return (SkipAction(unit), f"{unit.name} berserks stuck")
        path, final_position = built
        return (MoveAction(unit, path, final_position=final_position),
                f"{unit.name} berserks onward")

    def _analyze_battle_state(self, battle: BattleState, unit: Unit) -> None:
        return analysis._analyze_battle_state(self, battle, unit)

    def _in_defended_area(self, battle: BattleState, unit: Unit, pos: Tuple[int, int]) -> bool:
        return analysis._in_defended_area(self, battle, unit, pos)

    def _retreat_decision(self, battle: BattleState, unit: Unit) -> int:
        return retreat._retreat_decision(self, battle, unit)

    def _retreat_outcome(self, battle: BattleState, unit: Unit) -> Optional[Tuple[Action, str]]:
        return retreat._retreat_outcome(self, battle, unit)

    def _archer_decision(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        return archer_logic._archer_decision(self, battle, unit)

    def _best_melee_target_adjacent(self, battle: BattleState, attacker: Unit, enemies: List[Unit]) -> Optional[Unit]:
        return archer_logic._best_melee_target_adjacent(self, battle, attacker, enemies)

    def _best_ranged_target(self, battle: BattleState, attacker: Unit, enemies: List[Unit]) -> Optional[Unit]:
        return archer_logic._best_ranged_target(self, battle, attacker, enemies)

    def _ranged_target_priority(self, battle: BattleState, attacker: Unit, target: Unit) -> float:
        return archer_logic._ranged_target_priority(self, battle, attacker, target)

    def _archer_retreat_position(self, battle: BattleState, archer: Unit, enemies: List[Unit]) -> Optional[Tuple[int, int]]:
        return archer_logic._archer_retreat_position(self, battle, archer, enemies)

    def _cell_is_threatened(self, battle: BattleState, enemy: Unit, pos: Tuple[int, int]) -> bool:
        return archer_logic._cell_is_threatened(self, battle, enemy, pos)

    def _melee_turn(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        return melee._melee_turn(self, battle, unit)

    def _melee_offense(self, battle: BattleState, unit: Unit, enemies: List[Unit]) -> '_TargetPair':
        return melee._melee_offense(self, battle, unit, enemies)

    def _melee_defense(self, battle: BattleState, unit: Unit, enemies: List[Unit]) -> '_TargetPair':
        return melee._melee_defense(self, battle, unit, enemies)

    def _melee_best_outcome(self, battle: BattleState, unit: Unit, enemies: List[Unit]) -> melee._BestOutcomeResult:
        return melee._get_melee_best_outcome(self, battle, unit, enemies)

    def _best_attack_outcome(self, battle: BattleState, unit: Unit, enemies: List[Unit]) -> '_MeleeAttackOutcome':
        return melee._best_attack_outcome(self, battle, unit, enemies)

    def _optimal_attack_value(self, battle: BattleState, atk: Unit, tgt: Unit, from_cell: Optional[Tuple[int, int]] = None) -> float:
        return melee._optimal_attack_value(self, battle, atk, tgt, from_cell=from_cell)

    def _attack_score(self, battle: BattleState, atk: Unit, tgt: Unit, dmg: int, from_cell: Tuple[int, int]) -> float:
        return melee._evaluate_attack_value(self, battle, atk, tgt, dmg, from_cell)

    def _choose_distant_target(self, battle: BattleState, unit: Unit, enemies: List[Unit], enemy_predicate) -> Optional[Tuple[int, int]]:
        return melee._choose_distant_target(self, battle, unit, enemies, enemy_predicate)

    def _optimal_next_attack_cell(self, battle: BattleState, unit: Unit, path: List[Tuple[int, int]], enemies: List[Unit]) -> Optional[Tuple[int, int]]:
        return melee._optimal_next_attack_cell(self, battle, unit, path, enemies)

    def _nearest_wall_cell(self, battle: BattleState, unit: Unit) -> Optional[Tuple[int, int]]:
        return melee._nearest_wall_cell(self, battle, unit)

    def _archer_cover_cells(self, battle: BattleState, unit: Unit, archer: Unit) -> Dict[Tuple[int, int], int]:
        return melee._archer_cover_cells(self, battle, unit, archer)

    def _reachable_cells(self, battle: BattleState, unit: Unit) -> Dict[Tuple[int, int], int]:
        return movement._reachable_cells(self, battle, unit)

    def _path_to(self, battle: BattleState, unit: Unit, goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        return movement._path_to(self, battle, unit, goal)

    def _unit_movement_target(self, battle: BattleState, unit: Unit, dest: Tuple[int, int]) -> Tuple[int, int]:
        return movement._unit_movement_target(self, battle, unit, dest)

    def maybe_cast_spell(self, battle: BattleState, unit: Unit) -> Optional[Tuple[CastAction, str]]:
        return spells.maybe_cast_spell(self, battle, unit)

    def _score_spell(self, battle: BattleState, hero, spell, unit: Unit) -> Tuple[float, Optional[Unit], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        return spells._score_spell(self, battle, hero, spell, unit)

    @staticmethod
    def _spell_damage_score(dmg: int, spell) -> float:
        return spells._spell_damage_score(dmg, spell)

    def _score_utility_spell(self, battle: BattleState, spell, team: int) -> Tuple[float, Optional[Unit], Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        return spells._score_utility_spell(self, battle, spell, team)

    def _spell_value_threshold(self, battle: BattleState, unit: Unit, hero) -> float:
        return spells._spell_value_threshold(self, battle, unit, hero)

    def _best_damage_target(self, battle: BattleState, spell, team: int) -> Tuple[Optional[Unit], int]:
        return spells._best_damage_target(self, battle, spell, team)

    def _best_aoe_target(self, battle: BattleState, spell, team: int) -> Tuple[Optional[Unit], int]:
        return spells._best_aoe_target(self, battle, spell, team)

    def _strongest_enemy(self, battle: BattleState, team: int) -> Optional[Unit]:
        return spells._strongest_enemy(self, battle, team)

    def _weakest_friend(self, battle: BattleState, team: int) -> Optional[Unit]:
        return spells._weakest_friend(self, battle, team)

    def _maybe_farewell_spell(self, battle: BattleState, hero, unit: Unit, ignore_threshold: bool=False) -> Optional[Tuple[CastAction, str]]:
        return spells._maybe_farewell_spell(self, battle, hero, unit, ignore_threshold)
