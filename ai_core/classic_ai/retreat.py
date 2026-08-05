"""ClassicAI retreat and surrender decision helpers."""

from __future__ import annotations

from typing import Optional, Tuple

from engine.actions import Action, RetreatAction
from engine.battle_state import BattleState
from engine.unit import Unit

from .constants import (
    RETREAT_NONE,
    RETREAT_RETREAT,
    RETREAT_STRENGTH_RATIO,
    RETREAT_SURRENDER,
)
from .forces import (
    _can_retreat_opponent,
    _can_surrender,
    _can_surrender_opponent,
    _is_possible_to_rehire,
    _total_primary_skill_level,
)


def _retreat_decision(self, battle: BattleState, unit: Unit) -> int:
    """Port of the ``Outcome`` lambda inside ``BattlePlanner::planUnitTurn``.

    Mirrors the C++ branches in this exact order:

      1. ``!_considerRetreat`` → ContinueBattle
      2. ``actualHero->isControlHuman()`` → ContinueBattle
      3. strength-ratio guard by difficulty
      4. ``!CanRetreatOpponent`` branch (surrender-only)
      5. ``!CanSurrenderOpponent`` fallback
      6. ``hasValuableArtifacts`` → Retreat/Surrender
      7. ``!isPossibleToReHire`` → ContinueBattle
      8. ``getTotalPrimarySkillLevel() >= 10`` → Retreat/Surrender

    The lambda is keyed off ``_myColor`` — the colour of whichever army is
    currently acting — so it applies to the defender too.  There is no
    attacker-only guard in the original: ``CanRetreatOpponent`` is what keeps
    a castle-bound defender in place, and the AI legitimately surrenders when
    it commands the defending side.
    """
    hero = battle.heroes.get(unit.team)
    if hero is None:
        return RETREAT_NONE

    # 1. do not even consider retreat if not required
    if not self._consider_retreat:
        return RETREAT_NONE

    # 2. Human-controlled heroes do not retreat during auto-combat
    if getattr(hero, "is_control_human", False):
        return RETREAT_NONE

    # 3. strength ratio guard
    ratio = RETREAT_STRENGTH_RATIO.get(
        self.difficulty, RETREAT_STRENGTH_RATIO["Normal"])
    if self._my_army_strength * ratio >= self._enemy_army_strength:
        return RETREAT_NONE

    # ``arena.CanRetreatOpponent( _myColor )`` / ``CanSurrenderOpponent`` —
    # evaluated against the arena, never assumed.
    can_retreat = _can_retreat_opponent(battle, unit.team)
    # C++ ``isAbleToSurrender`` = arena gate AND ``kingdom.AllowPayment``.
    can_surrender = (
        _can_surrender_opponent(battle, unit.team)
        and _can_surrender(battle, unit.team)
    )
    has_valuable_artifacts = bool(
        getattr(hero, "has_valuable_artifacts", False))
    rehire_possible = _is_possible_to_rehire(battle, hero)
    primary = _total_primary_skill_level(hero)
    min_skill_for_retreat = 10

    # 4-5. Cannot retreat — only consider surrender
    if not can_retreat:
        if not can_surrender:
            return RETREAT_NONE
        if has_valuable_artifacts:
            return RETREAT_SURRENDER
        if not rehire_possible:
            return RETREAT_NONE
        if primary >= min_skill_for_retreat:
            return RETREAT_SURRENDER
        return RETREAT_NONE

    # Can retreat.
    # 6. keep artifacts off the enemy
    if has_valuable_artifacts:
        return RETREAT_RETREAT
    # 7. worth keeping the hero around to re-hire
    if not rehire_possible:
        return RETREAT_NONE
    # 8. experienced hero — retreat so we can re-hire later
    if primary >= min_skill_for_retreat:
        return RETREAT_RETREAT
    return RETREAT_NONE


def _retreat_outcome(self, battle: BattleState, unit: Unit
                      ) -> Optional[Tuple[Action, str]]:
    """Port of the ``Outcome`` switch in ``planUnitTurn``.

    Mirrors the C++ flow:

      * ``farewellSpellcast()`` may push a damage spell first
        (handled here as a side-effect / cached for the caller).
      * Then the hero RETREATs or SURRENDERs.

    Returns ``(action, reason)``.  When a farewell spell is
    available it is exposed via ``self._pending_farewell_spell`` so
    the orchestrator can dispatch the cast before the retreat.
    """
    outcome = self._retreat_decision(battle, unit)
    if outcome == RETREAT_NONE:
        self._pending_farewell_spell = None
        return None
    hero = battle.heroes.get(unit.team)
    self._pending_farewell_spell = None
    if hero is not None and not hero._cast_this_round:
        farewell = self._maybe_farewell_spell(
            battle, hero, unit, ignore_threshold=True)
        if farewell is not None:
            self._pending_farewell_spell = farewell
    if outcome == RETREAT_SURRENDER:
        return (RetreatAction(unit.team),
                f"{hero.name if hero else 'Hero'} surrenders")
    return (RetreatAction(unit.team),
            f"{hero.name if hero else 'Hero'} retreats")
