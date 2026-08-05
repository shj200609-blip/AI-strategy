"""Constants mirrored from fheroes2 ``ai_battle.cpp`` and ``ai_battle_spell.cpp``.

The spell ratios are intentionally NOT precomputed constants: fheroes2
computes them per-target via ``getSpellSlowRatio`` / ``getSpellHasteRatio``
/ ``getSpellDisruptingRayRatio``. They live in ``spells.py`` now.
"""

MAX_TURNS_WITHOUT_DEATHS = 50
# Engineering sentinel — fheroes2 uses ``isPositionReachable`` to enumerate
# reachable cells (no upper bound). Our reachability search is bounded by
# the caller's max-move budget, so a "huge" value just means "no cap"
# for archer-target filtering. Renamed slightly to make the intent clear.
DEFAULT_RANGED_RANGE = 99  # Python sentinel — "no cap" for archer reach.

# fheroes2 Difficulty::getArmyStrengthRatioForAIRetreat
# (C++ difficulty.cpp:162-176 — "Easy" falls through to the default branch
# and EXPERT shares HARD's threshold.)
RETREAT_STRENGTH_RATIO = {
    "Easy": 100.0 / 6.0,
    "Normal": 100.0 / 7.5,
    "Hard": 100.0 / 8.5,
    "Expert": 100.0 / 8.5,
    "Impossible": 100.0 / 10.0,
}

# fheroes2 analyzeBattleState — defensive / cautious thresholds
DEFENSE_OVERPOWER_RATIO_WALKER = 10.0
DEFENSE_OVERPOWER_RATIO_FLYER = 6.0
MIN_DEFENSIVE_SHOOTER_RATIO = 0.15
MAX_OFFENSIVE_ENEMY_SHOOTER_RATIO = 0.66
CAUTIOUS_OFFENSE_SHOOTER_RATIO = 0.15
AREA_ATTACK_STACKING_RATIO = 0.10

# fheroes2 GameStatic::getCastleWallRangedPenalty — raw 50% penalty
# (game_static.cpp:278). Applied as divisor ``1 + penalty/100`` so the
# effective ranged multiplier is ``1 / (1 + 50/100) = 2/3`` — NOT 0.5.
CASTLE_WALL_RANGED_PENALTY = 50


def castle_wall_ranged_multiplier() -> float:
    """fheroes2: ranged strength /= 1 + (penalty/100) = 1.5, so *2/3."""
    return 1.0 / (1.0 + CASTLE_WALL_RANGED_PENALTY / 100.0)

# fheroes2 ai_battle_spell.cpp constants.
ANTIMAGIC_LOW_LIMIT = 200.0
BLOOD_LUST_RATIO = 0.1

# fheroes2 selectBestSpell spellValueThreshold — see spells.py.
SPELL_VALUE_RATIO = 0.04

# fheroes2 Spell::SLOW/Haste speed delta — read at runtime from the Spell
# dataclass (``SPELLS["Slow"].speed_delta`` / ``SPELLS["Haste"].speed_delta``)
# because C++ scales the delta with the casting hero's spell power.
# These legacy constants are kept only as the data-driven default that the
# Spell table happens to ship with.
SPELL_SLOW_LOST_SPEED = 2
SPELL_HASTE_GAINED_SPEED = 2

# fheroes2 Spell::DISRUPTINGRAY ExtraValue() — read at runtime from
# ``SPELLS["Disrupting Ray"].defense_delta``. Legacy constant below
# matches the default spell-power entry.
DISRUPTING_RAY_DEF_REDUCTION = 3

# Three-state retreat classifier (mirrors the C++ Outcome enum)
RETREAT_NONE = 0
RETREAT_RETREAT = 1
RETREAT_SURRENDER = 2

# Mode flag names — *deprecated*. The engine tracks Hypnotize / Berserker
# via Effect.is_hypnotize / is_berserk flags on Unit.effects (see
# engine.unit.is_hypnotized / is_berserk). ClassicAI now reads those
# properties directly. These string constants are kept only for callers
# that imported them as keys into a ``unit.modes`` dict — they no longer
# match any engine state and are recognised by no AI path.
HYPNOTIZED = "HYPNOTIZED"
BERSERKER = "BERSERKER"

# fheroes2 Difficulty::isBasicAIBattleLogicApplicable gates the "smart"
# area-shot penalty.  C++ difficulty.cpp: returns False on EASY when no
# human controls the side, and False on every difficulty when the side
# IS human-controlled. The planner exposes ``is_basic_ai_logic()``
# which consults ``difficulty`` and the hero's control flag.
USE_BASIC_AI_BATTLE_LOGIC = True


def is_basic_ai_logic(difficulty: str, hero=None) -> bool:
    """fheroes2 ``Difficulty::isBasicAIBattleLogicApplicable``.

    True except for: EASY without human control, or *any* difficulty with
    a human-controlled hero (auto-combat AI is "basic" only when the AI is
    actually running).
    """
    if hero is not None and getattr(hero, "is_control_human", False):
        return False
    if difficulty == "Easy":
        return False
    return True
