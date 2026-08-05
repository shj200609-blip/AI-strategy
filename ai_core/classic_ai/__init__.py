"""ClassicAI — Python port of fheroes2 ``AI::BattlePlanner``.

The package is split by battle-planning responsibility. Existing imports from
``ai_core.classic_ai`` remain stable through the exports below.
"""

from .constants import (
    ANTIMAGIC_LOW_LIMIT,
    AREA_ATTACK_STACKING_RATIO,
    BERSERKER,
    BLOOD_LUST_RATIO,
    CAUTIOUS_OFFENSE_SHOOTER_RATIO,
    CASTLE_WALL_RANGED_PENALTY,
    DEFAULT_RANGED_RANGE,
    DEFENSE_OVERPOWER_RATIO_FLYER,
    DEFENSE_OVERPOWER_RATIO_WALKER,
    DISRUPTING_RAY_DEF_REDUCTION,
    HYPNOTIZED,
    MAX_OFFENSIVE_ENEMY_SHOOTER_RATIO,
    MAX_TURNS_WITHOUT_DEATHS,
    MIN_DEFENSIVE_SHOOTER_RATIO,
    RETREAT_NONE,
    RETREAT_RETREAT,
    RETREAT_STRENGTH_RATIO,
    RETREAT_SURRENDER,
    SPELL_HASTE_GAINED_SPEED,
    SPELL_SLOW_LOST_SPEED,
    SPELL_VALUE_RATIO,
)
from .forces import (
    _can_surrender,
    _commander_max_spell_damage_value,
    _effective_enemies,
    _effective_friends,
    _effective_team,
    _enemy_for,
    _is_hypnotized,
    _is_possible_to_rehire,
    _stalemate_reached,
    _survivor_strength,
    _total_primary_skill_level,
)
from .models import (
    _MeleeAttackOutcome,
    _PositionCharacteristics,
    _SpellcastOutcome,
    _SpellSelection,
    _TargetPair,
)
from .planner import ClassicAI

__all__ = [
    "ClassicAI",
    "RETREAT_NONE",
    "RETREAT_RETREAT",
    "RETREAT_SURRENDER",
]
