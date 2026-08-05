"""Combat spell definitions and timed status effects.

Reimplemented from fheroes2's spell system (spell.cpp). Defines 48
hero-learnable combat spells plus the built-in-only Petrification placeholder.
Damage spells deal ``base_damage * power``; buffs/debuffs attach a timed Effect
that modifies unit stats and expires after ``power`` rounds.
"""

from dataclasses import dataclass
from typing import Optional

# ── Spell kinds ──────────────────────────────────────────────────
# Determine targeting, AI scoring, and _cast dispatch.

DAMAGE = "damage"       # single-target damage
AOE = "aoe"             # area / army-wide damage
BUFF = "buff"           # buff friendly (or all friends if is_mass)
DEBUFF = "debuff"       # debuff enemy (or all enemies if is_mass)
CONTROL = "control"     # Blind / Paralyze — skip_turn + break_on_damage
DISPEL = "dispel"       # remove effects from target
CURE = "cure"           # remove debuffs + heal HP
UTILITY = "utility"     # Teleport, Earthquake
RESURRECT = "resurrect" # Resurrect / Resurrect True / Animate Dead
SUMMON = "summon"       # Summon Elemental (Spell::isSummon)
MIRROR_IMAGE = "mirror_image" # Mirror Image is single-target, not isSummon()
HYPNOTIZE = "hypnotize" # mind control (HP < 25 * power)
BERSERKER = "berserker" # forced attack nearest neighbor


@dataclass(frozen=True)
class Spell:
    name: str
    kind: str
    cost: int
    base_damage: int = 0
    speed_delta: int = 0
    halves_speed: bool = False
    attack_delta: int = 0
    defense_delta: int = 0
    side_friendly: bool = False
    is_mass: bool = False
    # AOE targeting
    aoe_pattern: str = ""       # ring1 / ring2 / ring_outer / chain / all_tagged / all_units
    target_tags: tuple = ()     # only hit units with these tags (e.g. ("undead",))
    exclude_tags: tuple = ()    # skip units with these tags
    # Cure
    heal_base: int = 0
    # Elemental flag mirrors Spell::isElementalSpell exactly.
    elemental: bool = False
    # Effect behavior (for BUFF / DEBUFF / CONTROL effects)
    effect_break_on_damage: bool = False
    effect_skip_turn: bool = False
    effect_stackable: bool = False
    effect_ranged_shield: float = 1.0
    effect_anti_magic: bool = False
    # Hypnotize / Berserker / Resurrect / Summon
    hp_threshold_per_power: int = 0   # Hypnotize: max HP = 25*power
    resurrect_per_power: int = 0      # Resurrect / Animate Dead: HP restore per power
    resurrect_permanent: bool = False # True for Resurrect True / Animate Dead
    summon_unit_type: str = ""        # Summon: which monster to spawn
    summon_count_per_power: int = 0   # Summon: count = power * this


# ── Spell data ───────────────────────────────────────────────────
# Values from fheroes2 spell.cpp spells[] table.
# Sorted by level within each kind.

SPELLS = {
    # ── Level 1 ──────────────────────────────────────────────────
    "Magic Arrow":    Spell("Magic Arrow",  DAMAGE, cost=3,  base_damage=10),
    "Blood Lust":     Spell("Blood Lust",   BUFF,   cost=3,  attack_delta=3,
                           side_friendly=True),
    "Bless":          Spell("Bless",        BUFF,   cost=3,
                           side_friendly=True, exclude_tags=("undead",)),
    "Cure":           Spell("Cure",         CURE,   cost=6,  heal_base=5,
                           side_friendly=True),
    "Curse":          Spell("Curse",        DEBUFF, cost=3,
                           exclude_tags=("undead",)),
    "Dispel Magic":   Spell("Dispel Magic", DISPEL, cost=5),
    "Haste":          Spell("Haste",        BUFF,   cost=3,  speed_delta=2,
                           side_friendly=True),
    "Shield":         Spell("Shield",       BUFF,   cost=3,
                           effect_ranged_shield=0.5, side_friendly=True),
    "Slow":           Spell("Slow",         DEBUFF, cost=3,  halves_speed=True),
    "Stoneskin":      Spell("Stoneskin",    BUFF,   cost=3,  defense_delta=3,
                           side_friendly=True),

    # ── Level 2 ──────────────────────────────────────────────────
    "Blind":          Spell("Blind",        CONTROL, cost=6,
                           effect_skip_turn=True, effect_break_on_damage=True),
    "Cold Ray":       Spell("Cold Ray",     DAMAGE,  cost=6,  base_damage=20,
                           elemental=True),
    "Death Ripple":   Spell("Death Ripple", AOE,     cost=6,  base_damage=5,
                           aoe_pattern="all_tagged", exclude_tags=("undead",)),
    "Disrupting Ray": Spell("Disrupting Ray", DEBUFF, cost=7, defense_delta=-3,
                           effect_stackable=True),
    "Dragon Slayer":  Spell("Dragon Slayer", BUFF,   cost=6,  attack_delta=5,
                           side_friendly=True),
    "Lightning Bolt": Spell("Lightning Bolt", DAMAGE, cost=7,  base_damage=25,
                           elemental=True),
    "Steelskin":      Spell("Steelskin",     BUFF,   cost=6,  defense_delta=5,
                           side_friendly=True),

    # ── Level 3 ──────────────────────────────────────────────────
    "Anti-Magic":     Spell("Anti-Magic",    BUFF,    cost=7,
                           effect_anti_magic=True, side_friendly=True),
    "Cold Ring":      Spell("Cold Ring",      AOE,     cost=9,  base_damage=10,
                           aoe_pattern="ring_outer", elemental=True),
    "Death Wave":     Spell("Death Wave",     AOE,     cost=10, base_damage=10,
                           aoe_pattern="all_tagged", exclude_tags=("undead",)),
    "Earthquake":     Spell("Earthquake",     UTILITY, cost=15),
    "Fireball":       Spell("Fireball",       AOE,     cost=9,  base_damage=10,
                           aoe_pattern="ring1", elemental=True),
    "Holy Word":      Spell("Holy Word",      AOE,     cost=9,  base_damage=10,
                           aoe_pattern="all_tagged", target_tags=("undead",)),
    "Mass Bless":     Spell("Mass Bless",     BUFF,    cost=12,
                           side_friendly=True, is_mass=True,
                           exclude_tags=("undead",)),
    "Mass Curse":     Spell("Mass Curse",     DEBUFF,  cost=12,
                           is_mass=True, exclude_tags=("undead",)),
    "Mass Dispel":    Spell("Mass Dispel",    DISPEL,  cost=12, is_mass=True),
    "Mass Haste":     Spell("Mass Haste",     BUFF,    cost=10, speed_delta=2,
                           side_friendly=True, is_mass=True),
    "Mass Slow":      Spell("Mass Slow",      DEBUFF,  cost=15, halves_speed=True,
                           is_mass=True),
    "Paralyze":       Spell("Paralyze",       CONTROL, cost=9,
                           effect_skip_turn=True, effect_break_on_damage=True),
    "Teleport":       Spell("Teleport",       UTILITY, cost=9,
                           side_friendly=True),

    # ── Level 4 ──────────────────────────────────────────────────
    "Chain Lightning": Spell("Chain Lightning", AOE, cost=15, base_damage=40,
                           aoe_pattern="chain", elemental=True),
    "Elemental Storm": Spell("Elemental Storm", AOE, cost=15, base_damage=25,
                           aoe_pattern="all_units", elemental=True),
    "Fireblast":      Spell("Fireblast",       AOE, cost=15, base_damage=10,
                           aoe_pattern="ring2", elemental=True),
    "Holy Shout":     Spell("Holy Shout",       AOE, cost=12, base_damage=20,
                           aoe_pattern="all_tagged", target_tags=("undead",)),
    "Mass Cure":      Spell("Mass Cure",        CURE, cost=15, heal_base=5,
                           side_friendly=True, is_mass=True),
    "Mass Shield":    Spell("Mass Shield",      BUFF, cost=7,
                           effect_ranged_shield=0.5, side_friendly=True,
                           is_mass=True),
    "Meteor Shower":  Spell("Meteor Shower",    AOE, cost=15, base_damage=25,
                           aoe_pattern="ring1"),
    "Petrification": Spell("Petrification", CONTROL, cost=0,
                           effect_skip_turn=True, effect_break_on_damage=False),

    # ── Level 5 ──────────────────────────────────────────────────
    "Armageddon":     Spell("Armageddon",       AOE, cost=20, base_damage=50,
                           aoe_pattern="all_units"),

    # ── Hypnotize / Berserker (force-target control) ─────────────
    # Hypnotize: target must have HP < 25*power; allegiance flips to caster's side.
    "Hypnotize":       Spell("Hypnotize",       HYPNOTIZE, cost=15,
                             hp_threshold_per_power=25),
    "Berserker":       Spell("Berserker",       BERSERKER, cost=12),

    # ── Resurrection ─────────────────────────────────────────────
    # Resurrect: temporary until battle end in fheroes2.
    # Resurrect True / Animate Dead: permanent.
    "Resurrect":       Spell("Resurrect",       RESURRECT, cost=12,
                             resurrect_per_power=50, resurrect_permanent=False,
                             exclude_tags=("undead",), side_friendly=True),
    "Resurrect True":  Spell("Resurrect True",  RESURRECT, cost=15,
                             resurrect_per_power=50, resurrect_permanent=True,
                             exclude_tags=("undead",), side_friendly=True),
    "Animate Dead":    Spell("Animate Dead",    RESURRECT, cost=10,
                             resurrect_per_power=50, resurrect_permanent=True,
                             target_tags=("undead",), side_friendly=True),

    # ── Summon (creates a new unit on the battlefield) ───────────
    # Mirror Image: a phantom copy of the target that dies on any damage
    # Summon Elemental × 4: spawn `power * 3` Elementals next to a friend
    "Mirror Image":   Spell("Mirror Image",   MIRROR_IMAGE, cost=25),
    "Summon Earth Elemental":
                       Spell("Summon Earth Elemental", SUMMON, cost=30,
                             summon_unit_type="Earth Elemental",
                             summon_count_per_power=3, side_friendly=True),
    "Summon Air Elemental":
                       Spell("Summon Air Elemental",  SUMMON, cost=30,
                             summon_unit_type="Air Elemental",
                             summon_count_per_power=3, side_friendly=True),
    "Summon Fire Elemental":
                       Spell("Summon Fire Elemental", SUMMON, cost=30,
                             summon_unit_type="Fire Elemental",
                             summon_count_per_power=3, side_friendly=True),
    "Summon Water Elemental":
                       Spell("Summon Water Elemental", SUMMON, cost=30,
                             summon_unit_type="Water Elemental",
                             summon_count_per_power=3, side_friendly=True),
}

DEFAULT_SPELLBOOK = [
    name for name in SPELLS
    if name != "Petrification"
]


@dataclass
class Effect:
    """A timed status on a unit (one spell's lingering effect)."""
    name: str
    remaining: int               # rounds left (0 = expired, removed by tick)
    speed_delta: int = 0
    halves_speed: bool = False
    attack_delta: int = 0
    defense_delta: int = 0
    skip_turn: bool = False
    break_on_damage: bool = False
    stackable: bool = False
    is_positive: bool = True     # False for debuffs (used by Cure)
    ranged_shield: float = 1.0   # multiplier on incoming ranged damage
    anti_magic: bool = False     # immune to all spells while active
    is_hypnotize: bool = False   # Hypnotize control — flips team allegiance
    is_berserk: bool = False     # Berserker control — forced nearest-neighbor


def spell_damage(spell: Spell, power: int) -> int:
    """Damage dealt by a DAMAGE / AOE spell at the given hero power."""
    return spell.base_damage * power


def make_effect(spell: Spell, power: int) -> Optional[Effect]:
    """Build the timed Effect for a buff / debuff / control spell.

    Effect lasts ``power`` rounds.  Control spells (Blind, Paralyze) last
    indefinitely (remaining=100) until broken by damage.
    """
    if spell.kind in (DAMAGE, AOE, DISPEL, CURE, UTILITY, RESURRECT,
                       SUMMON, MIRROR_IMAGE):
        return None

    if spell.kind == HYPNOTIZE:
        # Hypnotize is permanent (until dispelled / unit dies).
        return Effect(name="Hypnotize", remaining=100,
                      is_positive=False, is_hypnotize=True,
                      break_on_damage=False)

    if spell.kind == BERSERKER:
        # Berserker is permanent (until dispelled / unit dies).
        return Effect(name="Berserker", remaining=100,
                      is_positive=False, is_berserk=True,
                      break_on_damage=False)

    remaining = power
    is_positive = spell.kind == BUFF

    if spell.effect_skip_turn:
        # Control spells last until broken by damage.
        remaining = 100

    return Effect(
        name=spell.name,
        remaining=remaining,
        speed_delta=spell.speed_delta,
        halves_speed=spell.halves_speed,
        attack_delta=spell.attack_delta,
        defense_delta=spell.defense_delta,
        skip_turn=spell.effect_skip_turn,
        break_on_damage=spell.effect_break_on_damage,
        stackable=spell.effect_stackable,
        is_positive=is_positive,
        ranged_shield=spell.effect_ranged_shield,
        anti_magic=spell.effect_anti_magic,
    )


# ── spell_caster combat ability effects ───────────────────────
# These are applied by monster abilities (not hero spellcasting).
# Blind/Paralyze/Petrify skip the unit's turn; broken when damaged.
# Curse is the same effect as the hero spell.

_CONTROL_EFFECTS = {
    "Blind":    lambda: Effect("Blind",    remaining=100, skip_turn=True,
                              break_on_damage=True, is_positive=False),
    "Paralyze": lambda: Effect("Paralyze", remaining=100, skip_turn=True,
                              break_on_damage=True, is_positive=False),
    "Petrification": lambda: Effect("Petrification", remaining=100,
                                    skip_turn=True, break_on_damage=False,
                                    is_positive=False),
    "Curse":    lambda: Effect("Curse",    remaining=3,
                              is_positive=False),
    # "Dispel" is handled specially: remove all effects from target.
}


def make_spell_caster_effect(spell_name: str):
    """Return a monster spell-caster Effect, accepting canonical names."""
    canonical_name = {
        "Petrify": "Petrification",
    }.get(spell_name.title(), spell_name.title())
    factory = _CONTROL_EFFECTS.get(canonical_name)
    if factory is not None:
        return factory()
    return None
