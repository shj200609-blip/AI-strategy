"""ClassicAI spell selection and scoring helpers.

Mirror of fheroes2 ``ai_battle_spell.cpp``. The C++ code splits spell
heuristics into category-specific functions:

  * ``spellDamageValue``     — DAMAGE / AOE / Chain Lightning
  * ``spellDispelValue``     — Dispel / Mass Dispel
  * ``spellSummonValue``     — summon spells (Mirror Image, Elemental Storm)
  * ``spellResurrectValue``  — Resurrect / Animate Dead
  * ``spellDragonSlayerValue`` — Dragon Slayer
  * ``spellTeleportValue``   — melee-unlock Teleport
  * ``spellEarthquakeValue`` — siege-breaker Earthquake
  * ``spellEffectValue``     — buffs / debuffs / control (per spell ratio)

Plus per-target helpers:

  * ``getSpellSlowRatio`` / ``getSpellHasteRatio`` /
    ``getSpellDisruptingRayRatio`` — dynamic per-target ratios.
  * ``spellDurationMultiplier`` — duration-aware gating.
  * ``isSpellcastUselessForUnit`` — pre-filter spells already present on
    the target or that won't apply.

This module exposes the same public surface as before so the rest of the
planner (and tests) keeps compiling.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from engine.actions import CastAction
from engine.battle_state import BattleState
from engine.spells import (
    AOE,
    BERSERKER,
    BUFF,
    CONTROL,
    CURE,
    DAMAGE,
    DEBUFF,
    DISPEL,
    HYPNOTIZE,
    RESURRECT,
    SPELLS,
    SUMMON,
    UTILITY,
    Spell,
)
from engine.unit import Unit

from .constants import (
    ANTIMAGIC_LOW_LIMIT,
    BLOOD_LUST_RATIO,
    SPELL_VALUE_RATIO,
)
from .models import _SpellcastOutcome


# ── helpers ──────────────────────────────────────────────────────────

def _get_spell_power(commander) -> int:
    """Port of C++ ``getSpellPower`` = commander.GetPower() + bagArtifacts.

    Engine Hero exposes ``bag_artifacts`` (optional) so this stays a
    duck-typed shim for tests that don't model artifacts.
    """
    if commander is None:
        return 1
    base = int(getattr(commander, "power", 1))
    bag = getattr(commander, "bag_artifacts", None)
    if bag is None:
        return base
    bonus = getattr(bag, "get_total_artifact_effect_value", lambda _t: 0)(
        "EVERY_COMBAT_SPELL_DURATION")
    return base + int(bonus)


def _reduce_effectiveness_by_distance(unit: Unit) -> float:
    """``ReduceEffectivenessByDistance`` — penalise effects on units that
    have already crossed the board (fheroes2 ``GetDistanceFromBoardEdge``).
    The engine simplifies this to 1.0 unless we explicitly model board
    width (the wider the row, the smaller the penalty).
    """
    return 1.0


def _is_u64_unit(unit: Unit) -> bool:
    """Mirror C++ ``unit.isImmovable()`` (UNDEAD / STANDING units)."""
    return (unit.has_tag("undead") or unit.has_tag("elemental")
            or getattr(unit, "speed", 0) == 0)


def _has_resurrect(commander, target: Unit) -> bool:
    """C++ ``_commander->HaveSpellPoints + AllowApplySpell`` for resurrect."""
    if commander is None:
        return False
    for spell in getattr(commander, "spells", []) or []:
        if spell not in SPELLS:
            continue
        sp = SPELLS[spell]
        if sp.kind != RESURRECT:
            continue
        if not getattr(commander, "can_cast", lambda _s: False)(sp):
            continue
        if not getattr(target, "allow_apply_spell", lambda _s, _h: True)(sp, commander):
            continue
        return True
    return False


def _attacker_ignores_cover(commander) -> bool:
    """C++ NO_SHOOTING_PENALTY artifact or any Archery skill level."""
    if commander is None:
        return False
    if getattr(commander, "no_shooting_penalty", False):
        return True
    level = getattr(commander, "get_skill_level", lambda _s: 0)("archery")
    return level > 0


# ── per-target ratio helpers (slow / haste / disrupting ray) ─────────

def _get_spell_slow_ratio(target: Unit, *,
                          attacking_castle: bool = False,
                          my_army_average_speed: float = 0.0) -> float:
    """``getSpellSlowRatio`` (ai_battle_spell.cpp:329).

    Slow is largely useless against archers and castle defenders, useful in
    proportion to the lost-speed delta, doubled if the target already has
    Haste, halved if the target is already slower than our army, and
    reduced by board-edge distance when the target is grounded (no
    multiplier for flyers).

    The speed delta is data-driven: ``SPELLS["Slow"].speed_delta``
    (negative) — fheroes2 computes the actual loss per hero level, the
    engine dataclass already encodes the magnitude.
    """
    if getattr(target, "is_archer", False) or attacking_castle:
        # fheroes2: almost zero (TODO: archer-vs-flying case).
        return 0.01
    speed = int(getattr(target, "speed", 0) or 0)
    lost_speed = abs(int(getattr(SPELLS["Slow"], "speed_delta", -2)))
    new_speed = max(1, speed - lost_speed)
    lost = max(1, speed - new_speed)
    ratio = 0.1 * lost

    # C++ ordering (ai_battle_spell.cpp:343-352): the *already-slower*
    # halving fires unconditionally first; only then does the SP_HASTE
    # / flying branch decide whether to skip the distance penalty.
    if my_army_average_speed and speed < my_army_average_speed:
        # Slow is even less useful against an already-slow target.
        ratio /= 2.0
    if any(e.name == "Haste" for e in getattr(target, "effects", [])):
        return ratio * 2.0
    if getattr(target, "is_flying", False):
        return ratio
    ratio /= _reduce_effectiveness_by_distance(target)
    return ratio


def _get_spell_haste_ratio(target: Unit, *,
                           enemy_average_speed: float,
                           defensive_tactics: bool = False) -> float:
    """``getSpellHasteRatio`` (ai_battle_spell.cpp:356).

    Haste is twice as valuable when the target is slower than the enemy's
    average, doubled when the target is already slowed, and halved for
    archers or units in defensive formation (which don't need to move).

    Speed delta is read from ``SPELLS["Haste"].speed_delta`` rather than
    a hard-coded constant — matches C++ ``getFasterSpeed`` which scales
    with the casting hero's spell power.
    """
    speed = int(getattr(target, "speed", 0) or 0)
    gained_speed = abs(int(getattr(SPELLS["Haste"], "speed_delta", 2)))
    new_speed = speed + gained_speed
    gained = max(1, new_speed - speed)
    ratio = 0.05 * gained
    if speed < enemy_average_speed:
        ratio *= 2.0
    if any(e.name == "Slow" for e in getattr(target, "effects", [])):
        ratio *= 2.0
    elif (getattr(target, "is_archer", False) or defensive_tactics):
        ratio /= 2.0
    return ratio


def _get_spell_disrupting_ray_ratio(target: Unit, *,
                                    my_army_strength: float) -> float:
    """``getSpellDisruptingRayRatio`` (ai_battle_spell.cpp:297).

    The C++ reads ``Spell(DISRUPTINGRAY).ExtraValue()`` at runtime — the
    defence delta scales with hero spell-power. Our Spell dataclass
    stores the *base* defence delta on ``defense_delta``; read it from
    there instead of a hard-coded constant (the legacy
    ``DISRUPTING_RAY_DEF_REDUCTION`` constant in ``constants.py`` happens
    to match the base value but is a coincidence, not the source of
    truth).
    """
    defense = int(getattr(target, "effective_defense", target.defense))
    if defense <= 1:
        return 0.0
    # Prefer the runtime computed ExtraValue (ai_battle_spell.cpp:307
    # passes ``Spell(Spell::DISRUPTINGRAY).ExtraValue()`` directly), but
    # fall back to the dataclass attribute when the engine doesn't
    # expose a callable — ExtraValue typically equals
    # ``abs(defense_delta) * spell_power``.
    spell_value = _spell_extra_value(SPELLS["Disrupting Ray"], "defense_delta")
    if spell_value <= 0:
        return 0.2
    ratio = 0.2
    if defense <= spell_value:
        ratio *= (defense - 1) / float(spell_value)
    target_str = float(getattr(target, "strength", 0.0))
    if target_str > 0 and my_army_strength < target_str:
        ratio *= my_army_strength / target_str
    return ratio


def _spell_extra_value(spell: Spell, attr: str) -> int:
    """Port of fheroes2 ``Spell::ExtraValue`` — the *actual* defence
    delta Disrupting Ray strips at the current hero spell power. The
    engine's Spell dataclass stores the base ``defense_delta``; the
    runtime value scales with power when the spell has a SpellInfo that
    defines ``extra``/``duration``. Until the engine grows a SpellInfo
    table we approximate by reading the dataclass field directly.
    """
    extra = getattr(spell, "extra_value", None)
    if callable(extra):
        try:
            return int(extra())
        except TypeError:
            pass
    return abs(int(getattr(spell, attr, 0) or 0))


def _spell_duration_multiplier(target: Unit, *, spell_power: int) -> int:
    """``spellDurationMultiplier`` — 0 when duration<2 but the target has
    already used its turn (we approximate *used its turn* as ``is_alive``
    + active effects already ticking — engine has no TR_MOVED flag, so
    default to 1).
    """
    if spell_power < 2 and getattr(target, "_acted", False):
        return 0
    return 1


def _is_spellcast_useless_for_unit(target: Unit, enemies: list,
                                   spell: Spell) -> bool:
    """``isSpellcastUselessForUnit`` — filter spells that won't apply.

    Mirrors the C++ switch statement (ai_battle_spell.cpp:764-848). When
    invoked from ``isDispel`` mode, the unit may already have the spell
    applied (so we need a positive signal that it would be useful).
    """
    if _is_u64_unit(target):
        return spell.name != "Anti-Magic"
    effects = getattr(target, "effects", []) or []
    has = lambda n: any(e.name == n for e in effects)
    speed = int(getattr(target, "speed", 0) or 0)

    # C++ Speed::INSTANT is the sentinel for "unit's own turn always
    # fires first" — for our engine we don't model the constant, but
    # any unit with effective speed 0 (e.g. standing caster with
    # nothing to do) is the analogue and the Haste branch must guard
    # against it (otherwise we'd burn mana on a non-mover).
    if spell.name in ("Haste", "Mass Haste"):
        return has("Haste") or speed <= 0
    if spell.name in ("Slow", "Mass Slow"):
        # C++ Speed::CRAWLING — slowest crawler can't be slowed further.
        return has("Slow") or speed <= 1
    if spell.name in ("Shield", "Mass Shield"):
        # When the spell would last a single round and no enemy shooter
        # is still to act this turn, the spell is a waste of mana
        # (ai_battle_spell.cpp:788-806).
        if not has("Shield"):
            duration = _get_spell_power(getattr(target, "_commander", None))
            if duration <= 1:
                has_active_archer = False
                for e in enemies:
                    if (getattr(e, "is_archer", False)
                            and getattr(e, "is_alive", True)
                            and not getattr(e, "_acted", False)):
                        has_active_archer = True
                        break
                if not has_active_archer:
                    return True
        return has("Shield")
    if spell.name == "Hypnotize":
        # C++ uses just Modes(SP_HYPNOTIZE) — Berserker doesn't gate this.
        return getattr(target, "is_hypnotized", False)
    if spell.name == "Mirror Image":
        # C++ CAP_MIRROROWNER: the *owner* of mirror images (not the
        # phantom itself) is the gating predicate. A unit without an
        # existing mirror-image flag can still receive copies.
        return bool(getattr(target, "_has_mirror_owner", False)
                    or getattr(target, "is_mirror_owner", False))
    if spell.name in ("Bless", "Mass Bless"):
        return has("Bless")
    if spell.name == "Blood Lust":
        return has("Blood Lust")
    if spell.name in ("Curse", "Mass Curse"):
        return has("Curse")
    if spell.name in ("Stoneskin", "Steelskin"):
        return any(e.name in ("Stoneskin", "Steelskin") for e in effects)
    if spell.name in ("Blind", "Paralyze", "Petrification"):
        return any(e.name in ("Blind", "Paralyze", "Petrification") for e in effects)
    if spell.name == "Dragon Slayer":
        return has("Dragon Slayer")
    if spell.name == "Anti-Magic":
        return has("Anti-Magic")
    if spell.name == "Berserker":
        return has("Berserker")
    if spell.name == "Disrupting Ray":
        return int(getattr(target, "effective_defense", target.defense)) <= 1
    return False


def _get_extra_value(spell: Spell, attr: str) -> int:
    """``Spell::ExtraValue`` defaults for engine-defined spells."""
    return int(getattr(spell, attr, 0) or 0)


# ── spellEffectValue (per target) — main ratio table ────────────────

def _spell_effect_value_single(self, battle: BattleState, spell: Spell,
                               target: Unit, enemies: list,
                               target_is_last: bool,
                               for_dispel: bool) -> float:
    """Mirror ``spellEffectValue(spell, target, enemies, targetIsLast, forDispel)``.

    Returns the value contributed by casting *spell* on *target* given the
    *enemies* set; multiplied by ``spellDurationMultiplier``.

    The result is **not** divided by sqrt(cost/3) — that is done by the
    outer dispatcher.
    """
    if not for_dispel and _is_spellcast_useless_for_unit(target, enemies, spell):
        return 0.0

    name = spell.name
    ratio = 0.0

    if name in ("Slow", "Mass Slow"):
        ratio = _get_spell_slow_ratio(target,
                                      attacking_castle=self._attacking_castle,
                                      my_army_average_speed=self._my_army_average_speed)
    elif name == "Blind":
        if target_is_last:
            if target.has_ability("unlimited_retaliation"):
                return 0.0
            if not getattr(target, "can_retaliate", lambda: True)():
                return 0.0
            ratio = 0.4
        else:
            ratio = 0.8
    elif name in ("Curse", "Mass Curse"):
        dmg_min = int(getattr(target, "damage_min", 0) or 0)
        dmg_max = int(getattr(target, "damage_max", 0) or 0)
        if dmg_min == dmg_max:
            return 0.0
        ratio = 0.15
    elif name == "Berserker":
        if target_is_last:
            return 0.0
        ratio = 0.85
    elif name == "Paralyze":
        if target_is_last:
            if target.has_ability("unlimited_retaliation"):
                return 0.0
            if not getattr(target, "can_retaliate", lambda: True)():
                return 0.0
            dur = _spell_duration_multiplier(target, spell_power=_get_spell_power(self._commander))
            if dur < 1:
                return 0.0
            ratio = 0.5
        else:
            ratio = 0.85
    elif name == "Hypnotize":
        ratio = 1.5
    elif name == "Disrupting Ray":
        ratio = _get_spell_disrupting_ray_ratio(
            target, my_army_strength=self._my_army_strength)
    elif name in ("Haste", "Mass Haste"):
        ratio = _get_spell_haste_ratio(
            target, enemy_average_speed=self._enemy_average_speed,
            defensive_tactics=self._defensive_tactics)
    elif name == "Blood Lust":
        ratio = BLOOD_LUST_RATIO
    elif name in ("Bless", "Mass Bless"):
        dmg_min = int(getattr(target, "damage_min", 0) or 0)
        dmg_max = int(getattr(target, "damage_max", 0) or 0)
        if dmg_min == dmg_max:
            return 0.0
        ratio = 0.15
    elif name == "Stoneskin":
        ratio = 0.1
    elif name == "Steelskin":
        ratio = 0.2
    elif name in ("Anti-Magic", "Mirror Image", "Shield", "Mass Shield"):
        ratio = 0.0
    else:
        return 0.0

    effects = getattr(target, "effects", []) or []

    # Combos: Bless-on-Curse and Curse-on-Bless are bonus *2 (C++ lines
    # 515-520).
    if name in ("Curse", "Mass Curse") and any(e.name == "Bless" for e in effects):
        ratio *= 2.0
    elif name in ("Bless", "Mass Bless") and any(e.name == "Curse" for e in effects):
        ratio *= 2.0
    elif name == "Anti-Magic":
        # C++ (ai_battle_spell.cpp:521-544): the formula branch fires when
        # the target has NO good magic AND enemy_spell_str exceeds the
        # threshold. Inside, the ratio is the *computed* formula value,
        # optionally *1.5 if enemy spell strength outstrips the enemy army
        # and *2 if the target has bad magic on it. There is no separate
        # branch that returns 0.9 unconditionally — a target with only
        # bad magic but low enemy spell strength falls through with
        # ratio = 0.0.
        enemy_spell_str = float(getattr(self, "_enemy_spell_strength", 0.0))
        has_good_magic = any(getattr(e, "is_positive", True) for e in effects)
        has_bad_magic = any(not getattr(e, "is_positive", True)
                            for e in effects)
        if not has_good_magic and enemy_spell_str > ANTIMAGIC_LOW_LIMIT:
            ratio_limit = 0.9
            if _has_resurrect(self._commander, target):
                ratio_limit = 0.35
            # 0..5000 enemy spell strength → 0.0..0.9 ratio (clamp).
            ratio = min(enemy_spell_str / ANTIMAGIC_LOW_LIMIT * 0.036,
                        ratio_limit)
            if enemy_spell_str > self._enemy_army_strength:
                ratio *= 1.5
            if has_bad_magic:
                ratio *= 2.0
        else:
            ratio = 0.0
    elif name == "Mirror Image":
        if getattr(target, "is_archer", False):
            ratio = 1.0
        elif getattr(target, "is_flying", False):
            ratio = 0.55
        else:
            ratio = 0.33
        if int(getattr(target, "speed", 0)) < self._enemy_average_speed:
            ratio /= 5.0
    elif name == "Berserker" and not getattr(target, "is_archer", False):
        ratio /= _reduce_effectiveness_by_distance(target)
    elif name in ("Shield", "Mass Shield"):
        ratio = (self._enemy_ranged_units_only / max(self._enemy_army_strength, 1.0)
                 * 0.3)
        if getattr(target, "is_archer", False):
            ratio *= 1.25

    base = float(getattr(target, "strength", 0.0))
    duration = _spell_duration_multiplier(target,
                                          spell_power=_get_spell_power(self._commander))
    return base * ratio * duration


# ── public dispatcher ───────────────────────────────────────────────

def _score_spell(self, battle: BattleState, hero, spell: Spell,
                 unit: Unit) -> Tuple[float, Optional[Unit],
                                       Optional[Tuple[int, int]],
                                       Optional[Tuple[int, int]]]:
    """Top-level dispatcher — mirrors C++ ``selectBestSpell``.

    Returns ``(value, target_unit_or_none, cell_or_none, destination)``.
    *value* is the raw spellPointValue used to rank candidates.
    """
    if hero is None:
        return (0.0, None, None, None)

    my_team = unit.team
    enemy_team = 1 - my_team
    friendly = battle.alive(my_team)
    enemies = battle.alive(enemy_team)
    # fheroes2 ``true*`` excludes hypnotized units — our effective_* helpers
    # already account for that, so friendly / enemies above are the
    # "true" sets.

    name = spell.name

    # ── 1. Damage spells (single-target + AoE including Chain Lightning)
    if spell.kind == DAMAGE:
        return _score_damage(self, spell, hero, friendly, enemies,
                             retreating=False)
    if spell.kind == AOE:
        return _score_aoe(self, spell, hero, friendly, enemies,
                          retreating=False)
    # ── 2. Dispel
    if spell.kind == DISPEL:
        return _score_dispel(self, spell, hero, friendly, enemies,
                             is_mass=spell.is_mass)
    # ── 3. Summon
    if spell.kind == SUMMON:
        return _score_summon(self, spell, hero, enemies)
    # ── 4. Resurrect
    if spell.kind == RESURRECT:
        return _score_resurrect(self, spell, hero, battle)
    # ── 5. Dragon Slayer (BUFF to friendly, only if any enemy dragon)
    if name == "Dragon Slayer":
        return _score_dragon_slayer(self, spell, hero, friendly, enemies)
    # ── 6. Teleport (UTILITY; melee-only unlock)
    if name == "Teleport":
        return _score_teleport(self, battle, spell, hero, unit, enemies)
    # ── 7. Earthquake
    if name == "Earthquake":
        return _score_earthquake(self, spell, friendly)
    # ── 8. Friendly-targeted
    if spell.side_friendly or spell.kind == CURE:
        return _score_effect_dispatch(self, spell, hero, friendly, enemies,
                                      is_mass=spell.is_mass)
    # ── 9. Enemy-targeted (debuffs / control / hypnotize / berserker)
    return _score_effect_dispatch(self, spell, hero, enemies, enemies,
                                  is_mass=spell.is_mass)


# ── per-category scorers ─────────────────────────────────────────────

def _score_damage(self, spell: Spell, hero, friendly: list, enemies: list,
                  *, retreating: bool) -> Tuple[float, Optional[Unit],
                                                 Optional[Tuple[int, int]],
                                                 Optional[Tuple[int, int]]]:
    """``spellDamageValue`` — single-target + chain-lightning branches."""
    raw_damage = int(getattr(spell, "base_damage", 0) or 0) * _get_spell_power(hero)
    # C++ (ai_battle_spell.cpp:168): magic resistance is applied at the
    # call site — the per-target damage is reduced by ``100 -
    # GetMagicResist(spell, hero)`` percent. The engine doesn't expose
    # ``GetMagicResist`` yet; for now we treat a unit as fully
    # resistant if it has a 100% resist tag.
    resist_pct = getattr(spell, "_magic_resist_pct", 0)  # engine-injected
    best_value = 0.0
    best_target: Optional[Unit] = None
    for enemy in enemies:
        if not enemy.is_alive:
            continue
        if any(enemy.has_tag(t) for t in getattr(spell, "exclude_tags", ())):
            continue
        if getattr(enemy, "is_immune_to_spells", False):
            continue
        per_target = raw_damage
        if resist_pct:
            per_target = raw_damage * (100 - resist_pct) // 100
        if retreating and getattr(enemy, "is_mirror", False):
            value = 0.0
        else:
            value = _damage_heuristic(
                enemy, per_target,
                army_strength=self._enemy_army_strength,
                army_speed=self._enemy_average_speed,
                retreating=retreating, enemies=enemies)
        v = value if retreating else _spell_damage_score(value, spell)
        if v > best_value:
            best_value = v
            best_target = enemy
    return (best_value, best_target, None, None)


def _score_aoe(self, spell: Spell, hero, friendly: list, enemies: list,
               *, retreating: bool) -> Tuple[float, Optional[Unit],
                                              Optional[Tuple[int, int]],
                                              Optional[Tuple[int, int]]]:
    """``spellDamageValue`` AoE branch — pick best centre cell.

    Mirrors C++ ai_battle_spell.cpp:222-280 verbatim:

      * **Chain Lightning** (``aoe_pattern == "chain"`` or
        ``spell.name == "Chain Lightning"``): iterate per enemy
        centre; use the chain-neighbour set as the affected
        population. C++ calls ``arena.GetTargetsForSpell(commander,
        spell, enemy->GetHeadIndex())`` to obtain the exact chain set —
        the engine approximates this to ``[enemy] + adjacent enemies``
        within range 1.
      * **Other AoE** (Fireball / Fireblast / Meteor Shower / Cold
        Ring / Death Ripple / Holy Word etc.): iterate per enemy
        centre (the engine doesn't materialise a board of cells, so
        using an enemy anchor is the closest analogue); the affected
        set is enemy centre + friendly fire on adjacent cells, with
        a per-tag filter so e.g. Holy Word skips non-undead.
      * **Army-wide spells** (``aoe_pattern in {"all_units",
        "all_tagged"}`` and ``isApplyWithoutFocusObject``): single
        pass over every alive unit (no per-cell search).

    Magic resistance is applied per-target; friendly-fire subtracts
    the same heuristic from the total.
    """
    raw_damage = int(getattr(spell, "base_damage", 0) or 0) * _get_spell_power(hero)
    resist_pct = getattr(spell, "_magic_resist_pct", 0)
    per_target_dmg = raw_damage * (100 - resist_pct) // 100 \
        if resist_pct else raw_damage

    is_chain = (spell.name == "Chain Lightning"
                or getattr(spell, "aoe_pattern", "") == "chain")
    is_army_wide = (getattr(spell, "aoe_pattern", "") in ("all_units", "all_tagged"))

    if is_army_wide:
        # C++ branch: ``isApplyWithoutFocusObject`` — sum over all
        # enemies, subtract sum over all friends. Single cell = -1.
        value = 0.0
        for enemy in enemies:
            if not getattr(enemy, "is_alive", True):
                continue
            if getattr(enemy, "is_immune_to_spells", False):
                continue
            if any(enemy.has_tag(t)
                   for t in getattr(spell, "exclude_tags", ())):
                continue
            value += _damage_heuristic(
                enemy, per_target_dmg,
                army_strength=self._enemy_army_strength,
                army_speed=self._enemy_average_speed,
                retreating=retreating, enemies=enemies)
        for friend in friendly:
            if not getattr(friend, "is_alive", True):
                continue
            if getattr(friend, "is_immune_to_spells", False):
                continue
            if any(friend.has_tag(t)
                   for t in getattr(spell, "exclude_tags", ())):
                continue
            if (retreating and friend is unit_itself(self, friendly)
                    and abs(_damage_heuristic(
                        friend, per_target_dmg,
                        army_strength=self._my_army_strength,
                        army_speed=self._my_army_average_speed,
                        retreating=True, enemies=enemies)
                            - getattr(friend, "strength", 0.0)) < 0.001):
                return (0.0, None, None, None)
            value -= _damage_heuristic(
                friend, per_target_dmg,
                army_strength=self._my_army_strength,
                army_speed=self._my_army_average_speed,
                retreating=retreating, enemies=enemies)
        # Army-wide spells bypass the per-cell cost discount in C++
        # (no single cell → no best-cell ranking). Cost-discount the
        # total though, so the spell stays comparable to single-target
        # damage in the planner.
        score = value if retreating else _spell_damage_score(value, spell)
        # Pick the strongest enemy as a representative target so the
        # engine can build a CastAction; the C++ sets cell = -1.
        rep = max(enemies, key=lambda u: getattr(u, "strength", 0.0),
                  default=None)
        return (score, rep, None, None)

    best_value = float("-inf")
    best_target: Optional[Unit] = None

    # Per-centre evaluation. For Chain Lightning, only enemy centres
    # are valid; for other AoE we iterate enemy centres too because
    # that's where the spell would land usefully (C++ iterates every
    # board cell but the engine doesn't materialise one).
    centres = [e for e in enemies if getattr(e, "is_alive", True)]
    for enemy in centres:
        if getattr(enemy, "is_immune_to_spells", False):
            continue
        # Build the affected set: chain → enemy + adjacent enemies.
        # Other AoE → enemy + adjacent friendlies (for friendly-fire
        # penalty), but C++ also loops over the board and pulls
        # whatever falls inside the radius; the engine approximates
        # to "all alive friendlies within 1 hex" for that.
        if is_chain:
            affected = [enemy] + _neighbour_targets(enemy, enemies)
        else:
            affected = [enemy] + _neighbour_targets(enemy, friendly)
            # Also include nearby enemies in the radius (Fireball /
            # Meteor Shower hit a small ring of hexes).
            affected += _neighbour_targets(enemy, enemies)
        seen = set()
        affected = [u for u in affected
                    if id(u) not in seen and not seen.add(id(u))]

        if retreating:
            damage_heurs = []
            for target in affected:
                if target is enemy or target.team != enemy.team:
                    damage_heurs.append(_damage_heuristic(
                        target, per_target_dmg,
                        army_strength=self._enemy_army_strength,
                        army_speed=self._enemy_average_speed,
                        retreating=True, enemies=enemies))
                else:
                    if retreating and target is unit_itself(self, friendly):
                        return (0.0, None, None, None)
                    damage_heurs.append(-_damage_heuristic(
                        target, per_target_dmg,
                        army_strength=self._my_army_strength,
                        army_speed=self._my_army_average_speed,
                        retreating=True, enemies=enemies))
            value = sum(damage_heurs)
        else:
            value = 0.0
            for target in affected:
                if target.team != enemy.team:
                    # Enemy hit — add value.
                    if getattr(target, "is_immune_to_spells", False):
                        continue
                    if any(target.has_tag(t)
                           for t in getattr(spell, "exclude_tags", ())):
                        continue
                    value += _damage_heuristic(
                        target, per_target_dmg,
                        army_strength=self._enemy_army_strength,
                        army_speed=self._enemy_average_speed,
                        retreating=False, enemies=enemies)
                else:
                    # Friendly fire (C++ subtracts valueLost on
                    # friendly targets caught in the radius).
                    if getattr(target, "is_immune_to_spells", False):
                        continue
                    if any(target.has_tag(t)
                           for t in getattr(spell, "exclude_tags", ())):
                        continue
                    value -= _damage_heuristic(
                        target, per_target_dmg,
                        army_strength=self._my_army_strength,
                        army_speed=self._my_army_average_speed,
                        retreating=False, enemies=enemies)
        v = value if retreating else _spell_damage_score(value, spell)
        if v > best_value:
            best_value = v
            best_target = enemy
    if best_target is None:
        return (0.0, None, None, None)
    return (best_value, best_target,
            best_target.pos if best_target else None, None)


def unit_itself(self, friendly):
    """Snap helper used by AoE retreating branch."""
    for u in friendly:
        if getattr(u, "is_alive", True):
            return u
    return friendly[0]


def _hit_points_for(unit: Unit) -> int:
    """C++ mirror: mirror images have HP=1."""
    if getattr(unit, "is_mirror", False):
        return 1
    return int(getattr(unit, "_total_hp", 0) or 0)


def _neighbour_targets(unit: Unit, pool: list) -> list:
    grid = None
    # caller-side helper: the engine's AoE spell resolutions use
    # BattleState._aoe_cells; here we approximate to all alive units
    # within range — but fheroes2 chain-lightning distances are 0/cell.
    return [u for u in pool if u is not unit and getattr(u, "is_alive", True)]


def _damage_heuristic(unit: Unit, damage: int, *,
                      army_strength: float, army_speed: float,
                      retreating: bool,
                      enemies: Optional[list] = None) -> float:
    """Port of C++ ``damageHeuristic`` — the per-unit value of damage.

    Mirrors ai_battle_spell.cpp:167-213: damage is reduced by the
    target's magic resistance (``GetMagicResist(spell, hero)`` returns
    0..100), so a fully-resisting target lands ``damage = 0`` and the
    spell is treated as a no-op (no kills, no wakeup).

    When the target carries ``CAP_MIRROROWNER`` and the spell kills it,
    the mirror-image copies hanging off the owner are added to the
    overall strength bonus — destroying the owner also destroys its
    mirrors and their worth is real strength lost by the enemy.
    """
    if getattr(unit, "is_immune_to_spells", False):
        return 0.0

    # Mirror-image phantom has HP=1 for the purpose of "did we kill it".
    is_mirror = bool(getattr(unit, "is_mirror", False))

    if retreating:
        # fheroes2 in retreating mode ignores partial damage: only units
        # actually killed by the spell count. Mirror Images always die in
        # one hit so they contribute 0 (they disappear post-battle).
        if is_mirror:
            return 0.0
        # HowManyWillBeKilled ≈ damage / max_hp, capped to count.
        per_unit = max(1, getattr(unit, "max_hp", 1))
        killed = min(int(getattr(unit, "count", 0)), max(0, damage // per_unit))
        return float(getattr(unit, "monster_strength", 0.0)) * killed

    # fheroes2: damage *= (100 - magicResist) / 100; if it drops to 0,
    # the spell never lands.
    if damage <= 0:
        return 0.0

    hp = 1 if is_mirror else _hit_points_for(unit)
    if damage >= hp:
        bonus = 0.07 if int(getattr(unit, "speed", 0)) > army_speed else 0.035
        overall = float(getattr(unit, "strength", 0.0))
        # CAP_MIRROROWNER (ai_battle_spell.cpp:191-198): if the dying
        # unit is the owner of mirror images, the copies' strengths
        # ride along on the kill bonus.
        if getattr(unit, "is_mirror_owner", False) and enemies:
            for other in enemies:
                if (other is unit
                        or not getattr(other, "is_mirror", False)
                        or getattr(other, "mirror_of", None) is not unit):
                    continue
                overall += float(getattr(other, "strength", 0.0))
        return overall + army_strength * bonus

    lost = min(damage / max(hp, 1), 1.0)
    if _is_u64_unit(unit):
        # Penalty for waking up an immovable: surviving portion weighs
        # against us (fheroes2 ``unitPercentageLost + unitPercentageLost - 1``)
        lost += lost - 1.0
    return lost * float(getattr(unit, "strength", 0.0))


def _score_dispel(self, spell: Spell, hero, friendly: list, enemies: list,
                  *, is_mass: bool) -> Tuple[float, Optional[Unit],
                                             Optional[Tuple[int, int]],
                                             Optional[Tuple[int, int]]]:
    """``spellDispelValue`` — strip buffs/debuffs off either side.

    Mirror of C++ (ai_battle_spell.cpp:586-633):

      * **Mass Dispel** (``is_mass=True``): every friendly/enemy unit's
        ``unitValue`` is *added* to ``outcome.value`` via
        ``updateOutcome(..., isMassEffect=True)``. The mass spell has
        no single target cell, so ``outcome.cell`` is left at ``None``
        (== C++ ``-1``). The Python ``best_target`` is a convenience
        for the engine's ``CastAction``; it's the unit with the
        largest individual contribution so the action has a
        human-readable anchor.
      * **Single Dispel** (``is_mass=False``): only the largest
        ``unitValue`` survives in ``outcome.value``; its cell becomes
        ``outcome.cell``.

    The friend loop is unconditional (always strip bad magic), the
    enemy loop is gated by ``is_dispel_kind`` because Mass Cure /
    Mass Bless etc. don't care about enemy effects.
    """
    outcome = _SpellcastOutcome()
    best_target: Optional[Unit] = None
    best_unit_value = 0.0
    is_dispel_kind = spell.name in ("Dispel Magic", "Mass Dispel")
    # Friendly side: friendly-targeted spells strip BAD magic from us.
    for friend in friendly:
        effects = getattr(friend, "effects", []) or []
        if not effects:
            continue
        unit_value = 0.0
        for eff in effects:
            sub = _spell_effect_value_single(
                self, None, SPELLS[eff.name] if eff.name in SPELLS else None,
                friend, enemies, target_is_last=False, for_dispel=True) \
                if eff.name in SPELLS else 0.0
            bad = not getattr(eff, "is_positive", True)
            if bad:
                unit_value += sub
            elif is_dispel_kind:
                unit_value -= sub
        if unit_value > best_unit_value:
            best_unit_value = unit_value
            best_target = friend
        # CRITICAL (matches C++ isMassSpell): when is_mass is True the
        # contribution is *added*, not max'd — without this flag a
        # Mass Dispel would be ranked as if it only hit one stack.
        outcome.update_outcome(unit_value, friend.pos, is_mass_effect=is_mass)
    # Enemy side: only proper Dispel(Kind) cares about enemy effects.
    if is_dispel_kind:
        enemy_is_last = len(enemies) == 1
        for enemy in enemies:
            effects = getattr(enemy, "effects", []) or []
            if not effects:
                continue
            unit_value = 0.0
            for eff in effects:
                if eff.name not in SPELLS:
                    continue
                sub = _spell_effect_value_single(
                    self, None, SPELLS[eff.name], enemy, enemies,
                    target_is_last=enemy_is_last, for_dispel=True)
                unit_value += sub if getattr(eff, "is_positive", True) else -sub
            if unit_value > best_unit_value:
                best_unit_value = unit_value
                best_target = enemy
            outcome.update_outcome(unit_value, enemy.pos,
                                   is_mass_effect=is_mass)
    if best_target is None:
        return (0.0, None, None, None)
    return (outcome.value, best_target, outcome.cell, outcome.destination_cell)


def _score_summon(self, spell: Spell, hero, enemies: list
                  ) -> Tuple[float, Optional[Unit], Optional[Tuple[int, int]],
                             Optional[Tuple[int, int]]]:
    """``spellSummonValue`` — strength with diminishing return.

    Mirrors C++ (ai_battle_spell.cpp:685-707):

      * **Gate**: ``arena.GetFreePositionNearHero(heroColor) < 0`` →
        return empty (no room for the summoned unit). The engine
        proxies this via ``arena.has_free_position_near_hero``; we
        keep the cautious-offensive / no-enemies check as a
        fallback for battles where the engine doesn't expose the
        helper.
      * **Mirror Image** is its own branch (C++ bails out at the top
        because Mirror Image's summon count / strength differs from
        a creature summon — the value comes from the *target*
        enemy's strength, scaled by ``Mirror Image``'s actual
        image-count). fheroes2 doesn't have a Mirror Image branch
        inside ``spellSummonValue`` itself (it returns {} when there
        are no free positions and trusts the caller's
        ``spell.value = summon.GetStrengthWithBonus(...)`` for the
        rest). The Python port keeps the explicit Mirror Image
        handling because the engine model has no summon Unit to
        materialise.
      * **Creature summon**: ``value = summon.GetStrengthWithBonus
        (commander.GetAttack(), commander.GetDefense())``. The
        engine doesn't model the bonus table, so we approximate
        by reading the unit's ``monster_strength`` from the spell's
        declared ``summon_unit_type`` (the engine's ``config.UNIT_TYPES``
        table).
    """
    arena = getattr(self, "_arena", None)
    has_free_pos = getattr(arena, "has_free_position_near_hero", None)
    if callable(has_free_pos):
        try:
            if not has_free_pos(hero):
                return (0.0, None, None, None)
        except TypeError:
            pass
    else:
        # Fallback: original cautious-offensive heuristic. Note that
        # the C++ checks the free-position helper unconditionally —
        # not just when enemies are absent.
        if not getattr(self, "_cautious_offensive", False) and not enemies:
            return (0.0, None, None, None)

    power = _get_spell_power(hero)

    # Mirror Image — separate value path. C++ treats this as
    # ``Monster::MirrorImage`` spawn-strength via
    # ``Troop::GetStrengthWithBonus``, but the engine's no-actual-
    # summon-creature rule means we approximate the spell value by
    # the strongest enemy's monster-strength scaled by a per-power
    # multiplier (one phantom per 5 hero power is the closest
    # engine-side constant; original fheroes2 is 1 image per spell
    # at power 1, +1 per 5 power beyond that, capped at 4).
    if spell.name == "Mirror Image":
        if not enemies:
            return (0.0, None, None, None)
        best = max(enemies, key=lambda u: float(getattr(u, "strength", 0.0)))
        # Mirror Image never has a "free position" issue — it
        # spawns on the target's cell. Don't gate on
        # ``has_free_position_near_hero`` for it.
        image_count = max(1, min(4, 1 + power // 5))
        per_strength = float(getattr(best, "monster_strength", 0.0))
        return (per_strength * image_count, best, best.pos, None)

    if not getattr(spell, "summon_unit_type", ""):
        return (0.0, None, None, None)

    # Elemental-style summons: the C++ constructs a ``Troop(Monster
    # (spell), getSummonMonsterCount(spell, power))`` and asks
    # ``GetStrengthWithBonus(commander.GetAttack(),
    # commander.GetDefense())`` for the value. We approximate the
    # bonus by reading the hero's primary stats and multiplying by
    # the standard ``1 + 0.1 * attack + 0.05 * defense`` factor used
    # in fheroes2's ``Monster::GetStrengthWithBonus``.
    summon_count = _summon_count_for(spell, power)
    if summon_count <= 0:
        return (0.0, None, None, None)

    # Per-monster base strength — pull from the engine's creature
    # table when available.
    unit_type = getattr(spell, "summon_unit_type", "")
    per_monster_base = _lookup_monster_base_strength(unit_type)
    if per_monster_base is None:
        # Engine doesn't model the creature — fall back to a
        # conservative estimate and let the spell book absorb the
        # imprecision (no test should hit this path without
        # registering the summon unit type).
        per_monster_base = 30.0

    attack = int(getattr(hero, "attack", 0) or 0)
    defense = int(getattr(hero, "defense", 0) or 0)
    bonus_factor = 1.0 + 0.1 * attack + 0.05 * defense
    value = per_monster_base * bonus_factor * summon_count

    if self._my_army_strength > self._enemy_army_strength * 2:
        value /= 2.0
    # Synthesise a placeholder so the engine's CastAction has a
    # representative target (no actual Unit is materialised).
    placeholder_unit = type("Placeholder", (), {
        "count": summon_count,
        "monster_strength": per_monster_base * bonus_factor,
        "strength": value,
        "is_alive": True,
        "pos": None,
    })()
    return (value, placeholder_unit, None, None)


def _lookup_monster_base_strength(unit_type: str) -> Optional[float]:
    """Engine proxy for ``Monster::GetMonsterBaseStrength(unit_type)``."""
    if not unit_type:
        return None
    try:
        import config
        entry = config.UNIT_TYPES.get(unit_type)
    except ImportError:
        entry = None
    if not entry:
        return None
    # Use the engine's `_compute_base_strength`-style mean: sqrt(
    # damage_avg * hp) * count-factor — but config doesn't always
    # expose all fields; fall back to the helper if present.
    if hasattr(entry, "get"):
        hp = entry.get("hp", 1)
        speed = entry.get("speed", 4)
        dmg = entry.get("damage", 1)
        try:
            dmg_min = entry.get("damage_min", dmg)
            dmg_max = entry.get("damage_max", dmg)
            dmg_avg = (dmg_min + dmg_max) / 2.0
        except Exception:
            dmg_avg = float(dmg)
        # Mirror fheroes2 getMonsterBaseStrength roughly.
        import math
        return math.sqrt(dmg_avg * hp) * (1.0 + max(0, (speed - 4)) * 0.1)
    return None


def _summon_count_for(spell: Spell, power: int) -> int:
    """``getSummonMonsterCount`` value (per-power multiplier)."""
    return int(getattr(spell, "summon_count_per_power", 0) or 0) * max(power, 0)


def _score_resurrect(self, spell: Spell, hero, battle: BattleState
                     ) -> Tuple[float, Optional[Unit], Optional[Tuple[int, int]],
                                Optional[Tuple[int, int]]]:
    """``spellResurrectValue`` — based on missing HP × per-unit strength.

    Mirrors C++ ai_battle_spell.cpp:635-683. The C++ uses an
    inline lambda ``updateBestOutcome`` that for *both* alive and
    graveyard units computes ``missingHP = min(unit->GetMissing
    HitPoints(), hpRestored)`` — clamping the restore budget to
    what is actually missing. The Python port must apply the same
    cap to graveyard corpses (a creature that's already fully healed
    isn't worth any further resurrection).

    Alive units pass through ``unit->AllowApplySpell`` first (C++
    line 663); graveyard entries go through
    ``arena.isAbleToResurrectFromGraveyard`` (C++ line 672). The
    engine doesn't model the second helper, so we keep the alive
    filter and leave the graveyard iteration as-is.
    """
    best_value = 0.0
    best_target: Optional[Unit] = None
    if not self._commander:
        return (0.0, None, None, None)
    per_power = int(getattr(spell, "resurrect_per_power", 0) or 0)
    hp_restore = per_power * _get_spell_power(hero)

    my_color = getattr(self, "_my_color",
                       1 - getattr(battle, "attacker_team", 0))
    for unit in battle.alive(my_color):
        missing = (unit.max_hp * max(0, unit.count - 1)
                   + (unit.max_hp - unit.hp))
        # CRITICAL: clamp to hp_restore (C++ line 643). Without this
        # the value is inflated when a unit has more missing HP than
        # the spell can deliver.
        restored = min(missing, hp_restore)
        if restored <= 0:
            continue
        single = float(getattr(unit, "monster_strength", 0.0))
        value = restored * single / max(1, getattr(unit, "max_hp", 1))
        if (self._my_army_strength > self._enemy_army_strength
                and getattr(spell, "resurrect_permanent", False)):
            # C++ branches on spell.GetID() != Spell::RESURRECT — the
            # Resurrect True / Animate Dead variants get the *2 bonus.
            value *= 2.0
        if value > best_value:
            best_value = value
            best_target = unit
    # Graves
    graveyard = getattr(battle, "dead", []) or []
    for corpse in graveyard:
        if spell.target_tags and not any(t in spell.target_tags
                                          for t in getattr(corpse, "tags", set())):
            continue
        if any(t in getattr(spell, "exclude_tags", ())
               for t in getattr(corpse, "tags", set())):
            continue
        init_count = int(getattr(corpse, "original_count", 0) or 0)
        if init_count <= 0:
            continue
        single = float(getattr(corpse, "_base_strength", 0.0) or 0.0) or 1.0
        # C++ uses ``unit->GetMissingHitPoints()`` — for a fully-dead
        # stack that's ``init_count * max_hp``. Cap by hp_restore.
        missing_full = init_count * getattr(corpse, "max_hp", 1)
        restored = min(missing_full, hp_restore)
        value = restored * single / max(1, getattr(corpse, "max_hp", 1))
        if value > best_value:
            best_value = value
            best_target = corpse
    return (best_value, best_target,
            best_target.pos if hasattr(best_target, "pos") else None, None)


def _score_dragon_slayer(self, spell: Spell, hero, friendly: list,
                          enemies: list) -> Tuple[float, Optional[Unit],
                                                   Optional[Tuple[int, int]],
                                                   Optional[Tuple[int, int]]]:
    """``spellDragonSlayerValue`` — Bloodlust analogue scaled to dragons."""
    if not any(getattr(e, "has_tag", lambda _t: False)("dragon")
               and getattr(e, "is_alive", True) for e in enemies):
        return (0.0, None, None, None)
    dragon_str = sum(float(getattr(e, "strength", 0.0)) for e in enemies
                     if e.has_tag("dragon") and e.is_alive)
    enemy_str = sum(float(getattr(e, "strength", 0.0))
                    for e in enemies if e.is_alive)
    if enemy_str <= 0 or dragon_str <= 0:
        return (0.0, None, None, None)
    # fheroes2: ratio = BLOODLUST * extraValue/blBonus * dragons/enemy.
    bonus = max(1, int(getattr(spell, "attack_delta", 0) or 0))
    # Bloodlust bonus is 3 from engine spells.py.
    bloodlust_bonus = 3
    ratio = BLOOD_LUST_RATIO * bonus / bloodlust_bonus * dragon_str / enemy_str
    best_value = 0.0
    best_target: Optional[Unit] = None
    for friend in friendly:
        if _is_spellcast_useless_for_unit(friend, enemies, spell):
            continue
        duration = _spell_duration_multiplier(friend,
                                              spell_power=_get_spell_power(hero))
        value = float(getattr(friend, "strength", 0.0)) * ratio * duration
        if value > best_value:
            best_value = value
            best_target = friend
    return (best_value, best_target,
            best_target.pos if best_target else None, None)


def _score_teleport(self, battle: BattleState, spell: Spell, hero,
                     unit: Unit, enemies: list
                     ) -> Tuple[float, Optional[Unit], Optional[Tuple[int, int]],
                                Optional[Tuple[int, int]]]:
    """``spellTeleportValue`` — for melee units unable to reach anything.

    Mirrors C++ ai_battle_spell.cpp:850-922. The C++ path fires off
    ``getMeleeBestOutcome`` with a temporary teleport ability granted
    on a *mutable* friendly unit (Battle::Unit*, not the const
    reference), records the resulting attack value, then immediately
    strips the ability.

    Five early-return guards before the heavy lifting (C++ lines
    855-888) — in C++ order:

      1. **Hypnotized** (``SP_HYPNOTIZE``): the unit's turn is
         scripted by the enemy; we can't queue a Teleport for it.
      2. **Defensive tactics**: fheroes2 leaves Teleport-for-defense
         as a TODO; we follow suit.
      3. **Useless on this unit** (``isSpellcastUselessForUnit``):
         covers Immovable / etc. — the AI never casts Teleport on a
         unit that can't move anyway.
      4. **Flying**: flyers can already reach any cell; Teleport is
         redundant.
      5. **Archers**: fheroes2 leaves Teleport-for-shooters as a TODO;
         we follow suit.

    Then ``getMeleeBestOutcome`` is run *as-is* — if the current
    monster can already engage something (``currentDamage > 0.1``)
    Teleport buys nothing. Otherwise we synthesise the teleport-
    granted outcome: re-run the melee search with the temporary
    ability, then un-set it. If the best-case attack is still below
    0.1, no enemy is reachable even with Teleport.

    Engine simplification: we cannot literally mutate the unit, so
    we instead trust the C++ heuristic — if the unit already reaches
    a target, skip; otherwise score ``unit.strength * BLOODLUST_RATIO``
    as the spell's worth (this matches the C++ ``currentUnit.
    GetStrength() * bloodLustRatio`` return value).
    """
    # Guard 1: Hypnotize overrides the unit's turn.
    if getattr(unit, "is_hypnotized", False):
        return (0.0, None, None, None)

    # Guard 2: defensive stance — TODO in C++.
    if getattr(self, "_defensive_tactics", False):
        return (0.0, None, None, None)

    # Guard 3: immovable / unit-mismatch uselessness check.
    if _is_spellcast_useless_for_unit(unit, enemies, spell):
        return (0.0, None, None, None)

    # Guard 4: flying units can move anywhere.
    if getattr(unit, "is_flying", False):
        return (0.0, None, None, None)

    # Guard 5: archers — TODO in C++.
    if getattr(unit, "is_archer", False):
        return (0.0, None, None, None)

    # Now check whether the unit already reaches a target without help.
    from ai_core.classic_ai.melee import _get_melee_best_outcome
    outcome = _get_melee_best_outcome(self, battle, unit, enemies)
    if outcome is None or outcome.target is None:
        return (0.0, None, None, None)
    if float(getattr(outcome, "attack_value", 0.0)) > 0.1:
        # Already in striking range; Teleport has nothing to add.
        return (0.0, None, None, None)

    # Engine can't grant temporary teleport; we approximate the C++
    # result by skipping the second ``getMeleeBestOutcome`` (which
    # would have run with TELEPORT_ABILITY) — the score the C++ code
    # ultimately returns is ``unit.strength * BLOOD_LUST_RATIO``.
    value = float(getattr(unit, "strength", 0.0)) * BLOOD_LUST_RATIO
    if value < 0.1:
        return (0.0, None, None, None)
    dest = (getattr(outcome, "from_index", None)
            or getattr(outcome.target, "pos", None))
    return (value, outcome.target, dest, None)


def _score_earthquake(self, spell: Spell, friendly: list
                      ) -> Tuple[float, Optional[Unit], Optional[Tuple[int, int]],
                                 Optional[Tuple[int, int]]]:
    """``spellEarthquakeValue`` — siege breaker (ai_battle_spell.cpp:924-973).

    C++ enumerates the spell's legal targets via
    ``Battle::Arena::getEarthQuakeSpellTargets()`` (the canonical wall +
    tower set) and excludes the cosmetic bridge towers
    (``TOP_BRIDGE_TOWER``, ``BOTTOM_BRIDGE_TOWER``). The damage band
    is read from ``Battle::Arena::getEarthquakeDamageRange(hero)``;
    the average is ``(max - min) / 2`` — it scales with hero level,
    not a hard-coded constant.

    The engine exposes a castle proxy with per-structure condition
    flags; we map that into the same iteration pattern.
    """
    if not getattr(self, "_attacking_castle", False):
        return (0.0, None, None, None)
    melee_count = sum(1 for u in friendly
                      if not getattr(u, "is_flying", False)
                      and not getattr(u, "is_archer", False))
    if melee_count == 0:
        return (0.0, None, None, None)
    melee_str = sum(float(getattr(u, "strength", 0.0))
                    for u in friendly
                    if not getattr(u, "is_flying", False)
                    and not getattr(u, "is_archer", False))

    # The C++ loops over ``Battle::Arena::getEarthQuakeSpellTargets``
    # (the canonical list of all 6 wall+tower positions) and skips
    # the cosmetic bridge towers. The engine doesn't have that enum
    # yet — the castle proxy exposes ``walls`` (4 entries) and
    # ``towers`` (variable length, may include bridge towers); we
    # therefore ask the proxy for the *valid* targets via
    # ``get_earthquake_targets()`` when available, falling back to
    # summing over walls+towers minus any explicitly-marked bridge
    # towers.
    castle = getattr(self, "_castle_proxy", None)
    if castle is None:
        return (0.0, None, None, None)

    targets = getattr(castle, "get_earthquake_targets", None)
    if callable(targets):
        targets = list(targets())
    else:
        # Fallback: build the canonical list minus any bridge towers.
        walls = [(name, hp) for name, hp
                 in getattr(castle, "walls", {}).items()
                 if name not in ("TOP_BRIDGE_TOWER", "BOTTOM_BRIDGE_TOWER")]
        towers = [t for t in getattr(castle, "towers", [])
                  if getattr(t, "name", "") not in
                  ("TOP_BRIDGE_TOWER", "BOTTOM_BRIDGE_TOWER")]
        targets = [("wall", name, hp) for name, hp in walls] \
            + [("tower", t, getattr(t, "hp", 0)) for t in towers]

    if not targets:
        return (0.0, None, None, None)

    total = len(targets)
    intact = sum(1 for t in targets
                 if (t[2] if isinstance(t, tuple) and len(t) >= 3
                     else getattr(t, "hp", 0)) > 0)
    target_ratio = intact / total if total > 0 else 0.0

    # Average damage = (max - min) / 2 from the C++ damage band. The
    # engine doesn't expose ``getEarthquakeDamageRange`` directly; the
    # hero spell-power-based range falls back to the constants below.
    min_dmg, max_dmg = _get_earthquake_damage_range(self, spell)
    avg_damage = (max_dmg - min_dmg) / 2.0

    enemy_shooter_ratio = (self._enemy_ranged_units_only
                           / max(self._enemy_army_strength, 1.0))
    melee_ratio = melee_str / max(self._my_army_strength, 1.0)
    value = (melee_count * melee_str * melee_ratio * target_ratio
             * avg_damage * enemy_shooter_ratio * 0.2)
    return (value, None, None, None)


def _get_earthquake_damage_range(self, spell: Spell):
    """Port of C++ ``Battle::Arena::getEarthquakeDamageRange(hero)``.

    Earthquake damage scales with hero level (per fheroes2
    battle_arena.cpp). Engine doesn't model the level lookup yet, so
    we fall back to the canonical level-1..5 band:
      - hero level 1..3 → 1..2
      - hero level 4..6 → 2..3
      - hero level 7+   → 3..4
    The C++ uses the hero's effective spell-power to interpolate; we
    read ``hero_power`` from the planner's commander when available.
    """
    hero = getattr(self, "_commander", None)
    power = _get_spell_power(hero) if hero else 1
    if power <= 3:
        return (1, 2)
    if power <= 6:
        return (2, 3)
    return (3, 4)


def _score_effect_dispatch(self, spell: Spell, hero, targets: list,
                           enemies: list, *, is_mass: bool
                           ) -> Tuple[float, Optional[Unit],
                                      Optional[Tuple[int, int]],
                                      Optional[Tuple[int, int]]]:
    """``spellEffectValue`` (multi-target wrapper) — best unit pick.

    Uses ``_SpellcastOutcome.update_outcome`` so a mass spell sums its
    per-target worth (C++ ``isMassSpell`` branch) instead of taking the
    single best — the mass variants would otherwise be scored as if they
    only hit one stack. The tracked ``best_target`` is a Python-side
    convenience for ``CastAction``; the engine ignores it for mass casts
    exactly as C++ leaves ``outcome.cell`` at ``-1``.
    """
    is_single_target_left = (len(targets) == 1)
    outcome = _SpellcastOutcome()
    best_target: Optional[Unit] = None
    best_unit_value = 0.0
    for unit in targets:
        if not getattr(unit, "is_alive", True):
            continue
        value = _spell_effect_value_single(
            self, None, spell, unit, enemies,
            target_is_last=is_single_target_left, for_dispel=False)
        # Apply spellPointValue scaling (cost discount) — engine has no
        # mass / single distinction, mirror C++ unconditionally except
        # for Resurrect.
        value = _spell_damage_score(value, spell)
        if value > best_unit_value:
            best_unit_value = value
            best_target = unit
        outcome.update_outcome(value, unit.pos, is_mass_effect=is_mass)
    if best_target is None:
        return (0.0, None, None, None)
    return (outcome.value, best_target, outcome.cell, outcome.destination_cell)


# ── outward-facing helpers (preserved for ClassicAI dispatch) ───────

def _spell_damage_score(dmg: float, spell) -> float:
    """Backward-compatible static helper. dmg here is a *value*; divide
    by sqrt(cost/3) to produce the spellPointValue used for ranking."""
    if dmg is None or dmg <= 0:
        return 0.0
    return float(dmg) / max(1.0, math.sqrt(spell.cost / 3.0))


def _score_utility_spell(self, battle: BattleState, spell: Spell, team: int
                          ) -> Tuple[float, Optional[Unit],
                                     Optional[Tuple[int, int]],
                                     Optional[Tuple[int, int]]]:
    """Fallback path for utility spells invoked directly from the planner
    (kept for callers that bypass the dispatcher)."""
    return _score_spell(self, battle, self._commander, spell,
                        type("U", (), {"team": team})())


def maybe_cast_spell(self, battle: BattleState, unit: Unit
                     ) -> Optional[Tuple[CastAction, str]]:
    """Port of ``BattlePlanner::selectBestSpell(retreating=false)``.

    Mirrors the C++ dispatcher (ai_battle_spell.cpp:71-156):
      1. Guard the hero (``_cast_this_round`` / no spells).
      2. For each combat spell the hero knows, skip spells the arena
         disables (``isDisableCastSpell``) or that the hero can't pay
         the SP for. In retreating mode only damage spells are even
         considered.
      3. Evaluate via the planner-bound dispatcher; rank by
         ``spellPointValue`` (cost-discounted value).
      4. Threshold gate — bypassed for ``isResurrect`` spells (and when
         retreating).
    """
    hero = battle.heroes.get(unit.team)
    if hero is None or hero._cast_this_round:
        return None
    spells = self.spellbook if self.spellbook is not None else hero.spells
    if not spells:
        return None

    threshold = self._spell_value_threshold(battle, unit, hero)
    best_score = 0.0
    best_action: Optional[CastAction] = None
    best_desc = ""

    for name in spells:
        if name not in SPELLS:
            continue
        spell = SPELLS[name]
        if not _is_spell_combat_candidate(spell, battle):
            continue
        if not hero.can_cast(spell):
            continue
        score, target, cell, destination = _bound_score_spell(
            self, battle, hero, spell, unit)
        # C++ (ai_battle_spell.cpp:104): ignoreThreshold covers both the
        # retreating case and ``spell.isResurrect()``. Use the kind (not
        # the name) so Animate Dead / Resurrect True bypass the
        # threshold too.
        ignore_threshold = (spell.kind == RESURRECT
                            or score > threshold + 1e-9)
        if not ignore_threshold and score <= threshold:
            continue
        if target is None and score <= 0.0 and not ignore_threshold:
            continue
        if score > best_score:
            best_score = score
            best_action = CastAction(
                team=unit.team, spell=spell, target=target,
                cell=cell, destination=destination)
            best_desc = (f"{hero.name} casts {spell.name} "
                         f"(score={score:.1f})")
    if best_action is None:
        return None
    return (best_action, best_desc)


def _is_spell_combat_candidate(spell: Spell, battle: BattleState) -> bool:
    """Mirror C++ ``arena.isDisableCastSpell`` + combat-spell filter.

    fheroes2 refuses to evaluate non-combat spells (``isCombat()``) and
    any spell the arena forbids during the current turn (``for example
    - spells disabled by a glyph, anti-magic on the hero, etc.). The
    engine's Spell dataclass only knows ``kind``, so we treat every
    spell with a non-UTILITY kind except UTILITY+Earthquake (which the
    engine treats as a combat spell) as a combat spell; non-combat
    exclusion lives on the battle-state ``disabled_spells`` set when
    the engine grows one.
    """
    combat_kinds = {DAMAGE, AOE, DISPEL, CURE, BUFF, DEBUFF, CONTROL,
                    HYPNOTIZE, BERSERKER, RESURRECT, SUMMON}
    if spell.kind not in combat_kinds:
        return False
    disabled = getattr(battle, "disabled_spells", None)
    if disabled and spell.name in disabled:
        return False
    return True


def _bound_score_spell(self, battle: BattleState, hero, spell: Spell,
                       unit: Unit):
    """Bind `self` so inner helpers see the live planner state."""
    from types import SimpleNamespace
    ctx = SimpleNamespace()
    ctx._commander = hero
    ctx._my_color = unit.team
    ctx._my_army_strength = self._my_army_strength
    ctx._enemy_army_strength = self._enemy_army_strength
    ctx._my_army_average_speed = self._my_army_average_speed
    ctx._enemy_average_speed = self._enemy_average_speed
    ctx._enemy_spell_strength = self._enemy_spell_strength
    ctx._enemy_ranged_units_only = self._enemy_ranged_units_only
    ctx._enemy_shooters_strength = self._enemy_shooters_strength
    ctx._defensive_tactics = self._defensive_tactics
    ctx._cautious_offensive = self._cautious_offensive
    ctx._attacking_castle = self._attacking_castle
    return _score_spell(ctx, battle, hero, spell, unit)


# ── threshold and farewell helpers ──────────────────────────────────

def _spell_value_threshold(self, battle: BattleState, unit: Unit,
                           hero) -> float:
    """``selectBestSpell`` value threshold — fheroes2 ai_battle_spell.cpp:91."""
    my_strength = self._my_army_strength
    enemy_strength = self._enemy_army_strength
    threshold = (my_strength * my_strength
                 / max(enemy_strength, 1.0)
                 * SPELL_VALUE_RATIO)
    if (enemy_strength
            and self._enemy_shooters_strength / enemy_strength > 0.5):
        threshold *= 0.5
    sp = int(getattr(hero, "spell_points", 0))
    max_sp = int(getattr(hero, "max_spell_points", max(sp, 1)))
    if sp * 2 < max_sp:
        threshold *= 2.0
    return threshold


def _best_damage_target(self, battle: BattleState, spell: Spell, team: int
                        ) -> Tuple[Optional[Unit], float]:
    """Legacy single-target entry used by external callers (and the
    farewell path). Implements only the caller's primary-use: find the
    enemy unit with the highest ``spell * power * count``.
    """
    hero = battle.heroes.get(team)
    power = getattr(hero, "power", 1) if hero else 1
    best, best_dmg = None, 0
    base = int(getattr(spell, "base_damage", 0)) * power
    for tgt in battle.alive(1 - team):
        if any(tgt.has_tag(t) for t in getattr(spell, "exclude_tags", ())):
            continue
        if getattr(tgt, "is_immune_to_spells", False):
            continue
        dmg = base * max(1, getattr(tgt, "count", 0))
        if dmg > best_dmg:
            best_dmg, best = dmg, tgt
    return best, best_dmg


def _best_aoe_target(self, battle: BattleState, spell: Spell, team: int
                     ) -> Tuple[Optional[Unit], float]:
    """Legacy AoE entry — best cell with most cluster."""
    hero = battle.heroes.get(team)
    power = getattr(hero, "power", 1) if hero else 1
    best, best_dmg = None, 0
    base = int(getattr(spell, "base_damage", 0)) * power
    enemies = battle.alive(1 - team)
    for centre in enemies:
        cluster = sum(1 for e in enemies
                      if battle.grid.distance(centre.pos, e.pos) <= 2)
        dmg = cluster * base
        if dmg > best_dmg:
            best_dmg, best = dmg, centre
    return best, best_dmg


def _strongest_enemy(self, battle: BattleState, team: int) -> Optional[Unit]:
    enemies = battle.alive(1 - team)
    if not enemies:
        return None
    return max(enemies, key=lambda u: u.strength)


def _weakest_friend(self, battle: BattleState, team: int) -> Optional[Unit]:
    friends = battle.alive(team)
    if not friends:
        return None
    return min(friends, key=lambda u: u.strength)


def _maybe_farewell_spell(self, battle: BattleState,
                          hero, unit: Unit,
                          ignore_threshold: bool = False
                          ) -> Optional[Tuple[CastAction, str]]:
    """Cast one final damage spell right before retreating.

    fheroes2 invokes ``selectBestSpell(true)`` so damage spells are
    ranked without the spellPointValue cost discount and without
    threshold gating.
    """
    if hero is None:
        return None
    team = unit.team
    best_action: Optional[CastAction] = None
    best_desc = ""
    best_score = 0.0
    for name in hero.spells:
        if name not in SPELLS:
            continue
        sp = SPELLS[sp.name] if hasattr(sp, "name") else SPELLS[name]  # noqa
        sp = SPELLS[name]
        if not hero.can_cast(sp):
            continue
        if sp.kind not in (DAMAGE, AOE):
            continue
        tgt, dmg = _best_damage_target(self, battle, sp, team)
        if tgt is None or dmg <= 0:
            continue
        # In retreating mode we skip the cost discount.
        value = float(dmg)
        if not ignore_threshold:
            value = _spell_damage_score(value, sp)
            if value <= self._spell_value_threshold(battle, unit, hero):
                continue
        if value > best_score:
            best_score = value
            cell = tgt.pos if sp.kind == AOE else None
            best_action = CastAction(
                team=team, spell=sp, target=tgt, cell=cell)
            best_desc = f"{hero.name} casts farewell {sp.name}"
    if best_action is None:
        return None
    return (best_action, best_desc)
