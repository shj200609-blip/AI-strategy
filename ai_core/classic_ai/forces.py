"""Force classification and battle-outcome helpers.

These ports mirror the fheroes2 C++ implementation in
``fheroes2/src/fheroes2/ai/ai_battle.cpp`` and the surrounding battle-arena
helpers (battle_arena.cpp, battle_troop.cpp, heroes_base.cpp, kingdom.cpp).

The semantics here intentionally match the originals so that the AI port
behaves identically to the reference C++ AI.
"""

from __future__ import annotations

from typing import List

from engine.battle_state import BattleState
from engine.spells import spell_damage
from engine.unit import Unit


def _is_hypnotized(unit: Unit) -> bool:
    """fheroes2 ``Battle::Unit::Modes( Battle::SP_HYPNOTIZE )``.

    The engine stores Hypnotize on an ``Effect`` whose ``is_hypnotize`` flag
    is set; ``Unit.is_hypnotized`` surfaces that property (unit.py:278-280).
    """
    return bool(getattr(unit, "is_hypnotized", False))


def _effective_team(unit: Unit) -> int:
    """Hypnotize inverts team allegiance — fheroes2 ``Battle::Unit::GetCurrentColor``.

    A hypnotized unit's *current* allegiance flips to the opposing side
    (battle_troop.cpp handles the SP_HYPNOTIZE mode flip), while the
    *original* allegiance (``unit.team``) is preserved. Combat-facing
    queries (``arena.getForce/getEnemyForce``) are keyed off the
    CurrentColor, so matching effective teams = matching forces.
    """
    return 1 - unit.team if _is_hypnotized(unit) else unit.team


def _effective_friends(battle: BattleState, unit: Unit) -> List[Unit]:
    """fheroes2 ``Arena::getForce(_myColor)`` with the Hypnotize filter applied.

    Matches every alive unit whose ``CurrentColor`` equals ``unit``'s.
    """
    team = _effective_team(unit)
    return [u for u in battle.alive() if _effective_team(u) == team]


def _effective_enemies(battle: BattleState, unit: Unit) -> List[Unit]:
    """fheroes2 ``Arena::getEnemyForce(_myColor)`` with the Hypnotize filter.

    A hypnotized former teammate counts as an enemy to ``unit`` because
    CurrentColor flipped — this is exactly what fheroes2 does when it
    iterates enemies in ``evaluatePotentialAttackPositions`` / melee paths.
    """
    team = _effective_team(unit)
    return [u for u in battle.alive() if _effective_team(u) != team]


def _enemy_for(unit: Unit, target: Unit) -> bool:
    """Two units are enemies iff their effective teams differ.

    Mirror of the same check in ``optimalAttackValue`` /
    ``isAllAdjacentCellsAttack`` — C++ compares ``GetCurrentColor()`` on
    attacker vs target.
    """
    return _effective_team(unit) != _effective_team(target)


def _survivor_strength(battle: BattleState, team: int) -> float:
    """Sum of fheroes2 ``Troop::GetStrength()`` across alive units on ``team``.

    fheroes2 ``Battle::Unit::GetStrength()`` is ``monsterStrength * count``
    — there is *no* HP-proportional weighting (battle_troop.cpp). The
    previous HP-weighted form over-counted healthy stacks and under-counted
    wounded-but-still-deadly ones, drifting from the reference AI.

    Args:
        team: fheroes2 ``Battle::Force`` color. Filtering by ``team`` uses
            the *original* allegiance so a unit whose CurrentColor was
            flipped by Hypnotize still contributes strength to its original
            side (matches ``ArmyColor`` semantics in ai_battle.cpp).
    """
    return sum(u.strength for u in battle.alive(team))


def _stalemate_reached(battle: BattleState, team: int) -> bool:
    """fheroes2 ``BattlePlanner::isLimitOfTurnsExceeded``.

    The death-free-rounds stalemate rule only fires for the *attacker*
    (ai_battle.cpp:633-637 — ``if (currentColor != arena.getAttackingArmyColor())
    return false;``). The defender never auto-retreats from a stalemate;
    only the attacker retreats after ``MAX_TURNS_WITHOUT_DEATHS`` rounds
    with no deaths on either side.

    Args:
        team: The team being checked. Non-attackers always return False.
    """
    if team != battle.attacker_team:
        return False
    return bool(battle.is_stalemate())


def _total_primary_skill_level(hero) -> int:
    """Surrogate for fheroes2 ``HeroBase::getTotalPrimarySkillLevel``.

    Engine heroes only expose attack/defense/power/knowledge. Summing them
    reproduces Heroes of Might & Magic II's primary-skill gate at 10 that
    the AI's retreat lambda uses.
    """
    if hero is None:
        return 0
    return (int(getattr(hero, "attack", 0))
            + int(getattr(hero, "defense", 0))
            + int(getattr(hero, "power", 0))
            + int(getattr(hero, "knowledge", 0)))


def _can_retreat_opponent(battle: BattleState, team: int) -> bool:
    """fheroes2 ``Battle::Arena::CanRetreatOpponent`` (battle_arena.cpp:906-910).

    The C++ predicate is::

        hero = getCommander( color );
        return hero && hero->isHeroes()
               && ( color == _attackingArmy->GetColor() || hero->inCastle() == nullptr );

    ``BattleState`` implements this directly, so the method is preferred when
    present.  The fallback re-evaluates the same predicate from the hero
    roster rather than assuming ``True`` — the old ``lambda _t: True`` shim
    granted retreat to captain-led armies and to defender heroes garrisoning
    their own castle, both of which the original refuses.
    """
    method = getattr(battle, "can_retreat_opponent", None)
    if callable(method):
        return bool(method(team))

    hero = (getattr(battle, "heroes", None) or {}).get(team)
    # ``hero && hero->isHeroes()`` — captains and native AI cannot retreat.
    if hero is None or not getattr(hero, "is_hero", True):
        return False
    # ``color == _attackingArmy->GetColor()`` — the attacker is never pinned.
    if team == getattr(battle, "attacker_team", 0):
        return True
    # ``hero->inCastle() == nullptr`` — a defender inside the castle stays.
    return not bool(getattr(hero, "in_castle", False))


def _can_surrender_opponent(battle: BattleState, team: int) -> bool:
    """fheroes2 ``Battle::Arena::CanSurrenderOpponent`` (battle_arena.cpp:899-904).

    The C++ predicate is::

        hero = getCommander( color );
        enemyHero = getEnemyCommander( color );
        return hero && hero->isHeroes()
               && enemyHero && ( enemyHero->isHeroes() || enemyHero->isCaptain() );

    This is only the *arena* half of the C++ ``isAbleToSurrender`` lambda; the
    kingdom's ``AllowPayment`` gate lives in :func:`_can_surrender`.  As with
    :func:`_can_retreat_opponent`, the fallback evaluates the predicate rather
    than defaulting to ``True``.
    """
    method = getattr(battle, "can_surrender_opponent", None)
    if callable(method):
        return bool(method(team))

    heroes = getattr(battle, "heroes", None) or {}
    hero = heroes.get(team)
    if hero is None or not getattr(hero, "is_hero", True):
        return False
    # ``getEnemyCommander( color )`` — you can only surrender to a commander.
    attacker_team = getattr(battle, "attacker_team", 0)
    enemy_team = (1 - attacker_team) if team == attacker_team else attacker_team
    enemy_hero = heroes.get(enemy_team)
    if enemy_hero is None:
        return False
    return bool(getattr(enemy_hero, "is_hero", True)
                or getattr(enemy_hero, "is_captain", False))


def _can_surrender(battle: BattleState, team: int) -> bool:
    """fheroes2 ``Battle::Arena::CanSurrenderOpponent`` + payment gate.

    A team may surrender when ALL of the following hold
    (battle_arena.cpp:899-904):

      * the battle is still ongoing (no prior retreat / not over);
      * the team's commander is a hero (not a captain / native AI);
      * the opposing team is also commanded by a hero or captain — you
        can only surrender to a sentient opponent;
      * the kingdom can pay the surrender cost —
        ``kingdom.AllowPayment({Resource::GOLD, GetSurrenderCost()})``.

    The ``GetSurrenderCost()`` in fheroes2 depends on the army strength of
    both sides (battle.cpp); the engine exposes it via
    ``hero.surrender_cost`` (Battle::Force::GetSurrenderCost).

    The C++ rule is symmetric: the *defender* must also have enough gold.
    The previous Python only checked the attacker's wallet and bypassed
    the commander-type / enemy-commander checks entirely, so a hero with
    zero gold could be "permitted" to surrender and a captain-led army
    could be surrendered into.
    """
    if battle.is_over():
        return False
    if not battle.alive(team):
        return False
    hero = battle.heroes.get(team)
    if hero is None or not getattr(hero, "is_hero", True):
        return False
    enemy = battle.defender_team() if team == battle.attacker_team else battle.attacker_team
    enemy_hero = battle.heroes.get(enemy)
    if enemy_hero is None:
        return False
    if not (getattr(enemy_hero, "is_hero", True)
            or getattr(enemy_hero, "is_captain", False)):
        return False
    # fheroes2 ``kingdom.AllowPayment({ GOLD, GetSurrenderCost() })`` —
    # the kingdom's gold must cover the full surrender price. Cost lives
    # on the engine hero shim as ``surrender_cost`` (default 0). A cost
    # of 0 with no gold is still rejected because fheroes2's
    # ``GetSurrenderCost`` always evaluates to at least 50 gold for any
    # non-empty army; if the engine reports 0 we treat that as "the hero
    # explicitly cannot afford it" and require gold > 0.
    cost = int(getattr(hero, "surrender_cost", 0))
    gold = int(getattr(hero, "gold", 0))
    if cost > 0:
        return gold >= cost
    return gold > 0


def _is_possible_to_rehire(battle: BattleState, hero) -> bool:
    """fheroes2 ``isPossibleToReHire`` from the Outcome lambda
    (ai_battle.cpp:750-776).

    Rehire is possible when:
      * the kingdom has more than one hero → always possible (another
        hero can capture an enemy castle); OR
      * the kingdom owns at least one castle; AND
      * the hero is *not* defending the last (only) castle in the kingdom
        — surrendering him would leave the kingdom castle-less and he
        couldn't be hired again.

    Implementation notes: the hero shim doesn't carry its own
    ``team`` attribute, so we resolve it from the battle's hero roster
    (``battle.heroes`` keyed by team) before asking the kingdom-level
    accessors. With no resolvable team we fall back to the per-hero
    proxy attributes, which is the previous behaviour — but unlike the
    previous code, the *defends-the-last-castle* check now also inspects
    ``hero.in_castle`` (the C++ ``HeroBase::inCastle()`` predicate) so
    the lone defending hero without an explicit flag is correctly
    classified as not-rehirable. That's the audit's "守城英雄被错误判
    为可重新雇佣" bug.
    """
    if hero is None:
        return False

    # Resolve the hero's team by reverse-lookup in the battle roster.
    # ``battle.heroes`` is a {0: heroA, 1: heroB} dict; an identity match
    # against either slot tells us which team this hero belongs to.
    team = None
    try:
        heroes_map = getattr(battle, "heroes", None) or {}
        for t, h in heroes_map.items():
            if h is hero:
                team = t
                break
    except Exception:
        team = None

    if team is not None:

        def hero_count() -> int:
            try:
                return int(battle.kingdom_hero_count(team))
            except Exception:
                return int(getattr(hero, "kingdom_hero_count", 1))

        def castle_count() -> int:
            try:
                return int(battle.kingdom_castle_count(team))
            except Exception:
                return int(getattr(hero, "kingdom_castle_count", 1))

        def defending_last_castle() -> bool:
            # Mirror C++ ``actualHero->inCastle() && castles.size() == 1``
            # (ai_battle.cpp:767-772). The C++ predicate has two parts:
            #   (1) the hero is currently inside the (only) castle, AND
            #   (2) the kingdom owns exactly one castle.
            # If either is false, the hero can be rehired (some other
            # hero or castle can carry the kingdom).
            #
            # Python port: combine the explicit ``defends_last_castle``
            # flag with the ``in_castle + castle_count==1`` fallback.
            # The flag is treated as authoritative when explicitly set
            # via ``Hero.__init__(defends_last_castle=...)`` for tests
            # that need to override the in_castle default; the fallback
            # covers engine-managed defender heroes where the field
            # exists but defaults to False (the audit's "lone defender
            # of the only castle was incorrectly flagged rehirable").
            explicit = getattr(hero, "defends_last_castle", None)
            if explicit is True:
                return True
            try:
                if bool(battle.defends_last_castle(team)):
                    return True
            except Exception:
                pass
            if bool(getattr(hero, "in_castle", False)):
                try:
                    if int(battle.kingdom_castle_count(team)) <= 1:
                        return True
                except Exception:
                    return True
            return False

        if hero_count() > 1:
            return True
        if castle_count() == 0:
            return False
        if defending_last_castle():
            return False
        return True

    # Team unknown — fall back to per-hero proxies only. This branch
    # preserves the prior behaviour for callers that haven't wired the
    # hero into BattleState.heroes; the rehire bug only matters when
    # there *is* a battle, so this path is purely a safety net.
    hero_count = int(getattr(hero, "kingdom_hero_count", 1))
    if hero_count > 1:
        return True
    castle_count = int(getattr(hero, "kingdom_castle_count", 1))
    if castle_count == 0:
        return False
    if bool(getattr(hero, "defends_last_castle", False)) \
            or bool(getattr(hero, "in_castle", False)):
        return False
    return True


def _commander_max_spell_damage_value(commander) -> float:
    """fheroes2 ``commanderMaximumSpellDamageValue`` (ai_battle.cpp:489-506).

    Walks every combat spell the commander knows, filters out non-damage
    / non-combat spells and any spell whose cost exceeds the commander's
    remaining spell points, then returns the maximum ``getSpellDamage``
    for the survivors. The engine's hero exposes ``spellbook`` (a list of
    ``Spell`` objects) and ``spell_points`` / ``power`` for these
    computations.

    The previous Python implementation iterated ``commander.spells`` (a
    list of spell *names*) and tried to read attributes like
    ``spell.is_combat`` / ``spell.spell_points`` off strings — every
    attribute lookup fell through to the ``getattr`` default and the
    function always returned 0, silently neutering the AI's
    ``_my_shooters_strength / _enemy_shooters_strength`` augmentation.
    """
    if commander is None:
        return 0.0
    sp = getattr(commander, "spell_points", 0)
    power = getattr(commander, "power", 0)
    best = 0.0
    # Prefer the resolved ``spellbook`` (a list of Spell objects); fall
    # back to the raw ``spells`` list if the hero shim doesn't expose
    # the property — both Spell objects and ``spell_damage`` lookups
    # would otherwise short-circuit to 0.
    spellbook = getattr(commander, "spellbook", None)
    if spellbook is None:
        spells_attr = getattr(commander, "spells", None) or []
        from engine.spells import SPELLS
        spellbook = [SPELLS[s] for s in spells_attr if s in SPELLS]
    for spell in spellbook:
        # fheroes2 ``spell.isCombat() && spell.isDamage()`` — combat-only,
        # damaging only. Other kinds (BUFF / DEBUFF / CONTROL / CURE /
        # DISPEL / UTILITY / RESURRECT / SUMMON / HYPNOTIZE / BERSERKER)
        # never produce spell-damage numbers.
        kind = getattr(spell, "kind", "")
        if kind not in ("damage", "aoe"):
            continue
        # fheroes2 ``commander.GetSpellPoints() < spell.spellPoints()`` —
        # the spell is unaffordable, skip it.
        cost = int(getattr(spell, "cost", 0))
        if sp < cost:
            continue
        # fheroes2 ``fheroes2::getSpellDamage(spell, comm.GetPower(), &comm)``
        # — the engine's analogue is ``spell_damage(spell, power)``.
        damage = float(spell_damage(spell, power))
        if damage > best:
            best = damage
    return best
