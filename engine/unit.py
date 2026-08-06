"""Unit class — a stack of identical creatures on the battlefield."""

import math
from typing import Optional

import config

# fheroes2 Speed::AVERAGE — the pivot for the base-strength speed remap.
SPEED_AVERAGE = 4


class Unit:
    def __init__(self, name: str, team: int, col: int, row: int, **kwargs):
        self.name = name
        self.team = team
        self.col = col
        self.row = row
        self.attack = kwargs["attack"]
        self.defense = kwargs["defense"]
        self.max_hp = kwargs["hp"]
        self.base_speed = kwargs["speed"]
        # fheroes2 damage is a per-creature [min, max] range rolled in combat.
        # Backward-compat: a single ``damage`` means min == max (no spread).
        if "damage_min" in kwargs and "damage_max" in kwargs:
            self.damage_min = kwargs["damage_min"]
            self.damage_max = kwargs["damage_max"]
        else:
            self.damage_min = self.damage_max = kwargs["damage"]
        self.is_archer = kwargs["is_archer"]
        self.is_flying = kwargs["is_flying"]
        # Wide units occupy two horizontally-adjacent cells (head + tail).
        # fheroes2 keeps the current reflection in Battle::Position.  Team
        # facing is only the initial orientation; a free U-turn can change it.
        self.is_wide = kwargs.get("is_wide", False)
        self._reflected = bool(kwargs.get("reflected", team == 1))
        self.abilities = set(kwargs.get("abilities", ()))
        self.ability_params = dict(kwargs.get("ability_params", {}))
        # fheroes2 MonsterWeaknessType (battle_troop.cpp:567-577): a separate
        # axis from abilities. We model it as its own set.
        self.weaknesses = set(kwargs.get("weaknesses", ()))
        self.tags = set(kwargs.get("tags", ()))
        self.symbol = kwargs.get("symbol", name[0])

        self.count = kwargs["count"]
        self._total_hp = self.count * self.max_hp
        self._max_total_hp = self._total_hp
        self._is_alive = True
        self.retaliated = False  # can retaliate once per round
        # fheroes2 Blind-style effect can be applied via spell (no retaliation)
        # or via monster ability (still retaliates, only damages reduced).
        # Default True so ability-blind units can still counter.
        self.blind_retaliation = True
        # Resurrect records the stack's full count for revival budgeting.
        self.original_count = self.count
        self._acted = False     # has acted this round (fheroes2 TR_MOVED)

        # Active timed spell effects (Haste / Slow / Bless / Curse …).
        self.effects = []

        # ── fheroes2 archer ammo (battle_troop.cpp _shotsLeft) ────
        # Archers fire one shot per ranged attack; when ammo runs out, an
        # archer unit must melee instead (and take the standard penalty).
        # Mirror Images have no ammo (shots=0).
        self.max_shots: int = int(kwargs.get("shots", 0))
        self.shots_left: int = self.max_shots if self.max_shots > 0 else 0

        # Mirror Image (CAP_MIRRORIMAGE): a phantom duplicate. Any damage
        # (including a missed attack's hit) instantly destroys the image.
        self.is_mirror: bool = bool(kwargs.get("is_mirror", False))
        # The unit we are an image of — used for damage mirroring / AI.
        self.mirror_of: Optional["Unit"] = kwargs.get("mirror_of", None)

        # Per-type base strength (fheroes2 getMonsterBaseStrength), computed once.
        self._base_strength = self._compute_base_strength()

    @staticmethod
    def from_type(type_name: str, team: int, col: int, row: int,
                  count: int = None, **overrides) -> "Unit":
        t = dict(config.UNIT_TYPES[type_name])
        if count is not None:
            t["count"] = count
        # Merge tags from UNIT_TAGS (M7b spell targeting).
        t["tags"] = config.UNIT_TAGS.get(type_name, [])
        t.update(overrides)
        return Unit(type_name, team, col, row, **t)

    @classmethod
    def mirror_image(cls, source: "Unit", team: int) -> "Unit":
        """Create a Mirror Image of *source* — fheroes2 CAP_MIRRORIMAGE.

        A single-creature phantom with the same stats; any hit kills it.
        """
        data = dict(
            attack=source.attack,
            defense=source.defense,
            hp=source.max_hp,            # 1 creature's worth of HP
            speed=source.base_speed,
            damage_min=source.damage_min,
            damage_max=source.damage_max,
            count=1,
            is_archer=source.is_archer,
            is_flying=source.is_flying,
            is_wide=False,
            abilities=(),
            tags=set(source.tags),
            symbol=source.symbol,
            shots=0,                     # mirror has no ammo
            is_mirror=True,
            mirror_of=source,
        )
        return cls(source.name, team, source.col, source.row, **data)

    def clone(self) -> "Unit":
        """Deep-copy enough state for MCTS sandboxing.

        Mirrors the public API surface (count, hp, position, effects,
        ammo, flags) without sharing mutable lists. ``_base_strength``
        is recomputed at construction time so we leave it off — MCTS only
        reads it via the property anyway.
        """
        import copy
        new = self.__class__.__new__(self.__class__)
        new.name = self.name
        new.team = self.team
        new.col = self.col
        new.row = self.row
        new.attack = self.attack
        new.defense = self.defense
        new.max_hp = self.max_hp
        new.base_speed = self.base_speed
        new.damage_min = self.damage_min
        new.damage_max = self.damage_max
        new.is_archer = self.is_archer
        new.is_flying = self.is_flying
        new.is_wide = self.is_wide
        new._reflected = self._reflected
        new.abilities = set(self.abilities)
        new.ability_params = dict(self.ability_params)
        new.weaknesses = set(self.weaknesses)
        new.tags = set(self.tags)
        new.symbol = self.symbol
        new.count = self.count
        new._total_hp = self._total_hp
        new._max_total_hp = self._max_total_hp
        new._is_alive = self._is_alive
        new.retaliated = self.retaliated
        new.blind_retaliation = self.blind_retaliation
        new.original_count = self.original_count
        new._acted = self._acted
        new.effects = [
            copy.copy(e) if hasattr(e, "__dict__") else e for e in self.effects
        ]
        new.max_shots = self.max_shots
        new.shots_left = self.shots_left
        new.is_mirror = self.is_mirror
        new.mirror_of = self.mirror_of
        new._base_strength = self._base_strength
        return new

    # ── properties ──────────────────────────────────────────

    @property
    def pos(self) -> tuple:
        """The head cell — the enemy-facing front of the unit."""
        return (self.col, self.row)

    @pos.setter
    def pos(self, value: tuple):
        self.col, self.row = value

    @property
    def tail_offset(self) -> Optional[int]:
        """Column offset of the tail in the unit's current reflection."""
        if not self.is_wide:
            return None
        return 1 if self._reflected else -1

    @property
    def is_reflected(self) -> bool:
        return self.is_wide and self._reflected

    def set_battle_position(self, head: tuple,
                            tail: Optional[tuple] = None) -> None:
        """Apply a complete pathfinder position, including reflection."""
        self.pos = head
        if self.is_wide and tail is not None:
            if tail[1] != head[1] or abs(tail[0] - head[0]) != 1:
                raise ValueError("wide unit head and tail must be horizontal neighbours")
            self._reflected = head < tail

    @property
    def tail_cell(self) -> Optional[tuple]:
        """The trailing cell for a wide unit, else None."""
        if not self.is_wide:
            return None
        return (self.col + self.tail_offset, self.row)

    def occupied_cells(self) -> set:
        """All cells this unit's body occupies — {head} or {head, tail}.

        For single-hex units this is exactly {pos}, so occupancy / collision
        logic is byte-for-byte unchanged from before wide units existed.
        """
        if self.is_wide:
            return {self.pos, self.tail_cell}
        return {self.pos}

    @property
    def hp(self) -> int:
        """HP of the top unit in the stack."""
        if self.count <= 0:
            return 0
        return self._total_hp - (self.count - 1) * self.max_hp

    @property
    def speed(self) -> int:
        """Effective speed after Haste/Slow, matching fheroes2 exactly."""
        if any(e.halves_speed for e in self.effects):
            return max(1, self.base_speed // 2)
        delta = sum(e.speed_delta for e in self.effects)
        return max(1, self.base_speed + delta)

    @property
    def damage_avg(self) -> float:
        """Average per-creature damage — used by expected_damage / strength."""
        return (self.damage_min + self.damage_max) / 2.0

    def damage_per_unit(self) -> float:
        """Per-creature base damage after Bless / Curse (fheroes2
        battle_troop.cpp:469-478 and getPotentialDamage:504-512).

          - SP_CURSE active  → damage_min
          - SP_BLESS active  → damage_max
          - else             → (damage_min + damage_max) / 2

        Curse takes precedence over Bless in the (illegal) co-active case,
        matching fheroes2. Use this everywhere the damage pipeline reads
        the unit's base damage — ``damage_avg`` stays around for callers
        that explicitly want the unconditional mean.
        """
        if self.has_effect("Curse"):
            return float(self.damage_min)
        if self.has_effect("Bless"):
            return float(self.damage_max)
        return (self.damage_min + self.damage_max) / 2.0

    @property
    def effective_attack(self) -> int:
        """Attack stat including spell bonuses (Bloodlust, Dragon Slayer)."""
        delta = sum(e.attack_delta for e in self.effects)
        return max(0, self.attack + delta)

    def effective_attack_with_hero(self, hero_attack: int = 0) -> int:
        """Attack including spell bonuses + hero primary attribute.

        fheroes2 ArmyTroop::GetAttack() = Troop::GetAttack() + hero.attack.
        Note: ``self.attack`` must NOT already include the hero bonus — the
        constructor reads it from ``UNIT_TYPES`` (the type stats), and the
        C++ code adds the hero attribute on top (army_troop.cpp:158-160).
        Bypassing this method and pre-baking the hero bonus will double-count.
        """
        return max(0, self.effective_attack + hero_attack)

    @property
    def effective_defense(self) -> int:
        """Defense stat including spell bonuses (Stone Skin, Disrupting Ray)."""
        delta = sum(e.defense_delta for e in self.effects)
        return max(0, self.defense + delta)

    def effective_defense_with_hero(self, hero_defense: int = 0) -> int:
        """Defense including spell bonuses + hero primary attribute.

        fheroes2 ArmyTroop::GetDefense() = Troop::GetDefense() + hero.defense.
        Same caveat as ``effective_attack_with_hero`` — never bake the
        hero bonus into ``self.defense`` at construction.
        """
        return max(0, self.effective_defense + hero_defense)

    @property
    def damage_factor(self) -> float:
        """Compatibility value; fheroes2 Bless/Curse do not use a multiplier."""
        return 1.0

    @property
    def incoming_ranged_factor(self) -> float:
        """Multiplier on incoming ranged damage (Shield effect)."""
        factor = 1.0
        for e in self.effects:
            factor *= e.ranged_shield
        return factor

    @property
    def skip_turn(self) -> bool:
        """True when a Blind / Paralyze / Petrify effect is active.

        Hypnotize / Berserker don't skip the unit's own turn — they redirect
        who/what they attack (handled in ``BattleState.execute``).
        """
        return any(e.skip_turn for e in self.effects)

    def can_retaliate(self) -> bool:
        """True if the unit can respond with a retaliatory strike.

        Mirrors ``Battle::Unit::isRetaliationAllowed`` in fheroes2:
          - Hypnotized units never retaliate (always).
          - Spell-Blind units never retaliate (``_blindRetaliation=False``).
          - Ability-Blind (blind_retaliation=True) still retaliates — its
            active turn is skipped separately by ``skip_turn``
            (``Battle::Unit::isImmovable``), but retaliation is allowed.
          - Paralyzed / Petrified (IS_PARALYZE_MAGIC) never retaliate.
          - Already-reacted units don't retaliate again unless they have
            ``unlimited_retaliation``.
        """
        if not self.is_alive or self.is_mirror:
            return False
        if self.is_hypnotized:
            return False
        # Paralyzed / Petrified (IS_PARALYZE_MAGIC) never retaliate.
        if (self.has_effect("Paralyze")
                or self.has_effect("Petrification")):
            return False
        # Spell-Blind forbids retaliation; ability-Blind (blind_retaliation
        # True) still retaliates.
        if self.has_effect("Blind") and not self.blind_retaliation:
            return False
        if self.retaliated and "unlimited_retaliation" not in self.abilities:
            return False
        return True

    @property
    def is_hypnotized(self) -> bool:
        """True if Hypnotize control is in effect — flips team allegiance."""
        return any(e.is_hypnotize for e in self.effects)

    @property
    def is_berserk(self) -> bool:
        """True if Berserker control is in effect — forced nearest-neighbor."""
        return any(e.is_berserk for e in self.effects)

    @property
    def effective_team(self) -> int:
        """Team the unit currently fights for.

        Hypnotize inverts allegiance (fheroes2 SP_HYPNOTIZE — CurrentColor).
        The 1-team flip is the C++ binary encoding; for non-0/1 teams the
        caller must adapt the team id before construction.
        """
        if self.team not in (0, 1):
            raise ValueError(
                f"effective_team is only defined for 0/1 teams (got {self.team})"
            )
        return 1 - self.team if self.is_hypnotized else self.team

    @property
    def is_immune_to_spells(self) -> bool:
        """True when an Anti-Magic effect is active."""
        return any(e.anti_magic for e in self.effects)

    @property
    def is_immune_to_mind(self) -> bool:
        """True when the unit is unaffected by mind-influence spells
        (Blind / Paralyze / Berserker / Hypnotize).

        fheroes2 monster_info.cpp:890-898 — Undead, Elemental, and any
        monster with MonsterAbilityType::MIND_SPELL_IMMUNITY have 100%
        resistance to mind-influence spells.
        """
        if self.has_tag("undead") or self.has_tag("elemental"):
            return True
        return self.has_ability("mind_spell_immunity")

    @property
    def can_shoot(self) -> bool:
        """True if the unit still has ranged ammo (archer)."""
        return self.is_archer and not self.is_mirror and self.shots_left > 0

    def has_tag(self, tag: str) -> bool:
        """Check if this unit has a specific tag (undead, dragon, elemental)."""
        return tag in self.tags

    def break_effects_on_damage(self) -> None:
        """Remove effects that break when the unit takes damage."""
        self.effects = [e for e in self.effects if not e.break_on_damage]

    # ── spell effects ───────────────────────────────────────

    def has_ability(self, name: str) -> bool:
        return name in self.abilities

    def has_weakness(self, name: str) -> bool:
        """True if the unit is weak to *name* (fheroes2 MonsterWeaknessType)."""
        return name in self.weaknesses

    def has_effect(self, name: str) -> bool:
        return any(e.name == name for e in self.effects)

    # Mutually-exclusive effect pairs enforced by fheroes2
    # ``_replaceAffection`` in battle_troop.cpp:1180-1250: a fresh spell
    # of one side expels the opposite side before the new one is added.
    # Failing to do so would let (e.g.) Bless and Curse coexist and
    # double-modify attack/damage, which the original engine never does.
    _EFFECT_MUTEX: dict = {
        "Bless":      {"Curse"},
        "Curse":      {"Bless"},
        "Haste":      {"Slow"},
        "Slow":       {"Haste"},
        "Stoneskin": {"Steelskin"},
        "Steelskin": {"Stoneskin"},
        "Berserker":  {"Hypnotize"},
        "Hypnotize":  {"Berserker"},
    }

    def add_effect(self, effect) -> None:
        """Apply (or refresh) a timed effect.

        Stackable effects (e.g. Disrupting Ray) are appended without removing
        existing ones.  Non-stackable effects replace any existing effect of
        the same name (one of each name at a time) AND remove any
        mutually-exclusive counterpart (Bless/Curse, Haste/Slow,
        Stone Skin/Steel Skin, Berserker/Hypnotize) — mirroring
        ``_replaceAffection`` in battle_troop.cpp.
        """
        if effect.stackable:
            self.effects.append(effect)
            return
        mutex = self._EFFECT_MUTEX.get(effect.name, set())
        self.effects = [
            e for e in self.effects
            if e.name != effect.name and e.name not in mutex
        ]
        self.effects.append(effect)

    def tick_effects(self) -> None:
        """Count down effects at the start of a round, dropping expired ones."""
        for e in self.effects:
            e.remaining -= 1
        self.effects = [e for e in self.effects if e.remaining > 0]

    def _compute_base_strength(self) -> float:
        """fheroes2 getMonsterBaseStrength() — depends only on the unit type.

        sqrt(damage * effectiveHP) * special, where special adds bonuses for
        being a shooter / flyer, a speed remap around Speed::AVERAGE, and the
        special-ability terms from getMonsterBaseStrength.
        """
        damage_potential = float(self.damage_avg)
        effective_hp = float(self.max_hp)

        # fheroes2: NO_ENEMY_RETALIATION → effectiveHP *= 1.4
        if "no_enemy_retaliation" in self.abilities:
            effective_hp *= 1.4

        # fheroes2: double attack abilities (mutually exclusive).
        if "double_shooting" in self.abilities:
            damage_potential *= 2
        elif "double_melee" in self.abilities:
            damage_potential *= (2.0 if "no_enemy_retaliation" in self.abilities
                                 else 1.75)

        if "double_damage_to_undead" in self.abilities:
            damage_potential *= 1.15

        if "two_cell_melee" in self.abilities:
            damage_potential *= 1.2

        if "unlimited_retaliation" in self.abilities:
            damage_potential *= 1.25

        # fheroes2: ALL_ADJACENT and AREA_SHOT share the same multiplier.
        # The legacy 1.2 figure in this port was a transcription error —
        # monster_info.cpp:77-80 is 1.3 for both Hydra and the Power Lich.
        if "all_adjacent_attack" in self.abilities or "area_shot" in self.abilities:
            damage_potential *= 1.3

        special = 1.0
        if self.is_archer:
            # fheroes2: NO_MELEE_PENALTY gives +0.5 instead of +0.4.
            # Use ``shots > 0`` (not ``is_archer``) so an archer with all
            # ammo spent falls back to melee-bonus semantics; the C++ engine
            # does the same via ``battleStats.shots``.
            if self.max_shots > 0:
                special += 0.5 if "no_melee_penalty" in self.abilities else 0.4
        if self.is_flying:
            special += 0.3
        # C++ (monster_info.cpp:92-94) only recognises ENEMY_HALVING (+1.0).
        # death_gaze is the legacy Medusa alias — at most one of the two
        # can be present on a given monster and they map to the same flag,
        # so a single ``+= 1.0`` covers both without double-counting.
        if "enemy_halving" in self.abilities or "death_gaze" in self.abilities:
            special += 1.0
        if "hp_drain" in self.abilities:
            special += 0.3
        if "soul_eater" in self.abilities:
            special += 2.0
        diff = self.base_speed - SPEED_AVERAGE
        special += diff * (0.1 if diff < 0 else 0.05)
        return math.sqrt(damage_potential * effective_hp) * special

    @property
    def monster_strength(self) -> float:
        """fheroes2 GetMonsterStrength() — single-creature strength."""
        return (1.0 + 0.1 * self.attack + 0.05 * self.defense) * self._base_strength

    @property
    def strength(self) -> float:
        """fheroes2 Troop::GetStrength() — stack strength (used by the AI)."""
        if not self.is_alive:
            return 0
        return self.monster_strength * self.count

    # ── combat ──────────────────────────────────────────────

    def take_damage(self, dmg: int) -> tuple:
        """Apply damage, return (actual_damage, killed_count).

        fheroes2: a Mirror Image (CAP_MIRRORIMAGE) is destroyed by ANY damage
        — even a single point — even if the stack would have survived. We
        follow that rule so the AI's Mirror Image cost calculus matches
        the original.

        Alignment fixes vs. the previous port:

        * Negative damage is a bug — fheroes2's ``_applyDamage`` clamps to
          ``_hitPoints`` and rejects 0 (no-op), so we raise instead of
          silently healing.
        * ``actual`` is always bounded by the unit's *current* HP — the
          C++ code uses ``std::min(_hitPoints, dmg)`` (battle_troop.cpp:662-667)
          so HP-drain / kill statistics get the true post-clamp figure.
        * Blind / Paralyze / Petrify are auto-broken on damage
          (battle_troop.cpp:634-649). We dispatch through
          ``break_effects_on_damage`` so callers can't forget to.
        """
        if dmg < 0:
            raise ValueError(f"take_damage: negative damage {dmg}")
        if dmg == 0 or self.count == 0:
            return 0, 0
        if self.is_mirror:
            self.count = 0
            self._is_alive = False
            self._total_hp = 0
            self.mirror_of = None
            self.break_effects_on_damage()
            return dmg, 1
        old_count = self.count
        actual = min(dmg, self._total_hp)
        self._total_hp -= actual
        if self._total_hp <= 0:
            self.count = 0
            self._is_alive = False
        else:
            self.count = (self._total_hp + self.max_hp - 1) // self.max_hp
        # fheroes2 auto-removes Blind/Paralyze/Petrify the moment the unit
        # takes damage. Failing to do so made retaliations illegal forever
        # after a single Blind hit.
        self.break_effects_on_damage()
        return actual, old_count - self.count

    def consume_shot(self) -> None:
        """Decrement the archer's ammo by one shot (battle_troop.cpp)."""
        if self.shots_left > 0:
            self.shots_left -= 1

    def get_uid(self) -> int:
        """fheroes2 ``Battle::Unit::GetUID`` — stable identity within a battle.

        The C++ arena assigns each unit a unique 32-bit UID at
        construction time. Our engine has no such generator, so we use
        ``id(self)`` as a stable per-process identity — the caller can
        index it against an initial roster (e.g. for graveyard tracking)
        but should not persist it across battle instances.
        """
        return id(self)

    def get_max_moves(self) -> int:
        """fheroes2 ``Battle::Unit::GetMaxMovePoints`` = current speed.

        Speed is movement points in this engine.  Slow halves speed
        (spell.cpp Slow.halvesSpeed == true) and Haste adds +2; we
        recompute on the fly so callers don't have to track spells.
        """
        sp = self.base_speed
        # Slow halves speed (round up so a Slowed Speed-1 unit can't
        # freeze).
        if self.has_effect("Slow"):
            sp = (sp + 1) // 2
        if self.has_effect("Haste"):
            sp += 2
        return max(1, sp)

    def is_hand_fighting(self, other: "Unit", grid) -> bool:
        """fheroes2 ``Battle::Unit::isHandFighting`` — adjacent melee range.

        True if the body cells of *self* and *other* share an edge
        (hex-neighbour).  Both head and tail (for wide units) count as
        attack reach.  ``grid`` is the battle's HexGrid (Unit is grid-
        agnostic so we don't bloat per-unit state).
        """
        if not self.is_alive or not other.is_alive:
            return False
        other_cells = other.occupied_cells()
        for cell in self.occupied_cells():
            if cell in other_cells:
                return True
            for nb in grid.neighbors(*cell):
                if nb in other_cells:
                    return True
        return False

    @property
    def is_alive(self) -> bool:
        return self._is_alive

    @is_alive.setter
    def is_alive(self, value: bool) -> None:
        """Toggle life state while keeping ``count`` / ``_total_hp`` in sync.

        The previous port let the caller set ``is_alive`` independently of
        ``count`` / ``_total_hp``, so ``strength`` could report > 0 for a
        zero-HP stack, or vice versa. fheroes2 derives alive from
        ``GetCount() > 0``; this setter enforces the same invariant.
        """
        value = bool(value)
        if value:
            # Revive path: if the unit was previously dead, restore the
            # best-known live count (Cure / Animate Dead allow resurrection
            # up to the original stack size — see ``heal`` below).
            if not self._is_alive:
                revive_count = self.count if self.count > 0 else self.original_count
                self.count = revive_count
                self._total_hp = revive_count * self.max_hp
        else:
            self.count = 0
            self._total_hp = 0
        self._is_alive = value

    def heal(self, amount: int) -> int:
        """Restore HP up to the original stack size (Cure may resurrect).

        fheroes2 ``Battle::Unit::ApplySpell`` (battle_troop.cpp:1439) caps
        healing at ``ArmyTroop::GetHitPoints() = _maxCount * Monster::GetHitPoints``,
        so a Cure cast on a half-dead stack whose top creature is the only
        living one can push the total HP past the current ``count`` and
        resurrect the creature underneath. The old ``count * max_hp`` cap
        silently dropped that overflow, so the resurrected creature never
        appeared.
        """
        if not self.is_alive:
            return 0
        if amount <= 0:
            return 0
        cap = self.original_count * self.max_hp
        healed = max(0, min(amount, cap - self._total_hp))
        self._total_hp += healed
        # If the heal crossed a creature boundary, the stack may have
        # resurrected. Recompute count from the new total HP.
        if self._total_hp > 0:
            self.count = min(
                self.original_count,
                (self._total_hp + self.max_hp - 1) // self.max_hp,
            )
        return healed

    def new_round(self):
        self.retaliated = False
        self._acted = False
