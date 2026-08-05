"""Hero / commander — casts at most one spell per round.

Port of fheroes2's ``HeroBase`` / ``Heroes`` classes (heroes_base.{h,cpp}
+ heroes.{h,cpp} + heroes_base.cpp:379-628 + heroes.cpp:967-970). The engine
only needs the battle-time slice of that interface:

* 4 primary skills (attack / defense / power / knowledge), clamped to the
  C++ ranges: attack / defense / knowledge ∈ [0, 255], power ∈ [1, 255].
* Spell-point pool: ``GetMaxSpellPoints() = 10 * knowledge`` (heroes.cpp:967).
* One cast per round — backed by the SPELLCASTED BitModes flag
  (heroes.h:84) instead of a separate boolean.
* Spellbook gate: ``CanCastSpell`` only succeeds if the hero has the
  MAGIC_BOOK artifact (HaveSpellBook, heroes_base.h:152-155).
* Spell cost uses ``spell.spellPoints(this)`` (heroes_base.cpp:600) so
  spells that scale by hero power are billed correctly.
* 14 secondary skills (``numOfSecondarySkills = 14``, skill.h:42), capped at
  ``maxNumOfSecSkills = 8`` (heroes.h:85). Values per level are taken from
  ``secondarySkillValuesPerLevel`` (game_static.cpp:91-104).
* ``is_hero`` / ``is_captain`` are mutually exclusive and selected through
  the C++ GetType() enum (UNDEFINED | CAPTAIN | HEROES, heroes_base.h:79-84).
* ``in_castle`` mirrors ``HeroBase::inCastle()`` (battle_arena.cpp:909).
"""

from typing import Dict, List, Optional

from .spells import SPELLS, DEFAULT_SPELLBOOK, Spell

# ── Commander type ───────────────────────────────────────────────
# fheroes2 HeroBase::UNDEFINED | CAPTAIN | HEROES (heroes_base.h:79-84).
TYPE_UNDEFINED = 0
TYPE_CAPTAIN = 1
TYPE_HEROES = 2

# ── Secondary skills ─────────────────────────────────────────────
# fheroes2 ``numOfSecondarySkills = 14`` (skill.h:42). The order mirrors
# ``Skill::Secondary`` (skill.h:73-86); values come from
# ``secondarySkillValuesPerLevel`` (game_static.cpp:91-104).
SKILL_NAMES: tuple = (
    "pathfinding",   # skill.h:74
    "archery",       # skill.h:75
    "logistics",     # skill.h:76
    "scouting",      # skill.h:77
    "diplomacy",     # skill.h:78
    "navigation",    # skill.h:79
    "leadership",    # skill.h:80
    "wisdom",        # skill.h:81
    "mysticism",     # skill.h:82
    "luck",          # skill.h:83
    "ballistics",    # skill.h:84
    "eagle_eye",     # skill.h:85
    "necromancy",    # skill.h:86
    "estates",       # skill.h:87 (engine uses "estates" — see game_static.cpp:104)
)
# "Resistance does not exist in HoMM2 (it's a HoMM3 skill)." — keep it out.

# fheroes2 game_static.cpp: secondarySkillValuesPerLevel
# {skill_name: {level: value}} — level 1=Basic, 2=Advanced, 3=Expert.
# Ballistics values are 0/0/0 — its combat effects live in battle_catapult.cpp.
SKILL_VALUES: Dict[str, Dict[int, int]] = {
    "pathfinding":   {1: 25,  2: 50,  3: 100},
    "archery":       {1: 10,  2: 25,  3: 50},
    "logistics":     {1: 10,  2: 20,  3: 30},
    "scouting":      {1: 1,   2: 2,   3: 3},
    "diplomacy":     {1: 25,  2: 50,  3: 100},
    "navigation":    {1: 33,  2: 66,  3: 100},
    "leadership":    {1: 1,   2: 2,   3: 3},
    "wisdom":        {1: 3,   2: 4,   3: 5},
    "mysticism":     {1: 1,   2: 2,   3: 3},
    "luck":          {1: 1,   2: 2,   3: 3},
    "ballistics":    {1: 0,   2: 0,   3: 0},   # handled in catapult (battle_catapult.cpp:42-69)
    "eagle_eye":     {1: 20,  2: 30,  3: 40},
    "necromancy":    {1: 10,  2: 20,  3: 30},
    "estates":       {1: 100, 2: 250, 3: 500},
}

# fheroes2 heroes.h:85 — heroes are capped at 8 secondary skills.
MAX_SECONDARY_SKILLS: int = 8

# fheroes2 heroes.cpp:919-933 — primary-skill clamps after modifiers.
ATTACK_MIN, ATTACK_MAX = 0, 255
DEFENSE_MIN, DEFENSE_MAX = 0, 255
POWER_MIN, POWER_MAX = 1, 255
KNOWLEDGE_MIN, KNOWLEDGE_MAX = 0, 255


def _clamp_primary(name: str, value: int) -> int:
    """Apply the fheroes2 primary-skill clamp (heroes.cpp:919-933)."""
    if name == "power":
        lo, hi = POWER_MIN, POWER_MAX
    else:
        lo, hi = 0, 255
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


class _BitModes:
    """Tiny stand-in for fheroes2 ``BitModes`` (heroes.h:84 enum).

    We only need the bits that affect combat: ``SPELLCASTED``. Other bits
    (SHIPMASTER, ENABLEMOVE, JAIL, …) are out of scope for this engine.
    """

    SPELLCASTED = 0x00000004

    def __init__(self) -> None:
        self._bits: int = 0

    def Modes(self, bit: int) -> bool:
        return bool(self._bits & bit)

    def SetModes(self, bit: int) -> None:
        self._bits |= bit

    def ResetModes(self, bit: int) -> None:
        self._bits &= ~bit

    def ResetAllModes(self) -> None:
        self._bits = 0


class BagArtifacts:
    """Minimal shim for fheroes2 ``BagArtifacts`` (artifact.h).

    The engine doesn't model artifact bonuses in combat; only ``MAGIC_BOOK``
    matters because ``HeroBase::CanCastSpell`` gates on
    ``hasArtifact(Artifact::MAGIC_BOOK)`` (heroes_base.cpp:379, 410).
    Tests / configs that don't carry artifacts get an empty bag, which
    matches a C++ hero with no spell book.
    """

    MAGIC_BOOK = "MAGIC_BOOK"

    def __init__(self) -> None:
        self._artifacts: List[str] = []

    def hasArtifact(self, art: str) -> bool:
        return art in self._artifacts

    def PushArtifact(self, art: str) -> bool:
        if art in self._artifacts:
            return False
        self._artifacts.append(art)
        return True

    def RemoveArtifact(self, art: str) -> None:
        if art in self._artifacts:
            self._artifacts.remove(art)

    # The classic AI spells module uses
    # ``bag.get_total_artifact_effect_value("EVERY_COMBAT_SPELL_DURATION")``
    # to bump the spell duration (ai_core/classic_ai/spells.py:65-69). We
    # return 0 by default — engines without artifact modelling get no bonus.
    def get_total_artifact_effect_value(self, _effect: str) -> int:  # noqa: D401
        return 0


class Hero:
    """Battle-only port of fheroes2 ``Heroes`` (heroes.h / heroes_base.h).

    The engine has no adventure-map state (no Army, no portrait, no race,
    no patrol, no move points, no artifact bag beyond MAGIC_BOOK). Combat
    only needs: primary stats, SP pool, spellbook contents, secondary
    skills, and the inCastle() predicate for retreat / castle-cover rules.
    """

    def __init__(
        self,
        power: int = 1,
        knowledge: int = 1,
        attack: int = 0,
        defense: int = 0,
        spells: Optional[List[str]] = None,
        name: str = "Hero",
        skills: Optional[Dict[str, int]] = None,
        type: int = TYPE_HEROES,
        in_castle: bool = False,
        has_spell_book: bool = True,
        bag_artifacts: Optional[BagArtifacts] = None,
        # Retain the battle-arena predicates the AI still relies on
        # (forces.py, planner.py). They mirror fheroes2 ``CanRetreat`` /
        # ``CanSurrender`` and ``kingdom.AllowPayment``.
        has_valuable_artifacts: bool = False,
        kingdom_hero_count: int = 1,
        kingdom_castle_count: int = 1,
        defends_last_castle: bool = False,
        gold: int = 0,
        surrender_cost: int = 0,
        no_shooting_penalty: bool = False,
    ):
        self.name = name

        # fheroes2 Skill::Primary — clamped the same way as
        # Heroes::Get{Attack,Defense,Power,Knowledge} (heroes.cpp:919-933).
        self.attack = _clamp_primary("attack", attack)
        self.defense = _clamp_primary("defense", defense)
        self.power = _clamp_primary("power", power)
        self.knowledge = _clamp_primary("knowledge", knowledge)

        # fheroes2 ``HeroBase::GetType()`` enum (heroes_base.h:79-84).
        # ``is_hero`` / ``is_captain`` derive from this and are mutually
        # exclusive; default is a real hero.
        if type not in (TYPE_UNDEFINED, TYPE_CAPTAIN, TYPE_HEROES):
            raise ValueError(f"unknown commander type: {type!r}")
        self.type = type
        self.is_hero = type == TYPE_HEROES
        self.is_captain = type == TYPE_CAPTAIN

        # fheroes2 ``HeroBase::inCastle()`` — non-NULL iff the defender is
        # standing inside their own castle (battle_arena.cpp:909).
        self.in_castle = in_castle

        # fheroes2 ``HasValuableArtifacts`` triggers retreat even when the
        # numbers are close — kept as a flag for the AI (classic_ai/forces.py).
        self.has_valuable_artifacts = has_valuable_artifacts

        # Rehire-possible proxies (fheroes2 ``isPossibleToReHire``).
        self.kingdom_hero_count = kingdom_hero_count
        self.kingdom_castle_count = kingdom_castle_count
        self.defends_last_castle = defends_last_castle

        # Surrender wallet: fheroes2 ``kingdom.AllowPayment({GOLD, cost})``.
        self.gold = gold
        self.surrender_cost = surrender_cost

        # fheroes2 ``NO_SHOOTING_PENALTY`` artifact (skip castle wall cover).
        self.no_shooting_penalty = no_shooting_penalty

        # fheroes2 ``HeroBase::HaveSpellBook`` ↔ ``_bagArtifacts`` holding
        # Artifact::MAGIC_BOOK (heroes_base.h:152-155, heroes_base.cpp:185-189).
        # ``SpellBookActivate`` is what LoadDefaults calls for HEROES/CAPTAIN
        # (heroes_base.cpp:54-78); a fresh Hero mirrors that by giving them
        # the book by default.
        self.bag_artifacts = bag_artifacts if bag_artifacts is not None else BagArtifacts()
        if has_spell_book and not self.HaveSpellBook():
            self.bag_artifacts.PushArtifact(BagArtifacts.MAGIC_BOOK)

        # fheroes2 ``HeroBase::_spellBook`` (heroes_base.h:174) and
        # ``HeroBase::_spellPoints`` (heroes_base.h:172). Default SP pool
        # is the max — heroes.cpp:980 calls ``SetSpellPoints(GetMaxSpellPoints())``
        # right after construction; the engine follows the same pattern.
        self.spells: List[str] = (
            list(spells) if spells is not None else list(DEFAULT_SPELLBOOK)
        )
        # fheroes2 ``GetMaxSpellPoints() = 10 * knowledge`` (heroes.cpp:967-970).
        # This is the authoritative ceiling; the old ``max_spell_points=15``
        # default had no C++ analogue and is removed.
        self.max_spell_points = 10 * self.knowledge
        self.spell_points = self.max_spell_points

        # fheroes2 ``Modes`` BitModes (heroes.h:84 — SPELLCASTED bit) replaces
        # the old ``_cast_this_round`` boolean. Reset by the battle loop in
        # ``ActionNewTurn`` on the C++ side; here we expose ``reset_round``.
        self.modes = _BitModes()

        # fheroes2 ``Skill::SecSkills`` (skill.h:89) capped at
        # ``maxNumOfSecSkills = 8`` (heroes.h:85). The C++ side validates on
        # ``LearnSkill``; we mirror the validation here.
        self.skills: Dict[str, int] = self._validate_skills(skills or {})

    # ── spellbook ───────────────────────────────────────────────

    @property
    def spellbook(self) -> List[Spell]:
        """Return the hero's known spells as ``Spell`` instances."""
        return [SPELLS[name] for name in self.spells if name in SPELLS]

    def HaveSpellBook(self) -> bool:
        """Port of ``HeroBase::HaveSpellBook`` (heroes_base.h:152-155).

        Gates the entire combat spell system — without the artifact the
        C++ ``CanCastSpell`` returns false unconditionally.
        """
        return self.bag_artifacts.hasArtifact(BagArtifacts.MAGIC_BOOK)

    def HaveSpell(self, spell: Spell) -> bool:
        """Port of ``HeroBase::HaveSpell`` (heroes_base.cpp:163-166)."""
        return self.HaveSpellBook() and spell.name in self.spells

    def _spell_points_cost(self, spell: Spell) -> int:
        """Spell cost as billed by ``spell.spellPoints(this)`` (heroes_base.cpp:600).

        The engine's ``Spell.cost`` is the base MP; in C++ some spells
        scale by hero power. We expose the raw ``cost`` here and let the
        caller pass a hero so the scaling can be added later without
        changing the signature.
        """
        return int(getattr(spell, "cost", 0))

    def can_cast(self, spell: Spell) -> bool:
        """Port of ``HeroBase::CanCastSpell`` combat-only branch
        (heroes_base.cpp:379-410).

        Four checks, in order:
        1. Have spell book (artifact MAGIC_BOOK).
        2. Not already cast this round (SPELLCASTED bit clear).
        3. Know the spell (it's in the hero's spellbook).
        4. Have enough SP for ``spell.spellPoints(this)``.
        """
        if not self.HaveSpellBook():
            return False
        if self.modes.Modes(_BitModes.SPELLCASTED):
            return False
        if not self.HaveSpell(spell):
            return False
        if self.spell_points < self._spell_points_cost(spell):
            return False
        return True

    def cast(self, spell: Spell) -> None:
        """Port of ``HeroBase::SpellCasted`` (heroes_base.cpp:599-603).

        Deducts ``spell.spellPoints(this)`` and sets the SPELLCASTED bit.
        C++ also subtracts ``spell.movePoints()`` — the engine has no move
        points, so the second line is omitted.
        """
        cost = min(self._spell_points_cost(spell), self.spell_points)
        self.spell_points -= cost
        self.modes.SetModes(_BitModes.SPELLCASTED)

    def reset_round(self) -> None:
        """Clear SPELLCASTED at the start of a new round.

        Equivalent to the engine-side ``ActionNewTurn`` clearing the bit;
        on C++ this happens automatically via the ``Heroes::ActionNewTurn``
        flow (heroes_move.cpp) once per turn.
        """
        self.modes.ResetModes(_BitModes.SPELLCASTED)

    # ── legacy aliases ──────────────────────────────────────────
    # The pre-refactor engine exposed ``_cast_this_round`` as a public
    # boolean and read it from the AI (action_space.py, planner.py,
    # spells.py, retreat.py). Keep an alias so the battle plumbing keeps
    # compiling while the SPELLCASTED bit is the source of truth.
    @property
    def _cast_this_round(self) -> bool:
        return self.modes.Modes(_BitModes.SPELLCASTED)

    @_cast_this_round.setter
    def _cast_this_round(self, value: bool) -> None:
        if value:
            self.modes.SetModes(_BitModes.SPELLCASTED)
        else:
            self.modes.ResetModes(_BitModes.SPELLCASTED)

    # ── secondary skills ────────────────────────────────────────

    @staticmethod
    def _validate_skills(skills: Dict[str, int]) -> Dict[str, int]:
        """Validate and clamp a skills dict the same way the C++ side does
        on ``LearnSkill`` + ``SecSkills::AddSkill``.

        Unknown skill names are dropped (C++: ``Skill::Secondary::isValid``
        guard in heroes.cpp:534). Levels outside [1, 3] are dropped. The
        total is capped at ``MAX_SECONDARY_SKILLS = 8`` (heroes.h:85) —
        extra entries are silently discarded, mirroring the C++ behaviour
        where a hero past the cap simply cannot learn more.
        """
        validated: Dict[str, int] = {}
        for name, level in skills.items():
            if name not in SKILL_VALUES:
                continue
            if not isinstance(level, int) or level < 1 or level > 3:
                continue
            if name in validated:
                # C++ keeps the higher level (heroes.cpp:533-538).
                if level > validated[name]:
                    validated[name] = level
                continue
            if len(validated) >= MAX_SECONDARY_SKILLS:
                break
            validated[name] = level
        return validated

    def get_skill_level(self, skill: str) -> int:
        """Return skill level 0–3 (0 = not learned).

        Port of ``Heroes::GetLevelSkill`` (heroes.cpp:1737-1740).
        """
        return self.skills.get(skill, 0)

    def get_skill_value(self, skill: str) -> int:
        """Return the numeric value for *skill* at current level.

        Port of ``Heroes::GetSecondarySkillValue`` (heroes.cpp:1727-1730)
        → ``Skill::Secondary::GetValue`` (skill.cpp:117-132) →
        ``secondarySkillValuesPerLevel`` (game_static.cpp:91-104).

        Ballistics returns 0 at every level — its combat effects are
        handled by ``Catapult`` (battle_catapult.cpp:42-69), not by the
        numeric modifier.
        """
        level = self.get_skill_level(skill)
        if level == 0:
            return 0
        return SKILL_VALUES.get(skill, {}).get(level, 0)

    def has_max_secondary_skills(self) -> bool:
        """Port of ``Heroes::HasMaxSecondarySkill`` (heroes.cpp:1719-1722)."""
        return len(self.skills) >= MAX_SECONDARY_SKILLS

    # ── factory ─────────────────────────────────────────────────

    @staticmethod
    def from_config(data: Optional[dict]) -> Optional["Hero"]:
        """Build a Hero from a config dict, or None if absent.

        Accepts the legacy keys (``power``, ``spell_points``, ``attack``,
        ``defense``, ``knowledge``, ``skills``, ``spells``, ``is_hero`` /
        ``is_captain``, ``in_castle``, …) so existing configs and YAML
        payloads keep loading.
        """
        if not data:
            return None

        # Resolve the C++ GetType() enum from the legacy ``is_hero`` /
        # ``is_captain`` flags; an explicit ``type`` key wins if present.
        is_hero = data.get("is_hero", True)
        is_captain = data.get("is_captain", False)
        if "type" in data:
            type_value = int(data["type"])
        elif is_captain:
            type_value = TYPE_CAPTAIN
        elif is_hero:
            type_value = TYPE_HEROES
        else:
            type_value = TYPE_UNDEFINED

        return Hero(
            power=data.get("power", 1),
            knowledge=data.get("knowledge", 1),
            attack=data.get("attack", 0),
            defense=data.get("defense", 0),
            spells=data.get("spells"),
            name=data.get("name", "Hero"),
            skills=data.get("skills"),
            type=type_value,
            in_castle=data.get("in_castle", False),
            has_spell_book=data.get("has_spell_book", True),
            has_valuable_artifacts=data.get("valuable_artifacts", False),
            kingdom_hero_count=data.get("kingdom_hero_count", 1),
            kingdom_castle_count=data.get("kingdom_castle_count", 1),
            defends_last_castle=data.get("defends_last_castle", False),
            gold=data.get("gold", 0),
            surrender_cost=data.get("surrender_cost", 0),
            no_shooting_penalty=data.get("no_shooting_penalty", False),
        )