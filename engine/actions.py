"""Action types for the battle AI and animation engine.

Mirror fheroes2 ``Battle::Command`` + ``Arena::ApplyAction*`` (battle_command.h
+ battle_action.cpp).  Each Python class corresponds to one C++ CommandType,
and the engine dispatches with ``isinstance(action, X)`` exactly like the
C++ ``switch (cmd.GetType())`` in ``Battle::Arena::ApplyAction``.

Unlike the C++ side, which packs every payload as a LIFO of ints in a
``std::vector<int>`` and pulls them with ``GetNextValue()``, we use plain
Python dataclasses — the field names below map directly to the C++ payload
ints in their semantic order.  There is no ``updatePCG32Stream`` rewinding
or replay here; determinism in this port comes from the seeded RNG that
``BattleState`` already threads through every roll.
"""

from typing import Optional, List, Tuple, TYPE_CHECKING
from .unit import Unit

if TYPE_CHECKING:
    from .battle_pathfinding import BattlePosition


# ── CommandType analogue ──────────────────────────────────────────────
# fheroes2 Battle::CommandType enum (battle_command.h:40-48).  Used as
# ``action.command_type`` for log / debug introspection — dispatch itself
# uses isinstance so subclasses can layer on without an enum explosion.

class CommandType:
    MOVE = "MOVE"
    ATTACK = "ATTACK"
    SPELLCAST = "SPELLCAST"
    MORALE = "MORALE"
    CATAPULT = "CATAPULT"
    TOWER = "TOWER"
    RETREAT = "RETREAT"
    SURRENDER = "SURRENDER"
    SKIP = "SKIP"


class Action:
    """Base class — every command knows its C++ CommandType tag."""
    command_type: str = ""


# ── CommandType::MOVE (battle_command.h:74) ───────────────────────────
# C++ payload: ``UID, dst`` — dst is the destination cell index, or ``-1``
# for "no move" (an attacker's AttackCommand may move first, see below).
# Python: keep ``unit`` + ``path`` (the cells visited, head cells) and the
# resolved final ``BattlePosition`` so wide-unit tail placement is exact.

class MoveAction(Action):
    command_type = CommandType.MOVE

    def __init__(self, unit: Unit, path: List[Tuple[int, int]],
                 final_position: Optional['BattlePosition'] = None):
        self.unit = unit
        # fheroes2 ``BattlePathfinder`` yields head cells; for wide units the
        # tail cell is implicit (horizontal neighbour).  Python port stores
        # the head trajectory so callers can replay it for animation; the
        # destination is the *complete* ``BattlePosition`` (head + optional
        # tail + reflection) so wide units don't get stuck head-only.
        self.path = path
        self.final_position = final_position


# ── CommandType::ATTACK (battle_command.h:75-77) ──────────────────────
# C++ payload: ``attacker UID, defender UID, dst, tgt, dir``
#   dst = cell index to move to first (-1 = no move, attack in place)
#   tgt = cell index to attack (head/tail of defender for wide targets)
#   dir = attack direction (CellDirection).  -1 = "let the engine pick";
#        archer-vs-non-handfighting must be CellDirection::UNKNOWN.
# Python keeps the typed dataclass so we don't lose the direction.

class AttackAction(Action):
    command_type = CommandType.ATTACK

    def __init__(self, attacker: Unit, target: Unit,
                 from_pos: Optional[Tuple[int, int]] = None,
                 ranged: bool = False,
                 cell: Optional[Tuple[int, int]] = None,
                 from_position: Optional['BattlePosition'] = None,
                 dir: Optional[int] = None):
        self.attacker = attacker
        self.target = target
        # C++ ``dst`` — the head cell the attacker moves to before the
        # attack.  ``None`` means "attack in place" (equivalent to C++ -1).
        # For an archer with no adjacent melee, ``from_pos`` MUST be None
        # — see ApplyActionAttack (battle_action.cpp:505) and the
        # CellDirection::UNKNOWN gate.
        self.from_pos = from_pos
        # fheroes2 derives ``ranged`` from
        # ``attacker->isArchers() && !isHandFighting()``; the Python port
        # kept an explicit flag for AI plumbing, but the engine itself
        # must recompute the flag (battle_action.cpp:515, 540) so a C++
        # archer who *can* shoot but is hand-fighting goes through the
        # melee branch.  AttackAction.ranged is a hint for callers, not
        # a guarantee — the engine recomputes from attacker state.
        self.ranged = ranged
        # C++ ``tgt`` — head or tail cell of the defender (for wide
        # targets) or the aimed cell for area-shot.  ``None`` lets the
        # engine pick via ``calculateAttackTarget``.
        self.cell = cell
        # Complete pathfinder position for ``from_pos`` (head + tail +
        # reflection).  ``None`` for an in-place archer.
        self.from_position = from_position
        # C++ ``dir`` — CellDirection of the attack.  ``None`` = "let the
        # engine pick" (matches C++ -1).  For non-blocked archers the
        # value MUST be CellDirection::UNKNOWN (-1 in our port) or the
        # command is rejected (battle_action.cpp:524-528).
        self.dir = dir


# ── CommandType::SKIP (battle_command.h:82) ──────────────────────────
# C++ payload: ``UID`` — applies ``TR_SKIP | TR_MOVED``.  There's no
# "defending" parameter: fheroes2 has no per-skip defend flag (the
# Defend hot-key is its own command type via the UI, not a SKIP variant).
# Setting ``TR_MOVED`` matters: without it the engine happily replays more
# commands for the same unit (battle_action.cpp:699-700 + 728-729).

class SkipAction(Action):
    command_type = CommandType.SKIP

    def __init__(self, unit: Unit):
        self.unit = unit


# ── CommandType::SPELLCAST (battle_command.h:78-93) ──────────────────
# C++ payload varies by spell:
#   MIRRORIMAGE → Spell, UID
#   TELEPORT    → Spell, src index, dst index
#   everything else → Spell, cell index  (target unit is *implicit* via the
#                                        cell; Earthquake doesn't even read
#                                        a target — battle_action.cpp:458).
# fheroes2 pulls the commander from ``GetCurrentForce().GetCommander()``
# rather than threading a team id through the command.  Python callers
# keep the explicit ``team`` because the AI planner doesn't necessarily
# have a "current force" handle, but the engine must treat it as
# authoritative and not derive it from elsewhere.

class CastAction(Action):
    command_type = CommandType.SPELLCAST

    def __init__(self, team: int, spell, target: Optional[Unit] = None,
                 cell: Optional[Tuple[int, int]] = None,
                 destination: Optional[Tuple[int, int]] = None):
        self.team = team
        self.spell = spell
        # fheroes2 only stores a unit when the spell resolves to one
        # (MIRRORIMAGE).  Pure cell spells (Earthquake, Death Ripple,
        # Holy Word, …) pass just a cell index — Python uses ``None``
        # for those so the engine doesn't crash on the missing target.
        self.target = target
        # C++ ``cell index``: AOE centre, Teleport destination, or any
        # other cell-targeted spell.  Same role as fheroes2's 2nd int.
        self.cell = cell
        # C++ TELEPORT's 3rd payload int — where the unit lands.
        self.destination = destination


# ── CommandType::MORALE (battle_command.h:79) ────────────────────────
# C++ payload: ``UID, morale``.  ``morale`` is bool — True clears
# ``TR_MOVED | MORALE_GOOD`` (the "bonus turn" / good-morale branch,
# battle_action.cpp:728-732), False clears ``MORALE_BAD`` and sets
# ``TR_MOVED`` (the "lost the turn to bad morale" branch, lines 746-749).
# Without this command, bad-morale's TR_MOVED never gets set and good-
# morale's TR_MOVED never gets cleared — both surface as bugs (the engine
# thinks the unit still owes an action, or never gets its bonus turn).

class MoraleAction(Action):
    command_type = CommandType.MORALE

    def __init__(self, unit: Unit, morale: bool):
        self.unit = unit
        # True = good morale (bonus turn → clear TR_MOVED + MORALE_GOOD).
        # False = bad morale (lose turn → clear MORALE_BAD, set TR_MOVED).
        self.morale = bool(morale)


# ── CommandType::RETREAT (battle_command.h:80) ───────────────────────
# C++ payload: empty.  The arena gates the command via
# ``CanRetreatOpponent(color)`` (battle_action.cpp:765-773) and the
# commander team comes from ``GetCurrentColor()``; the command carries
# no team id.

class RetreatAction(Action):
    command_type = CommandType.RETREAT

    def __init__(self, team: int):
        self.team = team


# ── CommandType::SURRENDER (battle_command.h:81) ─────────────────────
# C++ payload: empty.  Gates: ``CanSurrenderOpponent(color)`` AND the
# surrendering kingdom can pay ``GetSurrenderCost()`` gold (battle_action.cpp
# ApplyActionSurrender).  The Python port keeps the same shape — the
# ``team`` field is consumed by the engine to look up the cost, not to
# bypass the gate.

class SurrenderAction(Action):
    command_type = CommandType.SURRENDER

    def __init__(self, team: int, cost: int = 0):
        self.team = team
        # C++ derives the cost from the army; this Python port accepts
        # an explicit hint so the AI can pass ``get_surrender_cost()``
        # rather than the engine re-deriving it.
        self.cost = int(cost)


# ── CommandType::TOWER (battle_command.h:83) ─────────────────────────
# C++ payload: ``tower type, target UID``.  One shot per command — the
# arena fires one tower per turn (battle_action.cpp ApplyActionTower).
# ``tower_type`` mirrors fheroes2 ``Battle::TowerType`` (battle_tower.h):
# TWR_LEFT=0, TWR_CENTER=1, TWR_RIGHT=2.

class TowerAction(Action):
    command_type = CommandType.TOWER

    TWR_LEFT = 0
    TWR_CENTER = 1
    TWR_RIGHT = 2

    def __init__(self, tower_type: int, target: Unit):
        self.tower_type = int(tower_type)
        self.target = target


# ── CommandType::CATAPULT (battle_command.h:84) ──────────────────────
# C++ payload: ``shots, [target, damage, hit] * shots`` — the catapult
# has one command that packs all the wall/tower hits it will make during
# this turn (battle_action.cpp ApplyActionCatapult).  Python keeps the
# same shape so a catapult turn is a single Action.

class CatapultAction(Action):
    command_type = CommandType.CATAPULT

    # Mirror fheroes2 CastleDefenseStructure enum (battle.h:135-147).
    # Note the C++ enum has no shared values with TowerAction.TWR_* —
    # WALL1..WALL4 = 1..4 (NOT 0..3!), TOWER1/TOWER2 are the *side*
    # turrets, CENTRAL_TOWER is the Ballista.  Catapult::getAllowedTargets
    # only returns {WALL1..WALL4, TOWER1, TOWER2, BRIDGE, CENTRAL_TOWER}.
    NONE = 0
    WALL1 = 1
    WALL2 = 2
    WALL3 = 3
    WALL4 = 4
    TOWER1 = 5   # left side turret
    TOWER2 = 6   # right side turret
    BRIDGE = 7
    CENTRAL_TOWER = 8

    def __init__(self, shots: List[Tuple[int, int, bool]]):
        # shots: list of (target, damage, hit).  Empty list = catapult
        # exists but made no aimed shot this turn.
        self.shots = list(shots)
