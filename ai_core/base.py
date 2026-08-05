"""AIPlayer — the pluggable battle-AI contract.

Every battle AI (the faithful rule-based ``ClassicAI`` today, a future
deep-learning ``DeepAI``) implements this interface. Callers (headless runner,
GUI, tests) depend on this contract rather than a concrete implementation.

``battle_begins`` is called once per battle. The three decision methods then
mirror fheroes2's per-unit activation order (retreat -> hero spell -> unit
action). A learning agent may answer some of them trivially (e.g.
``check_retreat`` returning ``None``) and concentrate its policy in ``decide``;
the contract stays forward-compatible.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

from engine.actions import Action, CastAction, RetreatAction
from engine.battle_state import BattleState
from engine.unit import Unit


class AIPlayer(ABC):
    """Contract for a pluggable battle AI."""

    def battle_begins(self) -> None:
        """Reset per-battle state before the first decision.

        Stateless players can inherit this no-op implementation. Battle
        runners must call it once for every newly-created battle.
        """

    @abstractmethod
    def check_retreat(self, battle: BattleState, unit: Unit
                      ) -> Tuple[int, Optional[Tuple[Optional[Tuple[CastAction, str]],
                                                       RetreatAction]]]:
        """Before a unit acts, decide whether its hero flees.

        Returns one of three states — mirrors fheroes2 ``BattlePlanner``:

        * ``(RETREAT_NONE, None)`` — keep fighting
        * ``(RETREAT_RETREAT, (farewell, RetreatAction))`` — flee the field
        * ``(RETREAT_SURRENDER, (None, RetreatAction))`` — surrender (instant
          loss, used when the army is severely beaten and the hero is weak)
        """
        raise NotImplementedError

    @abstractmethod
    def maybe_cast_spell(self, battle: BattleState, unit: Unit
                         ) -> Optional[Tuple[CastAction, str]]:
        """Let the unit's hero cast one spell this round.

        Returns ``(CastAction, description)`` or ``None``.
        """
        raise NotImplementedError

    @abstractmethod
    def decide(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        """Choose the unit's own action.

        Returns ``(action, human-readable description)``.
        """
        raise NotImplementedError
