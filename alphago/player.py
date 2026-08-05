"""MCTSAIPlayer — AlphaGo-style AI implementing the AIPlayer interface.

Wraps MCTS + neural network so it can be plugged into the arena evaluator
or used for pit evaluation during training.
"""

from typing import Optional, Tuple

import torch

from ai_core.base import AIPlayer
from ai_core.model import BattleNet
from engine.actions import Action, RetreatAction, CastAction
from engine.battle_state import BattleState
from engine.unit import Unit
from alphago.config import AlphaGoConfig
from alphago.mcts import MCTS, sample_action_from_policy


class MCTSAIPlayer(AIPlayer):
    """AlphaGo Zero MCTS-based battle AI.

    Implements the AIPlayer contract so it can be used interchangeably
    with ClassicAI or DeepAI in headless battles and GUI mode.

    All decisions (retreat, spellcast, unit action) go through the unified
    MCTS search over the 13,566-dim action space.

    Parameters
    ----------
    model_path : str or None
        Path to a BattleNet checkpoint.  If None, uses a randomly
        initialised network (for testing only).
    config : AlphaGoConfig or None
        MCTS and evaluation hyperparameters.  Defaults to a fast-eval
        config with fewer simulations.
    device : str
        Torch device.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[AlphaGoConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or AlphaGoConfig()
        self.device = device

        self.model = BattleNet()
        if model_path is not None:
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()

        self.mcts = MCTS(self.config)

    # ── AIPlayer interface ───────────────────────────────────────────

    def check_retreat(self, battle: BattleState, unit: Unit) -> None:
        """MCTSAI handles retreat via the unified action space."""
        return None

    def maybe_cast_spell(self, battle: BattleState, unit: Unit) -> None:
        """MCTSAI handles spells via the unified action space."""
        return None

    def decide(self, battle: BattleState, unit: Unit) -> Tuple[Action, str]:
        """Run MCTS search and return the best action.

        Uses greedy selection (τ → 0) at inference time.
        """
        from ai_core.action_space import index_to_action

        policy = self.mcts.search(battle, unit, self.model, self.device)

        if not policy:
            # Fallback: skip
            from engine.actions import SkipAction
            return SkipAction(unit), "MCTSAI(skip-fallback)"

        # Greedy selection at inference time
        action_idx = sample_action_from_policy(policy, temperature=0.0)
        action = index_to_action(action_idx, battle, unit)

        return action, f"MCTSAI({action_idx})"

    @classmethod
    def from_model(
        cls,
        model: BattleNet,
        config: AlphaGoConfig,
        device: str = "cpu",
    ) -> 'MCTSAIPlayer':
        """Create a player from an already-loaded model (no disk read).

        Useful in the pipeline for pit evaluation where the model is
        already in memory.
        """
        player = cls.__new__(cls)
        cls.__init__(player, model_path=None, config=config, device=device)
        player.model = model.to(device)
        player.model.eval()
        return player
