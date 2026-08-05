"""AlphaGoConfig — single source of truth for all hyperparameters."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AlphaGoConfig:
    """Hyperparameters for AlphaGo Zero pipeline.

    All values have sensible defaults matching the AlphaGo Zero paper
    where applicable, scaled down for the fheroes2 battle domain.
    """

    # ── MCTS ───────────────────────────────────────────────────────
    num_simulations: int = 800
    """Number of MCTS simulations per move."""

    c_puct: float = 2.5
    """PUCT exploration constant. Higher = more exploration."""

    dirichlet_alpha: float = 0.03
    """Dirichlet noise concentration parameter. Smaller = spikier noise."""

    dirichlet_epsilon: float = 0.25
    """Dirichlet noise mixing ratio: (1-ε)*P + ε*Dir(α)."""

    temperature_threshold: int = 8  # ~25% of typical game moves
    """Use temperature τ=1 for the first N moves, then greedy after."""

    # ── Self-Play ──────────────────────────────────────────────────
    games_per_iteration: int = 100
    """Number of self-play games per pipeline iteration."""

    max_moves_per_game: int = 200
    """Safety cap on moves per game (BattleState.MAX_ROUNDS)."""

    # ── Replay Buffer ─────────────────────────────────────────────
    buffer_capacity: int = 500_000
    """Maximum number of (s, π, z) tuples stored (FIFO window)."""

    min_buffer_size: int = 10_000
    """Minimum buffer size before training begins."""

    # ── Training ──────────────────────────────────────────────────
    batch_size: int = 512
    """Mini-batch size for SGD updates."""

    learning_rate: float = 0.01
    """Initial learning rate (SGD + momentum)."""

    momentum: float = 0.9
    """SGD momentum coefficient."""

    weight_decay: float = 1e-4
    """L2 regularization coefficient (handled by optimizer)."""


    train_steps_per_iter: int = 1000
    """Number of mini-batch updates per pipeline iteration."""

    # ── Evaluation ────────────────────────────────────────────────
    eval_games: int = 50
    """Number of pit games for evaluating new vs best network."""

    win_rate_threshold: float = 0.55
    """New network must beat best by this margin to be promoted."""

    eval_mcts_simulations: int = 400
    """MCTS simulations during evaluation (can be lower than training)."""

    # ── Pipeline ──────────────────────────────────────────────────
    num_iterations: int = 100
    """Total self-play → train → eval loops to run."""

    checkpoint_dir: str = "checkpoints"
    """Directory for saving model checkpoints."""

    best_model_name: str = "best.pt"
    """Filename for the current best model."""

    # ── Battle Configs ────────────────────────────────────────────
    battle_configs: List[dict] = field(default_factory=lambda: [{
        "units": [
            {"team": 0, "type": "Swordsman", "col": 5, "row": 3},
            {"team": 1, "type": "Swordsman", "col": 5, "row": 6},
        ]
    }])
    """Battle configuration(s) used for self-play.  Multi-config mixes
    randomly per game to increase data diversity."""

    eval_battle_configs: Optional[List[dict]] = None
    """Battle config(s) for evaluation.  Defaults to battle_configs."""

    early_stop_patience: int = 0
    """Stop if eval win rate hasn't improved for N iterations. 0=off."""

    opponent_pool_size: int = 5
    """Number of historical models to keep as opponents. 0 = pure self-play."""

    # ── Opponent schedule (rule-based → self-play) ─────────────────
    use_classic_opponent: bool = True
    """If True, the first ``classic_opponent_iters`` iterations play against
    a frozen ClassicAI (fheroes2 default heuristic AI).  This gives the
    network a stable learning signal: "beat the baseline" is a much
    clearer objective than "beat a random init network".  After that
    schedule, the opponent pool / pure self-play kicks in."""

    classic_opponent_iters: int = 5
    """Number of initial iterations where team 1 is the ClassicAI."""

    classic_opponent_difficulty: str = "Normal"
    """Difficulty for the ClassicAI opponent during the rule-based phase."""

    classic_opponent_randomize: float = 0.05
    """Slight noise injection in ClassicAI (mirrors fheroes2 tie-break
    randomness).  0.0 = fully deterministic."""

    classic_opponent_decay: float = 0.5
    """Per-iteration decay of the ClassicAI use ratio (only meaningful when
    mixing classic + pool).  Range [0, 1]; 1.0 = never decay (always use
    ClassicAI), 0.0 = drop it immediately after ``classic_opponent_iters``."""

    # ── Hardware ──────────────────────────────────────────────────
    device: str = "cpu"
    """Torch device string: 'cpu', 'cuda', 'cuda:0', etc."""

    num_workers: int = 1  # TODO: parallel self-play not yet implemented
    """Number of parallel self-play worker processes."""

    # ── Misc ──────────────────────────────────────────────────────
    log_interval: int = 10
    """Log training metrics every N steps."""

    tensorboard: bool = False
    """If True, write TensorBoard logs to runs/."""

    # ── Final evaluation (best_model vs ClassicAI) ────────────────
    final_eval_enabled: bool = True
    """After training (or early stop), pit the saved best_model against
    the rule-based ClassicAI on a larger set and report the win rate.
    Useful for measuring how far above the heuristic baseline the
    final network is."""

    final_eval_games: int = 100
    """Number of games for the final best-vs-ClassicAI evaluation.
    Falls back to ``eval_games`` if 0.  0 disables the final eval."""

    final_eval_mcts_simulations: int = 800
    """MCTS simulations per move during the final evaluation.  Falls
    back to ``eval_mcts_simulations`` if 0."""
