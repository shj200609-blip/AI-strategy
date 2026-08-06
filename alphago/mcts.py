"""Monte Carlo Tree Search with PUCT, guided by a neural network.

AlphaGo Zero style:
  - select:  descend tree via PUCT until a leaf
  - expand:  call network f_θ(s) → (p, v), create child nodes for legal actions
  - backup:  propagate value v up the tree, flipping sign at each level

At the root, Dirichlet noise is added to the network prior for exploration.
After search, the visit-count distribution (with temperature) becomes the
improved policy π.
"""

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ai_core.action_space import ACTION_DIM, index_to_action, legal_mask
from ai_core.model import BattleNet
from ai_core.observation import encode_observation
from engine.actions import Action
from engine.battle_state import BattleState
from engine.unit import Unit


# ── MCTS Node ────────────────────────────────────────────────────────────────


@dataclass
class MCTSNode:
    """A node in the Monte Carlo search tree.

    Each node corresponds to a battle state where a specific unit is about to act.
    Statistics are tracked per child action (edge).

    Attributes
    ----------
    visit_counts : dict
        N(s, a) — how many times each action was taken from this node.
    total_values : dict
        W(s, a) — sum of backpropagated values for each action.
    prior_probs : dict
        P(s, a) — prior probability from the neural network.
    children : dict
        action_index → MCTSNode mapping for expanded actions.
    """

    visit_counts: Dict[int, int] = field(default_factory=dict)
    total_values: Dict[int, float] = field(default_factory=dict)
    prior_probs: Dict[int, float] = field(default_factory=dict)
    children: Dict[int, 'MCTSNode'] = field(default_factory=dict)

    is_terminal: bool = False
    terminal_value: float = 0.0

    def q_value(self, action: int) -> float:
        """Mean action value Q(s, a) = W(s, a) / N(s, a)."""
        n = self.visit_counts.get(action, 0)
        if n == 0:
            return 0.0
        return self.total_values.get(action, 0.0) / n

    def total_visits(self) -> int:
        """Sum of visit counts across all children."""
        return sum(self.visit_counts.values())


# ── PUCT Selection ──────────────────────────────────────────────────────────


def puct_score(node: MCTSNode, action: int, c_puct: float) -> float:
    """PUCT formula: Q(s,a) + c_puct * P(s,a) * sqrt(Σ_b N(s,b)) / (1 + N(s,a))."""
    q = node.q_value(action)
    p = node.prior_probs.get(action, 0.0)
    n = node.visit_counts.get(action, 0)
    sqrt_total = math.sqrt(max(node.total_visits(), 1))
    u = c_puct * p * sqrt_total / (1 + n)
    return q + u


def select_action(node: MCTSNode, c_puct: float) -> int:
    """Select the action with the highest PUCT score."""
    best_action = -1
    best_score = float("-inf")
    for action in node.prior_probs:
        score = puct_score(node, action, c_puct)
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


# ── Dirichlet Noise ─────────────────────────────────────────────────────────


def add_dirichlet_noise(node: MCTSNode, alpha: float, epsilon: float):
    """Add Dirichlet noise to root prior probabilities for exploration.

    P'(s,a) = (1 - ε) * P(s,a) + ε * Dir(α)
    """
    actions = list(node.prior_probs.keys())
    n = len(actions)
    if n == 0:
        return
    noise = np.random.dirichlet([alpha] * n)
    for i, action in enumerate(actions):
        node.prior_probs[action] = (
            (1.0 - epsilon) * node.prior_probs[action]
            + epsilon * noise[i]
        )


# ── State Cloning ───────────────────────────────────────────────────────────


def fast_clone_battle(battle: BattleState) -> BattleState:
    """Lightweight clone — only copies mutable fields that MCTS modifies.

    5-10x faster than clone_battle_state() because we skip:
      - re-creating Unit objects (just copy attrs that change)
      - re-creating the HexGrid (validity never changes)
    """
    # Shallow copy the battle state
    import copy
    c = copy.copy(battle)
    # Deep-copy only the things MCTS mutations touch.
    # NOTE: ``u.effects`` holds ``Effect`` dataclass instances whose
    # ``remaining`` counter is decremented in-place by ``Unit.tick_effects``
    # (called from ``BattleState.start_round`` during MCTS rollouts).
    # A plain ``copy.copy(u)`` + ``list(u.effects)`` rebuild would alias the
    # Effect objects with the original battle, causing every clone's
    # ``start_round`` to mutate the root state's effects and eventually
    # expire buffs/debuffs that were never actually consumed. Clone each
    # Effect as well so MCTS mutations stay confined to the simulation.
    c.units = [copy.copy(u) for u in battle.units]
    for u in c.units:
        u.effects = [copy.copy(e) for e in u.effects]
    c.heroes = {t: copy.copy(h) if h else None for t, h in battle.heroes.items()}
    c._stale_rounds = battle._stale_rounds
    c.deaths_this_round = battle.deaths_this_round
    # These are small dicts, safe shallow copy
    c.morale = dict(battle.morale)
    c.luck = dict(battle.luck)
    # Rebuild _initial_counts with new unit ids
    c._initial_counts = {id(u): u.count for u in c.units}
    return c


# ── Deterministic Turn Advance ───────────────────────────────────────────────


def advance_to_next_unit(battle: BattleState) -> Optional[Unit]:
    """Advance battle to the next unit that needs an AI decision.

    Deterministic version for MCTS: no morale randomness, no retreat checks.
    Walks the speed-based turn order, skipping dead units, and starts new
    rounds when the current round is exhausted.

    Returns the next unit to act, or None if the battle is over.
    """
    if battle.is_over():
        return None

    # Get current turn order and index
    order = battle.turn_order()
    if not order:
        return None

    # Find the first unacted alive unit in this round
    for unit in order:
        if unit.is_alive and not unit._acted:
            return unit

    # All units have acted — start a new round
    if battle.is_over():
        return None

    battle.start_round()
    new_order = battle.turn_order()
    for unit in new_order:
        if unit.is_alive:
            return unit

    return None


# ── MCTS Search ──────────────────────────────────────────────────────────────


class MCTS:
    """AlphaGo Zero-style Monte Carlo Tree Search.

    Usage
    -----
    >>> mcts = MCTS(config)
    >>> policy = mcts.search(battle_state, current_unit, network, config)
    >>> # policy is a dict: action_index → probability
    """

    def __init__(self, config: 'AlphaGoConfig'):
        self.config = config

    def search(
        self,
        root_state: BattleState,
        current_unit: Unit,
        network: BattleNet,
        device: str = "cpu",
    ) -> Dict[int, float]:
        """Run MCTS from the root state and return improved policy π.

        Parameters
        ----------
        root_state : BattleState
            The current battle state (will be cloned internally).
        current_unit : Unit
            The unit that must act now.
        network : BattleNet
            Neural network f_θ(s) → (p, v).
        device : str
            Torch device for network inference.

        Returns
        -------
        policy : dict
            Mapping from action index (int) to probability (float).
            Sums to 1.0 over legal actions.
        """
        # ── Prepare root ───────────────────────────────────────────────
        root = MCTSNode()

        # Encode root state for network
        grid, gvec = encode_observation(root_state, current_unit)
        mask = legal_mask(root_state, current_unit)

        # Network evaluation at root
        t_grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(device)
        t_gvec = torch.tensor(gvec, dtype=torch.float32).unsqueeze(0).to(device)
        t_mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, value = network(t_grid, t_gvec, t_mask)

        # Populate root priors from network policy
        probs = torch.softmax(logits[0], dim=-1).cpu().numpy()
        legal_indices = np.where(mask > 0)[0]
        for action in legal_indices:
            idx = int(action)
            root.prior_probs[idx] = float(probs[idx])
            root.children[idx] = MCTSNode()  # Create child for tree traversal

        # Add Dirichlet noise at root (applied once, won't be overwritten
        # because EXPAND skips nodes that already have priors)
        add_dirichlet_noise(
            root,
            self.config.dirichlet_alpha,
            self.config.dirichlet_epsilon,
        )

        # ── Run simulations ───────────────────────────────────────────
        for _ in range(self.config.num_simulations):
            # Clone state for this simulation
            sim_state = fast_clone_battle(root_state)

            # Find the corresponding unit in the cloned state
            sim_unit = self._find_unit(sim_state, current_unit)

            node = root
            path: List[Tuple[MCTSNode, int]] = []  # (parent_node, action_taken)

            # ---- SELECT ----
            while node.children and not node.is_terminal:
                action = select_action(node, self.config.c_puct)
                path.append((node, action))

                # Execute action on cloned state
                sim_unit = self._find_unit(sim_state, sim_unit)
                if sim_unit is None:
                    break  # unit died somehow — treat as terminal

                act = index_to_action(action, sim_state, sim_unit)
                sim_unit._acted = True  # 标记已行动，回合才能推进
                sim_state.execute(act)

                # Advance to next acting unit
                sim_unit = advance_to_next_unit(sim_state)
                if sim_unit is None:
                    # Battle ended
                    winner = sim_state.winner()
                    # Value is from the perspective of the team that just acted
                    if hasattr(act, 'unit'):
                        value = 1.0 if winner == act.unit.team else -1.0
                    else:
                        value = 0.0
                    break

                # Move to child node
                node = node.children.get(action)
                if node is None:
                    break

            # ---- EXPAND & EVALUATE ----
            if sim_unit is not None and not node.is_terminal:
                # Check if battle is over
                if sim_state.is_over():
                    winner = sim_state.winner()
                    # value from perspective of current acting unit's team
                    acting_team = sim_unit.team
                    leaf_value = 1.0 if winner == acting_team else -1.0
                    node.is_terminal = True
                    node.terminal_value = leaf_value
                else:
                    # Encode and evaluate with network
                    grid_s, gvec_s = encode_observation(sim_state, sim_unit)
                    mask_s = legal_mask(sim_state, sim_unit)

                    t_grid_s = torch.tensor(grid_s, dtype=torch.float32).unsqueeze(0).to(device)
                    t_gvec_s = torch.tensor(gvec_s, dtype=torch.float32).unsqueeze(0).to(device)
                    t_mask_s = torch.tensor(mask_s, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits_s, leaf_value_t = network(t_grid_s, t_gvec_s, t_mask_s)

                    leaf_value = float(leaf_value_t.item())

                    # Expand node: create children for all legal actions
                    probs_s = torch.softmax(logits_s[0], dim=-1).cpu().numpy()
                    legal_s = np.where(mask_s > 0)[0]
                    for act_idx in legal_s:
                        idx = int(act_idx)
                        node.prior_probs[idx] = float(probs_s[idx])
                        node.children[idx] = MCTSNode()  # CRITICAL: actually create the child!

            # If no sim_unit (battle ended during select), value is from winner
            if sim_unit is None and not node.is_terminal:
                leaf_value = value  # retained from select loop

            # ---- BACKUP ----
            # Propagate value up the path, flipping sign at each level
            # (because each level represents the opponent's perspective)
            current_value = leaf_value
            for parent_node, parent_action in reversed(path):
                parent_node.visit_counts[parent_action] = \
                    parent_node.visit_counts.get(parent_action, 0) + 1
                parent_node.total_values[parent_action] = \
                    parent_node.total_values.get(parent_action, 0.0) + current_value
                # Update Q cache
                n = parent_node.visit_counts[parent_action]
                w = parent_node.total_values[parent_action]
                # Flip value for opponent's perspective
                current_value = -current_value

        # ── Compute improved policy from visit counts ─────────────────
        return self._compute_policy(root)

    def _compute_policy(self, root: MCTSNode) -> Dict[int, float]:
        """Convert root visit counts to a probability distribution.

        π(a) ∝ N(s, a)^(1/τ) where τ = 1 (proportional to visit counts).
        """
        total = root.total_visits()
        if total == 0:
            # Fallback: uniform over priors
            total_p = sum(root.prior_probs.values())
            if total_p == 0:
                return {}
            return {a: p / total_p for a, p in root.prior_probs.items()}

        policy = {}
        for action in root.prior_probs:
            n = root.visit_counts.get(action, 0)
            policy[action] = n / total
        return policy

    @staticmethod
    def _find_unit(battle: BattleState, reference: Unit) -> Optional[Unit]:
        """Find the equivalent unit in a cloned BattleState.

        Matches by (team, col, row, name) since unit IDs differ after cloning.
        """
        for u in battle.units:
            if (u.team == reference.team
                    and u.col == reference.col
                    and u.row == reference.row
                    and u.name == reference.name
                    and u.is_alive):
                return u
        return None


# ── Policy Utilities ────────────────────────────────────────────────────────


def sample_action_from_policy(
    policy: Dict[int, float],
    temperature: float = 1.0,
) -> int:
    """Sample an action from the policy distribution with temperature.

    Parameters
    ----------
    policy : dict
        action_index → probability (should sum to ~1)
    temperature : float
        τ = 1.0 for proportional sampling, τ → 0 for argmax.

    Returns
    -------
    action_index : int
    """
    actions = list(policy.keys())
    if not actions:
        raise ValueError("Empty policy — no legal actions available")

    if temperature < 1e-6:
        # Greedy: argmax
        return max(policy, key=policy.get)

    # Apply temperature: π(a)^(1/τ) / Σ π(b)^(1/τ)
    probs = np.array([policy[a] for a in actions], dtype=np.float64)
    if temperature != 1.0:
        probs = probs ** (1.0 / temperature)
    probs /= probs.sum()

    idx = np.random.choice(len(actions), p=probs)
    return actions[idx]
