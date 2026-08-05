"""Replay Buffer — fixed-capacity FIFO storage for (s, π, z) training examples.

AlphaGo Zero uses a flat buffer with uniform sampling of the most recent
500K positions.  Each entry is one state-policy-outcome tuple from self-play.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class TrainingExample:
    """One training example from self-play.

    Attributes
    ----------
    grid : np.ndarray
        Encoded observation grid (35, 9, 11), float32.
    global_vec : np.ndarray
        Global feature vector (20,), float32.
    mask : np.ndarray
        Legal action mask (13566,), float32 (1 = legal).
    policy : np.ndarray
        MCTS improved policy π (13566,), float32.  Sparse — zeros for
        illegal / unvisited actions.
    outcome : float
        Final game outcome from the perspective of the acting team.
        +1.0 for win, -1.0 for loss.
    """

    grid: np.ndarray
    global_vec: np.ndarray
    mask: np.ndarray
    policy: np.ndarray
    outcome: float


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer with uniform sampling.

    Parameters
    ----------
    capacity : int
        Maximum number of examples to store.  When exceeded, oldest
        entries are evicted (ring-buffer semantics).
    """

    def __init__(self, capacity: int = 500_000):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._buffer: List[TrainingExample] = []
        self._write_pos = 0  # For ring-buffer eviction

    def add(self, examples: List[TrainingExample]) -> None:
        """Add a batch of training examples.

        Parameters
        ----------
        examples : list of TrainingExample
            Examples to add.  Each example must have valid numpy arrays.
        """
        for ex in examples:
            if len(self._buffer) < self._capacity:
                self._buffer.append(ex)
            else:
                self._buffer[self._write_pos] = ex
                self._write_pos = (self._write_pos + 1) % self._capacity

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Uniformly sample a mini-batch.

        Parameters
        ----------
        batch_size : int
            Number of examples to sample.

        Returns
        -------
        batch : dict
            Tensors stacked on dim 0:
              - "grid":    (B, 35, 9, 11)
              - "global":  (B, 20)
              - "mask":    (B, 13566)
              - "policy":  (B, 13566)
              - "outcome": (B, 1)

        Raises
        ------
        ValueError
            If batch_size > buffer size.
        """
        if batch_size > len(self._buffer):
            raise ValueError(
                f"batch_size ({batch_size}) exceeds buffer size ({len(self._buffer)})"
            )

        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        selected = [self._buffer[i] for i in indices]

        return {
            "grid": torch.tensor(
                np.stack([ex.grid for ex in selected]), dtype=torch.float32
            ),
            "global": torch.tensor(
                np.stack([ex.global_vec for ex in selected]), dtype=torch.float32
            ),
            "mask": torch.tensor(
                np.stack([ex.mask for ex in selected]), dtype=torch.float32
            ),
            "policy": torch.tensor(
                np.stack([ex.policy for ex in selected]), dtype=torch.float32
            ),
            "outcome": torch.tensor(
                [[ex.outcome] for ex in selected], dtype=torch.float32
            ),
        }

    def __len__(self) -> int:
        """Number of examples currently stored."""
        return len(self._buffer)

    def is_ready(self, min_size: int) -> bool:
        """Return True if the buffer has at least min_size examples."""
        return len(self._buffer) >= min_size

    def clear(self) -> None:
        """Remove all stored examples."""
        self._buffer.clear()
        self._write_pos = 0
