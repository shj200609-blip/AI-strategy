"""Simplified BattleNet — CNN residual backbone + Policy/Value dual heads.

AlphaGo Zero style: f_θ(s) → (p, v)
  - p: policy vector over actions (logits, masked for illegal actions)
  - v: scalar value in [-1, 1]

Stripped of MoE complexity. ~13.5M parameters.
Compatible with battle-ai-demo's encode_observation / action_space.

Architecture:
    grid (35, 9, 11)                   global (20,)
         |                                 |
    CNN path: grid[:, :33]                |
    Conv2d(33→128, 3x3) + GN + ReLU       |
         |                                 |
    ResBlock × 6  (128→128)               |
         |                                 |
    (B, 128, 9, 11)                        |
         |                                 |
    Embed path: grid[:, 33:35]            |
    Embedding(67, 16) × 2                 |
         |                                 |
    (B, 32, 9, 11)                         |
         |                                 |
    concat CNN + Embed ──────┐             |
         |                   |             |
    (B, 160, 9, 11)          |             |
         |                   |             |
    Flatten → 15840 ─ concat ─┘            |
              |                             |
           15860                            |
              |                             |
    Linear(15860, 384) + ReLU  ← shared backbone
              |
    ┌─────────┴─────────┐
    |                   |
  policy_head       value_head
  Linear(384,       Linear(384, 1)
   14655)              + tanh
    |
  logits (masked)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_core.action_space import ACTION_DIM
from ai_core.observation import NUM_GRID_CHANNELS, GRID_ROWS, GRID_COLS, GLOBAL_DIM

# -- Architecture constants ---------------------------------------------------

_CONV_CHANNELS = 128
_NUM_RES_BLOCKS = 6
_BOTTLENECK_DIM = 384
_EMBED_DIM = 16
_GN_GROUPS = 16

_NUM_ORIG_CHANNELS = 33
_NUM_UNIT_TYPES = 67
_MAX_TYPE_INDEX = 66

_FLAT_SPATIAL_DIM = (_CONV_CHANNELS + 2 * _EMBED_DIM) * GRID_ROWS * GRID_COLS  # 15840
_FUSED_DIM = _FLAT_SPATIAL_DIM + GLOBAL_DIM  # 15860


# -- Residual Block -----------------------------------------------------------


class ResidualBlock(nn.Module):
    """Conv2d → GN → ReLU → Conv2d → GN + skip → ReLU."""

    def __init__(self, channels: int, gn_groups: int = _GN_GROUPS):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


# -- Main network -------------------------------------------------------------


class BattleNet(nn.Module):
    """CNN residual backbone with unit-type embedding and dual heads.

    Produces (policy_logits, value) as required by AlphaGo Zero MCTS.
    No MoE — shared policy/value heads for simplicity (~13.5M params).
    """

    def __init__(
        self,
        grid_channels: int = NUM_GRID_CHANNELS,
        action_dim: int = ACTION_DIM,
    ):
        super().__init__()
        self._action_dim = action_dim

        # -- Stem ------------------------------------------------------------
        self.stem_conv = nn.Conv2d(_NUM_ORIG_CHANNELS, _CONV_CHANNELS,
                                   kernel_size=3, padding=1, bias=False)
        self.stem_gn = nn.GroupNorm(_GN_GROUPS, _CONV_CHANNELS)

        # -- Residual backbone -----------------------------------------------
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(_CONV_CHANNELS) for _ in range(_NUM_RES_BLOCKS)]
        )

        # -- Unit-type embedding ---------------------------------------------
        self.unit_embed = nn.Embedding(
            _NUM_UNIT_TYPES, _EMBED_DIM, padding_idx=0)

        # -- Shared bottleneck -----------------------------------------------
        self.fc_bottleneck = nn.Linear(_FUSED_DIM, _BOTTLENECK_DIM)
        self.dropout = nn.Dropout(0.15)  # 防过拟合(更保守)

        # -- Dual heads (AlphaGo Zero style) ---------------------------------
        self.policy_head = nn.Linear(_BOTTLENECK_DIM, action_dim)
        self.value_head = nn.Linear(_BOTTLENECK_DIM, 1)

        self._init_weights()

    @property
    def num_experts(self) -> int:
        """Always 0 — no MoE in this simplified variant."""
        return 0

    def forward(
        self,
        grid: torch.Tensor,
        global_vec: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            grid:       (B, 35, 9, 11) grid feature map
            global_vec: (B, 20) global scalar vector
            mask:       (B, ACTION_DIM) legality mask (1 = legal, 0 = illegal)

        Returns:
            policy_logits: (B, ACTION_DIM) — masked logits (illegal = -inf)
            value:         (B, 1) — value estimate in [-1, 1]
        """
        # 1. CNN backbone — original feature channels only (0–32)
        x = F.relu(self.stem_gn(self.stem_conv(grid[:, :_NUM_ORIG_CHANNELS])))
        x = self.res_blocks(x)  # (B, 128, 9, 11)

        # 2. Unit-type embedding — channels 33 (my) and 34 (enemy)
        my_type_idx = (grid[:, 33] * _MAX_TYPE_INDEX).round().long().clamp(0, _MAX_TYPE_INDEX)
        enemy_type_idx = (grid[:, 34] * _MAX_TYPE_INDEX).round().long().clamp(0, _MAX_TYPE_INDEX)

        my_emb = self.unit_embed(my_type_idx).permute(0, 3, 1, 2)   # (B, 16, 9, 11)
        enemy_emb = self.unit_embed(enemy_type_idx).permute(0, 3, 1, 2)

        # 3. Concatenate CNN features + embeddings
        x = torch.cat([x, my_emb, enemy_emb], dim=1)  # (B, 160, 9, 11)

        # 4. Flatten spatial dims + concat global vector
        x = x.flatten(start_dim=1)                     # (B, 15840)
        x = torch.cat([x, global_vec], dim=1)          # (B, 15860)

        # 5. Shared bottleneck
        x = F.relu(self.fc_bottleneck(x))               # (B, 384)
        x = self.dropout(x)                               # 防过拟合(dropout.train()时生效)

        # 6. Dual heads
        policy_logits = self.policy_head(x)              # (B, ACTION_DIM)
        value = torch.tanh(self.value_head(x))           # (B, 1)

        # Mask illegal actions
        policy_logits = policy_logits.masked_fill(
            mask == 0, float("-inf")
        )

        return policy_logits, value

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _init_weights(self):
        """Orthogonal initialisation (PPO / AlphaGo Zero standard)."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
