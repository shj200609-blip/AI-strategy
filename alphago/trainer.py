"""AlphaGo Trainer — supervised learning on self-play data.

Trains the neural network f_θ to predict:
  - policy p: match the MCTS improved policy π (cross-entropy loss)
  - value v:  match the actual game outcome z (MSE loss)

Loss = (z - v)² - πᵀ log p + c * ||θ||²

where the L2 weight decay c * ||θ||² is handled by the optimizer.
"""

from typing import Dict, Optional
import time

import numpy as np
import torch
import torch.nn.functional as F

from ai_core.model import BattleNet
from alphago.config import AlphaGoConfig
from alphago.replay_buffer import ReplayBuffer


class AlphaGoTrainer:
    """Trains BattleNet on (s, π, z) examples from self-play.

    Parameters
    ----------
    model : BattleNet
        The neural network to train.
    config : AlphaGoConfig
        Training hyperparameters.
    """

    def __init__(self, model: BattleNet, config: AlphaGoConfig):
        self.model = model.to(config.device)
        self.config = config

        # SGD + momentum (as in AlphaGo Zero paper)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        self._step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one mini-batch training step.

        Parameters
        ----------
        batch : dict
            From ReplayBuffer.sample(): grid, global, mask, policy, outcome.

        Returns
        -------
        metrics : dict
            total_loss, value_loss, policy_loss, accuracy.
        """
        device = self.config.device
        grid = batch["grid"].to(device)
        gvec = batch["global"].to(device)
        mask = batch["mask"].to(device)
        target_pi = batch["policy"].to(device)
        target_z = batch["outcome"].to(device)

        # Forward pass
        policy_logits, value = self.model(grid, gvec, mask)

        # ── Value loss: MSE(z, v) ─────────────────────────────────────
        value_loss = F.mse_loss(value, target_z)

        # ── Policy loss: cross-entropy H(π, p) = -Σ π(a) log p(a) ────
        # log_softmax over masked logits: illegal actions have -inf → nan*0 issue.
        # Fix: replace -inf with a large negative before log_softmax, or compute
        # CE only over legal actions.  We use a numerically stable approach:
        #   CE = -Σ π(a) * (logits[a] - logsumexp(logits))
        # where logsumexp is computed over the full (masked) logits.
        logits_stable = policy_logits.clone()
        logits_stable = logits_stable.masked_fill(
            mask == 0, -1e9  # replace -inf with very negative (not -inf)
        )
        log_sum_exp = torch.logsumexp(logits_stable, dim=-1, keepdim=True)
        log_probs = logits_stable - log_sum_exp  # (B, A)
        # Only legal actions contribute: target_pi = 0 for illegal actions
        policy_loss = -(target_pi * log_probs).sum(dim=-1).mean()

        # ── Total loss ────────────────────────────────────────────────
        total_loss = value_loss + policy_loss

        # ── Optimizer step ────────────────────────────────────────────
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._step += 1

        # ── Accuracy (for logging) ────────────────────────────────────
        with torch.no_grad():
            # Policy accuracy: does argmax(p) match argmax(π)?
            pred_action = policy_logits.argmax(dim=-1)
            target_action = target_pi.argmax(dim=-1)
            policy_acc = (pred_action == target_action).float().mean().item()

            # Value sign accuracy: does sign(v) match sign(z)?
            value_sign_acc = ((value.squeeze() * target_z.squeeze()) >= 0).float().mean().item()

        return {
            "total_loss": total_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "policy_acc": policy_acc,
            "value_sign_acc": value_sign_acc,
        }

    def train(
        self,
        buffer: ReplayBuffer,
        num_steps: int,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """Train for a fixed number of mini-batch updates.

        Parameters
        ----------
        buffer : ReplayBuffer
            Source of training examples.
        num_steps : int
            Number of mini-batch updates to perform.
        log_interval : int
            Print metrics every N steps.

        Returns
        -------
        avg_metrics : dict
            Averaged metrics over all steps.
        """
        metrics_sum = {"total_loss": 0.0, "value_loss": 0.0,
                       "policy_loss": 0.0, "policy_acc": 0.0,
                       "value_sign_acc": 0.0}

        self.model.train()
        t0 = time.time()

        for step in range(num_steps):
            batch = buffer.sample(self.config.batch_size)
            metrics = self.train_step(batch)

            for k in metrics_sum:
                metrics_sum[k] += metrics[k]

            if (step + 1) % log_interval == 0:
                elapsed = time.time() - t0
                steps_per_sec = (step + 1) / max(elapsed, 0.001)
                avg = {k: v / (step + 1) for k, v in metrics_sum.items()}
                print(f"  Train step {step + 1}/{num_steps} "
                      f"({steps_per_sec:.1f} steps/s): "
                      f"loss={avg['total_loss']:.4f}, "
                      f"v_loss={avg['value_loss']:.4f}, "
                      f"p_loss={avg['policy_loss']:.4f}, "
                      f"p_acc={avg['policy_acc']:.3f}, "
                      f"v_acc={avg['value_sign_acc']:.3f}")

        self.model.eval()

        n = max(num_steps, 1)
        return {k: v / n for k, v in metrics_sum.items()}

    @property
    def step(self) -> int:
        """Total number of train_step calls."""
        return self._step

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state."""
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
        }, path)

    def load_checkpoint(self, path: str) -> int:
        """Load model and optimizer state. Returns saved step."""
        ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._step = ckpt.get("step", 0)
        return self._step
