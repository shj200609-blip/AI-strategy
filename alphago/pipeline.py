"""AlphaGo Pipeline — orchestrates the self-play → train → evaluate loop.

AlphaGo Zero training cycle:
  1. Self-Play:  generate games with current best network + MCTS
  2. Training:   update network on (s, π, z) from replay buffer
  3. Evaluation: pit new network vs best network
  4. Promote:    if new network wins ≥ 55%, it becomes the new best
"""

import os
import random
import time
from typing import List, Optional

import numpy as np
import torch

from ai_core.base import AIPlayer
from ai_core.model import BattleNet
from alphago.config import AlphaGoConfig
from alphago.mcts import MCTS, advance_to_next_unit
from alphago.replay_buffer import ReplayBuffer
from alphago.self_play import SelfPlayRunner
from alphago.trainer import AlphaGoTrainer
from alphago.player import MCTSAIPlayer
from alphago.opponent_pool import OpponentPool


class AlphaGoPipeline:
    """Orchestrator for the full AlphaGo Zero training cycle.

    Parameters
    ----------
    config : AlphaGoConfig
        All hyperparameters for the pipeline.
    """

    def __init__(self, config: AlphaGoConfig):
        self.config = config
        self.device = config.device

        # Best network — used for self-play data generation
        self.best_model = BattleNet().to(self.device)
        self.best_model.eval()

        # New network — trained on replay buffer, then pitted vs best
        self.new_model = BattleNet().to(self.device)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_capacity)

        # Trainer (created once — optimizer state persists across iterations)
        self.trainer = AlphaGoTrainer(self.new_model, self.config)

        # Opponent pool (prevents strategy collapse)
        pool_size = getattr(self.config, 'opponent_pool_size', 5)
        self.opponent_pool = OpponentPool(
            capacity=max(1, pool_size),
            save_dir=os.path.join(self.config.checkpoint_dir, "opponent_pool"),
        )
        # Restore any historical opponents left on disk so resumed runs
        # don't lose the diversity accumulated in earlier sessions.
        self.opponent_pool.load_from_disk()
        import sys
        print(f"[Pool] Init cap={pool_size} dir={self.opponent_pool._save_dir} "
              f"loaded={len(self.opponent_pool)}", flush=True, file=sys.stderr)

        # TensorBoard logger
        self._tb = None
        if config.tensorboard:
            from alphago.visualization import TensorBoardLogger
            self._tb = TensorBoardLogger(
                os.path.join(config.checkpoint_dir, "tensorboard")
            )
            print(f"[TensorBoard] logging to {config.checkpoint_dir}/tensorboard", flush=True)

        # Checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Stats + early stopping
        # ``_iteration`` = index of the LAST COMPLETED iteration (-1 = none).
        # Resuming reads this value and continues from ``_iteration + 1``;
        # a fresh run starts from 0.
        self._iteration = -1
        self._total_games = 0
        self._best_win_rate = 0.0
        self._stale_iters = 0

        # Rule-based opponent (ClassicAI). We hold one instance and reuse it
        # across games; battle runners reset its per-battle state through the
        # AIPlayer.battle_begins lifecycle hook.
        self._classic_ai = None
        if getattr(self.config, 'use_classic_opponent', False):
            from ai_core.classic_ai import ClassicAI
            self._classic_ai = ClassicAI(
                spellbook=None,  # use whatever the hero carries
                difficulty=self.config.classic_opponent_difficulty,
                randomize=self.config.classic_opponent_randomize,
                ranged_range=self.config.max_attack_range
                if hasattr(self.config, 'max_attack_range') else 5,
            )
            print(f"[ClassicOpponent] difficulty={self.config.classic_opponent_difficulty} "
                  f"randomize={self.config.classic_opponent_randomize}", flush=True)

    def _classic_opponent_ai(self):
        """Return the singleton ClassicAI (or None if disabled)."""
        return self._classic_ai

    # ── Main Loop ───────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full training pipeline."""
        print(f"=== AlphaGo Zero Battle AI Training ===", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"MCTS sims: {self.config.num_simulations}", flush=True)
        print(f"Games/iter: {self.config.games_per_iteration}", flush=True)
        print(f"Buffer cap: {self.config.buffer_capacity}", flush=True)
        print(f"Batch size: {self.config.batch_size}", flush=True)
        print(f"Checkpoints: {self.config.checkpoint_dir}/", flush=True)
        print()

        # ── Resume: start from the saved iteration if we loaded one ──
        # ``_iteration`` stores the index of the LAST completed iteration
        # (0 means none completed yet).  When a checkpoint is loaded, the
        # caller sets ``_iteration`` to that value; running from 0 would
        # cause checkpoint-file collisions (iter_0010.pt gets overwritten).
        start_iter = max(0, self._iteration + 1)
        if start_iter > 0:
            print(f"Resuming from iteration {start_iter + 1}/"
                  f"{self.config.num_iterations}", flush=True)
            print()

        for iteration in range(start_iter, self.config.num_iterations):
            self._iteration = iteration
            t_iter_start = time.time()
            print(f"{'='*60}", flush=True)
            print(f"Iteration {iteration + 1}/{self.config.num_iterations}", flush=True)
            print(f"{'='*60}", flush=True)

            # ── 1. Self-Play ──────────────────────────────────────────
            t_sp = time.time()
            seeds = [iteration * 10000 + i for i in range(self.config.games_per_iteration)]
            runner = SelfPlayRunner(self.best_model, self.config, self.device)

            # Decide the *opponent* for this iteration.
            #
            # Curriculum:
            #   - If ``use_classic_opponent`` and we're still inside
            #     ``classic_opponent_iters``, the team-1 opponent is the
            #     rule-based ClassicAI for 100% of games.  This gives the
            #     network a stable "beat the baseline" signal.
            #   - After that, we transition through a mix of
            #     ClassicAI + pool + self-play until the schedule decays.
            #   - Eventually we end up in pure self-play / pool mode.
            opponent_net = None
            opponent_ai = None
            classic_ratio = 0.0

            in_classic_phase = (
                self.config.use_classic_opponent
                and iteration < self.config.classic_opponent_iters
            )
            if in_classic_phase:
                # Rule-based opponent for the entire batch — but allow
                # light ClassicAI randomization so the learner sees varied
                # tie-break decisions.
                opponent_ai = self._classic_opponent_ai()
                classic_ratio = 1.0
                import sys
                print(f"  [Opponent] ClassicAI (rule-based) — "
                      f"iter {iteration + 1}/{self.config.classic_opponent_iters}",
                      flush=True, file=sys.stderr)
            elif self.config.use_classic_opponent and self.config.classic_opponent_decay > 0:
                # Transition phase: ratio decays from 1.0 to 0.0 over the
                # subsequent iterations.  After 1 / (1 - decay) iterations
                # the ratio is below 1%, then we drop the rule-based path.
                steps_in_decay = iteration - self.config.classic_opponent_iters
                classic_ratio = (self.config.classic_opponent_decay
                                 ** max(1, steps_in_decay))
                if classic_ratio > 0.01:
                    opponent_ai = self._classic_opponent_ai()

            # Sample neural opponent from pool (50% of non-classic games)
            if opponent_ai is None and self.opponent_pool is not None and len(self.opponent_pool) > 0:
                opp_state = self.opponent_pool.sample()
                if opp_state is not None:
                    opponent_net = BattleNet().to(self.device)
                    opponent_net.load_state_dict(opp_state)
                    opponent_net.eval()
                    import sys
                    print(f"  [Pool] Playing vs historical opponent (pool size={len(self.opponent_pool)})",
                          flush=True, file=sys.stderr)

            examples = runner.run_batch(
                seeds,
                opponent_network=opponent_net,
                opponent_ai=opponent_ai,
                classic_ai_ratio=classic_ratio,
            )
            self.replay_buffer.add(examples)
            self._total_games += self.config.games_per_iteration
            t_sp_elapsed = time.time() - t_sp
            print(f"[Self-Play] {len(examples)} examples from "
                  f"{self.config.games_per_iteration} games "
                  f"({t_sp_elapsed:.1f}s). "
                  f"Buffer: {len(self.replay_buffer):,}/{self.config.buffer_capacity:,}")
            if self._tb and self._tb.active:
                self._tb.log_metrics({
                    "examples": len(examples),
                    "buffer_size": len(self.replay_buffer),
                    "self_play_time_s": t_sp_elapsed,
                    "total_games": self._total_games,
                }, self._iteration, prefix="self_play")

            # ── 2. Training ──────────────────────────────────────────
            if self.replay_buffer.is_ready(self.config.min_buffer_size):
                t_train = time.time()

                # Continue training new_model (optimizer state persists)
                avg_metrics = self.trainer.train(
                    self.replay_buffer,
                    self.config.train_steps_per_iter,
                    log_interval=self.config.log_interval,
                )
                t_train_elapsed = time.time() - t_train
                print(f"[Train] {self.config.train_steps_per_iter} steps "
                      f"({t_train_elapsed:.1f}s): "
                      f"loss={avg_metrics['total_loss']:.4f}, "
                      f"v_loss={avg_metrics['value_loss']:.4f}, "
                      f"p_loss={avg_metrics['policy_loss']:.4f}, "
                      f"p_acc={avg_metrics['policy_acc']:.3f}, "
                      f"v_acc={avg_metrics['value_sign_acc']:.3f}")
                if self._tb and self._tb.active:
                    self._tb.log_metrics(avg_metrics, self._iteration, prefix="train")

                # ── 3. Evaluation (Pit) ───────────────────────────────
                t_eval = time.time()
                # While the model is still random/below the classic baseline
                # threshold, we evaluate against the rule-based ClassicAI to
                # get a meaningful "how far above random" signal.  Once we
                # cross that threshold we switch to the standard new-vs-best
                # pit.  This avoids the very-first iteration reporting a
                # random-vs-random ~50% that the trainer can't act on.
                if (self.config.use_classic_opponent
                        and self._iteration < self.config.classic_opponent_iters):
                    win_rate = self._pit_evaluate_vs_classic()
                else:
                    win_rate = self._pit_evaluate()
                t_eval_elapsed = time.time() - t_eval
                print(f"[Eval] {self.config.eval_games} games "
                      f"({t_eval_elapsed:.1f}s): "
                      f"new vs best = {win_rate:.1%}")
                if self._tb and self._tb.active:
                    self._tb.log_metrics({
                        "win_rate": win_rate,
                        "eval_time_s": t_eval_elapsed,
                    }, self._iteration, prefix="eval")

                # ── 4. Promote ────────────────────────────────────────
                if win_rate >= self.config.win_rate_threshold:
                    print(f"[Promote] New model wins ({win_rate:.1%} >= "
                          f"{self.config.win_rate_threshold:.1%})! "
                          f"Updating best model.")
                    self.best_model.load_state_dict(self.new_model.state_dict())
                    self._save_checkpoint(
                        self.best_model,
                        os.path.join(self.config.checkpoint_dir,
                                     self.config.best_model_name),
                    )
                else:
                    print(f"[Keep] New model loses ({win_rate:.1%} < "
                          f"{self.config.win_rate_threshold:.1%}). "
                          f"Keeping old best.")
                    # Reset new_model back to best so the next iteration's
                    # training starts from the current strongest model.
                    # This is the standard AlphaGo Zero "challenger"
                    # pattern: every iteration is a fresh attempt against
                    # the current best, not a continuation of a failed
                    # challenger.
                    self.new_model.load_state_dict(self.best_model.state_dict())

                # ── Opponent pool save (every 5 iters) ────────────────
                if (iteration + 1) % 5 == 0 and self.opponent_pool is not None:
                    import sys, os as _os
                    marker = _os.path.join(self.config.checkpoint_dir,
                                          f".pool_marker_{iteration+1}")
                    with open(marker, 'w') as f:
                        f.write(f'iter={iteration+1}\n')
                    print(f"[Pool] TRACE iter={iteration+1} marker={marker}",
                          flush=True, file=sys.stderr)
                    try:
                        sd = {k: v.cpu() for k, v in self.new_model.state_dict().items()}
                        self.opponent_pool.add(
                            sd,
                            step=(iteration + 1) * self.config.train_steps_per_iter,
                        )
                        print(f"[Pool] SAVED iter={iteration+1} size={len(self.opponent_pool)}",
                              flush=True, file=sys.stderr)
                    except Exception as e:
                        import traceback
                        print(f"[Pool] ERROR {e}", flush=True, file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)

                # ── Early stopping ──────────────────────────────────────
                if self.config.early_stop_patience > 0:
                    if win_rate > self._best_win_rate:
                        self._best_win_rate = win_rate
                        self._stale_iters = 0
                    else:
                        self._stale_iters += 1
                        if self._stale_iters >= self.config.early_stop_patience:
                            print(f"[EarlyStop] Win rate stagnant for "
                                  f"{self._stale_iters} iters, stopping.", flush=True)
                            final_path = os.path.join(self.config.checkpoint_dir, "final.pt")
                            self._save_checkpoint(self.best_model, final_path)
                            print(f"Best model saved: {final_path}", flush=True)
                            self._run_final_eval(final_path)
                            return
            else:
                print(f"[Train] Skipped — buffer size {len(self.replay_buffer):,} "
                      f"< min {self.config.min_buffer_size:,}")

            # ── Periodic checkpoint ──────────────────────────────────
            if (iteration + 1) % 10 == 0:
                path = os.path.join(
                    self.config.checkpoint_dir,
                    f"iter_{iteration + 1:04d}.pt",
                )
                self._save_checkpoint(self.best_model, path)
                print(f"[Checkpoint] Saved {path}", flush=True)

            t_iter_elapsed = time.time() - t_iter_start
            print(f"[Iteration {iteration + 1}] Total: {t_iter_elapsed:.1f}s", flush=True)
            print()

        # Final save
        final_path = os.path.join(self.config.checkpoint_dir, "final.pt")
        self._save_checkpoint(self.best_model, final_path)
        print(f"Training complete! Best model: {final_path}", flush=True)
        print(f"Total games played: {self._total_games:,}", flush=True)
        self._run_final_eval(final_path)

    # ── Pit Evaluation ──────────────────────────────────────────────

    def _run_final_eval(self, best_path: str) -> None:
        """After training finishes (or early-stops), play the saved
        ``best_model`` against the rule-based ClassicAI on a larger
        evaluation set and report the final win rate.

        The same evaluation harness used during the early curriculum
        phase is reused here, but with potentially more games
        (``final_eval_games``) for a tighter estimate.
        """
        if not getattr(self.config, "final_eval_enabled", True):
            print("[FinalEval] skipped (final_eval_enabled=False).", flush=True)
            return

        n = getattr(self.config, "final_eval_games", 0) or self.config.eval_games
        sims = getattr(
            self.config, "final_eval_mcts_simulations", 0
        ) or self.config.eval_mcts_simulations

        if n <= 0:
            print("[FinalEval] skipped (final_eval_games<=0).", flush=True)
            return

        from ai_core.classic_ai import ClassicAI

        # Swap in evaluation-mode MCTS settings without mutating the
        # training config (we still want best_model to be the saved one).
        original_sims = self.config.num_simulations
        self.config.num_simulations = sims
        try:
            player = MCTSAIPlayer.from_model(
                self.best_model, self.config, self.device,
            )
            classic = ClassicAI(
                spellbook=None,
                difficulty=self.config.classic_opponent_difficulty,
                randomize=self.config.classic_opponent_randomize,
                ranged_range=5,
            )

            wins = 0
            n_half = n // 2
            for i in range(n_half):
                cfg = random.choice(
                    self.config.eval_battle_configs or self.config.battle_configs
                )
                if self._play_pit_game_hetero(
                    cfg, player, classic, seed=i * 2, learner_team=0,
                ):
                    wins += 1
                if self._play_pit_game_hetero(
                    cfg, player, classic, seed=i * 2 + 1, learner_team=1,
                ):
                    wins += 1
            if n % 2 != 0:
                cfg = random.choice(
                    self.config.eval_battle_configs or self.config.battle_configs
                )
                if self._play_pit_game_hetero(
                    cfg, player, classic, seed=n, learner_team=0,
                ):
                    wins += 1

            win_rate = wins / n
        finally:
            self.config.num_simulations = original_sims

        print(f"[FinalEval] best_model vs ClassicAI: {wins}/{n} "
              f"= {win_rate:.1%} win rate "
              f"(mcts_sims={sims}, n={n})", flush=True)
        if self._tb and self._tb.active:
            self._tb.log_metrics(
                {"win_rate": win_rate, "n_games": n, "mcts_sims": sims},
                self._iteration, prefix="final_eval",
            )

        # Persist a small JSON summary next to the checkpoint so external
        # tools / scripts can read the final result without parsing logs.
        summary_path = os.path.join(
            self.config.checkpoint_dir, "final_eval.json",
        )
        try:
            import json
            with open(summary_path, "w") as f:
                json.dump({
                    "best_model": os.path.basename(best_path),
                    "win_rate_vs_classic": win_rate,
                    "wins": wins,
                    "n_games": n,
                    "mcts_simulations": sims,
                    "total_iterations": self._iteration + 1,
                    "total_games": self._total_games,
                }, f, indent=2)
            print(f"[FinalEval] summary saved: {summary_path}", flush=True)
        except Exception as e:  # pragma: no cover - best-effort write
            print(f"[FinalEval] WARNING failed to write summary: {e}",
                  flush=True)

    def _pit_evaluate_vs_classic(self) -> float:
        """Pit new_model against the rule-based ClassicAI opponent.

        During the first few iterations, this gives a much more useful
        signal than new-vs-best (which is new-vs-random for iter 0).
        Win rate is reported as fraction of games the network wins.

        We alternate teams to cancel first-move bias.
        """
        from ai_core.classic_ai import ClassicAI
        n_games = self.config.eval_games
        if n_games <= 0:
            return 0.0
        n_half = n_games // 2

        eval_config = AlphaGoConfig(
            num_simulations=self.config.eval_mcts_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            device=self.config.device,
            max_moves_per_game=self.config.max_moves_per_game,
            battle_configs=self.config.battle_configs,
        )
        player_net = MCTSAIPlayer.from_model(
            self.new_model, eval_config, self.device,
        )
        classic = ClassicAI(
            spellbook=None,
            difficulty=self.config.classic_opponent_difficulty,
            randomize=self.config.classic_opponent_randomize,
            ranged_range=5,
        )

        wins = 0
        for i in range(n_half):
            cfg = random.choice(self.config.eval_battle_configs
                                or self.config.battle_configs)
            # Game 1: net = team 0, classic = team 1
            if self._play_pit_game_hetero(cfg, player_net, classic,
                                           seed=i * 2, learner_team=0):
                wins += 1
            # Game 2: classic = team 0, net = team 1
            if self._play_pit_game_hetero(cfg, player_net, classic,
                                           seed=i * 2 + 1, learner_team=1):
                wins += 1

        if n_games % 2 != 0:
            cfg = random.choice(self.config.eval_battle_configs
                                or self.config.battle_configs)
            if self._play_pit_game_hetero(cfg, player_net, classic,
                                           seed=n_games, learner_team=0):
                wins += 1

        return wins / n_games

    @staticmethod
    def _play_pit_game_hetero(
        battle_config: dict,
        player_net: "MCTSAIPlayer",
        player_rule: "AIPlayer",
        seed: int,
        learner_team: int,
    ) -> bool:
        """One pit game between an MCTS player and a rule-based player.

        ``learner_team`` is which team the *learner* plays; we report
        True if that team wins.
        """
        from alphago.self_play import _build_battle, _determine_winner_by_strength
        from ai_core.action_space import index_to_action

        random.seed(seed)
        np.random.seed(seed)

        battle = _build_battle(battle_config)
        player_net.battle_begins()
        player_rule.battle_begins()
        current_unit = advance_to_next_unit(battle)

        max_moves = 200
        step = 0

        while not battle.is_over() and current_unit is not None and step < max_moves:
            if current_unit.team == learner_team:
                action, _ = player_net.decide(battle, current_unit)
            else:
                action, _ = player_rule.decide(battle, current_unit)
            current_unit._acted = True
            battle.execute(action)

            step += 1
            current_unit = advance_to_next_unit(battle)

        if battle.is_over():
            winner = battle.winner()
        else:
            winner = _determine_winner_by_strength(battle)

        return winner == learner_team

    def _pit_evaluate(self) -> float:
        """Pit new_model vs best_model.

        Both use MCTS with the evaluation number of simulations.
        Teams alternate to cancel first-move bias.

        Returns
        -------
        win_rate : float
            New model's win rate against the best model.
        """
        n_games = self.config.eval_games
        if n_games <= 0:
            return 0.0
        n_half = n_games // 2

        # Configure for evaluation (potentially fewer sims; no Dirichlet
        # noise — evaluation must reflect the network's true strength).
        eval_config = AlphaGoConfig(
            num_simulations=self.config.eval_mcts_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=0.0,
            dirichlet_epsilon=0.0,
            device=self.config.device,
            max_moves_per_game=self.config.max_moves_per_game,
            battle_configs=self.config.battle_configs,
        )

        player_new = MCTSAIPlayer.from_model(
            self.new_model, eval_config, self.device,
        )
        player_best = MCTSAIPlayer.from_model(
            self.best_model, eval_config, self.device,
        )

        wins_new = 0

        for i in range(n_half):
            cfg = random.choice(self.config.eval_battle_configs or self.config.battle_configs)

            # Game 1: new = team 0, best = team 1
            if self._play_pit_game(cfg, player_new, player_best, seed=i * 2, new_team=0):
                wins_new += 1

            # Game 2: new = team 1, best = team 0
            if self._play_pit_game(cfg, player_best, player_new, seed=i * 2 + 1, new_team=1):
                wins_new += 1

        # Handle odd number
        if n_games % 2 != 0:
            cfg = random.choice(self.config.eval_battle_configs or self.config.battle_configs)
            if self._play_pit_game(cfg, player_new, player_best, seed=n_games, new_team=0):
                wins_new += 1

        return wins_new / n_games

    @staticmethod
    def _play_pit_game(
        battle_config: dict,
        player_a: MCTSAIPlayer,
        player_b: MCTSAIPlayer,
        seed: int,
        new_team: int,
    ) -> bool:
        """Play one pit game. Returns True if the 'new' model wins.

        Parameters
        ----------
        player_a : plays as team 0
        player_b : plays as team 1
        new_team : which team the new model is playing (0 or 1)
        """
        from alphago.self_play import _build_battle
        from ai_core.action_space import index_to_action

        random.seed(seed)
        np.random.seed(seed)

        battle = _build_battle(battle_config)
        player_a.battle_begins()
        player_b.battle_begins()
        current_unit = advance_to_next_unit(battle)

        max_moves = 200
        step = 0

        while not battle.is_over() and current_unit is not None and step < max_moves:
            team = current_unit.team
            player = player_a if team == 0 else player_b

            action, _ = player.decide(battle, current_unit)
            current_unit._acted = True  # 标记已行动
            battle.execute(action)

            step += 1
            current_unit = advance_to_next_unit(battle)

        if battle.is_over():
            winner = battle.winner()
        else:
            # Determine by strength
            from alphago.self_play import _determine_winner_by_strength
            winner = _determine_winner_by_strength(battle)

        return winner == new_team

    # ── Checkpoint Helpers ───────────────────────────────────────────

    def _save_checkpoint(self, model: BattleNet, path: str) -> None:
        """Save model + optimizer state to disk."""
        torch.save({
            "model": model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict() if self.trainer else {},
            "iteration": self._iteration,
            "total_games": self._total_games,
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load a saved checkpoint, restoring best_model, new_model, and optimizer."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.best_model.load_state_dict(ckpt["model"])
        self.new_model.load_state_dict(ckpt["model"])
        self.best_model.to(self.device).eval()
        self.new_model.to(self.device)
        # Restore optimizer if present
        if "optimizer" in ckpt and ckpt["optimizer"]:
            self.trainer.optimizer.load_state_dict(ckpt["optimizer"])
        # ``_iteration`` records the last COMPLETED iteration.  Default
        # ``-1`` (no completed iterations) so a fresh ``run()`` starts
        # at iteration 0; a restored checkpoint skips the iterations
        # already represented by the saved value.
        self._iteration = ckpt.get("iteration", -1)
        self._total_games = ckpt.get("total_games", 0)
        # Refresh opponent pool from disk — without this, a resumed run
        # would only ever see one opponent (best_model) for the rest of
        # training, which biases the policy.
        if self.opponent_pool is not None:
            self.opponent_pool.load_from_disk()
        print(f"Loaded checkpoint: {path}", flush=True)
        print(f"  Iteration: {self._iteration}, Games: {self._total_games:,}", flush=True)
