"""
The main RL training code.
"""
import collections
from typing import Dict, Iterator, List, Tuple

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    SequentialLR,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader, IterableDataset

from connectfour.embedding_net import EmbeddingNet
from connectfour.game import MutableBatchGameState
from connectfour.minimax import MiniMaxPolicyCorrector, minimax
from connectfour.nn import sample_masked_multinomial
from connectfour.play_state import PlayState, play_state_embedding_ix
from connectfour.policy import PolicyNet
from connectfour.value_net import ValueNet


class Placeholder(IterableDataset):
    """Iterable Dataset of game states.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        yield 1


def sample_move(
    bgs: MutableBatchGameState, policy_net: PolicyNet, board_state=None
) -> torch.Tensor:
    # board_state = self.bgs.cannonical_board_state
    if board_state is None:
        board_state = bgs.cannonical_board_state
    logits = policy_net(board_state)

    blank_ix = play_state_embedding_ix(PlayState.BLANK)
    blank_space_indicator = board_state[:, blank_ix, :, :]
    num_blank_spaces = torch.sum(blank_space_indicator, dim=1)
    actions = num_blank_spaces != 0

    mask = ~actions
    moves = sample_masked_multinomial(logits, mask, axis=1)
    return moves


class ConnectFourAI(pl.LightningModule):
    def __init__(
        self,
        policy_net_kwargs: dict,
        value_net_kwargs: dict,
        embedding_net_kwargs: dict,
        policy_lr,
        value_lr,
        gamma,
        run_length,
        max_epochs,
        value_net_burn_in_frac,
        weight_decay,
        n_play_ahead_steps,
        bootstrap_threshold,
        val_batch_size,
        opponent_policy_net: nn.Module,
        minimax_depth: int = 4,
        minimax_target=True,
        ce_loss_strength=1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["policy_net", "value_net", "opponent_policy_net"]
        )
        bgs = MutableBatchGameState(**kwargs)
        out_channels = embedding_net_kwargs["latent_dim"]
        kernel_size = embedding_net_kwargs["kernel_size"]
        depth = embedding_net_kwargs["depth"]
        self._embedding = EmbeddingNet(
            kernel_size=kernel_size, out_channels=out_channels, depth=depth
        )
        self._policy_net = PolicyNet(embedding=self._embedding, **policy_net_kwargs)
        self._value_net = ValueNet(embedding=self._embedding, **value_net_kwargs)
        self.policy_net = AveragedModel(self._policy_net)
        self.value_net = AveragedModel(self._value_net)
        if opponent_policy_net is None:
            opponent_policy_net = self.policy_net
        self.opponent_policy_net = opponent_policy_net
        device = next(opponent_policy_net.parameters()).device
        if minimax_depth > 0:
            self.minimax_corrector = MiniMaxPolicyCorrector(
                num_rows=kwargs["num_rows"],
                num_cols=kwargs["num_cols"],
                run_length=run_length,
                depth=4,
            )
        self.bgs = bgs
        self.val_bgs = MutableBatchGameState(**{**kwargs, "batch_size": val_batch_size})
        self.register_buffer("board_state", self.bgs._board_state)
        self.register_buffer("val_board_state", self.val_bgs._board_state)
        play_first = False
        if play_first:
            self.play_state = PlayState.X
            self.opponent_play_state = PlayState.O
        else:
            self.play_state = PlayState.O
            self.opponent_play_state = PlayState.X
            # Let the opponent move:
            board_state = self.bgs.cannonical_board_state.to(device=device)
            opponent_move = self.sample_move(
                self.call_opponent, board_state=board_state
            )
            self.bgs.play_at(opponent_move)
        self.automatic_optimization = False
        self.swa_start = self.hparams.max_epochs // 10
        self.policy_start = int(
            self.hparams.max_epochs * self.hparams.value_net_burn_in_frac
        )
        self.history = collections.deque()

    def call_opponent(self, x: torch.Tensor):
        if hasattr(self, "minimax_corrector"):
            return self.minimax_corrector(x, self.opponent_policy_net)
        else:
            return self.opponent_policy_net(x)

    def sample_move(self, policy_net, board_state) -> torch.Tensor:
        return sample_move(self.bgs, policy_net, board_state)

    def get_reward(self, winners):
        return (winners == play_state_embedding_ix(self.play_state)).to(
            dtype=torch.float
        ) - (winners == play_state_embedding_ix(self.opponent_play_state)).to(
            dtype=torch.float
        )

    def get_win_count(self, winners):
        return (winners == play_state_embedding_ix(self.play_state)).to(
            dtype=torch.float
        ) + (winners == play_state_embedding_ix(self.opponent_play_state)).to(
            dtype=torch.float
        )

    def take_composite_move_and_get_reward_delta(
        self, board_state
    ) -> Dict[str, torch.Tensor]:
        """
        Both the player and the opponent take a move. The state is updated and
        the (adjusted) reward:
            R +\gamma v(new_state) - v(old_state)
        is returned
        """

        # Choose a move:
        move = self.sample_move(self._policy_net, board_state=board_state)

        # Make the play:
        self.bgs.play_at(move)
        mid_board_state = self.bgs.cannonical_board_state

        # Now check if the game is over:

        winners = self.bgs.winners_numeric(run_length=self.hparams["run_length"])
        # compute the rewards:
        reward = self.get_reward(winners)

        n_winners = self.get_win_count(winners)
        # reset any dead games:
        first_resets = winners != play_state_embedding_ix(None)

        # Let the opponent move:
        opponent_move = self.sample_move(
            self.call_opponent, board_state=mid_board_state
        )
        self.bgs.play_at(opponent_move, first_resets)
        final_board_state = self.bgs.cannonical_board_state

        # Now check if the game is over:
        winners = self.bgs.winners_numeric(run_length=self.hparams["run_length"])
        # compute the rewards:
        reward += self.get_reward(winners)
        n_winners += self.get_win_count(winners)
        # reset any dead games:
        second_resets = winners != play_state_embedding_ix(None)

        # Get the output_value, ignoring any resets:
        final_value = self._value_net(final_board_state).flatten()

        # Finally, reset any dead games:
        self.bgs.reset_games(second_resets)

        return {
            "move": move,
            "reward": reward,
            "resets": first_resets | second_resets,
            "final_value": final_value,
            "n_winners": n_winners,
        }

    def training_step(self, _):
        p_opt, v_opt = self.optimizers()
        p_sch, v_sch = self.lr_schedulers()

        while len(self.history) < self.hparams.n_play_ahead_steps:
            board_state = self.bgs.cannonical_board_state.clone()
            with (torch.no_grad()):
                # compute the delta:
                state_updates = self.take_composite_move_and_get_reward_delta(
                    board_state
                )

            self.history.append({**state_updates, "board_state": board_state})

        # Compute the n-step-reward:
        reward = torch.zeros(
            board_state.shape[0], device=board_state.device, dtype=torch.float
        )
        any_reset = torch.zeros(
            board_state.shape[0], device=board_state.device, dtype=torch.bool
        )
        for i, val_dict in enumerate(self.history):
            next_reward = val_dict["reward"]
            # Only accumulate rewards up to a reset of the board state.
            reward += torch.where(
                any_reset,
                torch.zeros_like(reward),
                self.hparams["gamma"] ** i * next_reward,
            )
            any_reset |= val_dict["resets"]
        final_value = self.history[-1]["final_value"]
        amortized_reward = reward + torch.where(
            any_reset,
            torch.zeros_like(reward),
            final_value * self.hparams["gamma"] ** (i + 1),
        )
        assert (
            amortized_reward.abs() <= 1
        ).all(), "Amortized reward should be bounded by 1: it's either win or loose."

        # Now get the initial values to compute the gradient
        val_dict = self.history.popleft()
        board_state = val_dict["board_state"]
        move = val_dict["move"]
        n_winners = val_dict["n_winners"]

        initial_value = self._value_net(board_state).flatten()
        smooth_delta = amortized_reward - initial_value.detach()

        temp = 1
        # get the value loss:

        # The amortized reward = gamma*future_reward
        value_loss = 0.5 * (amortized_reward - initial_value) ** 2
        # We compute re-weighted gradient decent loss, as in https://arxiv.org/pdf/2306.09222.pdf
        value_loss_rew = value_loss  # * torch.exp(
        #     torch.clamp(value_loss.detach(), max=temp) / (temp + 1)
        # )

        # Optimize the value net:
        v_opt.zero_grad()
        self.manual_backward(value_loss_rew.mean())
        nn.utils.clip_grad_norm_(self._value_net.parameters(), 1.0)
        v_opt.step()
        if self.current_epoch > self.swa_start:
            self.value_net.update_parameters(self._value_net)
        v_sch.step(value_loss_rew.detach().mean())

        # compute the policy loss:
        full_logits = self._policy_net(board_state)
        move_logits = torch.take_along_dim(full_logits, move.reshape([-1, 1]), dim=1)
        policy_loss = -reward * move_logits.detach()
        policy_loss_smooth = -smooth_delta * move_logits
        # We compute re-weighted gradient decent loss, as in https://arxiv.org/pdf/2306.09222.pdf
        policy_loss_rew = policy_loss_smooth  # * torch.exp(
        #     torch.clamp(policy_loss_smooth.detach(), max=temp) / (temp + 1)
        # )
        total_policy_loss = policy_loss_rew

        ce_loss = 0
        if self.hparams.minimax_target and hasattr(self, "minimax_corrector"):
            mini_logits = self.minimax_corrector(board_state, self.policy_net)
            target = torch.argmax(mini_logits, dim=1)
            ce_loss = nn.functional.cross_entropy(
                full_logits, target.detach(), reduction="none"
            )
            print("using ce loss")
            total_policy_loss += self.hparams.ce_loss_strength * ce_loss

        if self.current_epoch > self.policy_start:
            # Optimize the policy net:
            p_opt.zero_grad()
            self.manual_backward(total_policy_loss.mean())
            nn.utils.clip_grad_norm_(self._policy_net.parameters(), 1.0)
            p_opt.step()
            if self.current_epoch > self.swa_start + self.policy_start:
                self.policy_net.update_parameters(self._policy_net)
            p_sch.step(total_policy_loss.detach().mean())

        self.logger.log_metrics(
            {
                "train_loss": (value_loss + total_policy_loss).mean(),
                "reward": reward.mean(),
                "avg_finishes": n_winners.mean(),
                "value_loss": value_loss.mean(),
                "value_loss_smooth": 0.5 * (smooth_delta**2).mean(),
                "policy_loss_smooth": policy_loss_smooth.mean(),
                "ce_loss": ce_loss.mean(),
                "policy_loss": policy_loss.mean(),
                "bootstrapped": (value_loss < self.hparams.bootstrap_threshold)
                .to(dtype=torch.float)
                .mean(),
                "good_delta_sign": ((smooth_delta * amortized_reward) >= 0)
                .to(dtype=torch.float)
                .mean(),
            }
        )

        del board_state
        del move
        del n_winners
        del val_dict["final_value"]

    def validation_step(self, *args):
        metrics = {"val_reward": [], "val_avg_finishes": [], "val_policy_loss": []}
        with (torch.no_grad()):
            for _ in range(self.hparams.n_play_ahead_steps):
                board_state = self.val_bgs.cannonical_board_state

                # Choose a move:
                move = self.sample_move(self._policy_net, board_state=board_state)
                move_logits = torch.take_along_dim(
                    self._policy_net(board_state), move.reshape([-1, 1]), dim=1
                )

                # Make the play:
                self.val_bgs.play_at(move)
                mid_board_state = self.val_bgs.cannonical_board_state

                # Now check if the game is over:

                winners = self.val_bgs.winners_numeric(
                    run_length=self.hparams["run_length"]
                )
                # compute the rewards:
                reward = self.get_reward(winners)
                n_winners = self.get_win_count(winners)
                # reset any dead games:
                resets = winners != play_state_embedding_ix(None)

                # Let a random opponent move:
                blank_ix = play_state_embedding_ix(PlayState.BLANK)
                blank_space_indicator = mid_board_state[:, blank_ix, :, :]
                num_blank_spaces = torch.sum(blank_space_indicator, dim=1)
                actions = num_blank_spaces != 0

                mask = ~actions
                opponent_move = sample_masked_multinomial(
                    torch.zeros_like(mask, dtype=torch.float), mask, axis=1
                )
                self.val_bgs.play_at(opponent_move, resets)

                # Now check if the game is over:
                winners = self.val_bgs.winners_numeric(
                    run_length=self.hparams["run_length"]
                )
                # compute the rewards:
                reward += self.get_reward(winners)
                n_winners += self.get_win_count(winners)

                # reset any dead games:
                resets = winners != play_state_embedding_ix(None)
                self.val_bgs.reset_games(resets)

                policy_loss = -reward * move_logits.detach()

                metrics["val_reward"].append(reward.mean())
                metrics["val_avg_finishes"].append(n_winners.mean())
                metrics["val_policy_loss"].append(policy_loss.mean())

        self.logger.log_metrics(
            {k: np.mean([v.cpu().numpy() for v in vals]) for k, vals in metrics.items()}
        )

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = Placeholder()
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def val_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        policy_optimizer = AdamW(
            list(self._policy_net.parameters()) + list(self._embedding.parameters()),
            lr=self.hparams.policy_lr,
            weight_decay=self.hparams.weight_decay,
        )
        policy_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer=policy_optimizer),
            # OneCycleLR(
            #     optimizer=policy_optimizer,
            #     max_lr=self.hparams.policy_lr * 10,
            #     steps_per_epoch=1,
            #     epochs=self.hparams.max_epochs,
            # ),
            # SequentialLR(
            #     policy_optimizer,
            #     [
            #         # CosineAnnealingLR(policy_optimizer, T_max=self.hparams.max_epochs),
            #         ConstantLR(policy_optimizer),
            #         SWALR(
            #             policy_optimizer,
            #             swa_lr=self.hparams.policy_lr / 10,
            #             anneal_epochs=self.hparams.max_epochs // 10,
            #             anneal_strategy="cos",
            #         ),
            #     ],
            #     milestones=[self.swa_start + self.policy_start],
            # ),
            "interval": "epoch",
        }

        value_optimizer = AdamW(
            list(self._value_net.parameters()) + list(self._embedding.parameters()),
            lr=self.hparams.value_lr,
            weight_decay=self.hparams.weight_decay,
        )
        value_scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer=value_optimizer),
            # OneCycleLR(
            #     optimizer=value_optimizer,
            #     max_lr=self.hparams.value_lr * 10,
            #     steps_per_epoch=1,
            #     epochs=self.hparams.max_epochs,
            # ),
            #     SequentialLR(
            #     value_optimizer,
            #     [
            #         # CosineAnnealingLR(value_optimizer, T_max=self.hparams.max_epochs),
            #         ConstantLR(value_optimizer),
            #         SWALR(
            #             value_optimizer,
            #             swa_lr=self.hparams.value_lr / 10,
            #             anneal_epochs=self.hparams.max_epochs // 10,
            #             anneal_strategy="cos",
            #         ),
            #     ],
            #     milestones=[self.swa_start],
            # ),
            "interval": "epoch",
        }
        return [policy_optimizer, value_optimizer], [
            policy_scheduler,
            value_scheduler,
        ]
