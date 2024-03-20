"""
The main RL training code.
"""
import math
import pickle as pkl
import random
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.utilities import grad_norm
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
from connectfour.mc_tree_search import mc_tree_search
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
        weight_decay,
        n_play_ahead_steps,
        val_batch_size,
        opponent_policy_net: nn.Module,
        **kwargs,
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
        self.swa = False
        # if self.swa:
        #     self.policy_net = AveragedModel(self._policy_net)
        #     self.value_net = AveragedModel(self._value_net)
        # else:
        #     self.policy_net = self._policy_net
        #     self.value_net = self._value_net
        device = next(self._policy_net.parameters()).device

        self.bgs = bgs
        self.val_bgs = MutableBatchGameState(**{**kwargs, "batch_size": val_batch_size})
        self.register_buffer("board_state", self.bgs._board_state)
        self.register_buffer("val_board_state", self.val_bgs._board_state)
        play_first = random.choice([True, False])
        if play_first:
            self.play_state = PlayState.X
            self.opponent_play_state = PlayState.O
        else:
            self.play_state = PlayState.O
            self.opponent_play_state = PlayState.X
            # Let the opponent move:
            board_state = self.bgs.cannonical_board_state.to(device=device)
            opponent_move = self.sample_move(self._policy_net, board_state=board_state)
            self.bgs.play_at(opponent_move)
        self.automatic_optimization = False
        self.swa_start = self.hparams.max_epochs // 10

    def save(self, log_path: Path):
        if log_path is None:
            log_path = Path(self.trainer.logger.log_dir)
        with (log_path / "model.pkl").open("wb") as f:
            pkl.dump(
                {
                    "model_state": self.state_dict(),
                    "model_hparams": self.hparams,
                },
                f,
            )

    @classmethod
    def load(cls, log_path: Union[Path, str]):
        model_file = Path(log_path) / "model.pkl"
        with open(model_file, "rb") as f:
            model_dict = pkl.load(f)
        full_model = cls(**model_dict["model_hparams"], opponent_policy_net=None)
        full_model.load_state_dict(model_dict["model_state"])
        return full_model

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

    def training_step(self, _):
        p_opt, v_opt = self.optimizers()
        # p_sch, v_sch = self.lr_schedulers()

        with torch.no_grad():
            tree_search = mc_tree_search(
                bgs=self.bgs,
                value_net=self._value_net,
                policy_net=self._policy_net,
                run_length=self.hparams.run_length,
                depth=8,
                breadth=30,
                discount=self.hparams["gamma"],
            )

        board_state = self.bgs.cannonical_board_state

        initial_value = self._value_net(board_state).flatten()
        # get the value loss:

        # The amortized reward = gamma*future_reward
        mc_value = tree_search["optimal_value"]
        value_loss = 0.5 * (self.hparams.gamma * mc_value - initial_value) ** 2

        # Optimize the value net:
        v_opt.zero_grad()
        self.manual_backward(value_loss.mean())
        total_val_norm = grad_norm(self._value_net, norm_type=2)["grad_2.0_norm_total"]
        embedding_val_norm = grad_norm(self._value_net.embedding, norm_type=2)[
            "grad_2.0_norm_total"
        ]
        nn.utils.clip_grad_norm_(self._value_net.parameters(), 1.0)
        v_opt.step()
        # if self.swa and self.current_epoch > self.swa_start:
        #     self.value_net.update_parameters(self._value_net)
        # v_sch.step()

        # compute the policy loss:
        mini_logits = self._policy_net(board_state)
        mc_move = tree_search["move"]
        policy_loss = nn.functional.cross_entropy(
            mini_logits, mc_move, reduction="none"
        )
        accuracy = (torch.argmax(mini_logits, dim=1) == mc_move).to(dtype=torch.float)

        # Compute the entropy:
        probs = torch.softmax(mini_logits, dim=1)
        entropy = (-probs * torch.log2(probs)).sum(dim=1)
        flat_entropy = math.log2(self.bgs._num_cols)

        # Optimize the policy net:
        p_opt.zero_grad()
        policy_loss_rw = (1 + mc_value**2).detach() * policy_loss
        self.manual_backward(policy_loss_rw.mean())
        total_policy_norm = grad_norm(self._policy_net, norm_type=2)[
            "grad_2.0_norm_total"
        ]
        embedding_pol_norm = grad_norm(self._value_net.embedding, norm_type=2)[
            "grad_2.0_norm_total"
        ]
        nn.utils.clip_grad_norm_(self._policy_net.parameters(), 1.0)
        p_opt.step()
        # if self.swa and self.current_epoch > self.swa_start:
        #     self.policy_net.update_parameters(self._policy_net)
        # p_sch.step()

        # Make the actual move:
        player = self.bgs.turn
        self.bgs.play_at(mc_move)

        # Now check if the game is over:

        winners = self.bgs.winners_numeric(run_length=self.hparams.run_length)
        # compute the rewards:
        def get_reward(winners, play_state: PlayState) -> torch.Tensor:
            opponent_play_state = (
                PlayState.O if play_state == PlayState.X else PlayState.X
            )
            return (winners == play_state_embedding_ix(play_state)).to(
                dtype=torch.float
            ) - (winners == play_state_embedding_ix(opponent_play_state)).to(
                dtype=torch.float
            )

        reward = get_reward(winners, player)
        # reset any dead games:
        resets = winners != play_state_embedding_ix(None)
        if resets.any() and self.current_epoch % 100 == 0:
            outs = self.bgs._board_state[resets, ...]
            with open(
                f"{self.trainer.logger.log_dir}/games_{self.current_epoch // 1000}.txt",
                "a",
            ) as f:
                f.write(f"\n\nPlaying as {player} on epoch {self.current_epoch}\n")
                f.write(str(MutableBatchGameState(state=outs)))

        self.bgs.reset_games(resets)

        self.logger.log_metrics(
            {
                "train_loss": (value_loss + policy_loss).mean(),
                "value_loss": value_loss.mean(),
                "policy_loss": policy_loss.mean(),
                "spread": (tree_search["counts"] != 0).to(dtype=torch.float).mean(),
                "reward": reward.mean(),
                "value": initial_value.mean(),
                "optimal_value": mc_value.mean(),
                "mc_val_err": ((reward - mc_value) ** 2).mean(),
                "mc_val_diff": ((reward - mc_value)).mean(),
                "accuracy": accuracy.mean(),
                "final_accuracy": accuracy[resets].mean(),
                "entropy": entropy.mean() - flat_entropy,
                "final_entropy": entropy[resets].mean() - flat_entropy,
                "total_val_norm": total_val_norm,
                "total_pol_norm": total_policy_norm,
                "embedding_val_norm": embedding_val_norm,
                "embedding_pol_norm": embedding_pol_norm,
            }
        )

    def validation_step(self, *args):
        self.save(None)
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
        # policy_scheduler = {
        #     "scheduler": OneCycleLR(  # ReduceLROnPlateau(optimizer=policy_optimizer),
        #         optimizer=policy_optimizer,
        #         max_lr=self.hparams.policy_lr * 10,
        #         steps_per_epoch=1,
        #         epochs=self.hparams.max_epochs,
        #     ),
        #     # SequentialLR(
        #     #     policy_optimizer,
        #     #     [
        #     #         # CosineAnnealingLR(policy_optimizer, T_max=self.hparams.max_epochs),
        #     #         ConstantLR(policy_optimizer),
        #     #         SWALR(
        #     #             policy_optimizer,
        #     #             swa_lr=self.hparams.policy_lr / 10,
        #     #             anneal_epochs=self.hparams.max_epochs // 10,
        #     #             anneal_strategy="cos",
        #     #         ),
        #     #     ],
        #     #     milestones=[self.swa_start + self.policy_start],
        #     # ),
        #     "interval": "epoch",
        # }

        value_optimizer = AdamW(
            list(self._value_net.parameters()) + list(self._embedding.parameters()),
            lr=self.hparams.value_lr,
            weight_decay=self.hparams.weight_decay,
        )
        # value_scheduler = {
        #     "scheduler": OneCycleLR(  # ReduceLROnPlateau(optimizer=value_optimizer),
        #         optimizer=value_optimizer,
        #         max_lr=self.hparams.value_lr * 10,
        #         steps_per_epoch=1,
        #         epochs=self.hparams.max_epochs,
        #     ),
        #     #     SequentialLR(
        #     #     value_optimizer,
        #     #     [
        #     #         # CosineAnnealingLR(value_optimizer, T_max=self.hparams.max_epochs),
        #     #         ConstantLR(value_optimizer),
        #     #         SWALR(
        #     #             value_optimizer,
        #     #             swa_lr=self.hparams.value_lr / 10,
        #     #             anneal_epochs=self.hparams.max_epochs // 10,
        #     #             anneal_strategy="cos",
        #     #         ),
        #     #     ],
        #     #     milestones=[self.swa_start],
        #     # ),
        #     "interval": "epoch",
        # }
        return [policy_optimizer, value_optimizer]  # , [
        #     policy_scheduler,
        #     value_scheduler,
        # ]
