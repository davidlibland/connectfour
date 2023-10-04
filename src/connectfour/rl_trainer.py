"""
The main RL training code.
"""
from typing import List, Tuple, Iterator, Dict

import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.utils.data import IterableDataset, DataLoader

from torch.optim.swa_utils import AveragedModel
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, ConstantLR
from torch.optim.swa_utils import SWALR


from connectfour.nn import sample_masked_multinomial

import lightning.pytorch as pl

from connectfour.game import MutableBatchGameState
from connectfour.play_state import PlayState, play_state_embedding_ix
from connectfour.policy import PolicyNet
from connectfour.value_net import ValueNet
from connectfour.embedding_net import EmbeddingNet


class RLDataset(IterableDataset):
    """Iterable Dataset of game states.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, bgs: MutableBatchGameState) -> None:
        self.bgs = bgs

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        board_state = self.bgs.cannonical_board_state
        for i in range(board_state.shape[0]):
            yield board_state[i, ...], play_state_embedding_ix(self.bgs.turn)


def sample_move(
    bgs: MutableBatchGameState, policy_net: PolicyNet, board_state=None
) -> torch.Tensor:
    # board_state = self.bgs.cannonical_board_state
    if board_state is None:
        board_state = bgs.cannonical_board_state.to(
            device=next(policy_net.parameters()).device
        )
    logits = policy_net(board_state.to(dtype=torch.float))
    mask = ~bgs.next_actions().to(device=logits.device)
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
        opponent_policy_net: nn.Module,
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
            self.opponent_policy_net = self.policy_net
        else:
            self.opponent_policy_net = opponent_policy_net
        self.bgs = bgs
        play_first = False
        if play_first:
            self.play_state = PlayState.X
            self.opponent_play_state = PlayState.O
        else:
            self.play_state = PlayState.O
            self.opponent_play_state = PlayState.X
            # Let the opponent move:
            board_state = self.bgs.cannonical_board_state.to(
                next(self.opponent_policy_net.parameters()).device
            )
            opponent_move = self.sample_move(
                self.opponent_policy_net, board_state=board_state
            )
            self.bgs.play_at(opponent_move)
        self.automatic_optimization = False
        self.swa_start = self.hparams.max_epochs // 10
        self.policy_start = int(
            self.hparams.max_epochs * self.hparams.value_net_burn_in_frac
        )

    def sample_move(self, policy_net, board_state) -> torch.Tensor:
        return sample_move(self.bgs, policy_net, board_state)

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
        mid_board_state = self.bgs.cannonical_board_state.to(board_state)

        # Now check if the game is over:
        def get_reward(win_state):
            if win_state == self.play_state:
                return 1
            if win_state == self.opponent_play_state:
                return -1
            return 0

        def get_win_count(win_state):
            if win_state == self.play_state:
                return 1
            if win_state == self.opponent_play_state:
                return 1
            return 0

        winners = self.bgs.winners(run_length=self.hparams["run_length"])
        # compute the rewards:
        reward = torch.Tensor([get_reward(win_state) for win_state in winners]).to(
            board_state
        )
        n_winners = torch.Tensor([get_win_count(win_state) for win_state in winners])
        # reset any dead games:
        resets = [win_state is not None for win_state in winners]

        # Let the opponent move:
        opponent_move = self.sample_move(
            self.opponent_policy_net, board_state=mid_board_state
        )
        self.bgs.play_at(opponent_move, resets)
        final_board_state = self.bgs.cannonical_board_state.to(board_state)

        # Now check if the game is over:
        winners = self.bgs.winners(run_length=self.hparams["run_length"])
        # compute the rewards:
        reward += torch.Tensor([get_reward(win_state) for win_state in winners]).to(
            board_state
        )
        n_winners += torch.Tensor([get_win_count(win_state) for win_state in winners])
        # reset any dead games:
        resets = [win_state is not None for win_state in winners]

        # Get the output_value, ignoring any resets:
        final_value_smooth = self.value_net(final_board_state).flatten()
        final_value = self._value_net(final_board_state).flatten()
        reset_masks = torch.Tensor(resets).to(
            device=final_value.device, dtype=torch.bool
        )
        zeros = torch.zeros_like(final_value)
        final_value = torch.where(reset_masks, zeros, final_value)
        final_value_smooth = torch.where(reset_masks, zeros, final_value_smooth)

        # Ok. Compute and return the delta:
        assert (
            ~(reward != 0) | (final_value == 0)
        ).all(), "reward != 0 -> final_value == 0"
        assert (
            ~(reward != 0) | (final_value_smooth == 0)
        ).all(), "reward != 0 -> final_value == 0"

        # Finally, reset any dead games:
        self.bgs.reset_games(reset_masks)
        amortized_reward = reward + final_value * self.hparams["gamma"]
        smooth_delta = (
            reward
            + final_value_smooth * self.hparams["gamma"]
            - self.value_net(board_state).flatten()
        )

        return {
            "move": move,
            "reward": reward,
            "amortized_reward": amortized_reward,
            "smooth_delta": smooth_delta,
            "n_winners": n_winners,
        }

    def eval_step(self, batch):
        board_state, turn = batch
        board_state = board_state.to(dtype=torch.float)

        with torch.no_grad():
            # compute the delta:
            state_updates = self.take_composite_move_and_get_reward_delta(board_state)

        return state_updates["reward"].mean()

    def training_step(self, batch):
        p_opt, v_opt = self.optimizers()
        p_sch, v_sch = self.lr_schedulers()

        board_state, turn = batch
        board_state = board_state.to(dtype=torch.float)
        initial_reward = self._value_net(board_state).flatten()

        with torch.no_grad():
            # compute the delta:
            state_updates = self.take_composite_move_and_get_reward_delta(board_state)
            move = state_updates["move"]
            amortized_reward = state_updates["amortized_reward"]
            smooth_delta = state_updates["smooth_delta"]
            true_reward = state_updates["reward"]

            # compute the gammas:
            I = torch.pow(
                torch.tensor(self.hparams["gamma"]).to(board_state),
                board_state[:, 1:, :, :].sum(dim=(1, 2, 3)),
            )

        temp = 1
        # get the value loss:

        # The amortized reward = gamma*future_reward
        value_loss = 0.5 * (amortized_reward - initial_reward) ** 2
        # We compute re-weighted gradient decent loss, as in https://arxiv.org/pdf/2306.09222.pdf
        value_loss_rew = value_loss * torch.exp(
            torch.clamp(value_loss.detach(), max=temp) / (temp + 1)
        )

        # Optimize the value net:
        v_opt.zero_grad()
        self.manual_backward(value_loss_rew.mean())
        nn.utils.clip_grad_norm_(self._value_net.parameters(), 1.0)
        v_opt.step()
        if self.current_epoch > self.swa_start:
            self.value_net.update_parameters(self._value_net)
        v_sch.step()

        # compute the policy loss:
        move_logits = torch.diag(self._policy_net(board_state)[:, move])
        policy_loss = -I * true_reward * move_logits.detach()
        policy_loss_smooth = -I * smooth_delta.detach() * move_logits
        # We compute re-weighted gradient decent loss, as in https://arxiv.org/pdf/2306.09222.pdf
        policy_loss_rew = policy_loss_smooth * torch.exp(
            torch.clamp(policy_loss_smooth.detach(), max=temp) / (temp + 1)
        )

        if self.current_epoch > self.policy_start:
            # Optimize the policy net:
            p_opt.zero_grad()
            self.manual_backward(policy_loss_rew.mean())
            nn.utils.clip_grad_norm_(self._policy_net.parameters(), 1.0)
            p_opt.step()
            if self.current_epoch > self.swa_start + self.policy_start:
                self.policy_net.update_parameters(self._policy_net)
            p_sch.step()

        self.logger.log_metrics(
            {
                "train_loss": (value_loss + policy_loss_smooth).mean(),
                "reward": true_reward.mean(),
                "avg_finishes": state_updates["n_winners"].mean(),
                "value_loss": value_loss.mean(),
                "value_loss_smooth": 0.5 * (smooth_delta**2).mean(),
                "policy_loss_smooth": policy_loss_smooth.mean(),
                "policy_loss": policy_loss.mean(),
            }
        )

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.bgs)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.bgs.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
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
            "scheduler": SequentialLR(
                policy_optimizer,
                [
                    # CosineAnnealingLR(policy_optimizer, T_max=self.hparams.max_epochs),
                    ConstantLR(policy_optimizer),
                    SWALR(policy_optimizer, swa_lr=0.05),
                ],
                milestones=[self.swa_start + self.policy_start],
            ),
            "interval": "epoch",
        }

        value_optimizer = AdamW(
            list(self._value_net.parameters()) + list(self._embedding.parameters()),
            lr=self.hparams.value_lr,
            weight_decay=self.hparams.weight_decay,
        )
        value_scheduler = {
            "scheduler": SequentialLR(
                value_optimizer,
                [
                    # CosineAnnealingLR(value_optimizer, T_max=self.hparams.max_epochs),
                    ConstantLR(value_optimizer),
                    SWALR(value_optimizer, swa_lr=0.05),
                ],
                milestones=[self.swa_start],
            ),
            "interval": "epoch",
        }
        return [policy_optimizer, value_optimizer], [
            policy_scheduler,
            value_scheduler,
        ]
