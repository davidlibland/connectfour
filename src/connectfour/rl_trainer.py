"""
The main RL training code.
"""
import math
from pathlib import Path
from typing import List, Tuple, Iterator, Dict
import pickle as pkl

import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning import Trainer
from torch.optim import Optimizer, Adam
from torch.utils.data import IterableDataset, DataLoader

from connectfour.nn import sample_masked_multinomial

import lightning.pytorch as pl

from connectfour.game import MutableBatchGameState
from connectfour.play_state import PlayState, play_state_embedding_ix
from connectfour.policy import PolicyNet
from connectfour.value_net import ValueNet


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


class ConnectFourAI(pl.LightningModule):
    def __init__(
        self,
        # policy_net: nn.Module,
        # value_net: nn.Module,
        # opponent_policy_net: nn.Module,
        # bgs: MutableBatchGameState,
        policy_net_kwargs: dict,
        value_net_kwargs: dict,
        lr,
        gamma,
        run_length,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["policy_net", "value_net", "opponent_policy_net"]
        )
        bgs = MutableBatchGameState(**kwargs)
        # self.policy_net = policy_net
        # self.value_net = value_net
        # self.opponent_policy_net = opponent_policy_net
        self.policy_net = PolicyNet(**policy_net_kwargs)
        self.value_net = ValueNet(**value_net_kwargs)
        self.opponent_policy_net = self.policy_net
        self.bgs = bgs
        self.play_state = PlayState.X
        self.opponent_play_state = PlayState.O

    def sample_move(self, policy_net, board_state) -> torch.Tensor:
        # board_state = self.bgs.cannonical_board_state
        logits = policy_net(board_state)
        mask = ~self.bgs.next_actions().to(device=logits.device)
        moves = sample_masked_multinomial(logits, mask, axis=1)
        return moves

    def take_composite_move_and_get_reward_delta(
        self, board_state
    ) -> Dict[str, torch.Tensor]:
        """
        Both the player and the opponent take a move. The state is updated and
        the (adjusted) reward:
            R +\gamma v(new_state) - v(old_state)
        is returned
        """
        # Initially, we assume that no board is a finished game (moves are allowed)
        initial_value = self.value_net(board_state).flatten()
        # if the board is empty, the initial value should be zero:
        empty_masks = board_state[:, 1:, :, :].sum(dim=(1, 2, 3)) == 0
        zeros = torch.zeros_like(initial_value)
        initial_value = torch.where(empty_masks, zeros, initial_value)

        # Now choose a move:
        move = self.sample_move(self.policy_net, board_state=board_state)

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
        reward = torch.Tensor(
            [get_reward(win_state) for win_state in winners]
        ).to(board_state)
        n_winners = torch.Tensor(
            [get_win_count(win_state) for win_state in winners]
        )
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
        reward += torch.Tensor(
            [get_reward(win_state) for win_state in winners]
        ).to(board_state)
        n_winners += torch.Tensor(
            [get_win_count(win_state) for win_state in winners]
        )
        # reset any dead games:
        resets = [win_state is not None for win_state in winners]

        # Get the output_value, ignoring any resets:
        final_value = self.value_net(final_board_state).flatten()
        reset_masks = torch.Tensor(resets).to(
            device=final_value.device, dtype=torch.bool
        )
        zeros = torch.zeros_like(final_value)
        final_value = torch.where(reset_masks, zeros, final_value)

        # Finally, reset any dead games:
        self.bgs.reset_games(reset_masks)

        # Ok. Compute and return the delta:
        delta = reward + final_value * self.hparams["gamma"] - initial_value

        return {
            "move": move,
            "delta": delta,
            "reward": reward,
            "n_winners": n_winners,
        }

    def training_step(self, batch):
        board_state, turn = batch
        board_state = board_state.to(dtype=torch.float)

        with torch.no_grad():
            # compute the delta:
            state_updates = self.take_composite_move_and_get_reward_delta(
                board_state
            )
            move = state_updates["move"]
            delta = state_updates["delta"]

            # compute the gammas:
            I = torch.pow(
                torch.tensor(self.hparams["gamma"]).to(board_state),
                board_state[:, 1:, :, :].sum(dim=(1, 2, 3)),
            )

        # get the value loss:
        value_loss = -delta * self.value_net(board_state).flatten()

        # compute the policy loss:
        policy_loss = (
            -I * delta * torch.diag(self.policy_net(board_state)[:, move])
        )

        total_loss = value_loss + policy_loss

        self.logger.log_metrics(
            {
                "train_loss": total_loss.mean(),
                "reward": state_updates["reward"].mean(),
                "avg_finishes": state_updates["n_winners"].mean(),
                "value_loss_proxy": value_loss.mean(),
                "value_loss": 0.5 * (delta**2).mean(),
                "policy_loss": policy_loss.mean(),
            }
        )
        return total_loss.mean()

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
        optimizer = Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


if __name__ == "__main__":
    # bgs = MutableBatchGameState(
    #
    # )
    # policy_net = PolicyNet(rows=7, cols=7)
    # opponent_policy_net = PolicyNet(rows=7, cols=7)
    # value_net = ValueNet(rows=7, cols=7)
    n_rows = 4
    n_cols = 3
    model = ConnectFourAI(
        # bgs=bgs,
        # policy_net=policy_net,
        # opponent_policy_net=policy_net,
        # value_net=value_net,
        policy_net_kwargs=dict(rows=n_rows, cols=n_cols),
        value_net_kwargs=dict(rows=n_rows, cols=n_cols),
        lr=1e-3,
        gamma=0.5,
        run_length=3,
        batch_size=2048,
        turn=PlayState.X,
        num_cols=n_cols,
        num_rows=n_rows,
        device="mps",
    )
    print(model.hparams)

    trainer = Trainer(
        accelerator="auto",
        # devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=3000,
        # val_check_interval=50,
        logger=True,
    )

    trainer.fit(model)

    with open("model.pkl", "wb") as f:
        pkl.dump(
            {
                "model_state": model.state_dict(),
                "model_hparams": model.hparams,
            },
            f,
        )

    log_path = Path(trainer.logger.log_dir)
    logs = log_path / "metrics.csv"
    log_df = pd.read_csv(logs).set_index("step")
    w = int(math.sqrt(len(log_df.columns)))
    h = math.ceil(len(log_df.columns) / w)
    fig, axs = plt.subplots(w, h, figsize=(4 * h, 6 * w))
    for col, ax in zip(log_df.columns, axs.flatten()):
        log_df.plot(y=col, ax=ax)
    fig.savefig(log_path / "metrics.png")
