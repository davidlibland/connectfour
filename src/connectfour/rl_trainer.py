"""
The main RL training code.
"""
import dataclasses
import math
import pathlib
import random
from pathlib import Path
from typing import List, Tuple, Iterator, Dict
import pickle as pkl

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import yaml
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


def sample_move(
    bgs: MutableBatchGameState, policy_net: PolicyNet, board_state=None
) -> torch.Tensor:
    # board_state = self.bgs.cannonical_board_state
    if board_state is None:
        board_state = bgs.cannonical_board_state.to(device=policy_net.device)
    logits = policy_net(board_state.to(dtype=torch.float))
    mask = ~bgs.next_actions().to(device=logits.device)
    moves = sample_masked_multinomial(logits, mask, axis=1)
    return moves


def load_policy_net(log_path):
    model_file = pathlib.Path(log_path) / "model.pkl"
    with open(model_file, "rb") as f:
        model_dict = pkl.load(f)
    full_model = ConnectFourAI(
        **model_dict["model_hparams"], opponent_policy_net=None
    )
    full_model.load_state_dict(model_dict["model_state"])
    policy_net = full_model.policy_net
    return policy_net


class ConnectFourAI(pl.LightningModule):
    def __init__(
        self,
        policy_net_kwargs: dict,
        value_net_kwargs: dict,
        lr,
        gamma,
        run_length,
        opponent_policy_net: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["policy_net", "value_net", "opponent_policy_net"]
        )
        bgs = MutableBatchGameState(**kwargs)
        self.policy_net = PolicyNet(**policy_net_kwargs)
        self.value_net = ValueNet(**value_net_kwargs)
        if opponent_policy_net is None:
            self.opponent_policy_net = self.policy_net
        else:
            self.opponent_policy_net = opponent_policy_net
        self.bgs = bgs
        self.play_state = PlayState.X
        self.opponent_play_state = PlayState.O

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
        # Initially, we assume that no board is a finished game (moves are allowed)
        initial_value = self.value_net(board_state).flatten()

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

    def eval_step(self, batch):
        board_state, turn = batch
        board_state = board_state.to(dtype=torch.float)

        with torch.no_grad():
            # compute the delta:
            state_updates = self.take_composite_move_and_get_reward_delta(
                board_state
            )

        return state_updates["reward"].mean()

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


def train_network(n_rows, n_cols, run_length, hparams, max_epochs, opponent):
    model = ConnectFourAI(
        opponent_policy_net=opponent,
        turn=PlayState.X,
        num_cols=n_cols,
        num_rows=n_rows,
        run_length=run_length,
        device="mps",
        **hparams
    )

    print(model.hparams)

    trainer = Trainer(
        accelerator="auto",
        # devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=max_epochs,
        # val_check_interval=50,
        logger=True,
    )

    trainer.fit(model)

    log_path = Path(trainer.logger.log_dir)

    with (log_path / "model.pkl").open("wb") as f:
        pkl.dump(
            {
                "model_state": model.state_dict(),
                "model_hparams": model.hparams,
            },
            f,
        )
    logs = log_path / "metrics.csv"
    log_df = pd.read_csv(logs).set_index("step")
    w = int(math.sqrt(len(log_df.columns)))
    h = math.ceil(len(log_df.columns) / w)
    fig, axs = plt.subplots(w, h, figsize=(4 * h, 6 * w))
    for col, ax in zip(log_df.columns, axs.flatten()):
        log_df[col].ewm(halflife=max_epochs / 10).mean().plot(y=col, ax=ax)
        ax.set_title(col)
    fig.savefig(log_path / "metrics.png")

    return log_path


def face_off(
    policy_net_1: PolicyNet,
    policy_net_2: PolicyNet,
    run_length,
    num_turns=100,
    batch_size=2048,
) -> float:
    def play_turn(
        bgs: MutableBatchGameState, play_state_1, play_state_2, run_length
    ):

        # Now choose a move:
        move = sample_move(bgs, policy_net_1, board_state=None)

        # Make the play:
        bgs.play_at(move)

        # Now check if the game is over:
        def get_reward(win_state):
            if win_state == play_state_1:
                return 1
            if win_state == play_state_2:
                return -1
            return 0

        def get_win_count(win_state):
            if win_state == play_state_1:
                return 1
            if win_state == play_state_2:
                return 1
            return 0

        winners = bgs.winners(run_length=run_length)
        # compute the rewards:
        reward = torch.Tensor(
            [get_reward(win_state) for win_state in winners]
        ).to(device=policy_net_1.device)
        n_winners = torch.Tensor(
            [get_win_count(win_state) for win_state in winners]
        )
        # reset any dead games:
        resets = [win_state is not None for win_state in winners]

        # Let the opponent move:
        opponent_move = sample_move(bgs=bgs, policy_net=policy_net_2)
        bgs.play_at(opponent_move, resets)

        # Now check if the game is over:
        winners = bgs.winners(run_length=run_length)
        # compute the rewards:
        reward += torch.Tensor(
            [get_reward(win_state) for win_state in winners]
        ).to(device=policy_net_1.device)
        n_winners += torch.Tensor(
            [get_win_count(win_state) for win_state in winners]
        )
        # reset any dead games:
        resets = [win_state is not None for win_state in winners]

        # Get the output_value, ignoring any resets:
        reset_masks = torch.Tensor(resets).to(
            device=policy_net_1.device, dtype=torch.bool
        )

        # Finally, reset any dead games:
        bgs.reset_games(reset_masks)

        return reward.mean()

    bgs = MutableBatchGameState(
        num_rows=n_rows,
        num_cols=n_cols,
        batch_size=batch_size,
        device=policy_net_1.device,
        turn=PlayState.X,
    )
    total_reward = 0
    for _ in range(num_turns):
        total_reward += play_turn(
            bgs,
            play_state_1=PlayState.X,
            play_state_2=PlayState.O,
            run_length=run_length,
        )
    return float(total_reward / num_turns)


@dataclasses.dataclass
class MatchData:
    """models: List of model paths"""

    models: List[str]
    """Matches: model_path, model_path, reward"""
    matches: Tuple[str, str, float]

    def top_performers(self):
        rewards_per_model = {model: [] for model in self.models}
        for model_1, model_2, reward in self.matches:
            rewards_per_model[model_1].append(reward)
            rewards_per_model[model_2].append(-reward)
        reward_per_model = {
            model: sum(rewards) / len(rewards) if rewards else 0
            for model, rewards in rewards_per_model.items()
        }
        top_model = sorted(self.models, key=reward_per_model.get, reverse=True)
        return top_model


def bootstrap_models(
    n_rows,
    n_cols,
    run_length,
    hparams,
    max_epochs,
    num_matches=3,
    match_file_path=None,
    faceoff_turns=None,
):
    if faceoff_turns is None:
        faceoff_turns = max(max_epochs // 3, 10)
    match_file_path = pathlib.Path(match_file_path)
    if match_file_path.exists():
        with match_file_path.open("r") as f:
            match_data = MatchData(**yaml.safe_load(f))
    else:
        log_path = train_network(
            n_rows=n_rows,
            n_cols=n_cols,
            run_length=run_length,
            hparams=hparams,
            max_epochs=max_epochs,
            opponent=None,
        )
        match_data = MatchData(models=[str(log_path)], matches=[])
    for _ in range(num_matches):
        opponent_path = match_data.top_performers()[0]
        opponent_policy_net = load_policy_net(opponent_path)

        log_path = train_network(
            n_rows=n_rows,
            n_cols=n_cols,
            run_length=run_length,
            hparams=hparams,
            max_epochs=max_epochs,
            opponent=opponent_policy_net,
        )
        policy_net = load_policy_net(log_path)

        # Toss up for first player:
        if random.choice([True, False]):
            reward = face_off(
                policy_net_1=policy_net,
                policy_net_2=opponent_policy_net,
                run_length=run_length,
                num_turns=faceoff_turns,
            )
        else:
            reward = -face_off(
                policy_net_2=policy_net,
                policy_net_1=opponent_policy_net,
                run_length=run_length,
                num_turns=faceoff_turns,
            )
        match_data.models.append(str(log_path))
        match_data.matches.append([str(log_path), str(opponent_path), reward])
    with match_file_path.open("w") as f:
        yaml.dump(dataclasses.asdict(match_data), f)


if __name__ == "__main__":
    hparams = {
        "policy_net_kwargs": dict(run_lengths=[3]),
        "value_net_kwargs": dict(run_lengths=[3]),
        "lr": 1e-3,
        "gamma": 0.8,
        "batch_size": 2048,
    }

    n_rows = 4
    n_cols = 3
    run_length = 3

    bootstrap_models(
        n_rows=n_rows,
        n_cols=n_cols,
        num_matches=3,
        run_length=run_length,
        hparams=hparams,
        max_epochs=5,
        match_file_path="matches.yml",
    )

    # log_path = train_network(
    #     n_rows=n_rows,
    #     n_cols=n_cols,
    #     run_length=run_length,
    #     hparams=hparams,
    #     max_epochs=30,
    #     opponent=None,
    # )
    # print(f"Model saved in {log_path}")
    #
    # model_file = log_path / "model.pkl"
    # with open(model_file, "rb") as f:
    #     model_dict = pkl.load(f)
    #     full_model = ConnectFourAI(**model_dict["model_hparams"], opponent_policy_net=None)
    #     full_model.load_state_dict(model_dict["model_state"])
    #     policy_net = full_model.policy_net
    #
    # policy_net_1 = policy_net
    # policy_net_2 = full_model.opponent_policy_net
    #
    # reward = face_off(policy_net_1, policy_net_2, run_length)
    # print(f"Avg reward: {reward}")
