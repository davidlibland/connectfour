import dataclasses
import math
from pathlib import Path
import pickle as pkl
import random

import pandas as pd
import torch
import yaml
from lightning import Trainer
from matplotlib import pyplot as plt

from connectfour.game import MutableBatchGameState
from connectfour.io import MatchData, load_policy_net, load_cfai
from connectfour.play_state import PlayState
from connectfour.policy import PolicyNet
from connectfour.rl_trainer import ConnectFourAI, sample_move


def train_network(model: ConnectFourAI, max_epochs):
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


def train_new_network(
    n_rows, n_cols, run_length, hparams, max_epochs, opponent
):
    model = ConnectFourAI(
        opponent_policy_net=opponent,
        turn=PlayState.X,
        num_cols=n_cols,
        num_rows=n_rows,
        run_length=run_length,
        device="mps",
        **hparams,
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
    match_file_path = Path(match_file_path)
    if match_file_path.exists():
        with match_file_path.open("r") as f:
            match_data = MatchData(**yaml.safe_load(f))
    else:
        log_path = train_new_network(
            n_rows=n_rows,
            n_cols=n_cols,
            run_length=run_length,
            hparams=hparams,
            max_epochs=max_epochs,
            opponent=None,
        )
        match_data = MatchData(models=[str(log_path)], matches=[])
    for _ in range(num_matches):
        if random.choice([True, False]):
            # Train a new network:
            opponent_path = match_data.top_performers()[0]
            opponent_policy_net = load_policy_net(opponent_path)
            print(f"Training a new network against {opponent_path}")

            log_path = train_new_network(
                n_rows=n_rows,
                n_cols=n_cols,
                run_length=run_length,
                hparams=hparams,
                max_epochs=max_epochs,
                opponent=opponent_policy_net,
            )
            policy_net = load_policy_net(log_path)
        else:
            # Improve an existing network:
            models = match_data.top_performers(eps=0.25)
            my_path = models[0]
            my_ai = load_cfai(my_path)

            opponent_path = models[1]
            opponent_policy_net = load_policy_net(opponent_path)
            print(f"Improving network: {my_path} against {opponent_path}")

            my_ai.opponent_policy_net = opponent_policy_net

            log_path = train_network(my_ai, max_epochs=max_epochs)
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
        print(f"Model saved at: {log_path}, with avg reward: {reward}")
        match_data.models.append(str(log_path))
        match_data.matches.append([str(log_path), str(opponent_path), reward])
        with match_file_path.open("w") as f:
            yaml.dump(dataclasses.asdict(match_data), f)


if __name__ == "__main__":
    hparams = {
        "policy_net_kwargs": dict(run_lengths=[3]),
        "value_net_kwargs": dict(run_lengths=[3]),
        "lr": 3e-3,
        "gamma": 0.8,
        "batch_size": 2048,
    }

    n_rows = 4
    n_cols = 3
    run_length = 3

    bootstrap_models(
        n_rows=n_rows,
        n_cols=n_cols,
        num_matches=100,
        run_length=run_length,
        hparams=hparams,
        max_epochs=100,
        match_file_path="matches.yml",
    )
