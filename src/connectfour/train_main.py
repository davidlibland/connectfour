import dataclasses
import itertools
import math
import pickle as pkl
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from lightning import Trainer
from matplotlib import pyplot as plt

from connectfour.game import MutableBatchGameState
from connectfour.io import MatchData, load_cfai, load_policy_net
from connectfour.play_state import PlayState, play_state_embedding_ix
from connectfour.policy import PolicyNet
from connectfour.rl_trainer import ConnectFourAI, sample_move


def train_network(model: ConnectFourAI, max_epochs, check_val_every_n_epoch):
    print(model.hparams)

    trainer = Trainer(
        accelerator="auto",
        check_val_every_n_epoch=check_val_every_n_epoch,
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
    plot_metrics(log_path, logs, max_epochs)

    return log_path


def train_new_network(
    n_rows, n_cols, run_length, hparams, max_epochs, opponent, check_val_every_n_epoch
):
    model = ConnectFourAI(
        opponent_policy_net=opponent,
        turn=PlayState.X,
        num_cols=n_cols,
        num_rows=n_rows,
        run_length=run_length,
        device="mps",
        max_epochs=max_epochs,
        **hparams,
    )
    return train_network(
        model=model,
        max_epochs=max_epochs,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )


def plot_metrics(log_path, logs, max_epochs):
    log_df = pd.read_csv(logs).set_index("step")
    val_met = re.compile(r"^val_(?P<rest>.*)")
    val_df = log_df[[v for v in log_df.columns if val_met.match(v)]]
    val_df = val_df.rename(
        columns={v: val_met.match(v)["rest"] for v in val_df.columns}
    )
    log_df = log_df[[v for v in log_df.columns if not val_met.match(v)]]
    w = int(math.sqrt(len(log_df.columns)))
    h = math.ceil(len(log_df.columns) / w)
    fig, axs = plt.subplots(w, h, figsize=(4 * h, 6 * w))
    for col, ax in zip(log_df.columns, axs.flatten()):
        log_df[col].ewm(halflife=max_epochs / 10).mean().plot(y=col, ax=ax)
        if col in val_df.columns:
            val_df[col].ewm(halflife=max_epochs / 10).mean().plot(
                y=col, ax=ax, color="red", ls=":"
            )
        ax.set_title(col)
    fig.savefig(log_path / "metrics.png")


def face_off(
    policy_net_1: PolicyNet,
    policy_net_2: PolicyNet,
    run_length,
    num_turns=100,
    batch_size=2048,
) -> float:
    device = next(policy_net_1.parameters()).device

    def play_turn(bgs: MutableBatchGameState, play_state_1, play_state_2, run_length):

        # Now choose a move:
        move = sample_move(bgs, policy_net_1, board_state=None)

        # Make the play:
        bgs.play_at(move)

        # Now check if the game is over:
        def get_reward(winners):
            return (winners == play_state_embedding_ix(play_state_1)).to(
                dtype=torch.float
            ) - (winners == play_state_embedding_ix(play_state_2)).to(dtype=torch.float)

        def get_win_count(winners):
            return (winners == play_state_embedding_ix(play_state_1)).to(
                dtype=torch.float
            ) + (winners == play_state_embedding_ix(play_state_2)).to(dtype=torch.float)

        winners = bgs.winners_numeric(run_length=run_length)
        # compute the rewards:
        reward = get_reward(winners)
        n_winners = get_win_count(winners)
        # reset any dead games:
        resets = winners != play_state_embedding_ix(None)

        # Let the opponent move:
        opponent_move = sample_move(bgs=bgs, policy_net=policy_net_2)
        bgs.play_at(opponent_move, resets)

        # Now check if the game is over:
        winners = bgs.winners_numeric(run_length=run_length)
        # compute the rewards:
        reward += get_reward(winners)
        n_winners += get_win_count(winners)
        # reset any dead games:
        resets = winners != play_state_embedding_ix(None)

        # Get the output_value, ignoring any resets:
        reset_masks = torch.Tensor(resets).to(device=device, dtype=torch.bool)

        # Finally, reset any dead games:
        bgs.reset_games(reset_masks)

        return reward.mean()

    bgs = MutableBatchGameState(
        num_rows=n_rows,
        num_cols=n_cols,
        batch_size=batch_size,
        device=device,
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
    improve_odds=0.9,
    n_test_opponents=5,
    train_last=False,
    run_challenges=True,
    check_val_every_n_epoch=10,
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
            check_val_every_n_epoch=check_val_every_n_epoch,
        )
        match_data = MatchData(models=[str(log_path)], matches=[])
    for _ in range(num_matches):
        new_model = not match_data.matches or (
            np.random.choice([False, True], p=[improve_odds, 1 - improve_odds])
            and not train_last
        )
        models = match_data.top_performers(eps=0.1)
        if new_model:
            # Train a new network:
            opponent_path = models[0]
            opponent_policy_net = load_policy_net(opponent_path)
            print(f"Training a new network against {opponent_path}")

            log_path = train_new_network(
                n_rows=n_rows,
                n_cols=n_cols,
                run_length=run_length,
                hparams=hparams,
                max_epochs=max_epochs,
                opponent=opponent_policy_net,
                check_val_every_n_epoch=check_val_every_n_epoch,
            )
            policy_net = load_policy_net(log_path)
        else:
            # Improve an existing network:
            if train_last:
                my_path = match_data.models[-1]
                opponent_path = my_path
            else:
                my_path = models[0]
                opponent_path = models[1]

            my_ai = load_cfai(my_path)
            opponent_policy_net = load_policy_net(opponent_path)
            print(f"Improving network: {my_path} against {opponent_path}")

            my_ai.opponent_policy_net = opponent_policy_net

            log_path = train_network(
                my_ai,
                max_epochs=max_epochs,
                check_val_every_n_epoch=check_val_every_n_epoch,
            )
            policy_net = load_policy_net(log_path)

        # Add the model to the match data:
        match_data.models.append(str(log_path))

        total_reward = 0
        # Test against some top opponents:
        for i, opponent_path in enumerate(models[:n_test_opponents]):
            print(f"Running match {i+1} of {n_test_opponents}")
            opponent_policy_net = load_policy_net(opponent_path)
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
            match_data.matches.append([str(log_path), str(opponent_path), reward])
            total_reward += reward

        print(
            f"Model saved at: {log_path}, with avg reward: {total_reward/n_test_opponents}"
        )

        # Run some random matches between top players:
        if run_challenges:
            challenge_matches = itertools.islice(
                itertools.combinations(models, 2), n_test_opponents
            )
            for i, (path_1, path_2) in enumerate(challenge_matches):
                print(f"Running random match {i+1} of {n_test_opponents}")
                policy_net = load_policy_net(path_1)
                opponent_policy_net = load_policy_net(path_2)
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
                match_data.matches.append([str(log_path), str(opponent_path), reward])

        with match_file_path.open("w") as f:
            yaml.dump(dataclasses.asdict(match_data), f)


if __name__ == "__main__":

    n_rows = 6
    n_cols = 7
    run_length = 4
    hparams = {
        "policy_net_kwargs": dict(kernel_size=3),
        "value_net_kwargs": dict(kernel_size=3, n_cols=n_cols, n_rows=n_rows),
        "embedding_net_kwargs": dict(kernel_size=3, latent_dim=10, depth=2),
        "policy_lr": 1e-3,
        "value_lr": 1e-2,
        "gamma": 0.95,
        "batch_size": 2048,
        "val_batch_size": 256,
        "value_net_burn_in_frac": 0.0,
        "weight_decay": 0.03,
    }

    bootstrap_models(
        n_rows=n_rows,
        n_cols=n_cols,
        num_matches=100,
        run_length=run_length,
        hparams=hparams,
        max_epochs=50,
        check_val_every_n_epoch=10,
        match_file_path="matches.yml",
        faceoff_turns=30,
        train_last=False,
        run_challenges=False,
    )
