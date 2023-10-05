"""The main entry point"""
import pathlib

import click
import pickle as pkl

import yaml

from connectfour.game import MutableBatchGameState
from connectfour.nn import sample_masked_multinomial
from connectfour.play_state import PlayState

from connectfour.io import MatchData, load_policy_net


@click.group()
def cli():
    pass


@cli.command()
@click.option("--rows", "-r", type=int, default=7)
@click.option("--cols", "-c", type=int, default=7)
@click.option("--run_length", "-l", type=int, default=4)
@click.option("--x/--o", type=bool, is_flag=True)
def two_players(rows, cols, run_length, x):
    """A simple two player game."""
    bgs = MutableBatchGameState(
        batch_size=1,
        turn=PlayState.X if x else PlayState.O,
        num_rows=rows,
        num_cols=cols,
    )
    while (winner := bgs.winners(run_length)[0]) is None:
        click.echo(bgs)
        loc = click.prompt(
            f"Which column would player {bgs.turn} like to play at?", type=int
        )
        bgs = bgs.play_at([loc])
    click.echo(bgs)
    click.echo(f"{winner} Won!")


@cli.command()
# @click.option("--rows", "-r", type=int, default=7)
# @click.option("--cols", "-c", type=int, default=7)
# @click.option("--run-length", "-l", type=int, default=4)
@click.option("--x/--o", type=bool, is_flag=True)
@click.option("--model-file", "-f", type=click.Path(), default=None)
@click.option("--match-file", "-m", type=click.Path(), default="matches.yml")
@click.option("--temperature", "-t", type=float, default=".01")
def one_player(x, model_file, match_file, temperature):
    """A simple two player game."""
    if match_file is not None:
        with open(match_file, "r") as f:
            match_data = MatchData(**yaml.safe_load(f))
        model_file = match_data.models[-1]
    with open(f"{model_file}/model.pkl", "rb") as f:
        model_dict = pkl.load(f)
    policy_net = load_policy_net(model_file)

    bgs = MutableBatchGameState(
        batch_size=1,
        turn=PlayState.X if x else PlayState.O,
        num_rows=model_dict["model_hparams"]["num_rows"],
        num_cols=model_dict["model_hparams"]["num_cols"],
    )
    while (winner := bgs.winners(model_dict["model_hparams"]["run_length"])[0]) is None:
        click.echo(bgs)
        if bgs.turn == PlayState.X:
            logits = policy_net(
                bgs.cannonical_board_state.to(next(policy_net.parameters()))
            )
            mask = ~bgs.next_actions().to(device=logits.device)
            moves = sample_masked_multinomial(logits / temperature, mask, axis=1)
            loc = moves[0]
        else:
            loc = click.prompt(
                f"Which column would player {bgs.turn} like to play at?",
                type=int,
            )
        bgs = bgs.play_at([loc])
    click.echo(bgs)
    click.echo(f"{winner} Won!")


@cli.command()
@click.option("--x/--o", type=bool, is_flag=True)
@click.option("--model-file", "-f", type=click.Path(), default=None)
@click.option("--match-file", "-m", type=click.Path(), default="matches.yml")
@click.option("--temperature", "-t", type=float, default=".01")
def ais(x, model_file, match_file, temperature):
    """A simple two player game."""
    if match_file is not None:
        with open(match_file, "r") as f:
            match_data = MatchData(**yaml.safe_load(f))
        # model_file = match_data.top_performers()[0]
        model_file1 = match_data.models[-1]
        model_file2 = match_data.models[-2]
    with open(f"{model_file1}/model.pkl", "rb") as f:
        model_dict = pkl.load(f)
    policy_net1 = load_policy_net(model_file1)
    policy_net2 = load_policy_net(model_file2)

    bgs = MutableBatchGameState(
        batch_size=1,
        turn=PlayState.X if x else PlayState.O,
        num_rows=model_dict["model_hparams"]["num_rows"],
        num_cols=model_dict["model_hparams"]["num_cols"],
    )
    while (winner := bgs.winners(model_dict["model_hparams"]["run_length"])[0]) is None:
        click.echo(bgs)
        if bgs.turn == PlayState.X:
            logits = policy_net1(
                bgs.cannonical_board_state.to(next(policy_net1.parameters()))
            )
        else:
            logits = policy_net1(
                bgs.cannonical_board_state.to(next(policy_net2.parameters()))
            )
        mask = ~bgs.next_actions().to(device=logits.device)
        moves = sample_masked_multinomial(logits / temperature, mask, axis=1)
        loc = moves[0]
        loc = click.prompt(
            f"Which column would player {bgs.turn} like to play at? ({loc})",
            type=int,
        )
        bgs = bgs.play_at([loc])

    click.echo(bgs)
    click.echo(f"{winner} Won!")


if __name__ == "__main__":
    cli()
