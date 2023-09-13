"""The main entry point"""
import click
import pickle as pkl

from connectfour.game import BatchGameState
from connectfour.nn import sample_masked_multinomial
from connectfour.play_state import PlayState
from connectfour.policy import PolicyNet

from connectfour.rl_trainer import ConnectFourAI
from connectfour.value_net import ValueNet


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
    bgs = BatchGameState(
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
@click.option("--model-file", "-f", type=click.Path(), default="model.pkl")
def one_player(x, model_file):
    """A simple two player game."""
    with open(model_file, "rb") as f:
        model_dict = pkl.load(f)
        full_model = ConnectFourAI(**model_dict["model_hparams"])
        full_model.load_state_dict(model_dict["model_state"])
        policy_net = full_model.policy_net

    bgs = BatchGameState(
        batch_size=1,
        turn=PlayState.X if x else PlayState.O,
        num_rows=model_dict["model_hparams"]["num_rows"],
        num_cols=model_dict["model_hparams"]["num_cols"],
    )
    while (
        winner := bgs.winners(model_dict["model_hparams"]["run_length"])[0]
    ) is None:
        click.echo(bgs)
        if bgs.turn == PlayState.X:
            logits = policy_net(
                bgs.cannonical_board_state.to(next(policy_net.parameters()))
            )
            mask = ~bgs.next_actions().to(device=logits.device)
            moves = sample_masked_multinomial(logits, mask, axis=1)
            loc = moves[0]
        else:
            loc = click.prompt(
                f"Which column would player {bgs.turn} like to play at?",
                type=int,
            )
        bgs = bgs.play_at([loc])
    click.echo(bgs)
    click.echo(f"{winner} Won!")


if __name__ == "__main__":
    cli()
