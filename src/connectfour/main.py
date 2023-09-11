"""The main entry point"""
import click

from connectfour.game import BatchGameState
from connectfour.play_state import PlayState


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
    bgs = BatchGameState(batch_size=1, turn=PlayState.X if x else PlayState.O, num_rows=rows, num_cols=cols)
    while (winner := bgs.winners(run_length)[0]) is None:
        click.echo(bgs)
        loc = click.prompt(f"Which column would player {bgs.turn} like to play at?", type=int)
        bgs = bgs.play_at([loc])
    click.echo(bgs)
    click.echo(f"{winner} Won!")


if __name__ == "__main__":
    cli()


