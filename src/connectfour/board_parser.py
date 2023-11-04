"""String representations of the board"""
import torch

from connectfour.play_state import (
    play_state_extraction,
    play_state_embedding_ix,
    PlayState,
)


def as_tuple(board_state: torch.Tensor):
    """Formats a board (player, num_rows, num_cols) as a tuple"""
    to_val = lambda x: x.value
    return tuple(
        [
            tuple([to_val(play_state_extraction(v)) for v in row])
            for row in torch.permute(board_state, (1, 2, 0)).tolist()
        ]
    )


def as_string(board_state: torch.Tensor) -> str:
    """Formats a board (player, num_rows, num_cols) as a string"""
    _, num_rows, num_cols = board_state.shape
    game_strs = []
    game_tuple = as_tuple(board_state)
    hor_line = "\n%s\n" % ("-" * (num_cols * 2 - 1))
    game_strs.append(hor_line.join(map(lambda row: "|".join(row), game_tuple)))
    hor_line = "\n\n%s\n\n" % ("*" * (num_cols * 2 + 1))
    return hor_line.join(game_strs)


def from_string(s: str) -> torch.Tensor:
    lines = s.split("\n")
    num_rows = (len(lines) + 1) // 2
    num_cols = (len(lines[0]) + 1) // 2
    board = torch.zeros((3, num_rows, num_cols))
    for i in range(num_rows):
        row = lines[i * 2]
        for j in range(num_cols):
            x = row[2 * j]
            for p in [PlayState.O, PlayState.X, PlayState.BLANK]:
                if x == p.value:
                    board[play_state_embedding_ix(p), i, j] = 1
    return board
