from enum import Enum

import numpy as np


class PlayState(Enum):
    X = "X"
    O = "O"
    BLANK = " "
    DRAW = " "  # This is an alias for BLANK


def play_state_embedding_ix(p: PlayState):
    if p == PlayState.BLANK:
        return 0
    elif p == PlayState.X:
        return 1
    elif p == PlayState.O:
        return 2
    raise ValueError("Unrecognized player %s" % p)


def play_state_embedding(p: PlayState):
    result = np.zeros([3], dtype=np.float32)
    result[play_state_embedding_ix(p)] = 1
    return result


def play_state_extraction(vect):
    for p in [PlayState.BLANK, PlayState.X, PlayState.O]:
        if vect[play_state_embedding_ix(p)] == 1:
            return p
    raise ValueError("Unparsable embedding %s" % vect)