from enum import Enum


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
    result = [0, 0, 0]
    result[play_state_embedding_ix(p)] = 1
    return result


def play_state_extraction(vect):
    for p in [PlayState.BLANK, PlayState.X, PlayState.O]:
        if vect[play_state_embedding_ix(p)] == 1:
            return p
    raise ValueError("Unparsable embedding %s" % vect)


def opponent(p: PlayState):
    if p == PlayState.X:
        return PlayState.O
    elif p == PlayState.O:
        return PlayState.X
    raise ValueError("No opponent for %s" % p)
