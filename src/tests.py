from functools import reduce
import tensorflow as tf
tf.enable_eager_execution()

from src.game import BatchGameState
from src.play_state import PlayState


def test_batch_winner():
    bgs = BatchGameState(batch_size=2)
    plays1 = [4,4,3,3,2]
    plays2 = [4,4,3,3,1]
    bgs = reduce(BatchGameState.play_at, zip(plays1,plays2), bgs)
    print(bgs)
    print(bgs.winners(3))
    assert bgs.winners(3)[0] == PlayState.X
    assert bgs.winners(3)[1] == None
