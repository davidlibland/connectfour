import numpy as np

from src.abstract import AbsAI
from src.game import BatchGameState
from src.play_state import PlayState, opponent, play_state_embedding_ix


class MCTreeSearchAI(AbsAI):
    def __init__(self, player: PlayState, sample_size: int=10, depth: int=10, discount_rate=0.9):
        self._player = player
        self._sample_size = sample_size
        self._depth = depth
        self._discount_rate = discount_rate

    @property
    def player(self):
        return self._player

    @property
    def opponent(self):
        return opponent(self._player)

    def next_moves(self, gs: BatchGameState):
        assert gs.turn == self.player, "Can't play on this turn"
        split_states = gs.split()
        moves = []
        for gs_i in split_states:
            moves.append(self._next_move_helper(gs_i))
        return np.array(moves)

    def _next_move_helper(self, gs: BatchGameState) -> int:
        """Assumes that the gs has a batch size of 1."""
        assert gs.turn == self.player, "Can't play on this turn"
        assert gs.batch_size == 1, "Can't do tree search on batched games."
        if gs.winners()[0] is not None:
            return 0

        def get_next_action_state_vals(gs):
            possible_actions = gs.next_actions().view(-1)

            # Initialize
            action_states = [
                (i, gs.play_at([i]))
                for i, possible in enumerate(possible_actions)
                if possible
            ]
            return [(i, gs, self.state_val(gs)[0]) for i, gs in action_states]


        # BFS through the tree
        # Global BFS state:
        action_vals = dict()  # average expected action
        def update(action, val):
            cur_val, cnt = action_vals.get(action, (0,0))
            action_vals[action] = cur_val+val, cnt+1

        # Initialize local BFS state:
        action_state_vals = get_next_action_state_vals(gs)
        # asv_stack = deque([asv+(1,) for asv in action_state_vals])
        cur_depth = 0
        while cur_depth < self._depth and action_state_vals:
            action_state_vals.sort(
                key=lambda a_s_v: 0 if a_s_v[2] == PlayState.DRAW else a_s_v[2],
                reverse=cur_depth % 2
            )
            next_asvs = []
            for action, gs, val in action_state_vals[:self._sample_size]:
                if val == PlayState.DRAW:
                    update(action, 0)
                    continue
                discounted_val = self._discount_rate**cur_depth * val
                update(action, discounted_val)
                if abs(val) == 1:
                    continue
                next_asvs.extend(get_next_action_state_vals(gs))
            cur_depth += 1
            action_state_vals = next_asvs
        if not action_vals:
            raise RuntimeError("Expected at least one action")
        return max(
            action_vals.keys(),
            key=lambda key: action_vals.get(key)[0]/action_vals.get(key)[1]
        )

    def state_val(self, gs: BatchGameState, run_length=4, hesitency=3):
        run_vals = gs.get_max_runs(max_run_length=run_length)
        return [
            1. if val[play_state_embedding_ix(self.player)] == run_length else
            -1. if val[play_state_embedding_ix(self.opponent)] == run_length else
            PlayState.DRAW if val[play_state_embedding_ix(PlayState.BLANK)] == 0 else
            (val[play_state_embedding_ix(self.player)]
            - val[play_state_embedding_ix(self.opponent)])/hesitency*run_length
            for val in run_vals
        ]
