"""Implements mc-tree search"""
from typing import Dict

import torch

from connectfour.game import MutableBatchGameState
from connectfour.nn import sample_masked_multinomial
from connectfour.play_state import play_state_embedding_ix, PlayState
from connectfour.policy import PolicyNet
from connectfour.value_net import ValueNet


def sample_move(policy_net: PolicyNet, board_state=None) -> torch.Tensor:
    logits = policy_net(board_state)

    blank_ix = play_state_embedding_ix(PlayState.BLANK)
    blank_space_indicator = board_state[:, blank_ix, :, :]
    num_blank_spaces = torch.sum(blank_space_indicator, dim=1)
    actions = num_blank_spaces != 0

    mask = ~actions
    moves = sample_masked_multinomial(logits, mask, axis=1)
    return moves


def get_reward(winners, play_state: PlayState) -> torch.Tensor:
    opponent_play_state = PlayState.O if play_state == PlayState.X else PlayState.X
    return (winners == play_state_embedding_ix(play_state)).to(dtype=torch.float) - (
        winners == play_state_embedding_ix(opponent_play_state)
    ).to(dtype=torch.float)


def fantasy_play(
    bgs: MutableBatchGameState,
    policy_net: PolicyNet,
    value_net: ValueNet,
    depth: int,
    run_length: int,
    discount: float,
) -> Dict[str, torch.Tensor]:
    """Returns the move we took along with the (discounted) reward"""
    player = bgs.turn
    initial_board_state = bgs.cannonical_board_state

    # Choose a move:
    move = sample_move(policy_net, board_state=initial_board_state)

    # Make the play:
    bgs.play_at(move)
    mid_board_state = bgs.cannonical_board_state

    # Now check if the game is over:

    winners = bgs.winners_numeric(run_length=run_length)
    # compute the rewards:
    reward = get_reward(winners, player)
    # reset any dead games:
    resets = winners != play_state_embedding_ix(None)
    bgs.reset_games(resets)

    if depth > 1:
        recursive_result = fantasy_play(
            bgs=bgs,
            policy_net=policy_net,
            value_net=value_net,
            depth=depth - 1,
            run_length=run_length,
            discount=discount,
        )
    else:
        # terminate recursion:
        recursive_result = {"value": value_net(mid_board_state)}

    opponent_value = recursive_result["value"]
    value = torch.where(
        resets,  # If reset, we have a real reward
        reward,  # so use the reward
        -opponent_value * discount,  # otherwise use the recursive estimate
    )

    return {"move": move, "value": value}


def mc_tree_search(
    bgs: MutableBatchGameState,
    policy_net: PolicyNet,
    value_net: ValueNet,
    depth: int,
    breadth: int,
    run_length: int,
    discount: float,
) -> Dict[str, torch.Tensor]:
    """Run `breadth` fantasy_plays through the game"""
    # Kick off `breadth` worth of runs:
    # Find the move with the highest value:
    batch_size, _, _, num_cols = bgs.cannonical_board_state.shape
    total_value_matrix = torch.zeros(
        (batch_size, num_cols), device=bgs.cannonical_board_state.device
    )
    count_matrix = torch.zeros(
        (batch_size, num_cols), device=bgs.cannonical_board_state.device
    )
    for _ in range(breadth):
        bgs_ = bgs.copy()
        result = fantasy_play(
            bgs=bgs_,
            policy_net=policy_net,
            value_net=value_net,
            depth=depth,
            run_length=run_length,
            discount=discount,
        )
        move = result["move"]
        value = result["value"]
        total_value_matrix.scatter_(dim=1, index=move[:, None], src=value[:, None])
        count_matrix.scatter_(
            dim=1, index=move[:, None], src=torch.ones_like(value)[:, None]
        )
    value_matrix = torch.where(
        count_matrix > 0,
        total_value_matrix / count_matrix,
        torch.zeros_like(total_value_matrix),
    )
    move = torch.argmax(value_matrix, dim=1)
    return {"move": move, "value": value_matrix.mean(dim=1), "counts": count_matrix}
