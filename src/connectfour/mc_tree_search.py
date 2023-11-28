"""Implements mc-tree search"""
from typing import Dict, List

import torch

from connectfour.game import MutableBatchGameState
from connectfour.nn import sample_masked_multinomial
from connectfour.play_state import play_state_embedding_ix, PlayState
from connectfour.policy import PolicyNet
from connectfour.value_net import ValueNet


def random_move(board_state: torch.Tensor) -> torch.Tensor:
    batch_size, _, _, num_cols = board_state.shape
    logits = torch.zeros(batch_size, num_cols, device=board_state.device)

    return masked_move(board_state, logits)


def sample_move(
    policy_net: PolicyNet, board_state: torch.Tensor, epsilon
) -> torch.Tensor:
    logits = policy_net(board_state)
    random_probs = torch.rand(board_state.shape[0], device=board_state.device) < epsilon
    logits = torch.where(random_probs[:, None], torch.zeros_like(logits), logits)

    return masked_move(board_state, logits)


def masked_move(board_state, logits):
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


def random_play(
    bgs: MutableBatchGameState,
    policy_net: PolicyNet,
    value_net: ValueNet,
    depth: int,
    run_length: int,
    discount: float,
    epsilon: float,
) -> Dict[str, torch.Tensor]:
    """Returns the move we took along with the (discounted) reward"""
    player = bgs.turn
    initial_board_state = bgs.cannonical_board_state

    # Choose a move:
    move = random_move(board_state=initial_board_state)

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
            epsilon=epsilon,
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


def fantasy_play(
    bgs: MutableBatchGameState,
    policy_net: PolicyNet,
    value_net: ValueNet,
    depth: int,
    run_length: int,
    discount: float,
    epsilon: float,
) -> Dict[str, torch.Tensor]:
    """Returns the move we took along with the (discounted) reward"""
    player = bgs.turn
    initial_board_state = bgs.cannonical_board_state

    # Choose a move:
    move = sample_move(policy_net, board_state=initial_board_state, epsilon=epsilon)

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
            epsilon=epsilon,
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
    epsilon: float = 0.1,
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
    logits = policy_net(bgs._board_state)
    probs = torch.softmax(logits, dim=1)
    wide_state = torch.clone(bgs._board_state).repeat(breadth, 1, 1, 1)
    bgs_ = MutableBatchGameState(
        state=wide_state,
        turn=bgs.turn,
        num_rows=bgs._num_rows,
        num_cols=bgs._num_cols,
        batch_size=bgs.batch_size * breadth,
        device=bgs._board_state.device,
    )
    result = random_play(
        bgs=bgs_,
        policy_net=policy_net,
        value_net=value_net,
        depth=depth,
        run_length=run_length,
        discount=discount,
        epsilon=epsilon,
    )
    for i in range(breadth):
        move = result["move"][i * batch_size : (i + 1) * batch_size]
        value = result["value"][i * batch_size : (i + 1) * batch_size]
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
    value = (value_matrix * probs).sum(dim=1)
    optimal_value = torch.gather(value_matrix, dim=1, index=move[:, None])
    return {
        "move": move,
        "value": value,
        "optimal_value": optimal_value,
        "counts": count_matrix,
    }
