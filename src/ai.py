import re
import glob

import torch
import torch.nn as nn

from src.abstract import AbsAI
from src.game import BatchGameState
from src.play_state import PlayState, opponent
from src.policy import Policy
from src.state_value import Value
from src.nn import sample_masked_multinomial, make_one_hot

# Helpers:
_root_path_regex = re.compile(r"(?P<root>[\w_/]*)_(\?|\d+)\.pt$")
_step_path_regex = re.compile(r"[\w_/]*_(?P<step>\d+)\.pt$")


class AI(nn.Module, AbsAI):
    def __init__(
            self,
            player: PlayState,
            value: Value=None,
            value_optimizer_state_dict: dict=None,
            policy: Policy=None,
            policy_optimizer_state_dict: dict=None,
            step: int=1,
            lr=1e-3,
            momentum=0.9):
        super().__init__()
        # Has both a policy and a value function.

        # Load the value model and its optimizer
        self.value = value if value else Value()
        self.value_optimizer = torch.optim.SGD(
            self.value.parameters(), lr=lr, momentum=momentum
        )
        if value_optimizer_state_dict:
            self.value_optimizer.load_state_dict(value_optimizer_state_dict)

        # Load the policy model and its optimizer
        self.policy = policy if policy else Policy()
        self.policy_optimizer = torch.optim.SGD(
            self.policy.parameters(), lr=lr, momentum=momentum
        )
        if policy_optimizer_state_dict:
            self.policy_optimizer.load_state_dict(policy_optimizer_state_dict)

        # Store metadata
        self._player = player
        self._step = step

    def next_moves(self, gs: BatchGameState):
        impossible_actions = ~gs.next_actions()
        logits = self.policy_logits(gs)
        return sample_masked_multinomial(logits, impossible_actions, axis=1)

    @property
    def player(self):
        return self._player

    @property
    def opponent(self):
        return opponent(self._player)

    @property
    def step(self):
        return self._step

    def reward_logits(self, gs: BatchGameState):
        assert gs.turn == self.player, "Can't play on this turn"
        gs_array = gs.as_array()
        return self.value(torch.Tensor(gs_array))

    def expected_reward(self, gs: BatchGameState):
        reward_logits = self.reward_logits(gs)
        return reward_logits[:, 0]

    def policy_logits(self, gs: BatchGameState):
        assert gs.turn == self.player, "Can't play on this turn"
        gs_array = gs.as_array()
        return self.policy(torch.Tensor(gs_array))

    def ln_pi(self, gs: BatchGameState):
        logits = self.policy_logits(gs)
        return logits - torch.logsumexp(logits, dim=1, keepdim=True)

    def learn_from_update(
            self,
            gs_0: BatchGameState,
            action: torch.Tensor,
            gs_2: BatchGameState,
            verbose=True
    ) -> float:
        """
        Actor critic update. Rewards are not discounted (episodic task).
        Note that gs_2 should be two plays after gs_0 (both player and opponent
        have played).
        """
        # Collect rewards
        winners = gs_2.winners()
        with torch.no_grad():
            expected_reward_2 = self.expected_reward(gs_2)
            # ToDo: rewrite as `torch.where` to optimize.
            rewards = torch.Tensor([
                1. if winner == self.player else
                -1. if winner == self.opponent else expected_reward_2[i]
                for i, winner in enumerate(winners)
            ])
        expected_reward_0 = self.expected_reward(gs_0)
        deltas = rewards - expected_reward_0
        if verbose:
            print("reward loss: %s" % float(deltas**2))

        # value update
        self.value.zero_grad()
        expected_reward_0.backward(-deltas)  # scale the grad by deltas.
        self.value_optimizer.step()

        # policy update:
        one_hot_action = make_one_hot(action, gs_0._num_cols)
        ln_pi = self.ln_pi(gs_0)

        self.policy.zero_grad()
        # Weight the updates by the deltas.
        ln_pi.backward(- deltas.view(-1, 1) * one_hot_action)
        self.policy_optimizer.step()

        # Increase the step count:
        self._step += gs_0.batch_size

        return float(torch.mean(deltas)**2)

    @property
    def root_fp(self):
        if not self._fp:
            return ""
        parsed_fp = _root_path_regex.match(self._fp)
        if parsed_fp:
            root_fp = parsed_fp["root"]
            return root_fp
        return self._fp

    def save(self, fp=None):
        if fp is None and self._fp:
            fp = self._fp
        parsed_fp = _root_path_regex.match(fp)
        if parsed_fp:
            root_fp = parsed_fp["root"]
            fp = "{}_{}.pt".format(root_fp, self._step)
        torch.save({
            "step": self._step,
            "value_state_dict": self.value.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "policy_state_dict": self.policy.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            # 'loss': loss,
        }, fp)

    @classmethod
    def load(cls, fp_regex=None, player: PlayState=PlayState.X):
        try:
            file_glob = glob.glob(fp_regex)
            files = sorted(file_glob, key=lambda fp: int(_step_path_regex.match(fp)["step"]))
            fp = files[-1]  # get the latest one
        except Exception as exc:
            print("Unable to find latest policy:", exc)
            fp = fp_regex.replace("_[0-9]*","_?")
        try:
            print("Loading %s" % fp)
            checkpoint = torch.load(fp)
            ai = AI(
                player=player,
                value=Value().load_state_dict(checkpoint["value_state_dict"]),
                value_optimizer_state_dict
                =checkpoint["value_optimizer_state_dict"],
                policy=Policy().load_state_dict(
                    checkpoint["policy_state_dict"]
                ),
                policy_optimizer_state_dict
                =checkpoint["policy_optimizer_state_dict"],
                step=checkpoint["step"]
            )
        except Exception as exc:
            print("Unable to load policy:", exc)
            ai = AI(player=player)
        ai._fp = fp
        return ai
