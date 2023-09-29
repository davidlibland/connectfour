import dataclasses
import pathlib
from typing import List, Tuple
import pickle as pkl

import numpy as np

from connectfour.rl_trainer import ConnectFourAI


@dataclasses.dataclass
class MatchData:
    """models: List of model paths"""

    models: List[str]
    """Matches: model_path, model_path, reward"""
    matches: Tuple[str, str, float]

    def top_performers(self, eps: float = 0):
        rewards_per_model = {model: [] for model in self.models}
        for model_1, model_2, reward in self.matches:
            rewards_per_model[model_1].append(reward)
            rewards_per_model[model_2].append(-reward)
        reward_per_model = {
            model: sum(rewards) / len(rewards) if rewards else 0
            for model, rewards in rewards_per_model.items()
        }
        if eps > 0:
            reward_array = np.array(list(reward_per_model.values()))
            reward_std = np.std(reward_array)
            noise = np.random.randn(len(reward_per_model)) * eps * reward_std
            reward_per_model = {
                model: reward + xi
                for (model, reward), xi in zip(
                    reward_per_model.items(), noise.tolist()
                )
            }
        top_models = sorted(
            self.models, key=reward_per_model.get, reverse=True
        )
        return top_models


def load_cfai(log_path):
    model_file = pathlib.Path(log_path) / "model.pkl"
    with open(model_file, "rb") as f:
        model_dict = pkl.load(f)
    full_model = ConnectFourAI(
        **model_dict["model_hparams"], opponent_policy_net=None
    )
    full_model.load_state_dict(model_dict["model_state"])
    return full_model


def load_policy_net(log_path):
    full_model = load_cfai(log_path)
    policy_net = full_model.policy_net
    return policy_net
