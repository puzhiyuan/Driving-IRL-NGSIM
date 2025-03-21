from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from NGSIM_env.envs.ngsim_env import NGSIMEnv
import numpy as np
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def train(model, total_timesteps, save_path):
    model.learn(total_timesteps=total_timesteps, log_interval=1)
    model.save(save_path + "DQN_NGSIM_320_1049")


def test(model, envs, total_timesteps):
    obs = envs.reset()
    for _ in range(total_timesteps):
        actions, _ = model.predict(obs)
        obs, rewards, dones, info = envs.step(actions)
        envs.render(mode="human")
        if dones:
            obs = envs.reset()


if __name__ == "__main__":
    env_case_ids = [8, 9, 88, 99, 888, 999]

    envs = SubprocVecEnv(
        [
            lambda: NGSIMEnv(scene="us-101", period=0, vehicle_id=case_id, IDM=False)
            for case_id in env_case_ids
        ]
    )

    model = DQN(
        "MlpPolicy",
        envs,
        verbose=1,
        tensorboard_log="./tensorboard/",
        device="cuda",
        learning_rate=1e-3,
        buffer_size=1000000,
        learning_starts=1000,
        batch_size=64,
        tau=0.01,
        gamma=0.9,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        max_grad_norm=10,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    train(model, total_timesteps=100000, save_path="./models/")
