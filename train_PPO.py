from datetime import datetime
import time
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
    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)


def test(model, envs, total_timesteps):
    obs = envs.reset()
    for _ in range(total_timesteps):
        actions, _ = model.predict(obs)
        obs, rewards, dones, info = envs.step(actions)
        envs.render(mode="human")
        if dones:
            obs = envs.reset()


if __name__ == "__main__":
    # env_case_ids = [8, 9, 88, 99, 888, 999]

    # envs = SubprocVecEnv(
    #     [
    #         lambda: NGSIMEnv(scene="us-101", period=0, vehicle_id=case_id, IDM=False)
    #         for case_id in env_case_ids
    #     ]
    # )

    # envs = NGSIMEnv(scene="us-101", period=0, vehicle_id=320, IDM=False)
    # torch.save(envs, "./envs/env_320")

    env_id = 320
    envs = torch.load("./envs/env_{}".format(env_id), weights_only=False)
    envs.reset()
    model = PPO(
        "MlpPolicy",
        envs,
        learning_rate=0.0001,
        verbose=1,
        tensorboard_log="./tensorboard/",
        n_steps=512,
        batch_size=64,
        n_epochs=5,
        gamma=0.9,
        device='cpu'
    )

    train(model, total_timesteps=100000, save_path="./models/PPO_{}_{}".format(env_id, datetime.now().strftime("%m%d_%H%M%S")))
