from NGSIM_env.envs.ngsim_env import NGSIMEnv
import torch
import numpy as np
# env = NGSIMEnv(scene="us-101", period=0, vehicle_id=999, IDM=True)

env = torch.load("./envs/env_888", weights_only=False)

env.reset()

# random action

action = env.action_space.sample()

observation, reward, done, info = env.step(None)
print("action: ", action)
print("observation: ", observation)
print("reward: ", reward)
print("done: ", done)
print("info: ", info)
