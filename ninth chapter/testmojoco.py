import tensorflow as tf
import numpy as np
import gym
from gym.envs.atari.atari_env import AtariEnv
env_name = 'pong'
# env_name = 'HalfCheetah-v1'
# env_name = 'Swimmer-v1'
# env_name = 'Hopper-v1'
# env_name = 'Humanoid-v1'
# env_name = 'HumanoidStandup-v1'
# env_name = 'BipedalWalker-v2'
# env = gym.make(env_name)
env = AtariEnv(game='breakout',obs_type="ram", frameskip=10)
env.reset()
while(1):
    env.render()