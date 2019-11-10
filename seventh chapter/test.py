import numpy as np
import tensorflow as tf
import random
# random.seed(1)
import gym
env = gym.make('Pendulum-v0')
env.reset()
env.step([1.5])
while True:
    env.render()
