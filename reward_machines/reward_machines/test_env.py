import gym
import numpy as np

from envs.robosuite_rm.my_block_stacking_env import MyBlockStackingEnvRM2, MyBlockStackingEnvRM1

my_env = MyBlockStackingEnvRM1()

my_env.seed(0)


my_env.reset()

while True:
    # print("robot joint positions", env.robots[0].dof)  # print robot joint positions
    action = np.random.randn(my_env.robots[0].dof) # sample random action (in this case, 8 joint positions)
    obs, reward, done, info = my_env.step(action)  # take action in the environment
    my_env.render()  # render on display