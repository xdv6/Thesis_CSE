import gym

from envs.robosuite_rm.my_block_stacking_env import MyBlockStackingEnvRM2, MyBlockStackingEnvRM1

# register the environment

gym.envs.registry.register(
    id='MyBlockStackingEnvRM1-v0',
    entry_point='envs.robosuite_rm.my_block_stacking_env:MyBlockStackingEnvRM1',
)

my_env = gym.make('MyBlockStackingEnvRM1-v0')