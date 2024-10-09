import robosuite as suite
from robosuite.wrappers import GymWrapper
from reward_machines.rm_environment import RewardMachineEnv
import gym
import gymnasium as gymnasium
import numpy as np

# Custom environment wrapper for block stacking using GymWrapper
class MyBlockStackingEnv(GymWrapper):
    def __init__(self):
        # Initialize the robosuite environment and wrap it with GymWrapper
        env = suite.make(
            "PickPlace",
            robots="Panda",               # Using Panda robot
            use_object_obs=True,           # Include object observations
            has_renderer=True,             # Enable rendering for visualization
            reward_shaping=True,           # Use dense rewards for easier learning
            control_freq=20                # Set control frequency for smooth simulation
        )
        super().__init__(env)              # Wrap the environment with GymWrapper

    def step(self, action):
        # Execute the action and return the next observation, reward, done status, and info
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info                   # Store info for event tracking
        return next_obs, original_reward, env_done, info

    def get_events(self):
        # Define events for the reward machine based on block states (grasped, stacked)
        events = ''
        if self.block_grasped():
            events += 'g'                  # 'g' event for block grasped
        if self.block_stacked():
            events += 's'                  # 's' event for block stacked
        return events

    def block_grasped(self):
        # Check if the block is grasped using robosuite's info keys
        return self.info.get('grasped_block', False)

    def block_stacked(self):
        # Check if the block is stacked using robosuite's info keys
        return self.info.get('stacked_block', False)

    def reset(self):
        # Reset the environment to its initial state and return the first observation
        return self.env.reset()


# RewardMachineEnv wrapper for the MyBlockStackingEnv using the first reward machine (t1.txt)
class MyBlockStackingEnvRM1(RewardMachineEnv):
    def __init__(self):
        # Use MyBlockStackingEnv as the base environment
        env = MyBlockStackingEnv()
        
        # Reward machine configuration file
        rm_files = ["./envs/robosuite_rm/reward_machines/t1.txt"]

        # Ensure compatibility by converting gymnasium Box space to gym Box space
        if isinstance(env.observation_space, gymnasium.spaces.Box):
            low = env.observation_space.low
            high = env.observation_space.high
            shape = env.observation_space.shape
            dtype = env.observation_space.dtype


            # Convert to gym.spaces.Box
            converted_observation_space = gym.spaces.Box(low=low, high=high, dtype=dtype)

            # Update the observation space
            env.observation_space = converted_observation_space

        # Initialize the RewardMachineEnv with the converted environment and reward machine files
        super().__init__(env, rm_files)


# RewardMachineEnv wrapper for the MyBlockStackingEnv using the second reward machine (t2.txt)
class MyBlockStackingEnvRM2(RewardMachineEnv):
    def __init__(self):
        # Use MyBlockStackingEnv as the base environment
        env = MyBlockStackingEnv()
        
        # Reward machine configuration file
        rm_files = ["./envs/robosuite_rm/reward_machines/t2.txt"]

        # Ensure compatibility by converting gymnasium Box space to gym Box space
        if isinstance(env.observation_space, gymnasium.spaces.Box):
            low = env.observation_space.low
            high = env.observation_space.high
            shape = env.observation_space.shape
            dtype = env.observation_space.dtype

            # Convert to gym.spaces.Box
            converted_observation_space = gym.spaces.Box(low=low, high=high, dtype=dtype)

            # Update the observation space
            env.observation_space = converted_observation_space

        # Initialize the RewardMachineEnv with the converted environment and reward machine files
        super().__init__(env, rm_files)
