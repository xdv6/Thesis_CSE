import robosuite as suite
from robosuite.wrappers import GymWrapper
from reward_machines.rm_environment import RewardMachineEnv
import gym
import gymnasium as gymnasium
import numpy as np
from gym import spaces



def flatten_observation(obs):
    # Flatten the observation dictionary into a single array
    flat_obs = []
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            flat_obs.extend(value.flatten())
        else:
            flat_obs.append(value)
    return np.array(flat_obs)

# Custom environment wrapper for block stacking using GymWrapper
class MyBlockStackingEnv(GymWrapper):
    def __init__(self):
        # Initialize the robosuite environment and wrap it with GymWrapper
        env = suite.make(
            "PickPlace",
            robots="Panda",  # Using Panda robot
            use_object_obs=True,  # Include object observations
            has_renderer=True,  # Enable rendering for visualization
            reward_shaping=True,  # Use dense rewards for easier learning
            control_freq=20,  # Set control frequency for smooth simulation
            use_camera_obs=False,  # Disable camera observations
        )
        super().__init__(env)  # Wrap the environment with GymWrapper

        # Flatten observation space
        flattened_observation = flatten_observation(env.reset())

        # Define the observation space based on the flattened observation
        self.obs_dim = flattened_observation.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)


    def step(self, action):
        # Step the environment and return the flattened observation, reward, done, and info
        next_obs, reward, done, info = self.env.step(action)
        return flatten_observation(next_obs), reward, done, info


    def get_events(self):
        # Define events for the reward machine based on block states (grasped, stacked)
        events = ''
        if self.block_grasped():
            events += 'g'  # 'g' event for block grasped
        if self.block_stacked():
            events += 's'  # 's' event for block stacked
        return events

    def block_grasped(self):
        # Check if the block is grasped using robosuite's info keys
        return self.info.get('grasped_block', False)

    def block_stacked(self):
        # Check if the block is stacked using robosuite's info keys
        return self.info.get('stacked_block', False)

    def reset(self):
        # Reset the environment and return the flattened observation
        obs = self.env.reset()
        return flatten_observation(obs)


# RewardMachineEnv wrapper for the MyBlockStackingEnv using the first reward machine (t1.txt)
class MyBlockStackingEnvRM1(RewardMachineEnv):
    def __init__(self):
        # Use MyBlockStackingEnv as the base environment
        env = MyBlockStackingEnv()

        # Reward machine configuration file
        rm_files = ["./envs/robosuite_rm/reward_machines/t1.txt"]

        # print("observation space before conversion: ", env.observation_space)
        #
        # # Ensure compatibility by converting gymnasium Box space to gym Box space
        # if isinstance(env.observation_space, gymnasium.spaces.Box):
        #     low = env.observation_space.low
        #     high = env.observation_space.high
        #     shape = env.observation_space.shape
        #     dtype = env.observation_space.dtype
        #
        #     # Convert to gym.spaces.Box
        #     converted_observation_space = gym.spaces.Box(low=low, high=high, dtype=dtype)
        #
        #     # Update the observation space
        #     env.observation_space = converted_observation_space
        #
        # print("observation space after conversion: ", env.observation_space)
        # print("observation", env.reset())
        # print("shape", env.observation_space.shape)
        # print("number of values in ordereddict", len(env.reset()))
        #
        # obs = env.reset()  # or the observation dictionary you provided above
        #
        # num_values = 0
        # for key, value in obs.items():
        #     print(f"Key: {key}, Number of values: {len(value)}")
        #     num_values += len(value)
        # print(f"Total number of values: {num_values}")

        # Initialize the RewardMachineEnv with the converted environment and reward machine files
        super().__init__(env, rm_files)


# RewardMachineEnv wrapper for the MyBlockStackingEnv using the second reward machine (t2.txt)
class MyBlockStackingEnvRM2(RewardMachineEnv):
    def __init__(self):
        # Use MyBlockStackingEnv as the base environment
        env = MyBlockStackingEnv()

        # Reward machine configuration file
        rm_files = ["./envs/robosuite_rm/reward_machines/t2.txt"]

        # Initialize the RewardMachineEnv with the converted environment and reward machine files
        super().__init__(env, rm_files)