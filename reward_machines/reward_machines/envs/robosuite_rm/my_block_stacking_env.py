import robosuite as suite
from robosuite.wrappers import GymWrapper
from reward_machines.rm_environment import RewardMachineEnv
import gym
import gymnasium as gymnasium
import numpy as np
import random


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
            "Stack",
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
        # Define events for the reward machine based on block states (grasped, stacked, above blockB)
        events = ''
        if self.block_grasped():
            events += 'g'  # 'g' event for block grasped
        if self.above_block_b_and_grasped():
            events += 'h'  # 'h' event for above cubeB in height while still holding cubeA
        if self.above_block_b_in_xy_and_grasped():
            events += 'p'  # 'p' event for above cubeB in x, y coordinates while still holding cubeA
        if self.cube_a_above_cube_b_and_in_contact():
            events += 'b'  # 'b' event for cubeA above cubeB and in contact
        if self.block_stacked():
            events += 's'  # 's' event for block stacked
        return events

    def block_grasped(self):
        # Check if the block cubeA is grasped by the gripper
        is_contact_with_cubeA = self.env.check_contact(
            geoms_1=["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"],
            geoms_2=["cubeA_g0"]
        )
        return is_contact_with_cubeA 

    def above_block_b_and_grasped(self):
        # Check if the end-effector is above the height of cubeB while still grasping cubeA
        obs = self.env._get_observation()
        eef_height = obs["robot0_eef_pos"][2]  # z-coordinate of end-effector position
        cube_b_height = obs["cubeB_pos"][2]  # z-coordinate of cubeB
        is_above_cube_b = eef_height > cube_b_height
        return is_above_cube_b and self.block_grasped()

    def above_block_b_in_xy_and_grasped(self):
        # Check if the end-effector is above cubeB in the x and y plane while still grasping cubeA
        obs = self.env._get_observation()
        eef_position = obs["robot0_eef_pos"][:2]  # x, y coordinates of end-effector position
        cube_b_position = obs["cubeB_pos"][:2]  # x, y coordinates of cubeB

        # Define an allowable margin to be considered "above" in the x, y plane
        margin = 0.025  # 5 cm margin
        is_above_cube_b_xy = (
            cube_b_position[0] - margin <= eef_position[0] <= cube_b_position[0] + margin and
            cube_b_position[1] - margin <= eef_position[1] <= cube_b_position[1] + margin
        )

        return is_above_cube_b_xy and self.block_grasped()

    def cube_a_above_cube_b_and_in_contact(self):
        # Check if cubeA is directly above cubeB and if they are in contact
        obs = self.env._get_observation()
        cube_a_pos = obs["cubeA_pos"]  # Position of cubeA
        cube_b_pos = obs["cubeB_pos"]  # Position of cubeB

        # Check z position - cubeA should be above cubeB
        is_above_cube_b_in_height = cube_a_pos[2] > cube_b_pos[2]

        # Check x, y alignment
        margin = 0.025  # Allowable margin for x, y alignment
        is_aligned_in_xy = (
            cube_b_pos[0] - margin <= cube_a_pos[0] <= cube_b_pos[0] + margin and
            cube_b_pos[1] - margin <= cube_a_pos[1] <= cube_b_pos[1] + margin
        )

        # Check contact between cubeA and cubeB
        is_contact_between_blocks = self.env.check_contact(
            geoms_1=["cubeA_g0"], geoms_2=["cubeB_g0"]
        )

        # Condition for cubeA being above cubeB and in contact
        return is_above_cube_b_in_height and is_aligned_in_xy and is_contact_between_blocks

    def block_stacked(self):
        # Placeholder for actual stacking logic, which could check relative positions
        # between cubeA and cubeB to verify stacking conditions.
        return self.cube_a_above_cube_b_and_in_contact()

    def reset(self):
        # Reset the environment and return the flattened observation
        obs = self.env.reset()
        return flatten_observation(obs)

    def seed(self, seed):
        # Set the random seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

# RewardMachineEnv wrapper for the MyBlockStackingEnv using the first reward machine (t1.txt)
class MyBlockStackingEnvRM1(RewardMachineEnv):
    def __init__(self):
        # Use MyBlockStackingEnv as the base environment
        env = MyBlockStackingEnv()

        # Reward machine configuration file
        rm_files = ["./envs/robosuite_rm/reward_machines/t1.txt"]

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
