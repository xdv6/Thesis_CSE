import robosuite as suite
from robosuite.wrappers import GymWrapper
from reward_machines.rm_environment import RewardMachineEnv
import gym
import gymnasium as gymnasium
import numpy as np
import random
import time
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite import load_controller_config
import imageio
import os


def flatten_observation(obs):
    # Flatten the observation dictionary into a single array while removing unneeded values
    flat_obs = []
    keys_to_keep = [
        "robot0_eef_pos",  # End-effector position
        "robot0_gripper_qpos",  # Gripper position
        "cubeA_pos",  # Position of cubeA
        "cubeB_pos",  # Position of cubeB
        "gripper_to_cubeA",  # Relative position vector from gripper to cubeA
        "gripper_to_cubeB",  # Relative position vector from gripper to cubeB
        "cubeA_to_cubeB"  # Relative position vector between cubeA and cubeB
    ]
    for key in keys_to_keep:
        if key in obs:
            value = obs[key]
            # Print the key and its value
            # print(f"Key: {key}, Value: {value}")
            if isinstance(value, np.ndarray):
                flat_obs.extend(value.flatten())
            else:
                flat_obs.append(value)
    return np.array(flat_obs)


# Custom environment wrapper for block stacking using GymWrapper
class MyBlockStackingEnv(GymWrapper):
    def __init__(self, video_path=os.path.join(os.environ.get("WORKDIR_PATH", "."), "training_video.mp4"), render_height=512, render_width=512):
        # Initialize the robosuite environment and wrap it with GymWrapper
        # Load controller configuration
        controller_config = load_controller_config(default_controller="OSC_POSITION")

        # Create environment instance with the given configuration
        env = suite.make(
            "Stack",
            robots="Panda",  # Using Panda robot
            controller_configs=controller_config,
            use_object_obs=True,  # Include object observations
            has_renderer=False,  # Enable rendering for visualization
            reward_shaping=True,  # Use dense rewards for easier learning
            control_freq=5,  # Set control frequency for smooth simulation
            horizon=50,
            use_camera_obs=False,  # Disable camera observations
        )

        # put line 358 en 359 from stack environment in comments
        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        placement_initializer.append_sampler(
            # Create a placement initializer with a y_range and dynamically updated x_range
            sampler = UniformRandomSampler(
                name="ObjectSamplerCubeA",
                x_range=[0.01, 0.01],
                y_range=[0.01, 0.01],
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0, 0, 0.8),
                z_offset=0.01,
            )
        )

        placement_initializer.append_sampler(
            # Create a placement initializer with a y_range and dynamically updated x_range
            sampler = UniformRandomSampler(
                name="ObjectSamplerCubeB",
                x_range=[0.2, 0.2],
                y_range=[0.2, 0.2],
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0, 0, 0.8),
                z_offset=0.01,
            )
        )

        placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeA", mujoco_objects=env.cubeA)
        placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeB", mujoco_objects=env.cubeB)

        # Update the environment to use the new placement initializer
        env.placement_initializer = placement_initializer
        super().__init__(env)  # Wrap the environment with GymWrapper

        # Video recording setup
        self.video_path = video_path
        self.writer = imageio.get_writer(self.video_path, fps=20)
        self.render_height = render_height
        self.render_width = render_width

        # for recognition in reward machine
        self.status = "robosuite"
        # Flatten observation space
        reset_env = env.reset()
        self.obs_dict = reset_env
        flattened_observation = flatten_observation(reset_env)

        # Define the observation space based on the flattened observation
        self.obs_dim = flattened_observation.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # Timer to track how long cubeA is above cubeB and in contact
        self.stack_timer = 0.0
        self.stack_threshold = 5.0  # Threshold time in seconds to consider cubeA as "stacked" on cubeB
        self.start_time = time.time()

    def step(self, action):
        # Step the environment and return the flattened observation, reward, done, and info
        next_obs, reward, done, info = self.env.step(action)
        self.obs_dict = next_obs

        # Render and save the frame to the video
        frame = self.env.sim.render(
            height=self.render_height,
            width=self.render_width,
            camera_name="frontview"
        )
        self.writer.append_data(frame)

        flattened_observation = flatten_observation(next_obs)
        # self.env.render()
        return flattened_observation, reward, done, info

    def get_events(self):
        # Define events for the reward machine based on block states (grasped, stacked, above blockB)
        events = ''
        if self.block_grasped():
            events += 'g'  # 'g' event for block grasped, robot in contact with cubeA
        if self.above_block_b_and_grasped():
            events += 'h'  # 'h' event for above cubeB in height
        if self.above_block_b_in_xy_and_grasped():
            events += 'p'  # 'p' event for above cubeB in x, y coordinates
        if self.cube_a_above_cube_b_and_in_contact():
            events += 'b'  # 'b' event for cubeA above cubeB and in contact
        if self.cube_a_above_cube_b_long_contact():
            events += 'l'  # 'l' event for cubeA above cubeB, in contact for more than 5 seconds, and robot not in contact with cubeA
        if self.block_dropped():
            events += 'd'  # 'd' event for when the robot drops the block (not in contact with cubeA anymore)
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
        obs = self.obs_dict
        eef_height = obs["robot0_eef_pos"][2]  # z-coordinate of end-effector position
        cube_b_height = obs["cubeB_pos"][2]  # z-coordinate of cubeB
        is_above_cube_b = eef_height > cube_b_height
        return is_above_cube_b

    def above_block_b_in_xy_and_grasped(self):
        # Check if the end-effector is above cubeB in the x and y plane while still grasping cubeA
        obs = self.obs_dict
        eef_position = obs["robot0_eef_pos"][:2]  # x, y coordinates of end-effector position
        cube_b_position = obs["cubeB_pos"][:2]  # x, y coordinates of cubeB

        # Define an allowable margin to be considered "above" in the x, y plane
        margin = 0.025  # 5 cm margin
        is_above_cube_b_xy = (
            cube_b_position[0] - margin <= eef_position[0] <= cube_b_position[0] + margin and
            cube_b_position[1] - margin <= eef_position[1] <= cube_b_position[1] + margin
        )

        return is_above_cube_b_xy

    def cube_a_above_cube_b_and_in_contact(self):
        # Check if cubeA is directly above cubeB and if they are in contact
        obs = self.obs_dict
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

    def cube_a_above_cube_b_long_contact(self):
        # Check if cubeA is above cubeB, in contact for more than the threshold time, and the robot is not in contact
        obs = self.obs_dict
        cube_a_pos = obs["cubeA_pos"]
        cube_b_pos = obs["cubeB_pos"]
        is_robot_not_in_contact = not self.block_grasped()

        # Check the conditions: cubeA above cubeB and in contact
        if self.cube_a_above_cube_b_and_in_contact():
            # Update the timer if the conditions hold
            self.stack_timer += time.time() - self.start_time
        else:
            # Reset the timer if conditions are not met
            self.stack_timer = 0.0
        self.start_time = time.time()

        # Condition for long contact
        return self.stack_timer > self.stack_threshold and is_robot_not_in_contact

    def block_dropped(self):
        # Check if the robot has dropped the block (i.e., no longer in contact with cubeA)
        return not self.block_grasped()

    def reset(self):
        # Reset the environment
        obs = self.env.reset()
        self.stack_timer = 0.0  # Reset timer on environment reset
        self.start_time = time.time()  # Reset start time on reset
        self.obs_dict = obs

        # # Move the gripper above the block (cubeA)
        # target_pos = obs["cubeA_pos"] + np.array([0, 0, 0.1])  # Target position above cubeA
        # for _ in range(100):
        #     curr_pos = obs["robot0_eef_pos"]
        #     delta_pos = target_pos - curr_pos
        #     action = np.concatenate([5 * delta_pos, [-1]])  # Gripper open
        #     obs, reward, done, info = self.env.step(action)
        #     self.obs_dict = obs
        #     if np.linalg.norm(delta_pos) < 0.01:  # Stop when close to target
        #         break
        #
        # # Move down to grasp the block
        # target_pos = obs["cubeA_pos"]
        # target_pos[-1] -= 0.01  # Lower the gripper slightly for grasping
        # for _ in range(100):
        #     curr_pos = obs["robot0_eef_pos"]
        #     delta_pos = target_pos - curr_pos
        #     action = np.concatenate([4 * delta_pos, [-1]])  # Gripper open
        #     obs, reward, done, info = self.env.step(action)
        #     self.obs_dict = obs
        #     if np.linalg.norm(delta_pos) < 0.01:  # Stop when close to target
        #         break
        #
        # # Close the gripper to grasp the block
        # for _ in range(25):
        #     action = np.concatenate([[0, 0, 0], [1]])  # Close gripper
        #     obs, reward, done, info = self.env.step(action)
        #     self.obs_dict = obs
        #     # Check for contact between gripper and cubeA
        #     is_contact = self.env.check_contact(
        #         geoms_1=["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"],
        #         geoms_2=["cubeA_g0"]
        #     )
        #     if is_contact:  # Stop if the block is grasped
        #         break

        # Render and save the initial frame to the video
        frame = self.env.sim.render(
            height=self.render_height,
            width=self.render_width,
            camera_name="frontview"
        )
        self.writer.append_data(frame)

        # Return the flattened observation
        flattened_observation = flatten_observation(obs)
        return flattened_observation

    def seed(self, seed):
        # Set the random seed for reproducibility
        # needed for gym compatibility
        pass

    def close(self):
        # Close the video writer
        if self.writer is not None:
            self.writer.close()
        # Close the environment
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()



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





