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
import wandb
import pickle


# Custom environment wrapper for block stacking using GymWrapper
class MyBlockStackingEnv(GymWrapper):

    def calculate_reward_gripper_to_cube(self):
        reward = 0.0
        geom_id = self.env.sim.model.geom_name2id(self.selected_cube_geom_name)
        cube_size = self.env.sim.model.geom_size[geom_id]
        cube_width = cube_size[0] * 2
        cube_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.selected_cube_body_name)]
        left_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint1_tip")]
        right_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint2_tip")]

        left_dist = np.linalg.norm(left_finger_pos - np.array([cube_pos[0], cube_pos[1] - cube_width / 2, cube_pos[2]]))
        right_dist = np.linalg.norm(right_finger_pos - np.array([cube_pos[0], cube_pos[1] + cube_width / 2, cube_pos[2]]))
        reward += 0.5 / (left_dist + right_dist + 0.01) # Adding 0.01 to avoid division by zero

        wandb.log({"left_dist": left_dist})
        wandb.log({"right_dist": right_dist})
        return reward

    def calculate_reward_cube_to_threshold_height(self):
        reward = 0.0
        cube_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.selected_cube_body_name)]

        # Find the geom ID corresponding to the cube
        geom_id = self.env.sim.model.geom_name2id(self.selected_cube_geom_name)
        cube_size = self.env.sim.model.geom_size[geom_id]

        bottom_of_cube = cube_pos[2] - cube_size[2]  # size[2] is the z half-size
        threshold_height = 0.91
        distance = abs(bottom_of_cube - threshold_height)
        wandb.log({"treshold_height_distance": distance})
        reward += 1 / (distance + 0.01)  # Penalize based on absolute distance

        if self.block_gripped and not self.block_grasped:
            reward = -20

        return reward

    def generate_option_reward_mapping(self):
        """
        Given a list of options and a cost dict for edges (from_node, to_node) → cost,
        return a mapping: option_id → reward (negative cost).
        Terminal states (-1) are ignored.
        """

        # cost_dict = {
        #     (0, 1): 1.0,
        #     (0, 6): 2.0,
        #     (0, 11): 1.0,
        #     (1, 2): 1.0,
        #     (1, 4): 1.2,
        #     (6, 7): 1.3,
        #     (6, 9): 1.2,
        #     (11, 12): 1.5,
        #     (11, 14): 1.0,
        #
        #     # Transitions into terminal nodes
        #     (2, -1): 1.1,  # 2 → 3
        #     (4, -1): 1.0,  # 4 → 5
        #     (7, -1): 1.0,  # 7 → 8
        #     (9, -1): 1.0,  # 9 → 10
        #     (12, -1): 1.2,  # 12 → 13
        #     (14, -1): 1.0  # 14 → 15
        # }

        cost_dict = {
            (0, 0): 1.35,
            (0, 1): 1.05,
            (0, 17): 1.6,
            (0, 33): 1.21,
            (0, 49): 1.75,
            (1, 1): 1.23,
            (1, 2): 1.09,
            (1, 7): 1.54,
            (1, 12): 1.33,
            (2, 2): 1.53,
            (2, 3): 1.41,
            (2, 5): 1.90,
            (3, 3): 1.72,
            (3, -1): 1.11,  # terminal
            (5, 5): 1.91,
            (5, -1): 1.04,  # terminal
            (7, 7): 1.64,
            (7, 8): 1.33,
            (7, 10): 1.16,
            (8, 8): 1.99,
            (8, -1): 1.85,  # terminal
            (10, 10): 1.96,
            (10, -1): 1.34,  # terminal
            (12, 12): 1.63,
            (12, 13): 1.25,
            (12, 15): 1.44,
            (13, 13): 1.23,
            (13, -1): 1.00,  # terminal
            (15, 15): 1.48,
            (15, -1): 1.09,  # terminal
            (17, 17): 1.18,
            (17, 18): 1.07,
            (17, 23): 1.14,
            (17, 28): 1.27,
            (18, 18): 1.99,
            (18, 19): 1.32,
            (18, 21): 1.91,
            (19, 19): 1.44,
            (19, -1): 1.45,  # terminal
            (21, 21): 1.75,
            (21, -1): 1.09,  # terminal
            (23, 23): 1.04,
            (23, 24): 1.79,
            (23, 26): 1.91,
            (24, 24): 1.13,
            (24, -1): 1.02,  # terminal
            (26, 26): 1.19,
            (26, -1): 1.57,  # terminal
            (28, 28): 1.26,
            (28, 29): 1.84,
            (28, 31): 1.34,
            (29, 29): 1.35,
            (29, -1): 1.77,  # terminal
            (31, 31): 1.27,
            (31, -1): 1.80,  # terminal
            (33, 33): 1.43,
            (33, 34): 1.61,
            (33, 39): 1.77,
            (33, 44): 1.89,
            (34, 34): 1.42,
            (34, 35): 1.53,
            (34, 37): 1.32,
            (35, 35): 1.11,
            (35, -1): 1.81,  # terminal
            (37, 37): 1.06,
            (37, -1): 1.40,  # terminal
            (39, 39): 1.36,
            (39, 40): 1.51,
            (39, 42): 1.64,
            (40, 40): 1.09,
            (40, -1): 1.68,  # terminal
            (42, 42): 1.43,
            (42, -1): 1.67,  # terminal
            (44, 44): 1.01,
            (44, 45): 1.49,
            (44, 47): 1.37,
            (45, 45): 1.54,
            (45, -1): 1.67,  # terminal
            (47, 47): 1.07,
            (47, -1): 1.03,  # terminal
            (49, 49): 1.69,
            (49, 50): 1.18,
            (49, 55): 1.34,
            (49, 60): 1.34,
            (50, 50): 1.40,
            (50, 51): 1.82,
            (50, 53): 1.66,
            (51, 51): 1.99,
            (51, -1): 1.91,  # terminal
            (53, 53): 1.26,
            (53, -1): 1.17,  # terminal
            (55, 55): 1.47,
            (55, 56): 1.39,
            (55, 58): 1.46,
            (56, 56): 1.75,
            (56, -1): 1.07,  # terminal
            (58, 58): 1.33,
            (58, -1): 1.57,  # terminal
            (60, 60): 1.35,
            (60, 61): 1.19,
            (60, 63): 1.63,
            (61, 61): 1.64,
            (61, -1): 1.09,  # terminal
            (63, 63): 1.97,
            (63, -1): 1.01,  # terminal
        }

        reward_mapping = {}
        for option_id, (_, from_node, to_node) in enumerate(self.options_list):
            cost = cost_dict.get((from_node, to_node))
            if cost is None:
                raise ValueError(f"No cost found for edge ({from_node}, {to_node})")
            reward_mapping[option_id] = -cost  # negative reward = penalty = cost
        return reward_mapping


    def set_option(self, option_id):
        self.option_id = option_id

    def set_options_list(self, options_list):
        self.options_list = options_list
        self.option_to_reward_mapping = self.generate_option_reward_mapping()

    def set_options_to_cube_mapping(self, options_to_cube_mapping):
        self.options_to_cube_mapping = options_to_cube_mapping


    def __init__(self, video_path=os.path.join(os.environ.get("WORKDIR_PATH", "./videos"), os.environ.get("WANDB_RUN_NAME", "default_run") + ".mp4"), render_height=512, render_width=512):
        # Initialize the robosuite environment and wrap it with GymWrapper
        # Load controller configuration
        controller_config = load_controller_config(default_controller="OSC_POSITION")

        # Check if rendering is enabled
        self.enable_renderer = False
        check_renderer = os.getenv("ENABLE_RENDERER", "False")
        if check_renderer == "True": # Enable rendering if the environment is set to render
            self.enable_renderer = True

        # checking start state
        self.start_state_value = int(os.getenv("START_STATE", "0"))

        # set the selected cube to grip
        self.selected_cube = os.getenv("SELECTED_CUBE", "cubeA")
        self.selected_cube_body_name = self.selected_cube + "_main"
        self.selected_cube_geom_name = self.selected_cube + "_g0"

        # set this to start on True if starting from state where cube is gripped
        self.block_gripped = False

        self.state_save_index = 0
        self.num_load_points = 17


        # Create environment instance with the given configuration
        env = suite.make(
            "Stack",
            robots="Panda",  # Using Panda robot
            controller_configs=controller_config,
            use_object_obs=True,  # Include object observations
            has_renderer=self.enable_renderer,  # Enable rendering for visualization
            reward_shaping=True,  # Use dense rewards for easier learning
            control_freq=10,  # Set control frequency for smooth simulation
            horizon=100,
            use_camera_obs=False,  # Disable camera observations
        )

        # put line 358 en 359 from stack environment in comments
        placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

        placement_initializer.append_sampler(
            # Create a placement initializer with a y_range and dynamically updated x_range
            sampler=UniformRandomSampler(
                name="ObjectSamplerCubeA",
                x_range=[0.05, 0.05],
                y_range=[0.05, 0.05],
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0, 0, 0.8),
                z_offset=0.01,
            )
        )

        placement_initializer.append_sampler(
            # Create a placement initializer with a y_range and dynamically updated x_range
            sampler=UniformRandomSampler(
                name="ObjectSamplerCubeB",
                x_range=[-0.05, -0.05],
                y_range=[0.05, 0.05],
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0, 0, 0.8),
                z_offset=0.01,
            )
        )

        placement_initializer.append_sampler(
            # Create a placement initializer with a y_range and dynamically updated x_range
            sampler=UniformRandomSampler(
                name="ObjectSamplerCubeC",
                x_range=[0.05, 0.05],
                y_range=[-0.05, -0.05],
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0, 0, 0.8),
                z_offset=0.01,
            )
        )

        placement_initializer.append_sampler(
            # Create a placement initializer with a y_range and dynamically updated x_range
            sampler=UniformRandomSampler(
                name="ObjectSamplerCubeD",
                x_range=[-0.05, -0.05],
                y_range=[-0.05, -0.05],
                rotation=0.0,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=(0, 0, 0.8),
                z_offset=0.01,
            )
        )

        placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeA", mujoco_objects=env.cubeA)
        placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeB", mujoco_objects=env.cubeB)

        placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeC", mujoco_objects=env.cubeC)
        placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeD", mujoco_objects=env.cubeD)

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
        flattened_observation = self.flatten_observation(reset_env)

        # Define the observation space based on the flattened observation
        self.obs_dim = flattened_observation.size
        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        self.amount_of_env_steps = 0


    def flatten_observation(self, obs):
        # Flatten the observation dictionary into a single array while removing unneeded values
        flat_obs = []
        keys_to_keep = [
            "robot0_eef_pos",  # End-effector position
            "robot0_gripper_qpos",  # Gripper position
            # "cubeA_pos",  # Position of cubeA
            # "cubeB_pos",  # Position of cubeB
            "gripper_to_cubeA",  # Relative position vector from gripper to cubeA
            "gripper_to_cubeB",  # Relative position vector from gripper to cubeB
            "gripper_to_cubeC",  # Relative position vector from gripper to cubeC
            "gripper_to_cubeD",  # Relative position vector from gripper to cubeD
            # "robot0_gripper_qvel"
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

        # add left and right finger positions to flat obs
        left_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint1_tip")]
        right_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint2_tip")]
        flat_obs.extend(left_finger_pos)
        flat_obs.extend(right_finger_pos)
        return np.array(flat_obs)

    def step(self, action):
        # Step the environment and return the flattened observation, reward, done, and info
        # clip gripper action, 1 is gripper closed, -1 is gripper open

        self.amount_of_env_steps += 1
        wandb.log({"env_steps_inside_sim": self.amount_of_env_steps})
        if action[-1] <= 0:
            action[-1] = -1
        else:
            action[-1] = 1

        next_obs, reward, done, info = self.env.step(action)
        reward = 0

        # if the gripper is grasping a cube, give the reward based on the mapping

        selected_cube = self.options_to_cube_mapping[self.option_id][0]
        cube_name = "cube" + selected_cube
        if self.block_grasped(cube_name):
            cost = self.option_to_reward_mapping[self.option_id]
            reward = cost
            info["cube_gripped"] = True

        self.obs_dict = next_obs
        # add the reward_for_gripper_to_cube to the obs_dict to pass it to the reward machine
        # also adapt the checks if the cube is changed
        self.obs_dict["reward_gripper_to_cube"] = self.calculate_reward_gripper_to_cube()
        self.obs_dict["reward_cube_lifted"] = self.calculate_reward_cube_to_threshold_height()

        # Render and save the frame to the video
        frame = self.env.sim.render(
            height=self.render_height,
            width=self.render_width,
            camera_name="frontview"
        )
        frame = np.flipud(frame)

        self.writer.append_data(frame)

        flattened_observation = self.flatten_observation(next_obs)

        if self.enable_renderer:
            self.env.render()

        return flattened_observation, reward, done, info

    def get_events(self):
        # Define events for the reward machine based on block states (grasped, stacked, etc)
        events = ''
        if self.block_grasped():
            events += 'g'  # 'g' event for block grasped, for individual block training
        if self.above_treshold():
            events += 'h'  # 'h' event for block above treshold, for individual block training

        # for block sequence training
        if self.block_grasped('cubeA'):
            events += 'gA'
            print("Block A gripped")
        if self.block_grasped('cubeB'):
            events += 'gB'
            print("Block B gripped")
        if self.block_grasped('cubeC'):
            events += 'gC'
            print("Block C gripped")
        if self.block_grasped('cubeD'):
            events += 'gD'


        if self.above_treshold('cubeA'):
            events += 'hA'
        if self.above_treshold('cubeB'):
            events += 'hB'
        if self.above_treshold('cubeC'):
            events += 'hC'
        if self.above_treshold('cubeD'):
            events += 'hD'

        return events

    def block_grasped(self, cube_name=None):

        # Define gripper collision geoms
        left_gripper_geom = ["gripper0_finger1_pad_collision"]  # Left gripper pad
        right_gripper_geom = ["gripper0_finger2_pad_collision"]  # Right gripper pad

        # Define cube collision geom
        if cube_name is None:
            cube_geom = self.selected_cube_geom_name
            cube_body = self.selected_cube_body_name
        else:
            cube_geom = cube_name + "_g0"
            cube_body = cube_name + "_main"

        cube_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(cube_body)]
        geom_id = self.env.sim.model.geom_name2id(cube_geom)
        cube_size = self.env.sim.model.geom_size[geom_id]
        cube_width = cube_size[0] * 2

        left_contact = self.env.check_contact(geoms_1=left_gripper_geom, geoms_2=cube_geom)
        right_contact = self.env.check_contact(geoms_1=right_gripper_geom, geoms_2=cube_geom)

        left_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_leftfinger")]
        right_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_rightfinger")]


        left_touching_left_face = left_contact and (abs(left_finger_pos[1] - (cube_pos[1] - cube_width / 2)) < 0.005)
        right_touching_right_face = right_contact and (abs(right_finger_pos[1] - (cube_pos[1] + cube_width / 2)) < 0.005)

        is_proper_grasp = left_touching_left_face and right_touching_right_face

        if is_proper_grasp:
            self.block_gripped = True

        return is_proper_grasp



    def above_treshold(self, cube_name=None):
        # Check if the end-effector is above the height
        if cube_name is None:
            cube_geom = self.selected_cube_geom_name
            cube_body = self.selected_cube_body_name
        else:
            cube_geom = cube_name + "_g0"
            cube_body = cube_name + "_main"

        cube_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(cube_body)]
        geom_id = self.env.sim.model.geom_name2id(cube_geom)
        cube_size = self.env.sim.model.geom_size[geom_id]
        cube_above_treshold = cube_pos[2] - cube_size[2] > 0.001

        # reset the environment after every lift
        # if cube_above_treshold:
        #     self.reset()

        return cube_above_treshold


    def reset(self):

        # folder_saving = "load_points_stacking"
        # # test saving simulation to file
        # state = self.env.sim.get_state()
        # with open(f'{folder_saving}/state_{self.state_save_index}.pkl', 'wb') as f:
        #     pickle.dump(state, f)
        # print(f"Simulation state saved to file: {folder_saving}/state_{self.state_save_index}.pkl")


        self.block_gripped = False
        # Reset the environment
        obs = self.env.reset()

        # folder_loading = "load_points_alligning"
        # current_load_state = self.state_save_index % self.num_load_points
        # with open(f'{folder_loading}/state_{current_load_state}.pkl', 'rb') as f:
        #     state = pickle.load(f)
        # self.env.sim.set_state(state)
        # self.env.sim.forward()
        # self.state_save_index += 1


        table_geom_id = self.env.sim.model.geom_name2id("table_collision")  # Correct table collision name

        # 🚀 ABSOLUTE MAXIMUM stiffness for contact resolution 🚀
        self.env.sim.model.geom_solref[table_geom_id] = [0.00001, 1]  # Hardest contact resolution possible
        self.env.sim.model.geom_solimp[table_geom_id] = [1, 1, 0.0, 0.0, 10]  # Hardest possible contact surface

        # 🚀 ABSOLUTE MAXIMUM friction to prevent sliding or sinking 🚀
        self.env.sim.model.geom_friction[table_geom_id] = [100.0, 10.0, 1.0]  #

        self.stack_timer = 0.0  # Reset timer on environment reset
        self.start_time = time.time()  # Reset start time on reset
        self.obs_dict = obs
        # add the reward_for_gripper_to_cube to the obs_dict
        # also adapt the checks if the cube is changed
        self.obs_dict["reward_gripper_to_cube"] = self.calculate_reward_gripper_to_cube()
        self.obs_dict["reward_cube_lifted"] = self.calculate_reward_cube_to_threshold_height()

        move_gripper_to_cube = False
        if self.start_state_value == 1:
            move_gripper_to_cube = True

        elif self.start_state_value == -1:
            # choose random start state
            move_gripper_to_cube = random.choice([True, False])

        if move_gripper_to_cube:
            # Move the gripper above CubeA
            cube_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id(self.selected_cube_body_name)]
            target_pos = cube_pos + np.array([0, 0, 0.1])  # Slightly above CubeA
            for _ in range(100):
                curr_pos = obs["robot0_eef_pos"]
                delta_pos = target_pos - curr_pos
                action = np.concatenate([5 * delta_pos, [-1]])  # Keep gripper open
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                self.obs_dict = obs
                if np.linalg.norm(delta_pos) < 0.01:  # Stop when close to target
                    break

            # Move down to grasp CubeA
            target_pos = cube_pos
            target_pos[-1] -= 0.01  # Lower slightly for grasping
            for _ in range(100):
                curr_pos = obs["robot0_eef_pos"]
                delta_pos = target_pos - curr_pos
                action = np.concatenate([4 * delta_pos, [-1]])  # Keep gripper open
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                self.obs_dict = obs
                if np.linalg.norm(delta_pos) < 0.01:
                    break

            # Close the gripper to grasp CubeA
            for _ in range(25):
                action = np.concatenate([[0, 0, 0], [1]])  # Close gripper
                obs, reward, done, info = self.env.step(action)
                self.env.render()
                self.obs_dict = obs

                is_contact = self.env.check_contact(
                    geoms_1=["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"],
                    geoms_2=[self.selected_cube_geom_name]
                )
                if is_contact:
                    break

            # # Lift CubeA
            # target_pos = obs["robot0_eef_pos"] + np.array([0, 0, 0.15])  # Lift cube up
            # for _ in range(100):
            #     curr_pos = obs["robot0_eef_pos"]
            #     delta_pos = target_pos - curr_pos
            #     action = np.concatenate([5 * delta_pos, [1]])  # Keep gripper closed
            #     obs, reward, done, info = self.env.step(action)
            #     self.env.render()
            #     self.obs_dict = obs
            #     if np.linalg.norm(delta_pos) < 0.01:
            #         break
            #
            # # Move above CubeB
            # target_pos = obs["cubeB_pos"] + np.array([0, 0, 0.15])  # Move above CubeB
            # for _ in range(100):
            #     curr_pos = obs["robot0_eef_pos"]
            #     delta_pos = target_pos - curr_pos
            #     action = np.concatenate([5 * delta_pos, [1]])  # Keep gripper closed
            #     obs, reward, done, info = self.env.step(action)
            #     self.env.render()
            #     self.obs_dict = obs
            #     if np.linalg.norm(delta_pos) < 0.01:
            #         break
            #
            # # Lower CubeA onto CubeB
            # target_pos = obs["cubeB_pos"] + np.array([0, 0, 0.05])  # Slightly above CubeB
            # for _ in range(100):
            #     curr_pos = obs["robot0_eef_pos"]
            #     delta_pos = target_pos - curr_pos
            #     action = np.concatenate([4 * delta_pos, [1]])  # Keep gripper closed
            #     obs, reward, done, info = self.env.step(action)
            #     self.env.render()
            #     self.obs_dict = obs
            #     if np.linalg.norm(delta_pos) < 0.01:
            #         break
            #
            # # Release the gripper
            # for _ in range(25):
            #     action = np.concatenate([[0, 0, 0], [-1]])  # Open gripper
            #     obs, reward, done, info = self.env.step(action)
            #     self.env.render()
            #     self.obs_dict = obs

        # Render and save the initial frame to the video
        frame = self.env.sim.render(
            height=self.render_height,
            width=self.render_width,
            camera_name="frontview"
        )

        frame = np.flipud(frame)
        self.writer.append_data(frame)

        # Return the flattened observation
        flattened_observation = self.flatten_observation(obs)
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



