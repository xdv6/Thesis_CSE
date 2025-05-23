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
        cube_width = self.env.cubeA.size[0] * 2
        cube_pos_A = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeA_main")]
        cube_pos_B = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeB_main")]

        left_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint1_tip")]
        right_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_finger_joint2_tip")]

        left_dist = np.linalg.norm(left_finger_pos - np.array([cube_pos_A[0], cube_pos_A[1] - cube_width / 2, cube_pos_A[2]]))
        right_dist = np.linalg.norm(right_finger_pos - np.array([cube_pos_A[0], cube_pos_A[1] + cube_width / 2, cube_pos_A[2]]))
        reward += 0.5 / (left_dist + right_dist + 0.01) # Adding 0.01 to avoid division by zero

        wandb.log({"left_dist": left_dist})
        wandb.log({"right_dist": right_dist})
        return reward

    def calculate_reward_cube_A_to_cube_B(self):

        reward = 0.0

        cube_pos_A = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeA_main")]
        cube_pos_B = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeB_main")]

        bottom_of_A = cube_pos_A[2] - self.env.cubeA.size[2]  # Bottom surface of cubeA
        top_of_B = cube_pos_B[2] + self.env.cubeB.size[2]  # Top surface of cubeB

        distance = abs(bottom_of_A - top_of_B)  # Correct distance
        reward += 1 / (distance + 0.01) # Penalize based on absolute distance
        # reward -= distance * 10  # Penalize based on absolute distance

        if self.block_gripped and not self.block_grasped():
            reward = -30
        wandb.log({"distance_between_cubeA_and_cubeB": distance})
        return reward

    def calculate_reward_cube_A_to_cube_B_xy(self):
        reward = 0.0
        cube_pos_A = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeA_main")]
        cube_pos_B = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeB_main")]

        # Compute XY-plane Euclidean distance
        distance_xy = np.linalg.norm(cube_pos_A[:2] - cube_pos_B[:2])

        # Penalize based on the XY distance
        reward += 2 * (5 / (distance_xy + 0.01))
        if self.block_gripped and not self.block_grasped():
            reward = -30
        wandb.log({"distance_xy_between_cubeA_and_cubeB": distance_xy})

        return reward


    def calculate_reward_cube_A_to_cube_B_full(self):
        reward = 0.0
        cube_pos_A = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeA_main")]
        cube_pos_B = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeB_main")]

        # Compute central points of the bottom face of cube A and top face of cube B
        bottom_of_A = np.array([
            cube_pos_A[0],  # x-coordinate remains the same
            cube_pos_A[1],  # y-coordinate remains the same
            cube_pos_A[2] - self.env.cubeA.size[2]  # Bottom surface of cubeA
        ])

        top_of_B = np.array([
            cube_pos_B[0],  # x-coordinate remains the same
            cube_pos_B[1],  # y-coordinate remains the same
            cube_pos_B[2] + self.env.cubeB.size[2]  # Top surface of cubeB
        ])

        # Compute full Euclidean distance
        distance = abs(np.linalg.norm(bottom_of_A - top_of_B))


        # Penalize based on the full distance (not just z)
        reward +=  5 / (distance + 0.01)

        if self.block_gripped and not self.block_grasped():
            reward = -30
        wandb.log({"distance_full_between_cubeA_and_cubeB": distance})

        return reward

    def calculate_reward_cube_A_to_tresh_above_cube_B(self):
        reward = 0.0
        cube_pos_A = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeA_main")]
        cube_pos_B = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeB_main")]

        # Compute central points of the bottom face of cube A and top face of cube B
        A = np.array([
            cube_pos_A[0],  # x-coordinate remains the same
            cube_pos_A[1],  # y-coordinate remains the same
            cube_pos_A[2]
        ])

        above_B_treshold = np.array([
            cube_pos_B[0],  # x-coordinate remains the same
            cube_pos_B[1],  # y-coordinate remains the same
            0.942
        ])

        # Compute full Euclidean distance
        distance = abs(np.linalg.norm(A - above_B_treshold))

        # Penalize based on the full distance (not just z)
        reward += 2 / (distance + 0.01)

        if self.block_gripped and not self.block_grasped():
            reward = -30

        wandb.log({"distance_above_tresh_between_cubeA_and_cubeB": distance})
        return reward



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
            horizon=1000,
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

        # placement_initializer.append_sampler(
        #     # Create a placement initializer with a y_range and dynamically updated x_range
        #     sampler=UniformRandomSampler(
        #         name="ObjectSamplerCubeC",
        #         x_range=[-0.2, -0.2],
        #         y_range=[-0.2, -0.2],
        #         rotation=0.0,
        #         ensure_object_boundary_in_range=False,
        #         ensure_valid_placement=True,
        #         reference_pos=(0, 0, 0.8),
        #         z_offset=0.01,
        #     )
        # )
        #
        # placement_initializer.append_sampler(
        #     # Create a placement initializer with a y_range and dynamically updated x_range
        #     sampler=UniformRandomSampler(
        #         name="ObjectSamplerCubeD",
        #         x_range=[-0.07, -0.07],
        #         y_range=[-0.07, -0.07],
        #         rotation=0.0,
        #         ensure_object_boundary_in_range=False,
        #         ensure_valid_placement=True,
        #         reference_pos=(0, 0, 0.8),
        #         z_offset=0.01,
        #     )
        # )

        placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeA", mujoco_objects=env.cubeA)
        placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeB", mujoco_objects=env.cubeB)

        # placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeC", mujoco_objects=env.cubeC)
        # placement_initializer.add_objects_to_sampler(sampler_name="ObjectSamplerCubeD", mujoco_objects=env.cubeD)

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

        # Timer to track how long cubeA is above cubeB and in contact
        self.stack_timer = 0.0
        self.stack_threshold = 5.0  # Threshold time in seconds to consider cubeA as "stacked" on cubeB
        self.start_time = time.time()

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
            "cubeA_to_cubeB",  # Relative position vector between cubeA and cubeB
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
        if action[-1] <= 0:
            action[-1] = -1
        else:
            action[-1] = 1

        next_obs, reward, done, info = self.env.step(action)

        # if cube is dropped after it was picked up, then the episode is done
        if self.block_gripped and not self.block_grasped():
            done = True

        self.obs_dict = next_obs
        # add the reward_for_gripper_to_cube to the obs_dict
        self.obs_dict["reward_gripper_to_cube"] = self.calculate_reward_gripper_to_cube()
        self.obs_dict["reward_cube_A_to_cube_B"] = self.calculate_reward_cube_A_to_cube_B()
        self.obs_dict["reward_cube_A_to_cube_B_xy"] = self.calculate_reward_cube_A_to_cube_B_xy()
        self.obs_dict["reward_cube_A_to_cube_B_full"] = self.calculate_reward_cube_A_to_cube_B_full()
        self.obs_dict["reward_cube_A_to_tresh_above_cube_B"] = self.calculate_reward_cube_A_to_tresh_above_cube_B()

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
        # is_contact_with_cubeA = self.env.check_contact(
        #     geoms_1=["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"],
        #     geoms_2=["cubeA_g0"]
        # )

        # Define gripper collision geoms
        left_gripper_geom = ["gripper0_finger1_pad_collision"]  # Left gripper pad
        right_gripper_geom = ["gripper0_finger2_pad_collision"]  # Right gripper pad

        # Define cube collision geom
        cube_geom = ["cubeA_g0"]

        # 1️⃣ Step 1: Check for contact
        left_contact = self.env.check_contact(geoms_1=left_gripper_geom, geoms_2=cube_geom)
        right_contact = self.env.check_contact(geoms_1=right_gripper_geom, geoms_2=cube_geom)

        # 2️⃣ Step 2: Get gripper pad positions
        left_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_leftfinger")]
        right_finger_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("gripper0_rightfinger")]

        # 3️⃣ Step 3: Get cube center position
        cube_pos = self.env.sim.data.body_xpos[self.env.sim.model.body_name2id("cubeA_main")]

        # 4️⃣ Step 4: Get cube width (assuming size is [0.02, 0.02, 0.02])
        cube_width = self.env.cubeA.size[0] * 2

        # 5️⃣ Step 5: Ensure contact is on the correct faces
        left_touching_left_face = left_contact and (abs(left_finger_pos[1] - (cube_pos[1] - cube_width / 2)) < 0.005)
        right_touching_right_face = right_contact and (abs(right_finger_pos[1] - (cube_pos[1] + cube_width / 2)) < 0.005)

        # 6️⃣ Step 6: Final check → Both contacts must be on the correct sides
        is_proper_grasp = left_touching_left_face and right_touching_right_face

        if is_proper_grasp:
            self.block_gripped = True

        return is_proper_grasp

    def above_block_b_and_grasped(self):
        # Check if the end-effector is above the height of cubeB while still grasping cubeA
        # obs = self.obs_dict
        # eef_height = obs["robot0_eef_pos"][2]  # z-coordinate of end-effector position
        # cube_b_height = obs["cubeB_pos"][2]  # z-coordinate of cubeB
        # is_above_cube_b = eef_height > cube_b_height
        # return is_above_cube_b
        block_A = self.obs_dict["cubeA_pos"]
        block_B = self.obs_dict["cubeB_pos"]
        # block_A_above_B = block_A[2] - self.env.cubeA.size[2] > block_B[2] + self.env.cubeB.size[2] + 0.02
        block_A_above_B = block_A[2] - self.env.cubeA.size[2] > 0.91
        return block_A_above_B


    def above_block_b_in_xy_and_grasped(self):
        # Check if the end-effector is above cubeB in the x and y plane while still grasping cubeA
        obs = self.obs_dict

        cube_a_pos = obs["cubeA_pos"]  # Position of cubeA
        cube_b_pos = obs["cubeB_pos"]  # Position of cubeB

        # Define an allowable margin to be considered "above" in the x, y plane
        is_above_cubeB = (
            cube_b_pos[0] - 0.025 <= cube_a_pos[0] <= cube_b_pos[0] + 0.025 and
            cube_b_pos[1] - 0.025 <= cube_a_pos[1] <= cube_b_pos[1] + 0.025 and
            0.93 <= cube_a_pos[2] <= 0.95  # Ensuring the height is within the range
        )

        return is_above_cubeB

    def     cube_a_above_cube_b_and_in_contact(self):
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
        return self.stack_timer > self.stack_threshold and is_robot_not_in_contact        # test loading simulation from file

    def block_dropped(self):
        # Check if the robot has dropped the block (i.e., no longer in contact with cubeA)
        return not self.block_grasped()

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
        self.obs_dict["reward_gripper_to_cube"] = self.calculate_reward_gripper_to_cube()
        self.obs_dict["reward_cube_A_to_cube_B"] = self.calculate_reward_cube_A_to_cube_B()
        self.obs_dict["reward_cube_A_to_cube_B_xy"] = self.calculate_reward_cube_A_to_cube_B_xy()
        self.obs_dict["reward_cube_A_to_cube_B_full"] = self.calculate_reward_cube_A_to_cube_B_full()
        self.obs_dict["reward_cube_A_to_tresh_above_cube_B"] = self.calculate_reward_cube_A_to_tresh_above_cube_B()

        move_gripper_to_cube = False
        if self.start_state_value == 1:
            move_gripper_to_cube = True

        elif self.start_state_value == -1:
            # choose random start state
            move_gripper_to_cube = random.choice([True, False])

        if move_gripper_to_cube:
            # Move the gripper above CubeA
            target_pos = obs["cubeA_pos"] + np.array([0, 0, 0.1])  # Slightly above CubeA
            for _ in range(100):
                curr_pos = obs["robot0_eef_pos"]
                delta_pos = target_pos - curr_pos
                action = np.concatenate([5 * delta_pos, [-1]])  # Keep gripper open
                obs, reward, done, info = self.env.step(action)
                # self.env.render()
                self.obs_dict = obs
                if np.linalg.norm(delta_pos) < 0.01:  # Stop when close to target
                    break

            # Move down to grasp CubeA
            target_pos = obs["cubeA_pos"]
            target_pos[-1] -= 0.01  # Lower slightly for grasping
            for _ in range(100):
                curr_pos = obs["robot0_eef_pos"]
                delta_pos = target_pos - curr_pos
                action = np.concatenate([4 * delta_pos, [-1]])  # Keep gripper open
                obs, reward, done, info = self.env.step(action)
                # self.env.render()
                self.obs_dict = obs
                if np.linalg.norm(delta_pos) < 0.01:
                    break

            # Close the gripper to grasp CubeA
            for _ in range(25):
                action = np.concatenate([[0, 0, 0], [1]])  # Close gripper
                obs, reward, done, info = self.env.step(action)
                # self.env.render()
                self.obs_dict = obs
                is_contact = self.env.check_contact(
                    geoms_1=["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"],
                    geoms_2=["cubeA_g0"]
                )
                if is_contact:
                    break

            # Lift CubeA
            target_pos = obs["robot0_eef_pos"] + np.array([0, 0, 0.15])  # Lift cube up
            for _ in range(100):
                curr_pos = obs["robot0_eef_pos"]
                delta_pos = target_pos - curr_pos
                action = np.concatenate([5 * delta_pos, [1]])  # Keep gripper closed
                obs, reward, done, info = self.env.step(action)
                # self.env.render()
                self.obs_dict = obs
                if np.linalg.norm(delta_pos) < 0.01:
                    break
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



