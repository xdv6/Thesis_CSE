import robosuite as suite
import numpy as np
from robosuite import load_controller_config
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
import time
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
import random
import math
import pickle


# stack env in file: miniconda3/envs/RM_robosuite/lib/python3.7/site-packages/robosuite/environments/manipulation/stack.py

# Load controller configuration
controller_config = load_controller_config(default_controller="OSC_POSE")

# Create environment instance with the given configuration
env = suite.make(
    "Stack",
    robots="Panda",  # Using Panda robot
    controller_configs=controller_config,
    use_object_obs=True,  # Include object observations
    has_renderer=True,  # Enable rendering for visualization
    reward_shaping=True,  # Use dense rewards for easier learning
    control_freq=20,  # Set control frequency for smooth simulation
    horizon=1000000,
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
#     sampler = UniformRandomSampler(
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

# placement_initializer.append_sampler(
#     # Create a placement initializer with a y_range and dynamically updated x_range
#     sampler = UniformRandomSampler(
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


# # Assign the cubes to the correct sampler after creating them
# cubes = [env.cubeA, env.cubeB]  # Assuming cubeA and cubeB are accessible here
# env.placement_initializer.add_objects(cubes)

# Wrap the environment with visualization wrapper
env = VisualizationWrapper(env, indicator_configs=None)

# Initialize the keyboard device for controlling the robot
device = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
env.viewer.add_keypress_callback(device.on_press)

# Set target height threshold for checking height
target_height_threshold = 0.85  # Adjust based on your environment setup

# Define gripper geometry names and cube geometry names
gripper_geom_names = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
cubeA_geom_name = ["cubeA_g0"]
cubeB_geom_name = ["cubeB_g0"]

# Timer to track how long cubeA is above cubeB and in contact
stack_timer = 0.0
stack_threshold = 5.0  # Threshold time in seconds to consider cubeA as "stacked" on cubeB


last_message = None

temp_left_finger_pos = 0.0

def map_values(value, min_value, max_value, linear=True, steepness=1):
    """
    Maps a value to a normalized range [0,1] using linear or exponential scaling.
    """
    # Normalize the value between 0 and 1
    normalized_value = (value - min_value) / (max_value - min_value) 
    
    # Apply exponential mapping if non-linear
    if not linear:
        normalized_value = math.exp(-normalized_value * steepness)
    else:
        normalized_value = 1 - normalized_value
    
    # Clip value between 0 and 1
    return max(0, min(1, normalized_value))



# def calculate_reward_gripper_to_cube():
#     reward = 0.0
#     cube_width = env.cubeA.size[0] * 2
#     cube_pos_A = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeA_main")]
#     cube_pos_B = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeB_main")]

#     left_finger_pos = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_finger_joint1_tip")]
#     right_finger_pos = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_finger_joint2_tip")]

#     left_dist = np.linalg.norm(left_finger_pos - np.array([cube_pos_A[0], cube_pos_A[1] - cube_width / 2, cube_pos_A[2]]))
#     right_dist = np.linalg.norm(right_finger_pos - np.array([cube_pos_A[0], cube_pos_A[1] + cube_width / 2, cube_pos_A[2]]))
#     reward -= (left_dist + right_dist) * 10
#     return reward

def calculate_reward_gripper_to_cube():
    reward = 0.0
    cube_width = env.cubeA.size[0] * 2
    cube_pos_A = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeA_main")]

    left_finger_pos = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_finger_joint1_tip")]
    right_finger_pos = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_finger_joint2_tip")]

    left_dist = np.linalg.norm(left_finger_pos - np.array([cube_pos_A[0], cube_pos_A[1] - cube_width / 2, cube_pos_A[2]]))
    right_dist = np.linalg.norm(right_finger_pos - np.array([cube_pos_A[0], cube_pos_A[1] + cube_width / 2, cube_pos_A[2]]))
    
    # Use negative distance as reward (closer = higher reward)
    reward +=  0.5 / (left_dist + right_dist + 0.01)   # Adding 0.01 to avoid division by zero

    return reward



def calculate_reward_cube_A_to_cube_B_full():
    reward = 0.0
    cube_pos_A = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeA_main")]
    cube_pos_B = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeB_main")]

    # Compute central points of the bottom face of cube A and top face of cube B
    bottom_of_A = np.array([
        cube_pos_A[0],  # x-coordinate remains the same
        cube_pos_A[1],  # y-coordinate remains the same
        cube_pos_A[2] - env.cubeA.size[2]  # Bottom surface of cubeA
    ])

    top_of_B = np.array([
        cube_pos_B[0],  # x-coordinate remains the same
        cube_pos_B[1],  # y-coordinate remains the same
        cube_pos_B[2] + env.cubeB.size[2]  # Top surface of cubeB
    ])

    # Compute full Euclidean distance
    distance = abs(np.linalg.norm(bottom_of_A - top_of_B) )

    # Penalize based on the full distance (not just z)
    reward += 30* (2 / (distance + 0.01))

    return reward

def calculate_reward_cube_A_to_cube_B_xy():
    reward = 0.0
    cube_pos_A = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeA_main")]
    cube_pos_B = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeB_main")]

    # Compute XY-plane Euclidean distance
    distance_xy = np.linalg.norm(cube_pos_A[:2] - cube_pos_B[:2])

    # Penalize based on the XY distance
    reward += 2* (5 / (distance_xy + 0.01))

    return reward



def calculate_reward_cube_A_to_cube_B():
    reward = 0.0
    cube_pos_A = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeA_main")]
    cube_pos_B = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeB_main")]

    bottom_of_A = cube_pos_A[2] - env.cubeA.size[2]  # Bottom surface of cubeA
    top_of_B = cube_pos_B[2] + env.cubeB.size[2]  # Top surface of cubeB

    distance = abs(bottom_of_A - top_of_B)  # Correct distance
    reward += 1 / (distance + 0.01)  # Penalize based on absolute distance
    return reward


contact_check = False
# Main control loop
while True:
    # Reset the environment
    obs = env.reset()

    device.start_control()  # Start listening for keyboard input
    stack_timer = 0.0  # Reset the timer
    start_time = time.time()


    table_geom_id = env.sim.model.geom_name2id("table_collision")  # Correct table collision name

    # 🚀 ABSOLUTE MAXIMUM stiffness for contact resolution 🚀
    env.sim.model.geom_solref[table_geom_id] = [0.00001, 1]  # Hardest contact resolution possible
    env.sim.model.geom_solimp[table_geom_id] = [1, 1, 0.0, 0.0, 10]  # Hardest possible contact surface

    # 🚀 ABSOLUTE MAXIMUM friction to prevent sliding or sinking 🚀
    env.sim.model.geom_friction[table_geom_id] = [100.0, 10.0, 1.0]  #


    # test loading simulation from file
    with open('state.pkl', 'rb') as f:
        state = pickle.load(f)
    env.sim.set_state(state)
    env.sim.forward()
    print("Simulation state loaded from file.")
    

    while True:
        # Get the newest action from the keyboard device
        action, grasp = input2action(
            device=device,
            robot=env.robots[0],
            active_arm="right",
            env_configuration="default"
        )

        # If action is None, it indicates a reset (e.g., pressing "q" on the keyboard)
        if action is None:
            # test saving simulation to file
            # state = env.sim.get_state()
            # with open('state.pkl', 'wb') as f:
            #     pickle.dump(state, f)
            # print("Simulation state saved to file.")

            break

        # Step the environment with the provided action
        obs, reward, done, info = env.step(action)
        

        # If the environment is done, break and reset
        if done:
            print("Episode finished, resetting environment.")
            break

        # Get the current end-effector position (z-coordinate)
        eef_position = obs["robot0_eef_pos"]  # End-effector position from observation
        eef_height = eef_position[2]  # Extracting the z-coordinate (height)

        # Get the current block positions for reference
        block_A = obs["cubeA_pos"]  # Get cubeA position
        block_B = obs["cubeB_pos"]  # Get cubeB position


        # Define gripper collision geoms
        left_gripper_geom = ["gripper0_finger1_pad_collision"]  # Left gripper pad
        right_gripper_geom = ["gripper0_finger2_pad_collision"]  # Right gripper pad
        

        # Define cube collision geom 
        cube_geom = ["cubeA_g0"]

        # 1️⃣ Step 1: Check for contact
        left_contact = env.check_contact(geoms_1=left_gripper_geom, geoms_2=cube_geom)
        right_contact = env.check_contact(geoms_1=right_gripper_geom, geoms_2=cube_geom)

        # 2️⃣ Step 2: Get gripper pad positions
        left_finger_pos_pad = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_leftfinger")]
        right_finger_pos_pad = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_rightfinger")]

        left_finger_pos = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_finger_joint1_tip")]
        right_finger_pos = env.sim.data.body_xpos[env.sim.model.body_name2id("gripper0_finger_joint2_tip")]

        # 3️⃣ Step 3: Get cube center position
        cube_pos = env.sim.data.body_xpos[env.sim.model.body_name2id("cubeA_main")]

        # 4️⃣ Step 4: Get cube width (assuming size is [0.02, 0.02, 0.02])
        cube_width = env.cubeA.size[0] * 2  

        # 5️⃣ Step 5: Ensure contact is on the correct faces
        left_touching_left_face = left_contact and (abs(left_finger_pos_pad[1] - (cube_pos[1] - cube_width / 2)) < 0.005)
        right_touching_right_face = right_contact and (abs(right_finger_pos_pad[1] - (cube_pos[1] + cube_width / 2)) < 0.005)

        # 6️⃣ Step 6: Final check → Both contacts must be on the correct sides
        is_proper_grasp = left_touching_left_face and right_touching_right_face

        # Condition: Check if the robot's end-effector is above the target height and in contact with cubeA
        is_above_target_height = eef_height > target_height_threshold

        # check if the bottom of cubeA is above the top of cubeB
        block_A_above_B = block_A[2] - env.cubeA.size[2] > 0.91

        # if height of cubeA is above 0.94 or below 0.88 then reset the environment
        # if last_message == "The robot is holding the block and block A is above block B." and (block_A[2] > 0.94 or block_A[2] < 0.88):
        #     print("Episode finished, resetting environment.")
        #     last_message = ""
        #     break

        # Condition for third event: Check if the robot is still above cubeB, still in contact with cubeA, and x, y coordinates are above cubeB
        is_above_cubeB = (
            block_B[0] - 0.025 <= eef_position[0] <= block_B[0] + 0.025 and
            block_B[1] - 0.025 <= eef_position[1] <= block_B[1] + 0.025
        )

        # Condition for fourth event: Check if cubeA is positioned above cubeB and they are in contact
        is_cubeA_above_cubeB = (
            block_B[0] - 0.025 <= block_A[0] <= block_B[0] + 0.025 and
            block_B[1] - 0.025 <= block_A[1] <= block_B[1] + 0.025 and
            block_A[2] > block_B[2]
        )
        are_blocks_in_contact = env.check_contact(geoms_1=cubeA_geom_name, geoms_2=cubeB_geom_name)

        # Condition for fifth event: Check if cubeA is above cubeB, in contact for more than 5 seconds, and the robot is not in contact with cubeA
        is_robot_not_in_contact_with_cubeA = not is_proper_grasp
        if is_cubeA_above_cubeB and are_blocks_in_contact:
            stack_timer += time.time() - start_time
        else:
            stack_timer = 0.0
        start_time = time.time()

        message = None


        if is_proper_grasp:
            message = "The robot is correctly in contact with the block."

        # if last message had block gripped, but now it doesn't, print message block is dropped
        if last_message and not is_proper_grasp:
            message = "Block is dropped."


        # Print message if conditions for second event are met: robot is holding the block and is above the target height
        if is_proper_grasp and block_A_above_B:
            message = "The robot is holding the block and block A is above block B."

        # Print message if conditions for third event are met: robot is holding block A, above block B in x, y, and above the target height
        if is_proper_grasp and is_above_cubeB:
            message = "The robot is holding block A and is positioned above block B."

        # # Print message if conditions for fourth event are met: cubeA is above cubeB and they are in contact
        if is_cubeA_above_cubeB and are_blocks_in_contact:
            message = "Cube A is above Cube B and they are in contact."

        # # Print message if conditions for fifth event are met: cubeA is stacked on cubeB for more than 5 seconds and the robot is not in contact with cubeA
        # if stack_timer > stack_threshold and is_robot_not_in_contact_with_cubeA:
        #     message = "Cube A is stacked on Cube B for more than 5 seconds and the robot is not in contact with Cube A."

        if message and message != last_message:
            print(message)
            last_message = message

        # reward debugging: 

        # reward = calculate_reward_gripper_to_cube()
        # reward = calculate_reward_cube_A_to_cube_B()
        reward = calculate_reward_cube_A_to_cube_B_xy()
        # reward = calculate_reward_cube_A_to_cube_B_full()
        # print("Reward gripper to cube: ", reward)


        # print("cubeA position: ", block_A)

        # Render the environment to visualize the robot's action
        env.render()