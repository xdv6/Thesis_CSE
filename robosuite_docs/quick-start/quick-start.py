import robosuite as suite
import numpy as np
from robosuite import load_controller_config
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
import time
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
import random

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
    horizon=1000,
    use_camera_obs=False,  # Disable camera observations
)

# put line 358 en 359 from stack environment in comments 
placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

placement_initializer.append_sampler(
    # Create a placement initializer with a y_range and dynamically updated x_range
    sampler = UniformRandomSampler(
        name="ObjectSamplerCubeA",
        x_range=[0.0, 0.0],
        y_range=[0.0, 0.0],
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
        x_range=[0.1, 0.1],
        y_range=[0.1, 0.1],
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

# Main control loop
while True:
    # Reset the environment
    obs = env.reset()

    # print location of cubeA and cubeB
    print("Cube A position: ", obs["cubeA_pos"])
    print("Cube B position: ", obs["cubeB_pos"])
    print("End-effector position: ", obs["robot0_eef_pos"])
    
    device.start_control()  # Start listening for keyboard input
    stack_timer = 0.0  # Reset the timer
    start_time = time.time()

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

        # Check if there's contact between the gripper and cubeA
        is_contact = env.check_contact(geoms_1=gripper_geom_names, geoms_2=cubeA_geom_name)

        # Condition: Check if the robot's end-effector is above the target height and in contact with cubeA
        is_above_target_height = eef_height > target_height_threshold

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
        is_robot_not_in_contact_with_cubeA = not is_contact
        if is_cubeA_above_cubeB and are_blocks_in_contact:
            stack_timer += time.time() - start_time
        else:
            stack_timer = 0.0
        start_time = time.time()

        # Print message if conditions for second event are met: robot is holding the block and is above the target height
        if is_contact and is_above_target_height:
            print("The robot is holding the block and is above the target height.")

        # Print message if conditions for third event are met: robot is holding block A, above block B in x, y, and above the target height
        if is_contact and is_above_cubeB:
            print("The robot is holding block A and is positioned above block B.")

        # Print message if conditions for fourth event are met: cubeA is above cubeB and they are in contact
        if is_cubeA_above_cubeB and are_blocks_in_contact:
            print("Cube A is above Cube B and they are in contact.")

        # Print message if conditions for fifth event are met: cubeA is stacked on cubeB for more than 5 seconds and the robot is not in contact with cubeA
        if stack_timer > stack_threshold and is_robot_not_in_contact_with_cubeA:
            print("Cube A is stacked on Cube B for more than 5 seconds and the robot is not in contact with Cube A.")

        # Render the environment to visualize the robot's action
        env.render()
