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
controller_config = load_controller_config(default_controller="OSC_POSITION")

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

# Set up placement initializer for consistent cube placement
placement_initializer = SequentialCompositeSampler(name="ObjectSampler")

placement_initializer.append_sampler(
    sampler=UniformRandomSampler(
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
    sampler=UniformRandomSampler(
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

# Wrap the environment with visualization wrapper
env = VisualizationWrapper(env, indicator_configs=None)

# Initialize the keyboard device for controlling the robot
device = Keyboard(pos_sensitivity=1.0, rot_sensitivity=1.0)
env.viewer.add_keypress_callback(device.on_press)

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

    # Move end-effector to initial position above cubeA
    target_pos = obs["cubeA_pos"] + np.array([0, 0, 0.1])  # Move above the cubeA
    for _ in range(100):
        curr_pos = obs['robot0_eef_pos']
        delta_pos = target_pos - curr_pos

        # Create action to move towards the block while keeping the gripper open
        action = np.concatenate([5 * delta_pos, [-1]])  # Gripper open
        obs, reward, done, info = env.step(action)
        env.render()

        # Stop moving if target is reached
        if np.linalg.norm(delta_pos) < 0.01:
            break

    # Move down to grasp the block
    target_pos = obs["cubeA_pos"]
    target_pos[-1] -= 0.01  # Move slightly lower to grasp
    for _ in range(100):
        curr_pos = obs['robot0_eef_pos']
        delta_pos = target_pos - curr_pos

        # Create action to move towards the block while keeping the gripper open
        action = np.concatenate([4 * delta_pos, [-1]])  # Gripper open
        obs, reward, done, info = env.step(action)
        env.render()

        # Stop moving if target is reached
        if np.linalg.norm(delta_pos) < 0.01:
            break

    # Close the gripper to grasp the block
    for _ in range(25):
        action = np.concatenate([[0, 0, 0], [1]])  # Close gripper
        obs, reward, done, info = env.step(action)

        # Check for contact between gripper fingers and cubeA
        is_contact = env.check_contact(
            geoms_1=gripper_geom_names,
            geoms_2=cubeA_geom_name
        )
        print("Contact with cubeA:", is_contact)

        env.render()

    # Start the manual control using keyboard
    device.start_control()  # Start listening for keyboard input
    stack_timer = 0.0  # Reset the timer
    start_time = time.time()

    print("Done grasping, starting manual control.")
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

        # Render the environment to visualize the robot's action
        env.render()
