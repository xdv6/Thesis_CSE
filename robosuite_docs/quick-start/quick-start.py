import robosuite as suite
import numpy as np
from robosuite import load_controller_config
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper

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
    use_camera_obs=False,  # Disable camera observations
)

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

# Main control loop
while True:
    # Reset the environment
    obs = env.reset()
    device.start_control()  # Start listening for keyboard input


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
            block_B[1] - 0.025 <= eef_position[1] <= block_B[1] + 0.0
        )

        # Print message if conditions for second event are met: robot is holding the block and is above the target height
        if is_contact and is_above_target_height:
            print("The robot is holding the block and is above the target height.")

        # Print message if conditions for third event are met: robot is holding block A, above block B in x, y, and above the target height
        if is_contact and is_above_cubeB:
            print("The robot is holding block A and is positioned above block B.")

        # Debug prints to show the current state of contact and position
        # print("Is the gripper in contact with the cube?", is_contact)
        # print(f"End-effector height: {eef_height}, Is above target height: {is_above_target_height}")
        # print(f"Is above cube B: {is_above_cubeB}")

        # Render the environment to visualize the robot's action
        env.render()
