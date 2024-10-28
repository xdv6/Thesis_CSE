import robosuite as suite
import numpy as np

# create environment instance
env = suite.make(
    "Stack",
    robots="Panda",  # Using Panda robot
    use_object_obs=True,  # Include object observations
    has_renderer=True,  # Enable rendering for visualization
    reward_shaping=True,  # Use dense rewards for easier learning
    control_freq=20,  # Set control frequency for smooth simulation
    use_camera_obs=False,  # Disable camera observations
)

# reset the environment
obs = env.reset()

# Define gripper geometry names and object geometry name
gripper_geom_names = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
cube_geom_name = ["cubeA_g0"]

# Define a target height threshold (for example, 0.1 meters above the table surface)

obs = env.reset()

target_height_threshold = 0.85  # Adjust based on your environment (e.g., cubeA height + extra clearance)

while True:
    # Sample random action (8 joint positions for the Panda)

    action = np.random.randn(env.robots[0].dof)
    obs, reward, done, info = env.step(action)

    # Get the current end-effector position (z-coordinate)
    eef_position = obs["robot0_eef_pos"]  # End-effector position
    eef_height = eef_position[2]  # z-coordinate

    block_A = obs["cubeA_pos"]  # Block A position

    # Check if there's contact between the gripper and the cube
    is_contact = env.check_contact(geoms_1=gripper_geom_names, geoms_2=cube_geom_name)

    # Condition: Check if robot is above a target height and in contact with the cube
    is_above_target_height = eef_height > target_height_threshold

    if is_contact and is_above_target_height:
        print("The robot is holding the block and is above the target height.")

    print("Is the gripper in contact with the cube?", is_contact)
    print(f"End-effector height: {eef_height}, Is above target height: {is_above_target_height}")

    env.render()  # Render on display
