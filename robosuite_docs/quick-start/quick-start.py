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
env.reset()

# Define gripper geometry names and object geometry name
gripper_geom_names = ["left_fingerpad", "right_fingerpad"]
cube_geom_name = "cubeA"  # Adjust this to the correct name for your object

print("Available Geometries in Environment:")
print(env.sim.model.geom_names)


while True:
    # sample random action (8 joint positions for the Panda)
    action = np.random.randn(env.robots[0].dof)
    obs, reward, done, info = env.step(action)

    # Check if there's contact between the gripper and the cube
    is_contact = env.check_contact(geoms_1=gripper_geom_names, geoms_2=[cube_geom_name])
    print("Is the gripper in contact with the cube?", is_contact)
    break
    env.render()  # Render on display



