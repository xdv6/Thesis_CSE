import numpy as np
import robosuite as suite

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

env._

while True:
    # print("robot joint positions", env.robots[0].dof)  # print robot joint positions
    action = np.random.randn(env.robots[0].dof)  # sample random action (in this case, 8 joint positions)
    obs, reward, done, info = env.step(action)  # take action in the environment
    gripper = env.robots[0].gripper  # get gripper object
    env.c
    print("gripper", gripper)  # print gripper object
    # print("obs", obs)
    # break
    env.render()  # render on display