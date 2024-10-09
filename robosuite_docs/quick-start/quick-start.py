import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Stack", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

while True:
    # print("robot joint positions", env.robots[0].dof)  # print robot joint positions
    action = np.random.randn(env.robots[0].dof) # sample random action (in this case, 8 joint positions)
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display