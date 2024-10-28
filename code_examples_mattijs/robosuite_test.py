import robosuite as suite
import numpy as np
import time
import pickle
from robosuite.utils.transform_utils import *
from robosuite.controllers import load_controller_config
from matplotlib import pyplot as plt

# Load controller config
controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config["control_delta"] = True

# Initialize environment
env = suite.make(
    env_name="Lift",                
    robots="Panda",                 
    controller_configs=controller_config,
    has_renderer=True,            
    has_offscreen_renderer=False, 
    use_camera_obs=False,         
    render_camera="frontview",    
)

demonstration = []
demonstration_imgs = []

init_block_pos = [0.2, 0.2]
goal_block_pos = [-0.05, -0.2]

env.placement_initializer.x_range = [init_block_pos[0], init_block_pos[0]]
env.placement_initializer.y_range = [init_block_pos[1], init_block_pos[1]]
env.placement_initializer.rotation = 0.0

obs = env.reset()
print(obs)

init_block_pos.append(obs['cube_pos'][-1])
goal_block_pos.append(obs['cube_pos'][-1])
init_eef_pos = obs['robot0_eef_pos']

# Define gripper actions
gripper_action_close = np.array([1])   # Close gripper
gripper_action_open = np.array([-1])   # Open gripper

# Move to initial position above the block
target_pos = obs["cube_pos"] + np.array([0, 0, 0.1])  # Move above block
env.render()
time.sleep(1)

# Move end-effector to initial target position above the cube
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos - curr_pos

    action = np.concatenate([5 * delta_pos, [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# Move down to the block for grasping
target_pos = obs["cube_pos"]
target_pos[-1] -= 0.01
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos - curr_pos
    
    action = np.concatenate([4 * delta_pos, [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# Close the gripper to grasp the block, checking for contact
for i in range(25):
    action = np.concatenate([[0, 0, 0], [0, 0, 0], gripper_action_close])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    
    # Check for contact between gripper fingers and cube
    is_contact = env.check_contact(
        geoms_1=["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"], 
        geoms_2=["cube_g0"]
    )
    print("Contact with cube:", is_contact)

    env.render()
    
# Lift the block up with the gripper closed
target_pos = obs["cube_pos"] + np.array([0, 0, 0.1])  # Lift block up
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos - curr_pos
    
    action = np.concatenate([4 * delta_pos, [0, 0, 0], gripper_action_close])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# Move to goal position
target_pos = np.array(goal_block_pos) + np.array([0, 0, 0.1])
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos - curr_pos

    action = np.concatenate([5 * delta_pos, [0, 0, 0], gripper_action_close])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# Place the block by lowering the gripper
target_pos = np.array(goal_block_pos)
target_pos[-1] -= 0.01
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos - curr_pos

    action = np.concatenate([5 * delta_pos, [0, 0, 0], gripper_action_close])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# Open the gripper to release the block
for i in range(25):
    action = np.concatenate([[0, 0, 0], [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()

# Move the end-effector up
target_pos = obs['robot0_eef_pos'] + [0, 0, 0.1]
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos - curr_pos

    action = np.concatenate([5 * delta_pos, [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# Move end-effector back to initial position
target_pos = init_eef_pos
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos - curr_pos

    action = np.concatenate([5 * delta_pos, [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# Plot demonstration trajectory
demonstration = np.array(demonstration)
plt.scatter(demonstration[:, 1], demonstration[:, 0])
plt.gca().invert_yaxis()
plt.show()

# Save demonstration trajectory
with open('demos.pkl', 'wb') as f:
    pickle.dump(demonstration, f)
