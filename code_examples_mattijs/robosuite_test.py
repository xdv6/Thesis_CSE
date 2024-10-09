import robosuite as suite
import time
import pickle
from robosuite.utils.transform_utils import *
from robosuite.controllers import load_controller_config
from transform import *
from matplotlib import pyplot as plt

controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config["control_delta"] = True

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

init_block_pos.append(obs['cube_pos'][-1])
goal_block_pos.append(obs['cube_pos'][-1])
init_eef_pos = obs['robot0_eef_pos']

# Define action parameters
gripper_action_close = np.array([1])   # Close gripper
gripper_action_open = np.array([-1])   # Open gripper

# Set initial position above the block
target_pos = obs["cube_pos"] + np.array([0, 0, 0.1])  # Move above block

env.render()
time.sleep(1)

for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos-curr_pos

    print(f'eef: {curr_pos}, target: {target_pos}, delta:{np.linalg.norm(delta_pos)}')
    
    # Move the end-effector to the target position (above the block)
    action = np.concatenate([5*delta_pos, [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# update target
target_pos = obs["cube_pos"]
target_pos[-1] -= 0.01

for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos-curr_pos
         
    # print(f'eef: {curr_ori}, target: {target_ori}, delta:{delta_ori[-1]}')
    print(f'eef: {curr_pos}, target: {target_pos}, delta:{np.linalg.norm(delta_pos)}')
    
    # Move the end-effector to the target position (above the block)
    action = np.concatenate([4*delta_pos, [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break
    
# Close the gripper to grasp the block
for i in range(25):
    action = np.concatenate([[0, 0, 0], [0, 0, 0], gripper_action_close])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
# Lift the block slightly
target_pos = obs["cube_pos"] + np.array([0, 0, 0.1])  # Lift the block up
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos-curr_pos
    
    action = np.concatenate([4*delta_pos, [0, 0, 0], gripper_action_close])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# udpate target
target_pos = goal_block_pos + np.array([0, 0, 0.1])

# move above goal pos
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos-curr_pos

    print(f'eef: {curr_pos}, target: {target_pos}, delta:{np.linalg.norm(delta_pos)}')
    
    # Move the end-effector to the target position (above the block)
    action = np.concatenate([5*delta_pos, [0, 0, 0], gripper_action_close])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break
    
# udpate target
target_pos = goal_block_pos
target_pos[-1] -= 0.01

# move above goal pos
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos-curr_pos

    print(f'eef: {curr_pos}, target: {target_pos}, delta:{np.linalg.norm(delta_pos)}')
    
    # Move the end-effector to the target position (above the block)
    action = np.concatenate([5*delta_pos, [0, 0, 0], gripper_action_close])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

# Open the gripper
for i in range(25):
    action = np.concatenate([[0, 0, 0], [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
# udpate target
target_pos = obs['robot0_eef_pos'] + [0, 0, 0.1]

# move eef up
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos-curr_pos

    print(f'eef: {curr_pos}, target: {target_pos}, delta:{np.linalg.norm(delta_pos)}')
    
    # Move the end-effector to the target position (above the block)
    action = np.concatenate([5*delta_pos, [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break
    
# move eef to initial position
for i in range(100):
    curr_pos = obs['robot0_eef_pos']
    delta_pos = target_pos-curr_pos

    print(f'eef: {curr_pos}, target: {target_pos}, delta:{np.linalg.norm(delta_pos)}')
    
    # Move the end-effector to the target position (above the block)
    action = np.concatenate([5*delta_pos, [0, 0, 0], gripper_action_open])
    obs, reward, done, info = env.step(action)
    demonstration.append(obs['cube_pos'])
    env.render()
    
    if np.linalg.norm(delta_pos) < 0.01:
        break

demonstration = np.array(demonstration)
plt.scatter(demonstration[:,1], demonstration[:,0])
plt.gca().invert_yaxis()
plt.show()

with open('demos.pkl', 'wb') as f:
    pickle.dump(demonstration, f)