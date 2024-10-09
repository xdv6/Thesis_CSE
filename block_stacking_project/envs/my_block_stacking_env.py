import robosuite as suite
import os
import sys

# Ensure block_stacking_project is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reward_machines.envs.reward_machine_env import RewardMachineEnv

class MyBlockStackingEnv(suite.make):
    def __init__(self):
        # Initialize robosuite environment for block stacking
        env = suite.make(
            "PickPlace",
            robots="Panda",  # Use Panda robot
            use_object_obs=True,  # Include object observations in observations
            has_renderer=True  # For rendering the environment visually
        )
        super().__init__(env)

    def step(self, action):
        # Execute action and store observation info for event tracking
        next_obs, original_reward, env_done, info = self.env.step(action)
        self.info = info  # Store information for event tracking
        return next_obs, original_reward, env_done, info

    def get_events(self):
        # Define the events for the reward machine
        events = ''
        if self.block_grasped():
            events += 'g'  # 'g' event when block is grasped
        if self.block_stacked():
            events += 's'  # 's' event when block is stacked
        return events

    def block_grasped(self):
        # Logic to detect if the block is grasped
        return self.info.get('grasped_block', False)

    def block_stacked(self):
        # Logic to detect if the block is stacked
        return self.info.get('stacked_block', False)

class MyBlockStackingEnvRM1(RewardMachineEnv):
    def __init__(self):
        env = MyBlockStackingEnv()
        # Use os.path.join to handle relative paths
        rm_files = [os.path.join(os.path.dirname(__file__), "../reward_machines/t1.txt")]
        super().__init__(env, rm_files)

class MyBlockStackingEnvRM2(RewardMachineEnv):
    def __init__(self):
        env = MyBlo
