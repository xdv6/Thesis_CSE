import math
import numpy as np
import wandb

class RewardFunction:
    def __init__(self):
        pass

    # To implement...
    def get_reward(self, s_info):
        raise NotImplementedError("To be implemented")

    def get_type(self):
        raise NotImplementedError("To be implemented")


class ConstantRewardFunction(RewardFunction):
    """
    Defines a constant reward for a 'simple reward machine'
    """
    def __init__(self, c):
        super().__init__()
        self.c = c

    def get_type(self):
        return "constant"

    def get_reward(self, s_info):
        return self.c

class RewardControl(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "ctrl"

    def get_reward(self, s_info):
        # euclidian distance
        distance_block_gripper = s_info['gripper_to_cubeA']
        distance_block_gripper_norm = np.linalg.norm(distance_block_gripper)

        eef_height = s_info['robot0_eef_pos'][2]
        height_cubeB = s_info['cubeB_pos'][2]
        height_diff_norm = np.linalg.norm([eef_height - height_cubeB])

        if s_info['current_u_id'] == 0:
            wandb.log({"distance": distance_block_gripper_norm})
            return -distance_block_gripper_norm
        elif s_info['current_u_id'] == 1:
            wandb.log({"height_diff": height_diff_norm})
            return -height_diff_norm
        else:
            return 0.0

        # original_reward = s_info['original_reward']
        # wandb.log({"robosuite_rs_reward": original_reward})
        # return original_reward


class RewardForward(RewardFunction):
    """
    Gives a reward for moving forward
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "forward"

    def get_reward(self, s_info):
        return s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah


class RewardBackwards(RewardFunction):
    """
    Gives a reward for moving backwards
    """
    def __init__(self):
        super().__init__()

    def get_type(self):
        return "backwards"

    def get_reward(self, s_info):
        return -s_info['reward_run'] + s_info['reward_ctrl']  #Cheetah
