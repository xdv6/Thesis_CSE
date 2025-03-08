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


        if s_info['current_u_id'] == 0:
            wandb.log({"gripper_to_cube_reward": s_info['reward_gripper_to_cube']})
            return s_info['reward_gripper_to_cube']
        elif s_info['current_u_id'] == 1:
            wandb.log({"reward_cube_A_to_cube_B": s_info['reward_cube_A_to_cube_B']})
            return s_info['reward_cube_A_to_cube_B']
        elif s_info['current_u_id'] == 2:
            wandb.log({"reward_cube_A_to_cube_B_xy": s_info['reward_cube_A_to_cube_B_xy']})
            return s_info["reward_cube_A_to_cube_B_xy"]
        else:
            return 0.0


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
