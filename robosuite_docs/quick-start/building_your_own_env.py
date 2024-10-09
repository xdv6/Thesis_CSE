from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from robosuite.utils.mjcf_utils import new_joint
import mujoco


# All mujoco object definitions are housed in an xml. We create a MujocoWorldBase class to do it
world = MujocoWorldBase()

# The class housing the xml of a robot can be created as follows.
robot = Panda()

# We can add a gripper to the robot by creating a gripper instance and calling the add_gripper method on a robot.
gripper = gripper_factory('PandaGripper')
robot.add_gripper(gripper)

# To add the robot to the world, we place the robot on to a desired position and merge it into the world
robot.set_base_xpos([0, 0, 0])
world.merge(robot)

# We can initialize the TableArena instance that creates a table and the floorplane
arena = TableArena()
arena.set_origin([0.8, 0, 0])
world.merge(arena)

# adding a ball object to the world
sphere = BallObject(
    name="sphere",
    size=[0.04],
    rgba=[0, 0.5, 0.5, 1]).get_obj()
sphere.set('pos', '1.0 0 1.0')
world.worldbody.append(sphere)

# run the simulation
model = world.get_model(mode="mujoco")


data = mujoco.MjData(model)
while data.time < 1:
    print(data.time)
    mujoco.mj_step(model, data)


