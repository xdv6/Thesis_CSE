import sys
import re
import multiprocessing
import os.path as osp

import gym
from collections import defaultdict
import tensorflow as tf
import numpy as np

from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import parse_unknown_args
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module
import wandb
from rl_agents.dhrm.dhrm import evaluate, evaluate_multiple_models

import datetime
import os

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_name = f"{timestamp}_PID{os.getpid()}"

# Store the run name as an environment variable
os.environ["WANDB_RUN_NAME"] = run_name

# Importing our environments and auxiliary functions
import envs
from envs.water.water_world import Ball, BallAgent
from reward_machines.rm_environment import RewardMachineWrapper
from cmd_util import make_vec_env, make_env, common_arg_parser

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    if env_type == "robosuite":
        alg_kwargs = get_learn_function_defaults(args.alg, 'half_cheetah_environment')
    else:
        alg_kwargs = get_learn_function_defaults(args.alg, env_type)
        alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    # Adding RM-related parameters
    alg_kwargs['use_rs']   = args.use_rs
    alg_kwargs['use_crm']  = args.use_crm
    alg_kwargs['gamma']    = args.gamma

    # import ipdb; ipdb.set_trace()
    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    if args.play:
        # if we are just playing we don't want to train the model, so we can pass None as the model
        model = None
    else:
        model = learn(
            env=env,
            seed=seed,
            total_timesteps=total_timesteps,
            **alg_kwargs
        )

    return model, env, alg_kwargs


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if alg in ['deepq', 'qlearning', 'hrm', 'dhrm']:
        env = make_env(env_id, env_type, args, seed=seed, logger_dir=logger.get_dir())
    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, args, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco' or env_type == 'robosuite':
            env = VecNormalize(env, use_tf=True)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Custom handling for robosuite environments
    if env_id in ["MyBlockStackingEnvRM1", "MyBlockStackingEnvRM2"]:
        return "robosuite", env_id

    # Default environment parsing from gym
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    library = 'rl_agents'
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join([library, alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['baselines', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def main(args):

    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    os.environ["SELECTED_CUBE"] = args.selected_cube

    os.environ["START_STATE"] = str(args.start_state)

    if args.enable_visualization:
        os.environ["ENABLE_RENDERER"] = "True"

    run = wandb.init(
        # Set the project where this run will be logged
        project="reward_machines",
        name=run_name,
        mode = "offline" if args.play else "online"  # Set offline mode dynamically
    )

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    model, env, alg_kwargs = train(args, extra_args)

    if args.play:
        logger.log("Running trained model")
        evaluate(env=env, seed=args.seed, total_timesteps=int(args.num_timesteps), **alg_kwargs)
        # evaluate_multiple_models(env=env, seed=args.seed, total_timesteps=int(args.num_timesteps), **alg_kwargs)


    env.close()

    return model

if __name__ == '__main__':



    # Examples over the office world:
    #    cross-product baseline:
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9
    #    cross-product baseline with reward shaping:
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_rs
    #    CRM:
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_crm
    #    CRM with reward shaping:
    #        >>> python3.6 run.py --alg=qlearning --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_crm --use_rs
    #    HRM:
    #        >>> python3.6 run.py --alg=hrm --env=Office-v0 --num_timesteps=1e5 --gamma=0.9
    #    HRM with reward shaping:
    #        >>> python3.6 run.py --alg=hrm --env=Office-v0 --num_timesteps=1e5 --gamma=0.9 --use_rs
    # NOTE: The complete list of experiments (that we reported in the paper) can be found on '../scripts'

    import time
    t_init = time.time()
    main(sys.argv)
    logger.log("Total time: " + str(time.time() - t_init) + " seconds")