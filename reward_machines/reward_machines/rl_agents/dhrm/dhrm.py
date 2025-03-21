import os
import tempfile
from datetime import time

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

from rl_agents.dhrm.options import OptionDQN, OptionDDPG
from rl_agents.dhrm.controller import ControllerDQN
import wandb


def learn(env,
          use_ddpg=False,
          gamma=0.90,
          use_rs=False,
          controller_kargs={},
          option_kargs={},
          seed=None,
          total_timesteps=100000,
          print_freq=100,
          callback=None,
          checkpoint_path="./checkpoints",
          checkpoint_freq=1000,
          load_path=None,
          **others):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    use_ddpg: bool
        whether to use DDPG or DQN to learn the option's policies
    gamma: float
        discount factor
    use_rs: bool
        use reward shaping
    controller_kargs
        arguments for learning the controller policy.
    option_kargs
        arguments for learning the option policies.
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    total_timesteps: int
        number of env steps to optimizer for
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    load_path: str
        path to load the model from. (default: None)

    Returns
    -------
    act: ActWrapper (meta-controller)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    act: ActWrapper (option policies)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    controller  = ControllerDQN(env, **controller_kargs)
    if use_ddpg:
        options = OptionDDPG(env, gamma, total_timesteps, **option_kargs)
    else:
        options = OptionDQN(env, gamma, total_timesteps, **option_kargs)
    option_s    = None # State where the option initiated
    option_id   = None # Id of the current option being executed
    option_rews = []   # Rewards obtained by the current option

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    options.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        # Get the base save location from the environment variable
        model_save_location = os.environ.get("CHECKPOINT_PATH", "./checkpoints")  # Default to "./checkpoints" if not set

        # Get the run name from the environment variable (default to "default_run" if not set)
        model_name = os.environ.get("WANDB_RUN_NAME", "default_run")

        # Create the full path
        run_save_path = os.path.join(model_save_location, model_name)

        # Create the directory if it doesn't exist
        os.makedirs(run_save_path, exist_ok=True)

        model_file = os.path.join(run_save_path, "best_model")
        model_saved = False

        if tf.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        elif load_path is not None:
            # load_variables(model_file)
            # tf.get_default_graph().finalize()  # 🔒 Finalize only after loading
            logger.log('Loaded model from {}'.format(load_path))


        num_steps_in_episode = 0
        for t in range(total_timesteps):
            wandb.log({"timestep": t})
            if callback is not None:
                if callback(locals(), globals()):
                    break

            # Selecting an option if needed
            if option_id is None:
                valid_options = env.get_valid_options()
                option_s    = obs
                option_id   = controller.get_action(option_s, valid_options)
                option_rews = []

            # Take action and update exploration to the newest value
            action = options.get_action(env.get_option_observation(option_id), t, reset)
            reset = False

            action = action.squeeze()
            new_obs, rew, done, info = env.step(action)
            num_steps_in_episode += 1

            # Saving the real reward that the option is getting
            if use_rs:
                option_rews.append(info["rs-reward"])
            else:
                wandb.log({"reward": rew})

                option_rews.append(rew)

            # Store transition for the option policies
            for _s,_a,_r,_sn,_done in env.get_experience():
                options.add_experience(_s,_a,_r,_sn,_done)

            # Learn and update the target networks if needed for the option policies
            options.learn(t)
            options.update_target_network(t)

            # Update the meta-controller if needed
            # Note that this condition always hold if done is True
            if env.did_option_terminate(option_id):
                option_sn = new_obs
                option_reward = sum([_r*gamma**_i for _i,_r in enumerate(option_rews)])
                valid_options = [] if done else env.get_valid_options()
                controller.add_experience(option_s, option_id, option_reward, option_sn, done, valid_options,gamma**(len(option_rews)))
                controller.learn()
                controller.update_target_network()
                controller.increase_step()
                option_id = None

            obs = new_obs
            episode_rewards[-1] += rew

            if done:
                wandb.log({"num_steps_in_episode": num_steps_in_episode})
                num_steps_in_episode = 0
                obs = env.reset()
                options.reset()
                episode_rewards.append(0.0)
                reset = True

            # General stats
            mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                logger.dump_tabular()
                save_variables(model_file)

            if (checkpoint_freq is not None and
                    num_episodes > 100 and t % checkpoint_freq == 0):

                # saving checkpoint model
                model_chekpoint_file = os.path.join(run_save_path, "model_" + str(t))
                save_variables(model_chekpoint_file)
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        logger.log("Saving model due to mean reward increase: {} -> {}".format(
                                   saved_mean_reward, mean_100ep_reward))
                    save_variables(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
        if model_saved:
            if print_freq is not None:
                logger.log("Restored model with mean reward: {}".format(saved_mean_reward))
            #load_variables(model_file)

    return controller.act, options.act


def evaluate(env,
          use_ddpg=False,
          gamma=0.9,
          use_rs=False,
          controller_kargs={},
          option_kargs={},
          seed=None,
          total_timesteps=1000,
          print_freq=100,
          callback=None,
          checkpoint_path="./checkpoints",
          checkpoint_freq=10000,
          load_path="./checkpoints",
          **others):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    use_ddpg: bool
        whether to use DDPG or DQN to learn the option's policies
    gamma: float
        discount factor
    use_rs: bool
        use reward shaping
    controller_kargs
        arguments for learning the controller policy.
    option_kargs
        arguments for learning the option policies.
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    total_timesteps: int
        number of env steps to optimizer for
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    load_path: str
        path to load the model from. (default: None)

    Returns
    -------
    act: ActWrapper (meta-controller)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    act: ActWrapper (option policies)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    controller  = ControllerDQN(env, **controller_kargs)
    options = OptionDDPG(env, gamma, total_timesteps, **option_kargs)


    # Override get_action to ensure deterministic execution (no noise)
    options.get_action = lambda obs, t, reset: options.agent.step(obs.reshape((1,) + obs.shape), apply_noise=False, compute_Q=True)[0] * options.max_action

    option_s    = None # State where the option initiated
    option_id   = None # Id of the current option being executed
    option_rews = []   # Rewards obtained by the current option

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    options.reset()
    reset = True


    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")

        if load_path is not None:
            load_variables(model_file)
            # tf.get_default_graph().finalize()  # 🔒 Finalize only after loading
            logger.log('Loaded model from {}'.format(load_path))

        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break

            # Selecting an option if needed
            if option_id is None:
                valid_options = env.get_valid_options()
                option_s    = obs
                option_id   = controller.get_action(option_s, valid_options)
                option_rews = []

            # Take action and update exploration to the newest value
            # print("option_id: ", option_id)
            # print(env.get_option_observation(option_id))
            action = options.get_action(env.get_option_observation(option_id), t, reset)
            reset = False

            action = action.squeeze()
            new_obs, rew, done, info = env.step(action)

            # Saving the real reward that the option is getting
            if use_rs:
                option_rews.append(info["rs-reward"])
            else:
                wandb.log({"reward": rew})
                option_rews.append(rew)

            obs = new_obs
            episode_rewards[-1] += rew

            if env.did_option_terminate(option_id):
                option_id = None

            # if rew > 2500:
            #     print(rew)
            #     print("SUCCESS: self.env.current_u_id == -1")
            #     break

            # print("rew: ", rew)

            if done:
                obs = env.reset()
                options.reset()
                episode_rewards.append(0.0)
                reset = True

    return controller.act, options.act





def evaluate_multiple_models(env,
          use_ddpg=False,
          gamma=0.9,
          use_rs=False,
          controller_kargs={},
          option_kargs={},
          seed=None,
          total_timesteps=1000,
          print_freq=100,
          callback=None,
          checkpoint_path="./checkpoints",
          checkpoint_freq=10000,
          load_path="./checkpoints",
          **others):
    """Train a deepq model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    use_ddpg: bool
        whether to use DDPG or DQN to learn the option's policies
    gamma: float
        discount factor
    use_rs: bool
        use reward shaping
    controller_kargs
        arguments for learning the controller policy.
    option_kargs
        arguments for learning the option policies.
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    total_timesteps: int
        number of env steps to optimizer for
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    load_path: str
        path to load the model from. (default: None)

    Returns
    -------
    act: ActWrapper (meta-controller)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    act: ActWrapper (option policies)
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # Create all the functions necessary to train the model

    sess = get_session()
    set_global_seeds(seed)

    controller  = ControllerDQN(env, **controller_kargs)
    options = OptionDDPG(env, gamma, total_timesteps, **option_kargs)


    # Override get_action to ensure deterministic execution (no noise)
    options.get_action = lambda obs, t, reset: options.agent.step(obs.reshape((1,) + obs.shape), apply_noise=False, compute_Q=True)[0] * options.max_action

    option_s    = None # State where the option initiated
    option_id   = None # Id of the current option being executed
    option_rews = []   # Rewards obtained by the current option

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    options.reset()
    reset = True


    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        solution_lifting = os.path.join(td, "solution_lifting")
        model_changed = False

        if load_path is not None:
            load_variables(solution_lifting)
            logger.log('Loaded model from {}'.format(load_path))

        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()):
                    break

            # Selecting an option if needed
            if option_id is None:
                valid_options = env.get_valid_options()
                option_s    = obs
                sub_rm_valid_option_s = option_s[:22]

                if not model_changed and valid_options[0] > 1:
                    # load new model
                    solution_alligning = os.path.join(td, "solution_alligning")
                    load_variables(solution_alligning)
                    print("model changed")
                    model_changed = True

                if valid_options[0] > 1:
                    valid_options[0] -= 2
                    # here we have to adapt the one-hot-encoding for the second model

                print("valid_options: ", valid_options)
                option_id   = controller.get_action(sub_rm_valid_option_s, valid_options)
                option_rews = []



            features = env.get_option_observation(option_id)
            sub_rm_features = features[:22]

            if option_id > 1:
                option_id -= 2
                # here we have to adapt the one-hot-encoding for the second model

            action = options.get_action(sub_rm_features, t, reset)
            reset = False

            action = action.squeeze()
            new_obs, rew, done, info = env.step(action)

            # Saving the real reward that the option is getting
            if use_rs:
                option_rews.append(info["rs-reward"])
            else:
                wandb.log({"reward": rew})
                option_rews.append(rew)

            obs = new_obs
            episode_rewards[-1] += rew

            if env.did_option_terminate(option_id):
                option_id = None

            # if rew > 2500:
            #     print(rew)
            #     print("SUCCESS: self.env.current_u_id == -1")
            #     break

            # print("rew: ", rew)

            if done:
                obs = env.reset()
                options.reset()
                episode_rewards.append(0.0)
                reset = True
                load_variables(solution_lifting)
                print("model changed back")
                break

    return controller.act, options.act

