import os

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
from baselines import logger
import numpy as np

import wandb
from wandb.integration.sb3 import WandbCallback

def learn(network, env,
          seed=None,
          use_crm=False,
          use_rs=False,
          total_timesteps=None,
          nb_epochs=None,
          nb_epoch_cycles=20,
          nb_rollout_steps=100,
          reward_scale=1.0,
          render=False,
          render_eval=False,
          noise_type='adaptive-param_0.2',
          normalize_returns=False,
          normalize_observations=True,
          critic_l2_reg=1e-2,
          actor_lr=1e-4,
          critic_lr=1e-3,
          popart=False,
          gamma=0.99,
          clip_norm=None,
          nb_train_steps=50,
          nb_eval_steps=100,
          batch_size=64,
          tau=0.01,
          eval_env=None,
          param_noise_adaption_interval=50,
          **network_kwargs):

    config = {
        "policy_type": "MlpPolicy",
        "env_name": "robosuite",
        "total_timesteps": total_timesteps,
        "learning_rate": actor_lr,
        "batch_size": batch_size,
        "gamma": gamma,
        "tau": tau,
        "reward_scale": reward_scale,
    }

    run = wandb.init(
        project="reward_machines",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # Set random seed
    seed = 42
    set_random_seed(seed)

    # Configure action noise if needed
    nb_actions = env.action_space.shape[-1]
    action_noise = None
    if noise_type is not None:
        for current_noise_type in noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            else:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))

    # Initialize the DDPG agent
    agent = DDPG("MlpPolicy", env, action_noise=action_noise, gamma=gamma, tau=tau, learning_rate=actor_lr,
                 batch_size=batch_size, buffer_size=int(1e6), verbose=1, tensorboard_log=f"runs/{run.id}")

    # Train the agent
    agent.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    # Save the trained model
    logdir = logger.get_dir()
    if logdir:
        agent.save(os.path.join(logdir, 'ddpg_agent'))

    # Finish the WandB run
    run.finish()

    return agent
