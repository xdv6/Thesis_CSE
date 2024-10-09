# run.py

from envs.my_block_stacking_env import MyBlockStackingEnvRM1, MyBlockStackingEnvRM2
from stable_baselines3 import DQN

def train(env_class):
    # Initialize the environment
    env = env_class()

    # Initialize DQN algorithm (or any other RL algorithm)
    model = DQN('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10000)

    # Save the model after training
    model.save("dqn_block_stacking")

if __name__ == "__main__":
    # Choose which environment (RM1 or RM2) to run
    train(MyBlockStackingEnvRM1)  # For the simple task
    # train(MyBlockStackingEnvRM2)  # Uncomment for the complex task
