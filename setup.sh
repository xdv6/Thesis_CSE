#!/bin/bash

# Ensure the WANDB_API_KEY is set at runtime
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "Error: WANDB_API_KEY is not set! Please provide it when running the container."
    exit 1  # Exit if the variable is missing
fi

# Save API key to the config file for Weights & Biases
echo "wandb_api_key=${WANDB_API_KEY}" > /root/.wandb_config

# Print confirmation
echo "✅ WANDB API key has been set in /root/.wandb_config"

cd /root/Thesis_CSE/reward_machines/reward_machines

python run_robosuite.py --env=MyBlockStackingEnvRM1 --num_timesteps=100000 --alg=dhrm