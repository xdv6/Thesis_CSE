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



# Run specified number of instances in parallel
for i in $(seq 1 $NUM_INSTANCES); do
    echo "Launching instance $i..."
    eval "${PYTHON_CMD} &"
done

# Wait for all background processes to finish
wait

echo "✅ All ${NUM_INSTANCES} instances have completed!"