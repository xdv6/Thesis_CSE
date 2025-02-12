#!/bin/bash



# change the installation of the cubes initialization
# Capture the indentation of line 356 (spaces or tabs) BEFORE modifying the file
indentation=$(sed -n '356s/^\([[:space:]]*\).*/\1/p' /root/miniconda3/envs/myenv/lib/python3.7/site-packages/robosuite/environments/manipulation/stack.py)

# Escape special characters in indentation (important for sed compatibility)
escaped_indentation=$(echo "$indentation" | sed 's/[&/\]/\\&/g')

# Comment out lines 356 and 357 in stack.py
sed -i '356,357s/^/#/' /root/miniconda3/envs/myenv/lib/python3.7/site-packages/robosuite/environments/manipulation/stack.py

# Insert "pass" on line 356 with the correct indentation
sed -i "356i\\${escaped_indentation}pass" /root/miniconda3/envs/myenv/lib/python3.7/site-packages/robosuite/environments/manipulation/stack.py

echo "✅ Lines 356-357 in stack.py have been commented out and 'pass' has been added on line 356 with correct indentation."



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