#!/bin/bash

cd /root/Thesis_CSE
git fetch origin
git checkout new-env || {
    echo "❌ Failed to checkout 'new-env' branch. Make sure it exists."
    exit 1
}
echo "✅ Checked out to branch 'new-env'"


# # Change the installation of the cubes initialization
# indentation=$(sed -n '356s/^\([[:space:]]*\).*/\1/p' /root/miniconda3/envs/myenv/lib/python3.7/site-packages/robosuite/environments/manipulation/stack.py)
# escaped_indentation=$(echo "$indentation" | sed 's/[&/\]/\\&/g')

# sed -i '356,357s/^/#/' /root/miniconda3/envs/myenv/lib/python3.7/site-packages/robosuite/environments/manipulation/stack.py
# sed -i "356i\\${escaped_indentation}pass" /root/miniconda3/envs/myenv/lib/python3.7/site-packages/robosuite/environments/manipulation/stack.py

# echo "✅ Lines 356-357 in stack.py have been commented out and 'pass' has been added on line 356 with correct indentation."


# Replace stack.py with custom my_stack.py
cp /root/Thesis_CSE/my_stack.py /root/miniconda3/envs/myenv/lib/python3.7/site-packages/robosuite/environments/manipulation/stack.py

echo "✅ Replaced stack.py with custom my_stack.py"

# Ensure the WANDB_API_KEY is set at runtime
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "Error: WANDB_API_KEY is not set! Please provide it when running the container."
    exit 1
fi

echo "wandb_api_key=${WANDB_API_KEY}" > /root/.wandb_config
echo "✅ WANDB API key has been set in /root/.wandb_config"

cd /root/Thesis_CSE/reward_machines/reward_machines

# Check if LOG_DIRECTORY is set and exists
if [[ -z "$LOG_DIRECTORY" ]]; then
    echo "Error: LOG_DIRECTORY is not set! Please set this variable before starting the container."
    exit 1
fi

mkdir -p "$LOG_DIRECTORY"  # Ensure the log directory exists

# # Start background logging of directory growth
# LOG_FILE="$LOG_DIRECTORY/file_growth.log"
# echo "🔍 Logging file growth in $LOG_FILE..."
# nohup bash -c 'while true; do
#     date >> "'"$LOG_FILE"'"
#     du -ah /root/Thesis_CSE/reward_machines | sort -rh | head -20 >> "'"$LOG_FILE"'"
#     echo "----------------" >> "'"$LOG_FILE"'"
#     sleep 300
# done' >/dev/null 2>&1 &

# Start background process to clean up all wandb logs every hour
CLEANUP_LOG="$LOG_DIRECTORY/cleanup.log"
echo "🗑️  Starting periodic full cleanup of wandb logs (every hour)..."
nohup bash -c 'while true; do
    echo "🗑️  Deleting all files and folders in wandb directory..." >> "'"$CLEANUP_LOG"'"
    rm -rf /root/Thesis_CSE/reward_machines/reward_machines/wandb/*
    echo "✅ Cleanup completed at $(date)" >> "'"$CLEANUP_LOG"'"
    sleep 3600  # Wait 1 hour before the next cleanup
done' >/dev/null 2>&1 &

# Run specified number of instances in parallel
for i in $(seq 1 $NUM_INSTANCES); do
    echo "Launching instance $i..."
    eval "${PYTHON_CMD} &"
done

# Wait for all background processes to finish
wait

echo "✅ All ${NUM_INSTANCES} instances have completed!"
