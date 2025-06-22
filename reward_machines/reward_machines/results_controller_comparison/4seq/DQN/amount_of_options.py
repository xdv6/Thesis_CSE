import os
import pandas as pd
import matplotlib.pyplot as plt

# Assume we are in: /.../results_controller_comparison/DQN/
base_dir = os.path.join("results_controller_comparison", "4seq", "DQN")
num_runs = 10

plt.figure(figsize=(10, 6))

for i in range(1, num_runs + 1):
    run_dir = os.path.join(base_dir, f"DQN{i}")
    options_path = os.path.join(run_dir, "amount_of_options_explored.csv")
    steps_path = os.path.join(run_dir, "env_steps_inside_sim.csv")

    if not os.path.exists(options_path) or not os.path.exists(steps_path):
        print(f"⚠️ Missing files in DQN{i}")
        continue

    # Load the CSVs
    options_df = pd.read_csv(options_path)
    steps_df = pd.read_csv(steps_path)

    # Get relevant columns (first timestamp column + main value column)
    env_time_col = "Relative Time (Process)"
    env_value_col = [col for col in steps_df.columns if "env_steps_inside_sim" in col and "__" not in col][0]

    opt_time_col = "Relative Time (Process)"
    opt_value_col = [col for col in options_df.columns if "amount_of_options_explored" in col and "__" not in col][0]

    # Match each options timestamp to the closest env_steps timestamp
    matched_env_steps = []
    for t in options_df[opt_time_col]:
        nearest_idx = (steps_df[env_time_col] - t).abs().idxmin()
        matched_env_steps.append(steps_df.iloc[nearest_idx][env_value_col])

    # Plot
    plt.plot(matched_env_steps, options_df[opt_value_col], label=f"Set {i}", marker='o')

# Final plot formatting
plt.xlabel("Environment Steps", fontsize=18)
plt.ylabel("Options Explored", fontsize=18)
plt.title("Options Explored vs. Environment Steps (DQN)", fontsize=22)
plt.legend(fontsize=16)
plt.grid(True)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.show()
