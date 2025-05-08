import os
import pandas as pd
import matplotlib.pyplot as plt

# Use relative paths assuming script is run from 'results_controller_comparison'
astar_dir = "astar"
dqn_dir = "DQN"

astar_costs = []
dqn_costs = []
labels = []

# Loop over 10 runs
for i in range(1, 11):
    run_label = f"Run {i}"
    astar_path = os.path.join(astar_dir, f"astar{i}", "cost_of_path.csv")
    dqn_path = os.path.join(dqn_dir, f"DQN{i}", "rewards_per_sequence.csv")
    dqn_col_name = f"DQN{i} - rewards_per_sequence"

    # Read A* cost
    try:
        df_astar = pd.read_csv(astar_path)
        astar_col = [col for col in df_astar.columns if "cost_of_path" in col]
        if not astar_col:
            raise KeyError(f"No column with 'cost_of_path' in {astar_path}")
        astar_cost = df_astar[astar_col[0]].dropna().iloc[-1]
    except Exception as e:
        print(f"[A*] Failed to read: {astar_path} | Error: {e}")
        continue

    # Read DQN reward
    try:
        df_dqn = pd.read_csv(dqn_path)
        if dqn_col_name not in df_dqn.columns:
            raise KeyError(f"Column '{dqn_col_name}' not found in {dqn_path}")
        dqn_reward = df_dqn[dqn_col_name].dropna().iloc[-1]
        dqn_cost = -dqn_reward
    except Exception as e:
        print(f"[DQN] Failed to read: {dqn_path} | Error: {e}")
        continue

    astar_costs.append(astar_cost)
    dqn_costs.append(dqn_cost)
    labels.append(run_label)

# Plotting
x = range(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar([i - width/2 for i in x], astar_costs, width, label="A* Cost")
ax.bar([i + width/2 for i in x], dqn_costs, width, label="DQN Cost (−Reward)")

ax.set_xlabel("Run")
ax.set_ylabel("Cost")
ax.set_title("Comparison of A* vs DQN Converged Costs")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig("cost_comparison_plot.png", dpi=300, bbox_inches="tight")  # <-- Save figure
plt.show()
