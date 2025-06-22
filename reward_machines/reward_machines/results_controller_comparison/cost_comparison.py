import matplotlib.pyplot as plt

# Updated steps taken
dqn_steps = [5846, 3226, 5900, 5908, 7030, 8148, 5790, 5338, 9188, 2844]
astar_steps = [706, 1764, 931, 1940, 1433, 1603, 682, 1335, 1589, 854]

# Updated costs
astar_costs = [4.63, 5.24, 5.18, 5.63, 5.05, 4.98, 4.52, 5.10, 5.02, 4.89]
dqn_costs =   [4.63, 5.24, 5.39, 5.63, 5.05, 4.98, 4.52, 5.10, 5.02, 4.89]

# Labels
labels = [f"Run {i}" for i in range(1, 11)]
x = range(len(labels))
width = 0.35

# Plot
fig, ax = plt.subplots(figsize=(16, 9))  # Larger figure size
bars_astar = ax.bar([i - width/2 for i in x], astar_steps, width, label="A* Steps")
bars_dqn = ax.bar([i + width/2 for i in x], dqn_steps, width, label="DQN Steps")

# Axis config with even larger font sizes
ax.set_xlabel("Run", fontsize=24)
ax.set_ylabel("Steps Taken", fontsize=24)
ax.set_title("Steps Taken by A* vs DQN Controllers", fontsize=30)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=20)
ax.tick_params(axis='y', labelsize=20)
ax.legend(fontsize=20)
ax.grid(True, axis='y')

# Adjust plot to make space at bottom
plt.subplots_adjust(bottom=0.35)

# Annotate cost underneath x-axis with larger font
for i, (c_astar, c_dqn) in enumerate(zip(astar_costs, dqn_costs)):
    ax.annotate(f"A*: {c_astar:.2f}\nDQN: {c_dqn:.2f}",
                xy=(i, 0), xytext=(0, -80),
                textcoords='offset points',
                ha='center', va='top', fontsize=16)

# Finalize and save
plt.tight_layout()
plt.savefig("updated_steps_with_separate_costs_larger_text.png", dpi=300, bbox_inches="tight")
plt.show()
