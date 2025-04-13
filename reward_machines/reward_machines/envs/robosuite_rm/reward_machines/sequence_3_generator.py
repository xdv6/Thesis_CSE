import itertools

# --- Configuration ---
cubes = ['A', 'B', 'C']              # Cubes used in the sequences
base_reward = 10000                  # Reward for best sequence
reward_step = 400                    # Decrease in reward per permutation
output_file = "cube_sequence_lifting.txt"  # Output filename

# --- Generate all permutations of cube sequences ---
sequences = list(itertools.permutations(cubes))

# --- Initialize ---
lines = []
lines.append("0  # initial state")

terminal_states = []
state_counter = 1  # start new states from 1

# Dictionary to keep track of already processed start cubes
already_processed_cubes = dict()

for idx, seq in enumerate(sequences):
    reward = base_reward - idx * reward_step
    current_state = 0  # always start from state 0

    for i, cube in enumerate(seq):
        # If the cube is first in the sequence and already processed before, skip it and build on the known state
        if i == 0:
            if cube in already_processed_cubes:
                current_state = already_processed_cubes[cube]
                continue
            else:
                already_processed_cubes[cube] = state_counter + 1  # h_state will become the new base

        g_event = f"g{cube}"
        h_event = f"h{cube}"
        g_state = state_counter
        h_state = state_counter + 1

        # Add gX transition from current_state → g_state
        lines.append(f"({current_state}, {g_state}, '{g_event}', ConstantRewardFunction())")

        # Add hX transition from g_state → h_state
        if i == len(seq) - 1:
            lines.append(f"({g_state}, {h_state}, '{h_event}', ConstantRewardFunction({reward}))")
        else:
            lines.append(f"({g_state}, {h_state}, '{h_event}', ConstantRewardFunction())")

        current_state = h_state
        state_counter += 2

    terminal_states.append(current_state)

# --- Add terminal state header ---
lines.insert(1, f"{' '.join(map(str, terminal_states))}  # terminal states")

# --- Write to file ---
with open(output_file, "w") as f:
    for line in lines:
        f.write(line + "\n")

print(f"✅ Reward Machine saved to '{output_file}' with {len(sequences)} sequences.")
