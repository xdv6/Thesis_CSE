import itertools

# Config
cubes = ['A', 'B', 'C', 'D']
base_reward = 10000
reward_step = 400  # reward difference between best and worst

# Start writing
lines = []
initial_state = 0
terminal_states = []

lines.append(f"{initial_state}  # initial state")
# terminal states will be from state index 8, 16, ..., so collect them
sequence_states = list(itertools.permutations(cubes))
final_state_counter = len(sequence_states) * 8
terminal_states = [i for i in range(8, final_state_counter + 1, 8)]
lines.append(f"{' '.join(map(str, terminal_states))}  # terminal states")

state_id = 0
seq_idx = 0

for seq in sequence_states:
    reward = base_reward - seq_idx * reward_step
    for cube in seq:
        # gX
        lines.append(f"({state_id}, {state_id}, '!g{cube}', ConstantRewardFunction())")
        lines.append(f"({state_id}, {state_id + 1}, 'g{cube}', ConstantRewardFunction())")
        state_id += 1
        # hX
        lines.append(f"({state_id}, {state_id}, '!h{cube}', ConstantRewardFunction())")
        if seq.index(cube) == 3:  # last cube → assign reward
            lines.append(f"({state_id}, {state_id + 1}, 'h{cube}', ConstantRewardFunction({reward}))")
        else:
            lines.append(f"({state_id}, {state_id + 1}, 'h{cube}', ConstantRewardFunction())")
        state_id += 1
    seq_idx += 1

# Output to file
with open("cube_sequence_lifting_rm.txt", "w") as f:
    for line in lines:
        f.write(line + "\n")

print("✅ Reward machine for all 24 sequences generated in 'cube_lifting_rm.txt'")
