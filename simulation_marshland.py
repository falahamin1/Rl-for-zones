
from marshland_example import MarshlandSprintEnv
from TabularQ import QLearningAgent
import numpy as np


def train_and_evaluate(episodes=5000):
    # Initialize environment and agent
    # We turn debug=False during training to speed it up
    env = MarshlandSprintEnv(debug=False)
    agent = QLearningAgent(action_size=4)

    print(f"Training agent for {episodes} episodes...")

    for _ in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
        
        # Decay exploration over time
        agent.decay_epsilon()

    print("Training Complete.\n")
    display_policy(agent, env)
    inspect_all_q_states(agent)
    print_symbolic_policy_atlas(agent, env)

def _get_learned_zones(agent, env):
    """Returns [(t_min, matrix_key), ...] for every zone the agent visited, sorted by t_min."""
    learned_matrices = {matrix for (_, _, matrix) in agent.q_table.keys()}
    zone_list = []
    seen = set()
    for t_min in range(env.max_time + 1):
        zone = (env.ctx.t_gate >= t_min) & (env.ctx.t_gate <= env.max_time)
        dbm_list = zone.to_dbm_list()
        if not dbm_list:
            continue
        matrix_key = tuple(tuple(row) for row in dbm_list[0].to_matrix(mode="raw"))
        if matrix_key in learned_matrices and matrix_key not in seen:
            zone_list.append((t_min, matrix_key))
            seen.add(matrix_key)
    return sorted(zone_list)


def display_policy(agent, env):
    """Prints the spatial policy grid for every symbolic zone the agent learned."""
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    for t_min, matrix_key in _get_learned_zones(agent, env):
        print(f"--- Policy for Zone: (t_gate >= {t_min}, budget: {env.max_time - t_min}s) ---")
        for y in range(env.grid_size - 1, -1, -1):
            row_str = ""
            for x in range(env.grid_size):
                state = ((x, y), True, matrix_key)
                if (x, y) == env.goal:
                    row_str += " [G] "
                elif (x, y) == env.obstacle:
                    row_str += " [X] "
                elif state not in agent.q_table:
                    row_str += "  .  "
                else:
                    best_act = np.argmax(agent.q_table[state])
                    row_str += f"  {action_symbols[best_act]}  "
            print(row_str)
        print()
    
def inspect_q_table(agent):
    print("\n=== Q-TABLE STATE INSPECTOR ===")
    print(f"Total Unique States Learned: {len(agent.q_table)}")
    
    # Group by position to see the 'Symbolic Depth'
    pos_states = {}
    for state in agent.q_table.keys():
        pos, gate_open, matrix = state
        if pos not in pos_states:
            pos_states[pos] = []
        pos_states[pos].append(matrix)

    for pos, matrices in sorted(pos_states.items()):
        print(f"\nPosition {pos}: Found {len(matrices)} unique symbolic zones.")
        # Print a few example actions for the first matrix found
        sample_state = (pos, True, matrices[0])
        q_values = agent.q_table[sample_state]
        print(f"   Example Actions (U, D, L, R): {np.round(q_values, 2)}")

def inspect_all_q_states(agent):
    print("\n=== FULL Q-TABLE INSPECTOR ===")
    print(f"Total Unique States Learned: {len(agent.q_table)}")
    
    # Group by position
    pos_states = {}
    for state in agent.q_table.keys():
        pos, gate_open, matrix = state
        if pos not in pos_states:
            pos_states[pos] = []
        pos_states[pos].append(matrix)

    # Action names for clarity
    action_names = ["UP", "DOWN", "LEFT", "RIGHT"]

    for pos, matrices in sorted(pos_states.items()):
        print(f"\nPosition {pos}:")
        
        for i, matrix in enumerate(matrices):
            state = (pos, True, matrix)
            q_values = agent.q_table[state]
            
            # We don't have the original Federation object here, but 
            # we can see the differences in the raw matrix values.
            # Printing the matrix tuple helps identify which 'time' this is.
            print(f"  Zone {i+1} (Matrix): {matrix}")
            
            # Format the Q-values nicely
            val_str = " | ".join([f"{action_names[a]}: {q_values[a]:.2f}" for a in range(4)])
            print(f"    Q-Values -> {val_str}")

def print_symbolic_policy_atlas(agent, env):
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    zones = _get_learned_zones(agent, env)

    print(f"\n=== THE SYMBOLIC POLICY ATLAS ({len(zones)} zones) ===")

    for t_min, matrix_key in zones:
        print(f"\n[t_gate >= {t_min}  |  budget: {env.max_time - t_min}s remaining]")
        print("-" * 20)

        for y in range(env.grid_size - 1, -1, -1):
            row = []
            for x in range(env.grid_size):
                state = ((x, y), True, matrix_key)
                if (x, y) == env.goal:
                    row.append(" [G] ")
                elif (x, y) == env.obstacle:
                    row.append(" [X] ")
                elif state in agent.q_table:
                    q_vals = agent.q_table[state]
                    if np.all(q_vals == 0):
                        row.append("  ?  ")
                    else:
                        best_action = np.argmax(q_vals)
                        row.append(f"  {action_symbols[best_action]}  ")
                else:
                    row.append("  .  ")
            print("".join(row))

if __name__ == "__main__":
    train_and_evaluate(episodes=100000)