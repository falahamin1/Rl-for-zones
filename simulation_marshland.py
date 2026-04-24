
from marshland_example import MarshlandSprintEnv
from TabularQ import QLearningAgent
import pyudbm
import numpy as np


def train_and_evaluate(episodes=5000):
    # Initialize environment and agent
    # We turn debug=False during training to speed it up
    env = MarshlandSprintEnv(debug=False)
    agent = QLearningAgent(action_size=5)

    print(f"Training agent for {episodes} episodes...")

    for i in range(episodes):
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

def display_policy(agent, env):
    """Prints the spatial policy for different symbolic time zones."""
    actions_map = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "Ⓑ"}
    
    # We sample two time profiles:
    # 1. Early (t_min = 3) - Plenty of time left
    # 2. Late  (t_min = 14) - Running out of time (after a slip)
    
    for t_val in [3, 14]:
        print(f"--- Policy for Zone: ({t_val} <= t_gate <= 20) ---")
        
        # Create a dummy DBM matrix for this time value to query the Q-table
        ctx = pyudbm.Context(['t_gate'], name='c')
        test_zone = (ctx.t_gate >= t_val) & (ctx.t_gate <= 20)
        raw_matrix = test_zone.to_dbm_list()[0].to_matrix(mode="raw")
        matrix_tuple = tuple(tuple(row) for row in raw_matrix)

        for y in range(2, -1, -1):
            row_str = ""
            for x in range(3):
                state = ((x, y), True, matrix_tuple)
                if (x, y) == env.goal:
                    row_str += " [G] "
                elif (x, y) == env.obstacle:
                    row_str += " [X] "
                elif state not in agent.q_table:
                    row_str += "  .  "
                else:
                    best_act = np.argmax(agent.q_table[state])
                    row_str += f"  {actions_map[best_act]}  "
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
        print(f"   Example Actions (U, D, L, R, B): {np.round(q_values, 2)}")

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
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "PRESS"]

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
            val_str = " | ".join([f"{action_names[a]}: {q_values[a]:.2f}" for a in range(5)])
            print(f"    Q-Values -> {val_str}")

def print_symbolic_policy_atlas(agent, env):
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "Ⓑ"}
    # The time points we want to inspect (matching your environment's steps)
    time_points = [3, 6, 11, 14, 17]
    
    print("\n=== THE SYMBOLIC POLICY ATLAS ===")
    
    for t in time_points:
        # 1. Recreate the DBM matrix for this specific time to use as a lookup key
        test_zone = (env.ctx.t_gate >= t) & (env.ctx.t_gate <= env.max_time)
        dbm_list = test_zone.to_dbm_list()
        
        if not dbm_list: continue # Skip if time is out of bounds
        
        raw_matrix = dbm_list[0].to_matrix(mode="raw")
        matrix_key = tuple(tuple(row) for row in raw_matrix)
        
        print(f"\n[Time: {t}s] Matrix Key: {matrix_key}")
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
                    # Check if the agent has actually learned something here
                    q_vals = agent.q_table[state]
                    if np.all(q_vals == 0):
                        row.append("  ?  ") # Explored but no reward propagated
                    else:
                        best_action = np.argmax(q_vals)
                        row.append(f"  {action_symbols[best_action]}  ")
                else:
                    row.append("  .  ") # Never visited this tile at this time
            print("".join(row))

if __name__ == "__main__":
    train_and_evaluate(episodes=100000)