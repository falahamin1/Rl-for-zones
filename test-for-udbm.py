# This is a test script to work with pyudbm. More information on 
# the functions can be found in the documentation: https://pyudbm.readthedocs.io/en/latest/foundations/reading-guide/index.html

import pyudbm

def test_ta_environment_logic():
    print("=== TIMED AUTOMATA ENV MODELING TEST (FINAL VERIFIED) ===\n")
    
    # 1. SETUP: Define Clocks
    # Clocks represent the continuous state of your RL environment.
    ctx = pyudbm.Context(['x', 'y'], name='c')
    
    # 2. INITIAL STATE: x=0 and y=0
    # Every episode usually starts with all clocks synchronized at zero.
    state = (ctx.x == 0) & (ctx.y == 0)
    print(f"Step 0: Initial State -> {state}")

    # 3. ELAPSE TIME (Wait Action)
    # Represents the 'Delay' action in RL. Clocks increase at the same rate.
    waiting_state = state.up()
    print(f"Step 1: After waiting (up) -> {waiting_state}")

    # 4. INVARIANT CHECK (Environmental Constraint)
    # If the RL agent stays in a location, it must satisfy that location's invariant.
    invariant = (ctx.x <= 5)
    legal_state = waiting_state & invariant
    print(f"Step 2: Apply Invariant (x <= 5) -> {legal_state}")

    # 5. GUARD CHECK (Action Availability)
    # Checks if an action (edge transition) is legally available for the agent.
    guard = (ctx.y > 2)
    # is_empty() checks if the intersection of current state and guard is possible.
    possible_to_act = not (legal_state & guard).is_empty()
    print(f"Step 3: Is action with guard 'y > 2' possible? {'Yes' if possible_to_act else 'No'}")

    # 6. RESET ACTION (Effect of Action)
    # When the agent takes a transition, clocks are often reset.
    if possible_to_act:
        # We take the intersection with the guard, then apply the reset.
        action_state = (legal_state & guard).reset_value(ctx.x)
        print(f"Step 4: Action taken! (reset_value x) -> {action_state}")

    # 7. MATRIX EXPORT (RL Input Generation)
    print("\n--- Matrix Representation for RL Input ---")
    
    # Converting the Federation into a list of individual DBM zones.
    # This is necessary because 'Federation' isn't directly iterable.
    dbms = action_state.to_dbm_list()
    
    for i, dbm in enumerate(dbms):
        # 'raw' mode gives you the numerical matrix for your Q-Table or Neural Network.
        # 'string' mode is perfect for logging and debugging.
        raw_matrix = dbm.to_matrix(mode="raw")
        string_matrix = dbm.to_matrix(mode="string")
        
        print(f"Zone {i} (Readable):")
        for row in string_matrix:
            print(f"  {row}")
            
        print(f"\nZone {i} (Raw Integers for RL):")
        # This nested list can be flattened or used as a key.
        for row in raw_matrix:
            print(f"  {row}")

if __name__ == "__main__":
    test_ta_environment_logic()