import pyudbm
import random
import numpy as np

class MarshlandSprintEnv:
    def __init__(self, debug=True):
        # 1. SETUP: Context and Constants
        self.ctx = pyudbm.Context(['t_gate'], name='c')
        self.grid_size = 3
        self.goal = (0, 2)
        self.button = (0, 0)
        self.muddy_tile = (1, 1)
        self.obstacle = (0, 1)
        self.max_time = 20
        self.debug = debug
        
        # Call reset but don't return its value here
        self.reset()

    def reset(self):
        self.pos = (0, 0)
        self.gate_open = True
        self.t_min = 0
        self.zone = (self.ctx.t_gate >= 0) & (self.ctx.t_gate <= self.max_time)
        
        if self.debug:
            print(f"\n[RESET] Position: {self.pos} | State: {self.render_symbolic_state()}")
        
        # Return the observation (used by RL training loops)
        return self._get_obs()

    def render_symbolic_state(self):
        """Returns the human-readable constraints of the current zone."""
        return str(self.zone)

    def _get_obs(self):
        """Generates the observation for the RL agent."""
        dbm_list = self.zone.to_dbm_list()
        
        # Check if the zone is empty (Terminal State)
        if len(dbm_list) == 0:
            # Return a 'Zero' matrix or a dummy matrix for the terminal state
            # This ensures the agent can still hash the observation
            matrix_tuple = ((0, 0), (0, 0)) 
        else:
            raw_matrix = dbm_list[0].to_matrix(mode="raw")
            matrix_tuple = tuple(tuple(row) for row in raw_matrix)
            
        return (self.pos, self.gate_open, matrix_tuple)

    def step(self, action):
        action_names = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

        # 1. Movement Logic
        moves = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0)}
        if action not in moves:
            return self._get_obs(), -1, False

        dx, dy = moves[action]
        new_x, new_y = self.pos[0] + dx, self.pos[1] + dy

        if not (0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size) or (new_x, new_y) == self.obstacle:
            if self.debug: print(f"--- Action: {action_names[action]} (Invalid Move) ---")
            return self._get_obs(), -5, False

        # 2. Probabilistic Time Passage
        time_inc = 3
        slipped = False
        if (new_x, new_y) == self.muddy_tile:
            if random.random() < 0.5:
                time_inc = 12
                slipped = True

        # 3. Symbolic Zone Update
        if self.gate_open:
            self.t_min += time_inc
            self.zone = (self.ctx.t_gate >= self.t_min) & (self.ctx.t_gate <= self.max_time)

        # 4. Debug Printout
        if self.debug:
            slip_str = " (SLIPPED!)" if slipped else ""
            print(f"--- Action: {action_names[action]}{slip_str} ---")
            print(f"    New Pos: {(new_x,new_y)} | Min Time Elapsed: {self.t_min}")
            print(f"    Zone: {self.render_symbolic_state()}")

        # 5. Update Position
        self.pos = (new_x, new_y)

        # 6. Auto-press: arriving at (0,0) opens the gate
        if self.pos == self.button and not self.gate_open:
            self.gate_open = True
            self.t_min = 0
            self.zone = (self.ctx.t_gate >= 0) & (self.ctx.t_gate <= self.max_time)
            if self.debug:
                print(f"    [AUTO-PRESS] Gate Opened at {self.pos}!")
                print(f"    Initial Budget: {self.render_symbolic_state()}")

        obs = self._get_obs()

        # 7. Termination & Rewards
        if self.zone.is_empty():
            if self.debug: print("!!! TERMINAL: TIME EXPIRED (ZONE EMPTY) !!!")
            return obs, -100, True

        if self.pos == self.goal and self.gate_open:
            if self.debug: print("!!! TERMINAL: GOAL REACHED !!!")
            return obs, 500, True

        return obs, -1, False

if __name__ == "__main__":
        env = MarshlandSprintEnv(debug=True)
        test_actions = [3, 0, 0, 2]
        for act in test_actions:
            obs, rew, done = env.step(act)
            if done: break