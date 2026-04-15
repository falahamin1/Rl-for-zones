import numpy as np
from collections import defaultdict

class QLearningAgent:
    def __init__(self, action_size, alpha=0.2, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        
        # Epsilon Parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def choose_action(self, state):
        """Epsilon-greedy with randomized tie-breaking."""
        # 1. Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # 2. Exploitation (with random tie-breaking)
        q_values = self.q_table[state]
        max_q = np.max(q_values)
        
        # Find all actions that share the maximum value
        actions_with_max_q = np.where(q_values == max_q)[0]
        
        # Pick randomly among the best (prevents getting stuck on Action 0)
        return np.random.choice(actions_with_max_q)

    def decay_epsilon(self):
        """Call this at the end of every episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def learn(self, state, action, reward, next_state, done):
        """Update Q-value based on the Bellman equation."""
        old_value = self.q_table[state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
            
        self.q_table[state][action] = (1 - self.alpha) * old_value + self.alpha * (target - old_value)