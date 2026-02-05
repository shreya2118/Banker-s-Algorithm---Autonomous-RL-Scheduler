import numpy as np
import pickle
from bankers_env import BankersEnvironment

# --- SETUP THE SCENARIO (MUST MATCH YOUR GUI DEFAULTS) ---
N = 5
M = 3
# Default Snapshot (P0 to P4)
allocation = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
max_req    = [[7, 5, 3], [3, 2, 2], [9, 0, 2], [2, 2, 2], [4, 3, 3]]
available  = [3, 3, 2]

# Initialize Environment
env = BankersEnvironment(N, M, allocation, max_req, available)

# Q-Table: Stores the "score" of every action in every state
# Keys: States (0,0,0,0,0) -> Values: [Score_P0, Score_P1, Score_P2...]
q_table = {}

# Hyperparameters
alpha = 0.1   # Learning Rate
gamma = 0.9   # Discount Factor
epsilon = 1.0 # Exploration Rate (start high)

print("Training the RL Agent...")

for episode in range(2000):
    state = env.reset()
    done = False
    
    while not done:
        # Initialize state in Q-table if new
        if state not in q_table:
            q_table[state] = np.zeros(N)

        # Epsilon-Greedy Action Selection
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, N) # Explore (Random)
        else:
            action = np.argmax(q_table[state]) # Exploit (Best known move)

        # Take action
        next_state, reward, done = env.step(action)
        
        # Update Q-Value (The Bellman Equation)
        old_value = q_table[state][action]
        next_max = np.max(q_table.get(next_state, np.zeros(N)))
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state][action] = new_value
        
        state = next_state

    # Decay exploration (stop guessing, start using brain)
    epsilon = max(epsilon * 0.995, 0.01)

# Save the brain
with open("q_table.pkl", "wb") as f:
    pickle.dump(q_table, f)

print("Training Complete! 'q_table.pkl' saved.")
print("The agent now knows the perfect Safe Sequence.")