import numpy as np
from bankers_env import BankersEnvironment

def train_agent_live(n, m, alloc, max_req, avail):
    """
    Optimized JIT Learner with Early Stopping.
    """
    env = BankersEnvironment(n, m, alloc, max_req, avail)
    q_table = {}
    
    # --- OPTIMIZATION 1: Lower limit ---
    # We don't need 800 tries. 300 is plenty for this problem size.
    MAX_EPISODES = 300 
    
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0 
    
    # --- OPTIMIZATION 2: Early Stopping variables ---
    success_streak = 0
    REQUIRED_STREAK = 5  # If we solve it 5 times in a row, we are ready.

    for episode in range(MAX_EPISODES):
        state = env.reset()
        done = False
        steps_taken = 0
        
        while not done:
            if state not in q_table:
                q_table[state] = np.zeros(n)

            # Epsilon-Greedy
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(0, n)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = env.step(action)
            
            # Update Q-Value
            old_val = q_table[state][action]
            next_max = np.max(q_table.get(next_state, np.zeros(n)))
            q_table[state][action] = (1 - alpha) * old_val + alpha * (reward + gamma * next_max)
            
            state = next_state
            steps_taken += 1
            
            # Safety break for infinite loops in bad states
            if steps_taken > n * 2: 
                break

        # Decay exploration
        epsilon = max(epsilon * 0.95, 0.05) # Faster decay
        
        # --- CHECK FOR EARLY STOPPING ---
        # If we got a positive reward at the end (means we finished safely)
        if reward > 0 and done:
            success_streak += 1
        else:
            success_streak = 0 # Reset streak if we failed
            
        if success_streak >= REQUIRED_STREAK:
            # print(f"Stopped early at episode {episode} (Brain is ready)")
            break
            
    return q_table