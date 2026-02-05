import time
import numpy as np
import pickle
import os
from bankers_env import BankersEnvironment
from rl_brain import train_agent_live

# File where the brain is stored
BRAIN_FILE = "global_brain.pkl"

class BenchmarkArena:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.global_q_table = {}
        
        # 1. Try to load existing brain
        if os.path.exists(BRAIN_FILE):
            print(f">>> Found {BRAIN_FILE}. Loading Offline Agent...")
            self._load_global_agent()
        else:
            # 2. If not found, train a new one (Fallback)
            print(">>> No brain found. Training Global Agent locally...")
            self._train_global_agent()

    def _train_global_agent(self):
        """Trains a fresh agent if no file exists."""
        EPISODES = 2000 
        # (Simplified training logic for local fallback)
        q_table = {}
        for _ in range(EPISODES):
            alloc = np.random.randint(0, 3, (self.n, self.m))
            max_req = alloc + np.random.randint(1, 4, (self.n, self.m))
            avail = np.random.randint(1, 5, self.m)
            env = BankersEnvironment(self.n, self.m, alloc, max_req, avail)
            state = env.reset()
            done = False
            steps = 0
            while not done and steps < 20: # Safety break
                steps += 1
                if state not in q_table: q_table[state] = np.zeros(self.n)
                action = np.random.randint(0, self.n)
                next_state, reward, done = env.step(action)
                old = q_table[state][action]
                nxt = np.max(q_table.get(next_state, np.zeros(self.n)))
                q_table[state][action] = old + 0.1 * (reward + 0.9 * nxt - old)
                state = next_state
        
        self.global_q_table = q_table
        with open(BRAIN_FILE, "wb") as f:
            pickle.dump(self.global_q_table, f)
        print(">>> Global Agent Trained and Saved.")

    def _load_global_agent(self):
        try:
            with open(BRAIN_FILE, "rb") as f:
                self.global_q_table = pickle.load(f)
        except Exception as e:
            print(f"Error loading brain: {e}")
            self.global_q_table = {}

    def run_global_test(self, alloc, mx, avail):
        """
        Runs the Offline Agent on YOUR specific data.
        Includes SAFETY BRAKES to prevent freezing.
        """
        start_time = time.time()
        env = BankersEnvironment(self.n, self.m, alloc, mx, avail)
        state = env.reset()
        done = False
        steps = []
        
        # --- SAFETY BRAKE ---
        # If it takes more than 2x the number of processes steps, it's stuck.
        max_steps = self.n * 2 
        steps_taken = 0

        while not done:
            steps_taken += 1
            if steps_taken > max_steps:
                return False, (time.time() - start_time), steps, "Failed (Timeout)"

            if state in self.global_q_table:
                action = np.argmax(self.global_q_table[state])
            else:
                action = np.random.randint(0, self.n) # Guess if unknown
            
            steps.append(f"P{action}")
            next_state, reward, done = env.step(action)
            
            if reward == -100: return False, (time.time() - start_time), steps, "Invalid Move"
            if reward == -50: return False, (time.time() - start_time), steps, "Deadlock"
                 
            state = next_state

        return True, (time.time() - start_time), steps, "Safe Sequence Found"

    def run_jit_test(self, alloc, mx, avail):
        """Runs the Online (JIT) Agent."""
        start_time = time.time()
        
        # Train on the spot
        q_table = train_agent_live(self.n, self.m, alloc, mx, avail)
        
        # Test immediately
        env = BankersEnvironment(self.n, self.m, alloc, mx, avail)
        state = env.reset()
        done = False
        steps = []
        max_steps = self.n * 2
        steps_taken = 0
        
        while not done:
            steps_taken += 1
            if steps_taken > max_steps:
                return False, (time.time() - start_time), steps, "Failed (Timeout)"

            action = np.argmax(q_table.get(state, np.zeros(self.n)))
            steps.append(f"P{action}")
            next_state, reward, done = env.step(action)
            
            if reward < 0: return False, (time.time() - start_time), steps, "Deadlock"
            state = next_state
            
        return True, (time.time() - start_time), steps, "Safe Sequence Found"