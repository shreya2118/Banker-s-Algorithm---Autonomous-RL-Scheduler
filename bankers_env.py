import numpy as np

class BankersEnvironment:
    def __init__(self, n_processes, m_resources, alloc, max_req, avail):
        self.n = n_processes
        self.m = m_resources
        # Initial state backup (to reset later)
        self.start_alloc = np.array(alloc)
        self.max_req = np.array(max_req)
        self.initial_avail = np.array(avail)
        self.start_max = np.array(max_req)
        self.start_avail = np.array(avail)
        self.alloc = None
        self.avail = None
        self.finished = None
        self.reset()


    def reset(self):
        """Resets the simulator to the start."""
        self.alloc = self.start_alloc.copy()
        self.avail = self.start_avail.copy()
        self.finished = [False] * self.n
        # State is represented as a tuple of finished processes (e.g., (0, 0, 1, 0, 0))
        return self.get_state()
    def get_state(self):
        return tuple(int(f) for f in self.finished)

    def step(self, action):
        # 1. CHECK: Is this process already finished?
        if self.finished[action]:
            # PENALTY! You cannot run a finished process again.
            # Reward = -100 (Severe punishment)
            # Done = False (Keep trying to find a valid move, or True to stop immediate)
            return self.get_state(), -100, False 

        # 2. CHECK: Is the move safe? (Banker's Logic)
        need = self.max_req[action] - self.alloc[action]
        
        if np.all(need <= self.avail):
            # SAFE MOVE
            self.avail += self.alloc[action] # Release resources
            self.alloc[action] = 0
            self.finished[action] = True     # <--- MARK AS DONE
            
            # Check if ALL are done (Victory)
            if all(self.finished):
                return self.get_state(), 100, True # BIG REWARD for solving it
            
            # Small reward for a good step
            return self.get_state(), 10, False
        
        else:
            # UNSAFE MOVE (Deadlock / Not enough resources)
            # Penalty = -50
            return self.get_state(), -50, False