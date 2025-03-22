import numpy as np
import pickle
import random 
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_state_key(obs):
    return obs

def get_action(obs):
    state = get_state_key(obs)
    if state not in q_table:
        # Avoid walls for unseen states
        valid_actions = [i for i in range(4) if not obs[i]]  # Only move actions (0-3)
        if valid_actions:
            return random.choice(valid_actions)
        return random.randint(0, 5)  # Fallback
    return np.argmax(q_table[state])