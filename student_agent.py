import numpy as np
import pickle
import random 
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_state_key(obs):
    # Extract only the surrounding wall info from the full observation
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east = obs[12]
    obstacle_west = obs[13]
    return (obstacle_north, obstacle_south, obstacle_east, obstacle_west)

def get_action(obs):
    state = get_state_key(obs)
    if state not in q_table:
        # Avoid walls for unseen states
        valid_actions = [i for i in range(4) if not obs[10+i]]  # Only move actions (0-3)
        return random.choice(valid_actions) if valid_actions else 0  # Default to 0 if no valid moves
    return np.argmax(q_table[state])