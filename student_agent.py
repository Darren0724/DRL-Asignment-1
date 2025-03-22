import numpy as np
import pickle
#from taxi_env import SimpleTaxiEnv
import random 

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_state_key(obs):
    return obs

def get_action(obs, env):
    """Use global variables from env to decide PICKUP/DROPOFF."""
    state = get_state_key(obs)
    
    # Check if we should PICKUP or DROPOFF based on global variables
    if env.taxi_pos == env.passenger_loc and not env.passenger_picked_up:
        return 4  # PICKUP
    if env.passenger_picked_up and env.taxi_pos == env.destination:
        return 5  # DROPOFF
    
    # Otherwise, use Q-table for movement, avoiding walls
    if state not in q_table:
        # Random move, avoiding walls based on state
        valid_actions = [i for i in range(4) if not obs[i]]
        return random.choice(valid_actions) if valid_actions else random.randint(0, 3)
    return np.argmax(q_table[state])