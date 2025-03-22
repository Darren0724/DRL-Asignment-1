import numpy as np
import pickle

with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_state_key(obs):
    return obs

def get_action(obs):
    state = get_state_key(obs)
    if state not in q_table:
        if obs[-3]:  # passenger_here
            return 4
        if obs[-2] and state[-1]:  # destination_here and passenger_picked
            return 5
        return np.random.randint(0, 4)
    return np.argmax(q_table[state])