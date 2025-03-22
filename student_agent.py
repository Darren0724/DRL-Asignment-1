import numpy as np
import pickle

# Load the pre-trained Q-table
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

def get_state_key(obs):
    """Convert observation to a state tuple for Q-table."""
    taxi_row, taxi_col, \
    r_rel_row, r_rel_col, g_rel_row, g_rel_col, y_rel_row, y_rel_col, b_rel_row, b_rel_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    return (
        r_rel_row, r_rel_col,
        g_rel_row, g_rel_col,
        y_rel_row, y_rel_col,
        b_rel_row, b_rel_col,
        obstacle_north, obstacle_south, obstacle_east, obstacle_west,
        passenger_look, destination_look
    )

def get_action(obs):
    """Select the best action using the trained Q-table."""
    state = get_state_key(obs)
    if state not in q_table:
        return np.random.randint(0, 6)  # Random action for unseen states
    return np.argmax(q_table[state])