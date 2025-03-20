import numpy as np
import pickle
import random

# Load the pre-trained Q-table
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    # If Q-table is missing, initialize an empty one (though training is recommended)
    q_table = {}

def get_action(obs):
    """
    Select an action based on the current observation using the Q-table.
    Args:
        obs: Tuple containing (taxi_row, taxi_col, r1, c1, r2, c2, r3, c3, r4, c4,
                             obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                             passenger_look, destination_look)
    Returns:
        int: Action (0-5)
    """
    # Unpack observation
    taxi_row, taxi_col, r1, c1, r2, c2, r3, c3, r4, c4, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs

    # Create state representation: relative distances to stations + environmental info
    state = (
        obstacle_north, obstacle_south, obstacle_east, obstacle_west,
    )

    # Convert state to tuple for hashing
    state_tuple = tuple(state)

    # Check if state exists in Q-table; if not, return a random action
    if state_tuple not in q_table:
        return random.randint(0, 5)  # Fallback strategy for unseen states

    # Select the action with the highest Q-value
    action = np.argmax(q_table[state_tuple])
    return action