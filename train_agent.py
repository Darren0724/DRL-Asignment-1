import numpy as np
import pickle
import random
import env1 as env2
from tqdm import tqdm

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 3000  # Number of training episodes

# Initialize Q-table
try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
        print("Q-table loaded:", q_table)
except FileNotFoundError:
    q_table = {}

def get_state_key(obs):
    """Convert observation to a state tuple for Q-table."""
    taxi_row, taxi_col, \
    r_rel_row, r_rel_col, g_rel_row, g_rel_col, y_rel_row, y_rel_col, b_rel_row, b_rel_col, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    return (
        r_rel_row, r_rel_col,  # Relative to R
        g_rel_row, g_rel_col,  # Relative to G
        y_rel_row, y_rel_col,  # Relative to Y
        b_rel_row, b_rel_col,  # Relative to B
        obstacle_north, obstacle_south, obstacle_east, obstacle_west,
        passenger_look, destination_look
    )

def train_agent():
    global q_table
    for episode in tqdm(range(episodes)):
        grid_size = random.randint(5, 10)  # Random grid size for generalization
        env = env2.SimpleTaxiEnv(grid_size=grid_size, fuel_limit=5000)
        obs, _ = env.reset()
        state = get_state_key(obs)
        
        if state not in q_table:
            q_table[state] = np.zeros(6)  # 6 actions
        
        done = False
        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                action = np.argmax(q_table[state])

            # Take step
            next_obs, reward, done, _ = env.step(action)
            next_state = get_state_key(next_obs)

            # Initialize Q-values for next state if unseen
            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)

            # Update Q-table
            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )

            # Move to next state
            state = next_state
            obs = next_obs

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} completed")

    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Training completed. Q-table saved to q_table.pkl")

if __name__ == "__main__":
    train_agent()