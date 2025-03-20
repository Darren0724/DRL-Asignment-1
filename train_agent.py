import numpy as np
import pickle
import random
from simple_custom_taxi_env import SimpleTaxiEnv  # Assume this is saved as simple_customtaxi_env.py
from tqdm import tqdm
import time 
# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 3000  # Number of training episodes

# Initialize Q-table
q_table = {}

# Training function
def train_agent():

    for episode in tqdm(range(episodes)):
        # Random grid size between 5 and 10 for generalization
        grid_size = random.randint(5, 10)
        env = SimpleTaxiEnv(grid_size=grid_size, fuel_limit=5000)
        obs, _ = env.reset()
        done = False
        while not done:
            # State representation
            taxi_row, taxi_col, r1, c1, r2, c2, r3, c3, r4, c4, \
            obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
            passenger_look, destination_look = obs
            state = (
                r1 - taxi_row, c1 - taxi_col,
                r2 - taxi_row, c2 - taxi_col,
                r3 - taxi_row, c3 - taxi_col,
                r4 - taxi_row, c4 - taxi_col,
                obstacle_north, obstacle_south, obstacle_east, obstacle_west,
                passenger_look, destination_look
            )
            state_tuple = tuple(state)

            # Initialize Q-values for new states
            if state_tuple not in q_table:
                q_table[state_tuple] = np.zeros(6)  # 6 actions

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                action = np.argmax(q_table[state_tuple])

            # Take action
            next_obs, reward, done, _ = env.step(action)

            # Update Q-table
            next_state = tuple(
                (next_obs[2] - next_obs[0], next_obs[3] - next_obs[1],  # R
                 next_obs[4] - next_obs[0], next_obs[5] - next_obs[1],  # G
                 next_obs[6] - next_obs[0], next_obs[7] - next_obs[1],  # Y
                 next_obs[8] - next_obs[0], next_obs[9] - next_obs[1],  # B
                 next_obs[10], next_obs[11], next_obs[12], next_obs[13],  # Obstacles
                 next_obs[14], next_obs[15])  # Passenger, Destination
            )
            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)

            # Q-learning update
            q_table[state_tuple][action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state_tuple][action]
            )
            obs = next_obs

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes} completed")

    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Training completed. Q-table saved to q_table.pkl")

if __name__ == "__main__":
    train_agent()