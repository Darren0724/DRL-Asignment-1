import numpy as np
import pickle
import random
from wall_1000_env import SimpleTaxiEnv
from tqdm import tqdm
# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 2000  # Number of training episodes

# Initialize Q-table
q_table = {}

def get_state_key(obs):
    """Convert observation to a state tuple for Q-table."""
    taxi_row, taxi_col, r1, c1, r2, c2, r3, c3, r4, c4, \
    obstacle_north, obstacle_south, obstacle_east, obstacle_west, \
    passenger_look, destination_look = obs
    return (
        r1 - taxi_row, c1 - taxi_col,  # Relative to R
        r2 - taxi_row, c2 - taxi_col,  # Relative to G
        r3 - taxi_row, c3 - taxi_col,  # Relative to Y
        r4 - taxi_row, c4 - taxi_col,  # Relative to B
        obstacle_north, obstacle_south, obstacle_east, obstacle_west,
        passenger_look, destination_look
    )

def train_agent():
    global q_table
    for episode in tqdm(range(episodes)):
        # Random grid size between 5 and 10 for generalization
        grid_size = random.randint(5, 10)
        env = SimpleTaxiEnv(grid_size=grid_size, fuel_limit=5000)
        obs, _ = env.reset()
        state = get_state_key(obs)
        done = False

        while not done:
            # Initialize Q-values for new states
            if state not in q_table:
                q_table[state] = np.zeros(6)  # 6 actions

            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, 5)
            else:
                action = np.argmax(q_table[state])

            # Take step and get next observation
            next_obs, reward, done, _ = env.step(action)
            next_state = get_state_key(next_obs)

            # Initialize Q-values for next state if unseen
            if next_state not in q_table:
                q_table[next_state] = np.zeros(6)

            # Update Q-table after one step
            q_table[state][action] += alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state][action]
            )

            # Move to next state
            state = next_state
            obs = next_obs

        if (episode + 1) % 1000 == 0:
            print(f"Episode {episode + 1}/{episodes} completed")

    # Save Q-table
    with open("q_table.pkl", "wb") as f:
        pickle.dump(q_table, f)
    print("Training completed. Q-table saved to q_table.pkl")

if __name__ == "__main__":
    train_agent()