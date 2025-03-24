import numpy as np
import pickle
import random
from time import sleep

# Load the trained Q-table
with open("q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

# Global variables (consistent with train_agent.py)
move_history = {}
now_doing = 0  # 0: moving1, 1: moving2, 2: moving3, 3: moving4, 4: move pickup, 5: pickup, 6: move dropoff, 7: dropoff
goal_r = -1
goal_c = -1
now_r = 0
now_c = 0
row = [0]*4
col = [0]*4
st = -1
ed = -1
last_action = 0
epsilon = 0.1  # Exploration rate for epsilon-greedy
alpha = 0.1    # Learning rate for Q-table update
gamma = 0.99   # Discount factor for Q-table update
rec_reward = 0
rec_state = None
step = 0
def sign(x):   
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def sign2(x):   
    if x > 3:
        return 2
    elif x > 0:
        return 1
    elif x < -3:
        return -2
    elif x < 0:
        return -1
    else:
        return 0

def get_state_key(obs):
    global move_history, now_r, now_c
    obstacle_north = obs[10]
    obstacle_south = obs[11]
    obstacle_east = obs[12]
    obstacle_west = obs[13]
    
    north_state = 1 if obstacle_north else (2 if (now_r, now_c, 1) in move_history else 0)
    south_state = 1 if obstacle_south else (2 if (now_r, now_c, 0) in move_history else 0)
    east_state = 1 if obstacle_east else (2 if (now_r, now_c, 2) in move_history else 0)
    west_state = 1 if obstacle_west else (2 if (now_r, now_c, 3) in move_history else 0)
    
    return (north_state, south_state, east_state, west_state, sign(goal_r - now_r), sign(goal_c - now_c))

def get_action(obs, reward=None, next_obs=None):
    
    global now_doing, goal_r, goal_c, now_r, now_c, row, col, st, ed, last_action, q_table, move_history, rec_reward, rec_state, step, epsilon
    step += 1
    if step > 200:
        epsilon = 0.2
    if rec_reward is not None and rec_state is not None:
        next_state = get_state_key(obs)
        #print('yes')
        if next_state not in q_table:
            q_table[next_state] = np.zeros(6)
        q_table[rec_state][last_action] += alpha * (
            rec_reward + gamma * np.max(q_table[next_state]) - q_table[rec_state][last_action]
        )
    # Update station positions and current position
    for i in range(4):
        row[i] = obs[2*i+2]
        col[i] = obs[2*i+3]
    now_r = obs[0]
    now_c = obs[1]
    
    # Initialize goal if not set (start of episode)
    if goal_r == -1:
        goal_r = row[0]
        goal_c = col[0]
        move_history.clear()  # Reset history at the start of an episode
    
    # State machine transitions (consistent with train_agent.py)
    if now_doing < 4:
        if now_r == goal_r and now_c == goal_c:
            if obs[14] == 1:
                st = now_doing
            if obs[15] == 1:
                ed = now_doing
            if now_doing == 3:
                now_doing = 4
                goal_r = row[st]
                goal_c = col[st]
            else:
                now_doing += 1
                goal_r = row[now_doing]
                goal_c = col[now_doing]
    elif now_doing == 4:
        if now_r == goal_r and now_c == goal_c:
            now_doing = 5
    elif now_doing == 5:
        if now_r == goal_r and now_c == goal_c and last_action == 4:
            now_doing = 6
            goal_r = row[ed]
            goal_c = col[ed]
    elif now_doing == 6:
        if now_r == goal_r and now_c == goal_c:
            now_doing = 7
    elif now_doing == 7:
        if now_r == goal_r and now_c == goal_c and last_action == 5:
            now_doing = 8
    
    # Define valid actions based on now_doing
    valid_actions = [i for i in range(4)]  # Movement actions by default
    if now_doing == 5 and now_r == goal_r and now_c == goal_c:
        valid_actions = [4]  # Only PICKUP when at pickup location
    elif now_doing == 7 and now_r == goal_r and now_c == goal_c:
        valid_actions = [5]  # Only DROPOFF when at dropoff location
    
    # Get current state
    state = get_state_key(obs)
    
    # ε-greedy action selection
    if state not in q_table:
        q_table[state] = np.zeros(6)  # Initialize if not in table
    q_values = q_table[state]
    
    if random.random() < epsilon:
        last_action = random.choice(valid_actions)  # Explore
    else:
        last_action = max(valid_actions, key=lambda a: q_values[a])  # Exploit
    
    # Update move history for movement actions
    if last_action in [0, 1, 2, 3] and not (last_action == 0 and obs[11] or last_action == 1 and obs[10] or last_action == 2 and obs[12] or last_action == 3 and obs[13]):
        move_history[(now_r, now_c, last_action)] = True
    shaped_reward = -0.1
    if last_action == 0:  # South
        if obs[11] == 1:
            shaped_reward = -100
    if last_action == 1:  # South
        if obs[10] == 1:
            shaped_reward = -100
    if last_action == 2:  # South
        if obs[12] == 1:
            shaped_reward = -100
    if last_action == 3:  # South
        if obs[13] == 1:
            shaped_reward = -100
    elif last_action == 4:  # PICKUP
        if now_doing == 5 and now_r == goal_r and now_c == goal_c:
            shaped_reward += 0
        else:
            shaped_reward = -100
    elif last_action == 5:  # DROPOFF
        if now_doing == 7 and now_r == goal_r and now_c == goal_c:
            shaped_reward += 0
        else:
            shaped_reward = -100

    # Update Q-table if reward and next_obs are provided
    reward = shaped_reward
    rec_reward = reward 
    rec_state = state
    
    return last_action

if __name__ == "__main__":
    print("Q-table loaded with size:", len(q_table))